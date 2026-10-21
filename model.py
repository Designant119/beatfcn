#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v22 模型定义（独立模块，训练和推理共用）。

把模型定义从训练脚本抽离，推理时不需要带整个训练代码库。
包含：常量、compute_flux、PositionalEncoding、BeatFCNContext。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

SR = 22050
HOP = 256
HR = SR / HOP
TOL = 0.030


# ---------- 谱通量（音头近似，在线算，零重提取） ----------
def compute_flux(mel):
    """mel (T,81) log-mel → flux (T,1)：帧间正差分均值 = 能量上升 = 音头线索。
    除以全局常驻尺度归一（log-mel 帧差通常在 0~0.5），clip 防尖峰。"""
    d = np.diff(mel, axis=0)
    flux = np.maximum(d, 0.0).mean(axis=1, keepdims=True)   # (T-1,1)
    flux = np.concatenate([flux, flux[-1:]], axis=0)        # (T,1) 末帧补复制
    return (flux / 0.1).clip(0, 2).astype(np.float32)


# ---------- v22 模型（Transformer；CNN下采样+全局attention+上采样） ----------
class PositionalEncoding(nn.Module):
    """正弦位置编码，给 Transformer 提供顺序信息。支持动态扩展（长歌不越界）。"""
    def __init__(self, d_model, max_len=100000):
        super().__init__()
        self.d_model = d_model
        self._max_len = max_len
        pe = self._make_pe(d_model, max_len)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    @staticmethod
    def _make_pe(d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    def forward(self, x):
        T = x.shape[1]
        if T > self.pe.shape[1]:
            if not getattr(self, '_warned_trunc', False):
                import warnings
                warnings.warn(f'PositionalEncoding: 输入长度{T}超过max_len={self.pe.shape[1]}，已截断（约19分钟以上，极少见）')
                self._warned_trunc = True
            T = self.pe.shape[1]  # 超长截断（19分钟以上，极少见）
            x = x[:, :T, :]
        return x + self.pe[:, :T, :]


class BeatFCNContext(nn.Module):
    """v22: CNN提取局部特征→Transformer全局attention（全分辨率）→输出。
    支持条件采音：输入已采音符（grid轴0/1/2/3/4），模型看前半段采后半段。
    o1=有没有音，o2=押数分类(1-4)。"""
    def __init__(self, n_mels=81, ch=96, ctx_off=True, n_pitch=46,
                 n_layers=4, nhead=4, downsample=1, n_press=2):
        super().__init__()
        self.ctx_off = ctx_off
        self.n_pitch = n_pitch
        self.downsample = downsample
        self.n_press = n_press  # one-hot类别数(0=无音,1=单押,2=双押)
        # 输入 = mel(81=80mel+1flux) + pitch(n_pitch) + notes_onehot(n_press+1) + mask(1)
        notes_dim = n_press + 1 + 1  # one-hot(0..n_press) + mask
        self.inp = nn.Sequential(
            nn.Conv1d(n_mels + n_pitch + notes_dim, ch, 5, padding=2), nn.GroupNorm(1, ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, stride=1, padding=2), nn.GroupNorm(1, ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, stride=1, padding=2), nn.GroupNorm(1, ch), nn.ReLU())
        self.pos_enc = PositionalEncoding(ch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ch, nhead=nhead, dim_feedforward=ch*4,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        def mk_up(out_ch=1):
            return nn.Sequential(
                nn.ConstantPad1d((2, 0), 0), nn.Conv1d(ch, ch, 3), nn.ReLU(),
                nn.Conv1d(ch, out_ch, 1))
        self.up1 = mk_up(1)       # 有没有音 (BCE二分类)
        self.up2 = mk_up(1)       # 是不是双押 (BCE二分类，只在有音位置算loss)

    def notes_to_frame(self, notes, grid, T):
        """notes (B,G) 0/1/2/3/4 → frame轴 (B,T,notes_dim) = one-hot(n_press+1) + mask(1)。
        遮罩位置(notes=-1): one-hot全0 + mask=0；有参考位置: one-hot正常 + mask=1。"""
        B, G = notes.shape
        device = notes.device
        if grid.dim() == 1:
            grid = grid.unsqueeze(0).expand(B, -1)
        frame_idx = (grid * HR).long().clamp(0, T - 1)
        valid = (notes >= 0).float()
        notes_clamped = notes.clamp(0, self.n_press)
        one_hot = torch.zeros(B, G, self.n_press + 1, device=device)
        one_hot.scatter_(2, notes_clamped.unsqueeze(-1).long(), 1.0)
        one_hot = one_hot * valid.unsqueeze(-1)
        notes_ch = torch.cat([one_hot, valid.unsqueeze(-1)], dim=-1)  # (B,G,n_press+2)
        notes_frame = torch.zeros(B, T, self.n_press + 2, device=device)
        # 参考段每格的押数（0/1/2），直接投影到帧轴，用amax取每帧最大押数
        # 这样参考段双押格(label=2)直接投影成frame_press=2，不会被压扁成单押
        frame_press_grid = notes_clamped * valid  # (B,G) 参考段押数，遮罩段为0
        frame_press = torch.zeros(B, T, device=device)
        frame_press.scatter_reduce_(1, frame_idx, frame_press_grid, reduce='amax', include_self=True)
        # frame_valid用valid投影，不是frame_press>0——参考空格(notes=0,valid=1)也要标成有参考
        # 否则参考空格和遮罩位在frame轴上一模一样，模型分不清"采过没音"和"没采要猜"
        frame_valid = torch.zeros(B, T, device=device)
        frame_valid.scatter_reduce_(1, frame_idx, valid, reduce='amax', include_self=True)
        # 转成one-hot，遮罩位乘frame_valid清零（和grid轴的one_hot*valid一致）
        frame_onehot = torch.zeros(B, T, self.n_press + 1, device=device)
        frame_onehot.scatter_(2, frame_press.long().unsqueeze(-1), 1.0)
        frame_onehot = frame_onehot * frame_valid.unsqueeze(-1)  # 遮罩位one_hot全0
        notes_frame = torch.cat([frame_onehot, frame_valid.unsqueeze(-1)], dim=-1)
        return notes_frame

    def to_grid_feat(self, h, grid):
        """h (B,ch,T) → grid轴特征 (B,ch,G)。grid 可以是 (G,) 或 (B,G)。"""
        B, ch, T = h.shape
        if grid.dim() == 1:
            grid = grid.unsqueeze(0).expand(B, -1)  # (B,G)
        G = grid.shape[1]
        x = (grid * HR).clamp(0, T - 1)  # (B,G)
        x0 = x.floor().long()  # (B,G)
        x1 = (x0 + 1).clamp(max=T - 1)  # (B,G)
        w = (x - x0.float()).unsqueeze(1)  # (B,1,G)
        # batch gather：从 h 的 T 维按 x0/x1 索引
        h0 = torch.gather(h, 2, x0.unsqueeze(1).expand(-1, ch, -1))  # (B,ch,G)
        h1 = torch.gather(h, 2, x1.unsqueeze(1).expand(-1, ch, -1))  # (B,ch,G)
        return h0 * (1 - w) + h1 * w

    def forward(self, mel, grid, notes=None, src_key_padding_mask=None):
        # mel (B,T,81+n_pitch), notes (B,G) 0/1/2 或 -1(遮罩)
        # src_key_padding_mask (B,T): True=padding位置，不参与注意力
        T = mel.shape[1]
        if notes is not None:
            notes_f = self.notes_to_frame(notes, grid, T)  # (B,T,n_press+2)
            mel = torch.cat([mel, notes_f], dim=-1)
        else:
            # 无条件采音：全零音符通道 + mask=0（全遮罩）
            zeros = torch.zeros(mel.shape[0], T, self.n_press + 2, device=mel.device)
            mel = torch.cat([mel, zeros], dim=-1)
        h = self.inp(mel.transpose(1, 2))                    # (B,ch,T)
        h = h.transpose(1, 2)                                 # (B,T,ch)
        h = self.pos_enc(h)
        # Gradient Checkpointing：训练时省显存，推理时不用（更快）
        if self.training:
            h = grad_ckpt(self.transformer, h, src_key_padding_mask=src_key_padding_mask, use_reentrant=False)
        else:
            h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B,T,ch)
        h = h.transpose(1, 2)                                 # (B,ch,T)
        hg = self.to_grid_feat(h, grid)                       # (B,ch,G)
        o1 = self.up1(hg).squeeze(1)                          # (B,G) 有没有音
        o2 = self.up2(hg).squeeze(1)                          # (B,G) 是不是双押
        return o1, o2

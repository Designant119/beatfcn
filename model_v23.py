#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v23 模型：网格对齐 token 化（Beat-Token Transformer）。

v22 的瓶颈：Transformer 直接在音频帧上做全局注意力（3 分钟歌≈15k 帧，O(T²)），
而真正要输出的只是网格格子上的“有没有音 / 是不是双押”。

v23 思路：
  1. 帧级小卷积 stem（stride 1，ch=64，2 层）提取局部音频特征——线性复杂度，很便宜。
  2. 按节拍网格把帧特征“收拢”成 token：每 K 个格子（默认 K=8，128 分网格下 = 1/4 拍）
     一个 token，在 token 覆盖的时间窗内等距采样 S 个点（线性插值）→ 拼接 → Linear。
     变速歌也天然支持（窗口由网格时间决定，不假设恒定 BPM）。
  3. 条件采音：token 内 K 个格子的已采音符 one-hot(3)+mask(1) 展平后线性投影加到 token 上。
  4. Transformer 只在 token 上做全局注意力：3 分钟歌 ≈ 1k token（v22 的 1/15），
     注意力开销降到 ~1/200，并且不再需要 gradient checkpointing（省掉重算）。
  5. 输出头逐格细化：token 上下文特征 + 该格时间点的帧特征（插值）→ 小 MLP → o1/o2，
     保留格级时间精度。
输入 / 输出接口与 v22 BeatFCNContext 相同：forward(mel, grid, notes) -> (o1, o2)，形状 (B,G)。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 22050
HOP = 256
HR = SR / HOP


def _interp(h, t_frames):
    """h (B,C,T)，t_frames (B,N) 帧坐标（浮点）→ (B,C,N) 线性插值。"""
    B, C, T = h.shape
    x = t_frames.clamp(0, T - 1)
    x0 = x.floor().long()
    x1 = (x0 + 1).clamp(max=T - 1)
    w = (x - x0.to(x.dtype)).unsqueeze(1).to(h.dtype)
    h0 = torch.gather(h, 2, x0.unsqueeze(1).expand(-1, C, -1))
    h1 = torch.gather(h, 2, x1.unsqueeze(1).expand(-1, C, -1))
    return h0 * (1 - w) + h1 * w


def sinusoid(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class BeatTokenNet(nn.Module):
    def __init__(self, n_mels=81, n_pitch=46, stem_ch=48, d_model=96, n_layers=4, nhead=4,
                 cells_per_token=16, samples_per_token=16, n_press=2, dropout=0.1, stem='sep'):
        super().__init__()
        self.n_press = n_press
        self.K = cells_per_token
        self.S = samples_per_token
        cin = n_mels + n_pitch
        if stem == 'sep':  # 便宜：k3 卷积 + 深度可分离 k5（感受野 7 帧≈81ms）
            self.stem = nn.Sequential(
                nn.Conv1d(cin, stem_ch, 3, padding=1), nn.GroupNorm(1, stem_ch), nn.GELU(),
                nn.Conv1d(stem_ch, stem_ch, 5, padding=2, groups=stem_ch),
                nn.Conv1d(stem_ch, stem_ch, 1), nn.GroupNorm(1, stem_ch), nn.GELU())
        else:      # 'conv'：两层普通 k5 卷积（更贵）
            self.stem = nn.Sequential(
                nn.Conv1d(cin, stem_ch, 5, padding=2), nn.GroupNorm(1, stem_ch), nn.GELU(),
                nn.Conv1d(stem_ch, stem_ch, 5, padding=2), nn.GroupNorm(1, stem_ch), nn.GELU())
        self.pool = nn.Linear(stem_ch * self.S, d_model)
        nd = n_press + 2
        self.note_proj = nn.Linear(nd * self.K, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout,
                                           batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)
        # 逐格头：token 上下文 + 格内位置编码 + 格时间点帧特征 + 该格音符输入
        self.cell_pos = nn.Parameter(torch.zeros(self.K, d_model))
        self.cell_audio = nn.Linear(stem_ch, d_model)
        self.cell_note = nn.Linear(nd, d_model)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def notes_onehot(self, notes):
        valid = (notes >= 0).to(torch.float32)
        oh = F.one_hot(notes.clamp(0, self.n_press).long(), self.n_press + 1).to(torch.float32)
        oh = oh * valid.unsqueeze(-1)
        return torch.cat([oh, valid.unsqueeze(-1)], -1)  # (B,G,nd)

    def forward(self, mel, grid, notes=None, grid_mask=None, src_key_padding_mask=None):
        """mel (B,T,81+n_pitch)；grid (G,) 或 (B,G) 秒；notes (B,G) 0/1/2 或 -1(遮罩)。
        grid_mask (B,G) True=有效格（batch padding 用；None=全有效）。返回 o1,o2 (B,G)。"""
        B, T, _ = mel.shape
        if grid.dim() == 1:
            grid = grid.unsqueeze(0).expand(B, -1)
        G = grid.shape[1]
        K, S = self.K, self.S
        dev = mel.device
        if notes is None:
            notes = torch.full((B, G), -1.0, device=dev)
        if grid_mask is None:
            grid_mask = torch.ones(B, G, dtype=torch.bool, device=dev)
        # 补齐到 K 的整数倍
        Gp = (G + K - 1) // K * K
        pad = Gp - G
        if pad:
            step = (grid[:, -1:] - grid[:, -2:-1]) if G > 1 else torch.full((B, 1), 0.015, device=dev)
            ext = grid[:, -1:] + step * torch.arange(1, pad + 1, device=dev).unsqueeze(0)
            grid = torch.cat([grid, ext], 1)
            notes = torch.cat([notes, torch.full((B, pad), -1.0, device=dev)], 1)
            grid_mask = torch.cat([grid_mask, torch.zeros(B, pad, dtype=torch.bool, device=dev)], 1)
        N = Gp // K
        h = self.stem(mel.transpose(1, 2))  # (B,C,T)
        C = h.shape[1]
        # token 时间窗：[第一个格 - 半格, 最后一个格 + 半格]
        gt = grid.view(B, N, K)
        half = (gt[:, :, -1] - gt[:, :, 0]) / max(K - 1, 1) / 2
        t0 = gt[:, :, 0] - half
        t1 = gt[:, :, -1] + half
        frac = (torch.arange(S, device=dev, dtype=grid.dtype) + 0.5) / S
        ts = t0.unsqueeze(-1) + (t1 - t0).unsqueeze(-1) * frac  # (B,N,S)
        pooled = _interp(h, (ts * HR).reshape(B, N * S))         # (B,C,N*S)
        pooled = pooled.view(B, C, N, S).permute(0, 2, 3, 1).reshape(B, N, S * C)
        nh = self.notes_onehot(notes)                             # (B,Gp,nd)
        x = self.pool(pooled) + self.note_proj(nh.view(B, N, -1))
        x = self.norm_in(x + sinusoid(N, x.shape[-1], dev).unsqueeze(0))
        tok_mask = grid_mask.view(B, N, K).any(-1)                # (B,N) True=有效
        x = self.transformer(x, src_key_padding_mask=~tok_mask)
        x = self.norm_out(x)                                      # (B,N,D)
        # 逐格细化
        ca = _interp(h, grid * HR).transpose(1, 2)                # (B,Gp,C)
        cell = (x.unsqueeze(2) + self.cell_pos).reshape(B, Gp, -1) + self.cell_audio(ca) + self.cell_note(nh)
        o = self.head(cell)                                       # (B,Gp,2)
        return o[:, :G, 0], o[:, :G, 1]


def load_any_model(ckpt_path, device='cpu', ch=96, n_pitch=46):
    """按 ckpt 的 hparams 自动构建 v22 (BeatFCNContext) 或 v23 (BeatTokenNet)。
    返回 (model, hparams, th1)；th1 = ckpt 里保存的最优阈值（v23 才有，v22 返回 None）。"""
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = d['model'] if isinstance(d, dict) and 'model' in d else d
    hp = d.get('hparams', {}) if isinstance(d, dict) else {}
    if hp.get('arch') == 'v23':
        m = BeatTokenNet(n_mels=hp.get('n_mels', 81), n_pitch=hp.get('n_pitch', 46), stem_ch=hp.get('stem_ch', 48),
                         d_model=hp.get('d_model', 96), n_layers=hp.get('n_layers', 4), nhead=hp.get('nhead', 4),
                         cells_per_token=hp.get('cells_per_token', 16),
                         samples_per_token=hp.get('samples_per_token', 16), stem=hp.get('stem', 'sep'))
    else:
        from model import BeatFCNContext
        m = BeatFCNContext(n_mels=hp.get('n_mels', 81), ch=hp.get('ch', ch), ctx_off=True,
                           n_pitch=hp.get('n_pitch', n_pitch))
    m.load_state_dict(sd)
    return m.to(device).eval(), hp, (d.get('th1') if isinstance(d, dict) else None)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v22 —— Transformer 全局注意力版 + 条件采音。

结构：CNN(inp, LayerNorm) → Transformer(4层4头, 全分辨率) → to_grid_feat → 输出
- stride=1 不下降采样，Transformer 直接看 11.6ms 帧距（满分辨率）
- 全局 self-attention，能看全曲上下文
- 加了正弦位置编码（Transformer 本身没有顺序感知）
- 条件采音：输入已采音符，模型看前半段采后半段
- 纯采音 ctx-off，输出每个格 0/1/2
"""
import os, sys, json, argparse, time, random, glob, math, signal
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import DataLoader, Sampler

SR = 22050; HOP = 256; HR = SR / HOP; TOL = 0.030

# 模型定义已抽离到 model.py，训练和推理共用
from model import BeatFCNContext, compute_flux, PositionalEncoding


# ---------- 数据（逐歌；v16：_load 时在线拼 flux） ----------
class Songs:
    def __init__(self, split, data_dir, split_file, pitch_dir=None, pitch_mode=None,
                 cache_size=32, n_pitch=0, flux=True):
        sp = json.load(open(split_file))
        base_ids = [k for k, v in sp.items() if v == split]
        # 展开多难度：如果 {id}.npz 不存在但 {id}_*.npz 存在，把所有难度都加进来
        # 兼容旧数据（单难度 {id}.npz）和新数据（多难度 {id}_ez.npz 等）
        self.fids = []
        for bid in base_ids:
            single = f"{data_dir}/{bid}.npz"
            if os.path.exists(single):
                self.fids.append(bid)  # 旧格式：单难度
            else:
                # 新格式：扫描所有难度
                diffs = sorted(glob.glob(f"{data_dir}/{bid}_*.npz"))
                for d in diffs:
                    fid = os.path.basename(d)[:-4]  # 去掉.npz
                    self.fids.append(fid)
        self.data_dir = data_dir
        self.pitch_dir = pitch_dir
        self.pitch_mode = pitch_mode
        self.n_pitch = n_pitch
        self.flux = flux
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.bad, self.bad_warned = {}, False
    def _load(self, fid):
        if fid in self.cache:
            self.cache.move_to_end(fid); return self.cache[fid]
        if fid in self.bad:
            return None
        try:
            z = np.load(f"{self.data_dir}/{fid}.npz")
            mel, grid, label = z["mel"], z["grid"], z["label"]
            # 只取前80维mel（去掉相位维，相位是数据泄漏，推理时没有）
            mel = mel[:, :80]
            if self.flux:
                mel = np.concatenate([mel, compute_flux(mel)], axis=1).astype(np.float32)  # (T,81)
            if self.pitch_dir:
                # pitch按歌曲提取（不分难度），从fid提取base_id（第一个_前面的数字ID）
                # 兼容旧格式fid=base_id、新格式fid=base_id_diff、新格式fid=base_id_song_[diff]
                if '_' in fid and not os.path.exists(f"{self.pitch_dir}/{fid}.npz"):
                    base_id = fid.split('_', 1)[0]  # 取第一个_前面的部分作为歌曲ID
                else:
                    base_id = fid
                pp = f"{self.pitch_dir}/{base_id}.npz"
                if os.path.exists(pp):
                    pz = np.load(pp)
                    pitch = pz["pitch"].astype(np.float32)
                    if pitch.shape[1] != self.n_pitch:
                        raise ValueError(f"pitch 通道={pitch.shape[1]} ≠ n_pitch={self.n_pitch}")
                else:
                    return None  # pitch缺失，跳过这首歌（不填全0）
                if pitch.shape[0] != mel.shape[0]:
                    n = min(pitch.shape[0], mel.shape[0])
                    mask = grid < (n * HOP / SR)
                    mel = mel[:n]
                    pitch = pitch[:n]
                    grid = grid[mask]
                    label = label[mask]
                if len(grid) != len(label):
                    raise ValueError(f"帧对齐后 grid({len(grid)}) 与 label({len(label)}) 长度不一致")
                # pitch每通道标准化（跨时间轴），和mel的标准化方式一致
                pitch_mean = pitch.mean(axis=0, keepdims=True)
                pitch_std = pitch.std(axis=0, keepdims=True) + 1e-5
                pitch = (pitch - pitch_mean) / pitch_std
                mel = np.concatenate([mel, pitch], axis=1).astype(np.float32)
        except Exception as e:
            self.bad[fid] = str(e)
            if not self.bad_warned:
                print(f"[数据自检] 坏文件示例 {fid}: {e}（后续同名跳过，不崩训练）", flush=True)
                self.bad_warned = True
            return None
        item = (mel, grid, label)
        self.cache[fid] = item
        if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return item
    def __len__(self): return len(self.fids)
    def __getitem__(self, i):
        item = self._load(self.fids[i])
        if item is None:
            return None
        mel, grid, label = item
        return mel.astype(np.float32), grid.astype(np.float32), label.astype(np.int64)


# ---------- 长度分桶 Sampler（减少batch内padding） ----------
class LengthBucketSampler(Sampler):
    def __init__(self, song_lens, batch_size, bucket_size=200, seed=0, skip=0):
        self.song_lens = song_lens
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.seed = seed
        self.skip = skip  # resume时跳过已处理的batch数

    def set_seed(self, seed, skip=0):
        self.seed = seed
        self.skip = skip

    def __iter__(self):
        order_by_len = sorted(range(len(self.song_lens)), key=lambda i: self.song_lens[i])
        buckets = [order_by_len[i:i+self.bucket_size] for i in range(0, len(order_by_len), self.bucket_size)]
        rng = random.Random(self.seed)
        rng.shuffle(buckets)
        for b in buckets:
            rng.shuffle(b)
        order = [idx for b in buckets for idx in b]
        batches = [order[i:i+self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.skip > 0:
            batches = batches[self.skip:]
        return iter(batches)

    def __len__(self):
        total = (len(self.song_lens) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip)


# ---------- collate_fn：数据增强 + padding ----------
def collate_fn(batch, max_frames=30000, crop_aug=True, crop_frames=None):
    # 过滤None和长歌
    items = []
    for x in batch:
        if x is None: continue
        mel, grid, label = x
        if mel.shape[0] > max_frames or len(grid) > 45000: continue
        # 随机裁剪增强：取60%-100%长度的连续片段
        if crop_aug and mel.shape[0] > 200:
            full_T = mel.shape[0]
            crop_ratio = random.uniform(0.6, 1.0)
            crop_T = int(full_T * crop_ratio)
            if crop_frames:  # 低显存/CPU：裁剪片段长度上限（默认None=不限制，和原行为一致）
                crop_T = min(crop_T, crop_frames)
            crop_start = random.randint(0, full_T - crop_T)
            crop_end = crop_start + crop_T
            mel = mel[crop_start:crop_end]
            # 不重新归一化：保持和推理一致（推理时用全曲归一化的mel）
            t_start = crop_start * HOP / SR
            t_end = crop_end * HOP / SR
            gmask = (grid >= t_start) & (grid < t_end)
            grid = grid[gmask] - t_start
            label = label[gmask]
            if len(grid) < 10: continue
        # 音量扰动 + 高斯噪声（只对前80维mel加，不污染pitch/flux通道）
        # log域加常数 = 线性域乘系数，模拟音量变化
        if random.random() < 0.5:
            mel_aug = mel[:, :80] + random.uniform(-0.1, 0.1)
            mel = np.concatenate([mel_aug, mel[:, 80:]], axis=1)
        if random.random() < 0.3:
            mel_aug = mel[:, :80] + np.random.normal(0, 0.05, mel[:, :80].shape).astype(np.float32)
            mel = np.concatenate([mel_aug, mel[:, 80:]], axis=1)
        items.append((mel, grid, label))
    if not items:
        return None
    B = len(items)
    Tmax = max(it[0].shape[0] for it in items)
    Gmax = max(len(it[1]) for it in items)
    C = items[0][0].shape[1]
    M = torch.zeros(B, Tmax, C, dtype=torch.float32)
    G = torch.zeros(B, Gmax, dtype=torch.float32)
    L = torch.zeros(B, Gmax, dtype=torch.long)
    mask = torch.zeros(B, Gmax, dtype=torch.float32)
    t_pad_mask = torch.ones(B, Tmax, dtype=torch.bool)
    for b, (mel, grid, label) in enumerate(items):
        T, Gc = mel.shape[0], len(grid)
        M[b, :T] = torch.from_numpy(mel)
        G[b, :Gc] = torch.from_numpy(grid)
        L[b, :Gc] = torch.from_numpy(label).long()
        mask[b, :Gc] = 1.0
        t_pad_mask[b, :T] = False
    return M, G, L, mask, t_pad_mask


# ---------- 评估（格轴） ----------
def f1_inst(pred_inst, gt_inst, tol=TOL):
    """双指针O(n+m)匹配：pred和gt都已排序，一次遍历完成匹配。
    比O(n×m)贪心快几十倍，reval全量评估时省大量时间。"""
    pred, gt = sorted(pred_inst), sorted(gt_inst)
    i, j, tp = 0, 0, 0
    while i < len(pred) and j < len(gt):
        if abs(pred[i] - gt[j]) <= tol:
            tp += 1; i += 1; j += 1
        elif pred[i] < gt[j]:
            i += 1  # 预测太早，往后移（gt递增，不可能和后面的匹配）
        else:
            j += 1  # 真实太早，往后移（pred递增，不可能和后面的匹配）
    fp, fn = len(pred) - tp, len(gt) - tp
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0), pr, rc

def evaluate(model, songs, th1=0.4, th2=0.5, max_songs=None, eval_seed=None, max_frames=30000, crop_frames=None):
    fids = songs.fids
    if max_songs:
        if eval_seed is not None:
            # 固定seed：可复现，适合调试，但200轮训练可能间接过拟合这200首
            fids = random.Random(eval_seed).sample(fids, min(max_songs, len(fids)))
        else:
            # 不固定seed：每轮随机抽不同的200首，避免间接过拟合
            fids = random.sample(fids, min(max_songs, len(fids)))
    # 局部Random：eval_seed=None时用随机seed，每轮不同
    rng = random.Random(eval_seed)
    inst, gridf, double_prec, double_rec = [], [], [], []
    inst_full, gridf_full = [], []  # 全遮罩（盲采）
    inst_half, gridf_half = [], []  # 半遮罩（条件采音）
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for fid in fids:
            mel, grid, label = songs._load(fid)
            if mel is None: continue
            if mel.shape[0] > max_frames or len(grid) > 45000:
                continue
            # 随机切片：起始位置随机，长度随机（60%-100%），和训练时随机裁剪口径完全一致
            T = mel.shape[0]
            if T > 100:
                crop_len = int(T * rng.uniform(0.6, 1.0))
                if crop_frames:
                    crop_len = min(crop_len, crop_frames)
                crop_start = rng.randint(0, T - crop_len)
                crop_end = crop_start + crop_len
                # 裁剪mel
                mel = mel[crop_start:crop_end]
                # 裁剪grid和label：只保留在裁剪时间范围内的grid
                t_start = crop_start / HR
                t_end = crop_end / HR
                mask = (grid >= t_start) & (grid < t_end)
                grid = grid[mask] - t_start  # 时间重新从0开始
                label = label[mask]
            M = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(device)
            G = torch.from_numpy(grid.astype(np.float32)).to(device)
            L = torch.from_numpy(label.astype(np.int64)).unsqueeze(0).to(device)
            # 评估：和训练分布一致——50%全遮罩（盲采）+ 50%半遮罩（条件采音）
            Gc = L.shape[1]
            notes_in = L.clone().float()
            if Gc > 10:
                if rng.random() < 0.5:
                    notes_in[:, :] = -1  # 全遮罩（纯盲采）
                    mask_pt = 0  # 全曲都算
                    is_full = True
                else:
                    mask_pt = rng.randint(int(Gc * 0.02), int(Gc * 0.9))
                    notes_in[:, mask_pt:] = -1  # 后半段遮罩（条件采音）
                    is_full = False
            else:
                mask_pt = Gc // 2
                notes_in[:, mask_pt:] = -1
                is_full = False
            o1, o2 = model(M, G, notes_in)
            s1 = torch.sigmoid(o1)[0].cpu().numpy(); s2 = torch.sigmoid(o2)[0].cpu().numpy()
            cls = np.zeros(Gc, int)
            hit = s1 >= th1
            cls[hit] = 1; cls[hit & (s2 >= th2)] = 2
            gt = label.astype(int)
            # 只评估遮罩部分（模型预测部分），避免前半段抄答案虚高
            # inst F1：双押只算一个音符位置（cls>0重复1次）
            has_note_pred = (cls[mask_pt:] > 0).astype(int)
            has_note_gt = (gt[mask_pt:] > 0).astype(int)
            # inst F1：0ms格级精确匹配（预测必须落在同一格）
            inst.append(f1_inst(np.repeat(grid[mask_pt:], has_note_pred), np.repeat(grid[mask_pt:], has_note_gt), 0)[0])
            # grid F1: 半格距容差（随BPM动态，120BPM≈7.8ms），和reval选冠军口径一致
            step_sec = float(np.median(np.diff(grid))) if len(grid) > 1 else 0.015
            tol_grid = step_sec * 0.5
            fg_val = f1_inst(grid[mask_pt:][cls[mask_pt:] > 0], grid[mask_pt:][gt[mask_pt:] > 0], tol_grid)[0]
            gridf.append(fg_val)
            # 分别统计全遮罩和半遮罩
            if is_full:
                inst_full.append(inst[-1]); gridf_full.append(fg_val)
            else:
                inst_half.append(inst[-1]); gridf_half.append(fg_val)
            # 双押 precision/recall
            pred_double = (cls[mask_pt:] == 2)
            gt_double = (gt[mask_pt:] == 2)
            if gt_double.sum() > 0:
                dtp = int((pred_double & gt_double).sum())
                dfp = int((pred_double & ~gt_double).sum())
                dfn = int((~pred_double & gt_double).sum())
                double_prec.append(dtp / max(dtp+dfp, 1))
                double_rec.append(dtp / max(dtp+dfn, 1))
    return (float(np.mean(inst)), float(np.mean(gridf)),
            float(np.mean(double_prec)) if double_prec else 0.0, float(np.mean(double_rec)) if double_rec else 0.0,
            float(np.mean(inst_full)) if inst_full else 0.0, float(np.mean(gridf_full)) if gridf_full else 0.0,
            float(np.mean(inst_half)) if inst_half else 0.0, float(np.mean(gridf_half)) if gridf_half else 0.0)


# ---------- 密度调制加权 loss ----------
def density_weighted_loss(o1, o2, tgt, w_cls, w_sparse, w_dense, mask=None, density_mask=None):
    """tgt (B,G) 格级 0/1/2。
    mask (B,G): 1=计算loss的位置（遮罩待预测位）,0=不算（参考位/padding）。
    density_mask (B,G): 1=用于计算密度的位置（全曲有效位，不含padding），
                        传None时回退为用mask算密度（兼容旧调用）。
    类别权重 w_cls=[w0,w1,w2] 基础上：
    每首歌 density = 音符格占比 → 稀疏歌音符格权重×w_sparse；密集歌空格权重×w_dense。
    损失按有效权重位置归一化（不是全局mean），避免o2双押头被大量零权重格稀释。"""
    y = (tgt > 0).float()
    if mask is None:
        mask = torch.ones_like(tgt, dtype=torch.float)
    else:
        mask = mask.float()
    if density_mask is None:
        density_mask = mask
    else:
        density_mask = density_mask.float()
    # 密度用全曲计算（density_mask=全曲有效位），避免全遮罩和半遮罩密度不一致
    dens = (y * density_mask).sum(dim=1, keepdim=True) / density_mask.sum(dim=1, keepdim=True).clamp(min=1)
    dens = dens.clamp(1e-4, 1)
    w0 = w_cls[0] * (1 + w_dense * dens)
    w1 = w_cls[1] * (1 + w_sparse * (1 - dens))
    w2 = w_cls[2] * (1 + w_sparse * (1 - dens))
    w_map = torch.where(tgt == 0, w0.expand_as(tgt),
             torch.where(tgt == 1, w1.expand_as(tgt), w2.expand_as(tgt)))
    w_map = w_map * mask  # 只在计算loss的位置有权重
    # o1: 按有效权重位置归一化（修复：原全局mean会被padding/参考位稀释）
    loss1_raw = F.binary_cross_entropy_with_logits(o1, y, weight=w_map, reduction='none')
    loss1 = loss1_raw.sum() / w_map.sum().clamp(min=1)
    # o2: 只在有音位置(tgt>0)算"是不是双押"，按有效权重位置归一化
    # 修复：原全局mean分母是全格数，双押有效格只占~5%，梯度被稀释约17倍
    o2_mask = (tgt > 0).float() * mask
    o2_weight = w_map * o2_mask
    if o2_weight.sum() > 0:
        loss2_raw = F.binary_cross_entropy_with_logits(o2, (tgt == 2).float(), weight=o2_weight, reduction='none')
        loss2 = loss2_raw.sum() / o2_weight.sum().clamp(min=1)
    else:
        loss2 = torch.tensor(0.0, device=o2.device)
    return loss1 + loss2


def soft_f1_loss(o1, tgt, mask=None):
    """可微 F1 近似：p=sigmoid(o1)，soft tp/fp/fn → 1-F1 项。mask=1有效，0=padding。"""
    p = torch.sigmoid(o1)
    y = (tgt > 0).float()
    if mask is None:
        mask = torch.ones_like(y)
    else:
        mask = mask.float()
    p = p * mask
    y = y * mask
    tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return 1 - f1


# ---------- 训练 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=99999)
    ap.add_argument("--tag", default="v22")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
                    help="cuda / mps(Apple Silicon) / cpu")
    ap.add_argument("--threads", type=int, default=8, help="CPU线程数（同时控制DataLoader worker数）")
    ap.add_argument("--batch-size", type=int, default=4, help="batch大小（按长度分桶，减少padding）")
    ap.add_argument("--data", default="grid_npz_full128")
    ap.add_argument("--split", default="split_full128_final.json")
    ap.add_argument("--max-songs", type=int, default=None)
    ap.add_argument("--pitch-dir", default=None)
    ap.add_argument("--pitch-mode", default='chroma', choices=['chroma'], help="pitch特征模式（当前只支持chroma=46维）")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="评估抽样固定seed（默认None=每轮随机抽，避免间接过拟合；调试时可设42）")
    ap.add_argument("--eval-songs", type=int, default=200, help="每轮评估歌曲数（默认200，越多越稳定但越慢）")
    ap.add_argument("--ch", type=int, default=96, help="CNN/Transformer 通道数（默认96，OOM可降64或48）")
    ap.add_argument("--w-sparse", type=float, default=1.5, help="稀疏歌音符格权重调制")
    ap.add_argument("--w-dense", type=float, default=0.8, help="密集歌空格权重调制")
    ap.add_argument("--w-double", type=float, default=4.5, help="双押类别基础权重（单押=3.0，双押更稀有应更高）")
    ap.add_argument("--soft-f1", action="store_true", help="加 soft-F1 损失项（权重 0.3）")
    ap.add_argument("--lr", type=float, default=5e-4, help="学习率（Transformer 建议 5e-4）")
    ap.add_argument("--max-frames", type=int, default=30000, help="长歌过滤阈值（帧数，默认30000≈5.8分钟）")
    ap.add_argument("--crop-frames", type=int, default=None, help="随机裁剪片段最大帧数（CPU/小显存调试用，如4000≈46秒；默认不限制）")
    ap.add_argument("--no-length-bucket", action="store_true", help="关闭按长度分桶（默认开，减少padding浪费）")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    print(f"v22 Transformer: device={device} ch={args.ch} "
          f"w_sparse={args.w_sparse} w_dense={args.w_dense} w_double={args.w_double} soft_f1={args.soft_f1}", flush=True)

    n_pitch = 0
    if args.pitch_dir:
        _fs = sorted(glob.glob(f"{args.pitch_dir}/*.npz"))
        if _fs:
            n_pitch = int(np.load(_fs[0])["pitch"].shape[1])
            print(f"pitch 维度探测: {n_pitch}", flush=True)
        else:
            n_pitch = {'chroma': 46}.get(args.pitch_mode, 46)
            print(f"[警告] pitch_dir 为空，使用默认维度 {n_pitch}", flush=True)

    tr = Songs("train", args.data, args.split, pitch_dir=args.pitch_dir,
               pitch_mode=args.pitch_mode, n_pitch=n_pitch, flux=True)
    va = Songs("val", args.data, args.split, pitch_dir=args.pitch_dir,
               pitch_mode=args.pitch_mode, n_pitch=n_pitch, flux=True)
    if args.max_songs:
        tr.fids = tr.fids[:args.max_songs]; va.fids = va.fids[:max(1, args.max_songs // 10)]
    print(f"train {len(tr)} val {len(va)} | 输入通道 = mel80+flux1+pitch{n_pitch}+notes_onehot3+mask1 = {85 + n_pitch}", flush=True)

    model = BeatFCNContext(n_mels=81, ch=args.ch, ctx_off=True, n_pitch=n_pitch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Warmup + Cosine：前5轮线性warmup，之后cosine decay
    warmup_ep = 5
    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep  # 线性从0.2升到1.0
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup_ep) / (args.epochs - warmup_ep)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # AMP 混合精度：GPU 上提速 2-3 倍，省显存
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_ep, best_inst, best_grid, wait, hist, skip_n = 1, 0.0, 0.0, 0, [], 0
    BS = args.batch_size
    w_cls = torch.tensor([1.0, 3.0, args.w_double]).to(device)  # 空格/单押/双押 类别权重（双押更稀有，权重更高）
    if args.resume:
        ck, js = f"ckpt_{args.tag}.pth", f"ckpt_{args.tag}.json"
        if os.path.exists(ck):
            try:
                d = torch.load(ck, map_location="cpu", weights_only=False)
                model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"])
                skip_n = d.get("batch_done", d.get("songs_done", 0))  # 兼容旧版songs_done
                n_batches_est = (len(tr) + BS - 1) // BS
                if skip_n >= n_batches_est:
                    start_ep = d["ep"] + 1; skip_n = 0; ep_done = True
                else:
                    start_ep = d["ep"]; skip_n = int(skip_n); ep_done = False
                best_inst = d.get("best_inst", d.get("best", 0.0))
                best_grid = d.get("best_grid", d.get("best", 0.0))
                wait = d.get("wait", 0); hist = d.get("hist", [])
                # LR调度：按epoch重算lr，并设置scheduler的last_epoch（避免step()从warmup重新开始）
                ep_for_lr = start_ep - 1
                lr_now = args.lr * lr_lambda(ep_for_lr)
                for g in opt.param_groups: g["lr"] = lr_now
                sched.last_epoch = ep_for_lr
                print(f"resume: ep {start_ep}, best_inst {best_inst:.4f}, best_grid {best_grid:.4f}, lr={lr_now:.2e}, skip_batch={skip_n}", flush=True)
            except Exception as e:
                print(f"[警告] ckpt 加载失败({e})，从头训练", flush=True)
                start_ep, best_inst, best_grid, wait, hist = 1, 0.0, 0.0, 0, []
    MAXF = args.max_frames
    # 预计算每首歌的长度（用于长度分桶）
    # 歌曲长度缓存（第一次启动算并存json，后续直接读，省5-15分钟）
    # 缓存格式: {"lens": [...], "data_mtime": 最新npz修改时间}
    # 数据目录有文件更新时自动失效重算
    len_cache = os.path.join(args.data, 'song_lens_cache.json')
    def _latest_data_mtime():
        """返回数据目录下最新npz文件的mtime，用于缓存失效检测"""
        try:
            latest = 0
            for f in os.listdir(args.data):
                if f.endswith('.npz'):
                    mt = os.path.getmtime(os.path.join(args.data, f))
                    if mt > latest:
                        latest = mt
            return latest
        except Exception:
            return 0
    song_lens = None
    if os.path.exists(len_cache):
        try:
            cached = json.load(open(len_cache))
            if isinstance(cached, dict) and 'lens' in cached:
                # 新格式：带mtime校验
                cached_mtime = cached.get('data_mtime', 0)
                current_mtime = _latest_data_mtime()
                if abs(cached_mtime - current_mtime) < 1.0 and len(cached['lens']) == len(tr):
                    song_lens = cached['lens']
                    print(f"长度缓存: 已加载 {len(song_lens)} 首歌长度（mtime校验通过）", flush=True)
                else:
                    print(f"长度缓存: 数据已更新（cached={cached_mtime:.0f}, current={current_mtime:.0f}），重新计算", flush=True)
            elif isinstance(cached, list) and len(cached) == len(tr):
                # 旧格式：只有长度列表，兼容但不校验mtime
                song_lens = cached
                print(f"长度缓存: 已加载 {len(song_lens)} 首歌长度（旧格式，无mtime校验）", flush=True)
        except Exception:
            song_lens = None
    if song_lens is None:
        song_lens = []
        for i in range(len(tr)):
            try:
                item = tr[i]
                if item is None:
                    song_lens.append(0)
                else:
                    song_lens.append(int(item[0].shape[0]))
            except Exception:
                song_lens.append(0)
        try:
            json.dump({"lens": song_lens, "data_mtime": _latest_data_mtime()}, open(len_cache, 'w'))
            print(f"长度缓存: 已保存到 {len_cache}", flush=True)
        except Exception:
            pass
    print(f"长度分桶: {'开' if not args.no_length_bucket else '关'} | 随机裁剪: 开 | max_frames={MAXF}", flush=True)
    hparams = {"n_mels": 81, "ch": args.ch, "n_pitch": n_pitch, "n_layers": 4, "nhead": 4}
    # DataLoader建在循环外，persistent_workers才能复用worker进程
    sampler = LengthBucketSampler(song_lens, BS, bucket_size=200, seed=start_ep, skip=skip_n)
    def _collate(batch):
        return collate_fn(batch, max_frames=MAXF, crop_aug=True, crop_frames=args.crop_frames)
    def _worker_init(worker_id):
        """每个worker进程独立播种，避免8个worker加的噪声/裁剪完全一样"""
        import random, numpy as np
        seed = torch.initial_seed() % (2**32) + worker_id
        random.seed(seed)
        np.random.seed(seed)
    # CPU 上 fork 出的 persistent worker 与 torch 线程池会在第2轮死锁（实测），CPU 模式退回单进程加载
    n_workers = min(args.threads, 8) if device.type != 'cpu' else 0
    if device.type == 'cpu' and args.threads > 0:
        print("[CPU模式] DataLoader 使用单进程加载（避免多进程死锁）", flush=True)
    loader = DataLoader(tr, batch_sampler=sampler, collate_fn=_collate,
                        num_workers=n_workers, prefetch_factor=2 if n_workers > 0 else None,
                        pin_memory=(device.type == 'cuda'), persistent_workers=(n_workers > 0),
                        worker_init_fn=_worker_init if n_workers > 0 else None)

    # ---------- 停止信号处理：收到Ctrl+C/SIGTERM时保存断点再退出，不丢训练 ----------
    stop_requested = [False]  # 用列表存标志，闭包里能修改
    def _stop_handler(signum, frame):
        print(f"\n[停止信号] 收到信号 {signum}，保存断点后退出（不会丢已训练的进度）...", flush=True)
        stop_requested[0] = True
    signal.signal(signal.SIGINT, _stop_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _stop_handler)  # kill / 网页停止按钮

    # ---------- best模型兜底：开局存一个UNTRAINED标记版，避免首轮崩溃后推理找不到文件 ----------
    best_path = f"beat_fcn_best_{args.tag}.pt"
    if not os.path.exists(best_path):
        untrained_path = best_path + ".UNTRAINED"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "hparams": hparams,
                    "ep": start_ep - 1, "best_inst": best_inst, "best_grid": best_grid,
                    "wait": wait, "hist": hist, "note": "UNTRAINED - 随机权重，不要用来推理"},
                   untrained_path)
        print(f"[兜底] 已保存未训练模型到 {untrained_path}（首轮崩溃后可用，但输出是噪声）", flush=True)
        print(f"       正式best.pt会在第一个epoch评估后自动生成", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        if tr.bad or va.bad:
            print(f"[数据自检] 坏文件累计: train {len(tr.bad)} / val {len(va.bad)}", flush=True)
        model.train(); tot, n = 0.0, 0; t0 = time.time()
        sampler.set_seed(ep, skip=skip_n)
        n_batches = len(sampler)
        for bi, batch in enumerate(loader):
            # 检查停止信号：收到就保存断点后退出，不丢进度
            if stop_requested[0]:
                print(f"\n[停止] 正在保存断点 ep={ep} batch={bi}/{n_batches}...", flush=True)
                save_ckpt(model, opt, sched, ep, best_grid, wait, hist, args.tag, bi, hparams=hparams)
                print(f"[停止] 断点已保存，退出。下次用 --resume 接着训。", flush=True)
                # 跳出两层循环
                break
            if batch is None: continue
            M, G, L, mask, t_pad_mask = batch
            B = M.shape[0]
            M, G, L, mask = M.to(device), G.to(device), L.to(device), mask.to(device)
            t_pad_mask = t_pad_mask.to(device)
            # 条件采音训练：50%全遮罩 + 50%随机遮罩后半段
            notes_in = L.clone().float()
            for b in range(B):
                Gc = int(mask[b].sum().item())
                notes_in[b, Gc:] = -1  # padding部分全部遮罩
                if Gc > 10 and random.random() < 0.5:
                    notes_in[b, :Gc] = -1  # 全遮罩（纯盲采）
                elif Gc > 10:
                    mpt = random.randint(int(Gc * 0.02), int(Gc * 0.9))
                    notes_in[b, mpt:Gc] = -1  # 后半段遮罩（条件采音）
            with torch.amp.autocast('cuda', enabled=use_amp):
                o1, o2 = model(M, G, notes_in, src_key_padding_mask=t_pad_mask)
                tgt = L[:, :o1.shape[1]]
                msk = mask[:, :o1.shape[1]]
                # loss只在遮罩待预测位置（notes_in==-1）计算
                loss_mask = msk * (notes_in[:, :o1.shape[1]] == -1).float()
                # 密度用全曲有效位mask算（msk），不是只用遮罩位（loss_mask）
                # 这样半遮罩时密度统计和全遮罩一致，和文档承诺一致
                loss = density_weighted_loss(o1, o2, tgt, w_cls, args.w_sparse, args.w_dense,
                                             mask=loss_mask, density_mask=msk)
                if args.soft_f1:
                    loss = loss + 0.3 * soft_f1_loss(o1, tgt, mask=loss_mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [警告] ep{ep} batch{bi}/{n_batches} loss={loss.item()}，跳过", flush=True)
                opt.zero_grad()
                scaler.update()  # 让GradScaler自动缩小scale，避免反复NaN
                continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); n += 1
            if bi % 300 == 0 and bi > 0:
                save_ckpt(model, opt, sched, ep, best_grid, wait, hist, args.tag, bi, hparams=hparams)
                print(f"  [{ep}] 中途ckpt @{bi}/{n_batches} batch ({(time.time()-t0):.0f}s)", flush=True)
        # 如果收到停止信号，跳过评估直接退出（断点已在batch循环里保存）
        if stop_requested[0]:
            print(f"[停止] 已退出 ep={ep}，断点已保存。", flush=True)
            break
        sched.step()
        skip_n = 0
        if True:  # 每轮都评估
            fi, fg, dprec, drec, fi_full, fg_full, fi_half, fg_half = evaluate(model, va, max_songs=args.eval_songs, eval_seed=args.eval_seed, max_frames=MAXF, crop_frames=args.crop_frames)
            improved_grid = fg > best_grid; wait = 0 if improved_grid else wait + 1
            if improved_grid: best_grid = fg
            if fi > best_inst: best_inst = fi
            lr_now = opt.param_groups[0]["lr"]
            print(f"[{ep}/{args.epochs}] inst={fi:.4f}(0ms) grid={fg:.4f}(半格) dprec={dprec:.3f} drec={drec:.3f} | "
                  f"盲采grid={fg_full:.4f} 条件grid={fg_half:.4f} | "
                  f"best_inst={best_inst:.4f} best_grid={best_grid:.4f} "
                  f"lr={lr_now:.2e} L={tot/max(n,1):.3f} ({time.time()-t0:.0f}s)", flush=True)
            hist.append({"ep": ep, "fi": fi, "fg": fg, "dprec": dprec, "drec": drec,
                         "best_inst": best_inst, "best_grid": best_grid})
            if improved_grid:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "hparams": hparams,
                            "ep": ep, "best_inst": best_inst, "best_grid": best_grid,
                            "wait": wait, "hist": hist},
                           f"beat_fcn_best_{args.tag}.pt")
            save_ckpt(model, opt, sched, ep, best_grid, wait, hist, args.tag, n_batches, hparams=hparams)
            torch.save({"model": model.state_dict(), "hparams": hparams}, f"ckpt_{args.tag}_ep{ep}.pt")
            if wait >= args.patience:
                print(f"early stop @ {ep}", flush=True); break
        else:
            save_ckpt(model, opt, sched, ep, best_grid, wait, hist, args.tag, n_batches)
    print(f"done. best_inst={best_inst:.4f} best_grid={best_grid:.4f}", flush=True)


def save_ckpt(model, opt, sched, ep, best_grid, wait, hist, tag, batch_done=0, hparams=None):
    if hparams is None:
        hparams = {"n_mels": 81, "ch": 96, "n_pitch": 46, "n_layers": 4, "nhead": 4}
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "hparams": hparams,
                "ep": ep, "best_grid": best_grid, "wait": wait, "hist": hist,
                "batch_done": batch_done},  # batch_done=已处理的batch数，resume时跳过对应数量
               f"ckpt_{tag}.pth")
    with open(f"ckpt_{tag}.json", "w") as f:
        json.dump({"ep": ep, "best_grid": best_grid, "hist": hist}, f)


if __name__ == "__main__":
    main()

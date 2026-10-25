#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v25 训练 —— v24b 输入 + 双头输出（onset + 双押），面向"编辑器草稿"用途。

在 v23dens 的基础上：
- 输入通道 127 → 130（+节拍引导 +重音曲线 +起音包络，见 feat_v25.py；训练/推理唯一入口）
- 输出 n_out=2（不要 v24 的类型/长条头）
- stem 默认 sep3（空洞深度卷积，±130ms 音符局部上下文）
- 训练增强：时间抖动 --jitter-ms（grid 相对 mel 随机平移，模拟谱师 offset 噪声）
- EMA 权重平均（--ema 0 关闭；评估/最优 ckpt 均用 EMA 权重）
- loss 可选 focal（--focal 1.5）与邻近格软标签（--soft-nb 0.3），默认关闭，A/B 用
- 双押权重 --w-double 沿用 v23；叠键折叠→双押标签由 extract_full 的 min(2,·) 天然实现

用法（详细流程见 USAGE_v25.md）：
  python beat_fcn_train_v25.py --data ./grid_npz_clean --pitch-dir ./pitch_npz_full \
      --split ./split_full128_final.json --guide-dir ./beats_clean --epochs 100 --device cuda
"""
import os, sys, json, argparse, time, random, math, signal, glob
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_fcn_train_v22 import collate_fn  # noqa: E402
from beat_fcn_train_v23 import (CachedSongs, TokenBudgetSampler, make_eval_set,  # noqa: E402
                                build_model, evaluate)
import feat_v25  # noqa: E402


class GuideSongs(CachedSongs):
    """CachedSongs + v25 三个通道（要求 gen/detect_all.py 产出的 beat_this 缓存已就位）。"""

    def __init__(self, *a, guide_dir=None, **kw):
        super().__init__(*a, **kw)
        self.guide_dir = guide_dir
        self.fids = [f for f in self.fids if os.path.exists(f'{guide_dir}/{f}.npz')]

    def __getitem__(self, i):
        it = super().__getitem__(i)
        if it is None:
            return None
        mel, grid, label = it
        z = np.load(f'{self.guide_dir}/{self.fids[i]}.npz')
        return feat_v25.append_channels(mel, z['beats'], z['downs']), grid, label


def loss_v25(o1, o2, tgt, w_cls, w_sparse, w_dense, mask=None, density_mask=None, focal=0.0, soft_nb=0.0):
    """v22 density_weighted_loss 的超集：+focal（o1）+ 邻近格软标签（o1 目标）。o2 语义与 v22 完全一致。

    软标签只改 BCE 的目标值；类别权重与密度加权仍按硬标签算（保持 v23 的平衡语义）。
    """
    y_hard = (tgt > 0).float()
    y = y_hard
    if soft_nb > 0.0:  # ±1 格有音 → 本格目标 soft_nb（对抗亚格/谱师 timing 抖动）
        nb = torch.zeros_like(y_hard)
        nb[:, 1:] = torch.maximum(nb[:, 1:], y_hard[:, :-1])
        nb[:, :-1] = torch.maximum(nb[:, :-1], y_hard[:, 1:])
        y = torch.where(y_hard > 0, y_hard, nb * soft_nb)
    if mask is None:
        mask = torch.ones_like(tgt, dtype=torch.float)
    else:
        mask = mask.float()
    if density_mask is None:
        density_mask = mask
    else:
        density_mask = density_mask.float()
    dens = ((y_hard > 0).float() * density_mask).sum(dim=1, keepdim=True) / density_mask.sum(dim=1, keepdim=True).clamp(min=1)
    dens = dens.clamp(1e-4, 1)
    w0 = w_cls[0] * (1 + w_dense * dens)
    w1 = w_cls[1] * (1 + w_sparse * (1 - dens))
    w2 = w_cls[2] * (1 + w_sparse * (1 - dens))
    w_map = torch.where(tgt == 0, w0.expand_as(tgt),
             torch.where(tgt == 1, w1.expand_as(tgt), w2.expand_as(tgt)))
    w_map = w_map * mask
    bce = F.binary_cross_entropy_with_logits(o1.float(), y, weight=w_map, reduction='none')
    if focal > 0.0:
        pt = torch.sigmoid(o1.float()).detach()
        pt = pt * y + (1 - pt) * (1 - y)
        bce = bce * (1 - pt).clamp(min=1e-4).pow(focal)
    loss1 = bce.sum() / w_map.sum().clamp(min=1)
    o2_mask = (tgt > 0).float() * mask
    o2_weight = w_map * o2_mask
    if o2_weight.sum() > 0:
        bce2 = F.binary_cross_entropy_with_logits(o2.float(), (tgt == 2).float(), weight=o2_weight, reduction='none')
        loss2 = bce2.sum() / o2_weight.sum().clamp(min=1)
    else:
        loss2 = torch.tensor(0.0, device=o1.device)
    return loss1 + loss2


def ema_update(model, ema, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(mv.detach(), alpha=1 - decay)
            else:
                v.copy_(mv)


def eval_with_ema(model, ema, songs, idxs, device):
    """有 EMA 时用 EMA 权重评估（评估后恢复原权重），否则直接评估。"""
    if ema is None:
        return evaluate(model, songs, idxs, device)
    bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    try:
        return evaluate(model, songs, idxs, device)
    finally:
        model.load_state_dict(bak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--tag', default='v25')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--data', default='grid_npz_clean')
    ap.add_argument('--pitch-dir', default='pitch_npz_full')
    ap.add_argument('--split', default='split_full128_final.json')
    ap.add_argument('--cache-dir', default='feat_cache_v25')
    ap.add_argument('--guide-dir', default='beats_clean', help='gen/detect_all.py 的 beat_this 缓存目录（必填，缺文件的歌曲自动跳过）')
    ap.add_argument('--strict-diff', action='store_true', help='只用文件名难度 [IN]/[AT] 的谱面')
    ap.add_argument('--max-songs', type=int, default=None)
    ap.add_argument('--max-tokens', type=int, default=120000)
    ap.add_argument('--max-frames', type=int, default=40000)
    ap.add_argument('--crop-frames', type=int, default=None)
    ap.add_argument('--eval-songs', type=int, default=300)
    ap.add_argument('--eval-seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--w-sparse', type=float, default=1.5)
    ap.add_argument('--w-dense', type=float, default=0.8)
    ap.add_argument('--w-double', type=float, default=4.5)
    ap.add_argument('--d-model', type=int, default=96)
    ap.add_argument('--stem-ch', type=int, default=48)
    ap.add_argument('--stem', default='sep3', choices=['sep', 'sep3', 'conv'])
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--cells-per-token', type=int, default=16)
    ap.add_argument('--samples-per-token', type=int, default=16)
    ap.add_argument('--no-cond-density', dest='cond_density', action='store_false')
    ap.add_argument('--jitter-ms', type=float, default=12.0, help='训练时 grid 相对特征随机平移幅度（0=关）')
    ap.add_argument('--focal', type=float, default=0.0, help='o1 focal gamma（0=普通 BCE）')
    ap.add_argument('--soft-nb', type=float, default=0.0, help='±1 邻格软标签目标值（0=关）')
    ap.add_argument('--ema', type=float, default=0.999, help='EMA 衰减（0=关）')
    ap.add_argument('--no-epoch-ckpt', action='store_true')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--time-budget', type=float, default=None)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    pz = sorted(glob.glob(f'{args.pitch_dir}/*.npz'))
    n_pitch = int(np.load(pz[0])['pitch'].shape[1]) if pz else 46
    kw = dict(pitch_dir=args.pitch_dir, pitch_mode='chroma', n_pitch=n_pitch, flux=True,
              cache_dir=args.cache_dir or None, strict_diff=args.strict_diff, guide_dir=args.guide_dir)
    tr = GuideSongs('train', args.data, args.split, **kw)
    va = GuideSongs('val', args.data, args.split, **kw)
    if args.max_songs:
        tr.fids = tr.fids[:args.max_songs]
    t0 = time.time()
    lens = [tr.length(i) for i in range(len(tr))]
    lens = [l if l <= args.max_frames else 0 for l in lens]
    eval_idx = make_eval_set(va, args.eval_songs, args.eval_seed, args.max_frames)
    print(f'v25: device={device} amp={use_amp} train={sum(l > 0 for l in lens)} eval={len(eval_idx)} '
          f'guide_dir={args.guide_dir} (长度扫描 {time.time()-t0:.0f}s)', flush=True)

    hp = dict(n_mels=81, n_pitch=n_pitch + 3, stem_ch=args.stem_ch, d_model=args.d_model, n_layers=args.layers,
              nhead=4, cells_per_token=args.cells_per_token, samples_per_token=args.samples_per_token,
              stem=args.stem, arch='v23', cond_density=args.cond_density, n_out=2, v25=True,
              jitter_ms=args.jitter_ms, focal=args.focal, soft_nb=args.soft_nb)
    model = build_model(hp).to(device)
    print(f'参数量 {sum(p.numel() for p in model.parameters()):,}  hparams={hp}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < args.warmup: return (ep + 1) / args.warmup
        p = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    fwd = torch.compile(model) if args.compile else model
    w_cls = torch.tensor([1.0, 3.0, args.w_double], device=device)

    start_ep, best, hist = 1, -1.0, []
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema > 0 else None
    ck = f'ckpt_{args.tag}.pth'
    if args.resume and os.path.exists(ck):
        d = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(d['model']); opt.load_state_dict(d['opt']); sched.load_state_dict(d['sched'])
        start_ep, best, hist = d['ep'] + 1, d['best'], d['hist']
        if ema is not None and 'ema' in d and d['ema'] is not None:
            for k in ema: ema[k].copy_(d['ema'][k])
        print(f'resume: ep {start_ep} best {best:.4f}', flush=True)

    sampler = TokenBudgetSampler(lens, args.max_tokens, args.crop_frames)
    nw = min(args.threads, 8) if device.type == 'cuda' else 0
    loader = torch.utils.data.DataLoader(
        tr, batch_sampler=sampler, num_workers=nw, prefetch_factor=4 if nw else None,
        persistent_workers=nw > 0, pin_memory=device.type == 'cuda',
        collate_fn=lambda b: collate_fn(b, max_frames=args.max_frames, crop_aug=True, crop_frames=args.crop_frames))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0, True))
    t_train0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train(); sampler.set_seed(ep)
        tot, n, t0 = 0.0, 0, time.time()
        for batch in loader:
            if stop[0]: break
            if batch is None: continue
            M, G, L, mask, _ = batch
            M, G, L, mask = M.to(device, non_blocking=True), G.to(device), L.to(device), mask.to(device)
            B, Gm = L.shape
            if args.jitter_ms > 0:  # 时间抖动：grid 整体平移 ±jitter（mel 不动）→ 模拟谱师 offset 噪声
                jit = (torch.rand(B, 1, device=device) * 2 - 1) * (args.jitter_ms / 1000.0)
                G = (G + jit).clamp(min=0.0)
            notes = L.clone().float()
            for b in range(B):
                Gc = int(mask[b].sum().item())
                notes[b, Gc:] = -1
                if Gc > 10 and random.random() < 0.5:
                    notes[b, :Gc] = -1
                elif Gc > 10:
                    notes[b, random.randint(int(Gc * 0.02), int(Gc * 0.9)):Gc] = -1
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                dens = None
                if args.cond_density:
                    Lb = (L > 0).float() * mask; Gx = G.masked_fill(mask == 0, -1e9)
                    dur = (Gx.max(1).values - G.masked_fill(mask == 0, 1e9).min(1).values).clamp(min=1.0)
                    dens = torch.log1p(Lb.sum(1) / dur)
                o1, o2 = fwd(M, G, notes, grid_mask=mask > 0, dens=dens)
            loss_mask = mask * (notes == -1).float()
            loss = loss_v25(o1, o2, L, w_cls, args.w_sparse, args.w_dense,
                            mask=loss_mask, density_mask=mask, focal=args.focal, soft_nb=args.soft_nb)
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema_update(model, ema, args.ema)
            tot += loss.item(); n += 1
        sched.step()
        t_ep = time.time() - t0
        te = time.time()
        res, th = eval_with_ema(model, ema, va, eval_idx, device)
        r = res[th]
        t_eval = time.time() - te
        improved = r['f20'] > best
        if improved: best = r['f20']
        hist.append(dict(ep=ep, loss=tot / max(n, 1), th1=th, train_s=t_ep, eval_s=t_eval, **{k: r[k] for k in
                    ('strict', 'f20', 'loose', 's_b', 's_c', 'l_b', 'l_c', 'dprec', 'drec')}))
        print(f'[{ep}/{args.epochs}] strict={r["strict"]:.4f} (盲{r["s_b"]:.3f}/条件{r["s_c"]:.3f}) '
              f'f20ms={r["f20"]:.4f} loose50ms={r["loose"]:.4f} dP={r["dprec"]:.3f} dR={r["drec"]:.3f} th1={th} best={best:.4f} '
              f'L={tot/max(n,1):.3f} lr={opt.param_groups[0]["lr"]:.1e} '
              f'train {t_ep:.0f}s eval {t_eval:.0f}s', flush=True)
        # 最优 ckpt 存 EMA 权重（推理端无感）；断点文件存原始权重 + EMA 副本
        if improved:
            if ema is not None:
                bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
                torch.save(dict(model=model.state_dict(), hparams=hp, th1=th, ep=ep, best=best), f'beat_fcn_best_{args.tag}.pt')
                model.load_state_dict(bak)
            else:
                torch.save(dict(model=model.state_dict(), hparams=hp, th1=th, ep=ep, best=best), f'beat_fcn_best_{args.tag}.pt')
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        ema=ema, hparams=hp, th1=th, ep=ep, best=best, hist=hist), ck)
        if not args.no_epoch_ckpt:
            torch.save(dict(model=model.state_dict(), hparams=hp, th1=th), f'ckpt_{args.tag}_ep{ep}.pt')
        json.dump(dict(best=best, hist=hist), open(f'ckpt_{args.tag}.json', 'w'), indent=1)
        if stop[0] or (args.time_budget and time.time() - t_train0 > args.time_budget):
            print('[停止] 到达时间预算或收到停止信号', flush=True); break
    print(f'done. best f20ms F1={best:.4f}', flush=True)


if __name__ == '__main__':
    main()

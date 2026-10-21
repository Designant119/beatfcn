#!/usr/bin/env python3
"""v24 训练加速版（同模型/同损失/同评估，参数兼容 beat_fcn_train_v24.py）。
剖析结论（CPU, 2 线程）：前向+反向 ≈96%，其中注意力 dropout 迫使 SDPA 走 math 路径（显式 N×N 矩阵 + bernoulli），
单这一项占步时 ~70%；数据加载+特征 ≈4%，评估 <5%。
加速开关：
  --attn-dropout 0   关掉注意力内 dropout（FFN/残差 dropout 保留）→ 走 fused flash 注意力，CPU 实测 ~3.6x
  --preload          一次性把 特征(含节拍引导/重音通道)+标签 读进内存，训练期零 npz 解码/重算
  --amp              bf16 autocast（CPU 有 AMX/avx512_bf16 时 ~+7%；CUDA/MPS 上收益更大）
  --device           cpu/cuda/mps（GPU 上主机侧 collate ≈50-90ms/步会成为瓶颈：配 --preload --workers 2-4）
  --workers          DataLoader 进程数
  --eval-ths         评估阈值列表（默认同 v24）
  --crop-aware      先定裁剪再按裁剪后长度分桶（同增强分布），padding 76%→96% 填充、每 epoch 步数 -23%
用 --fast 一次性打开 attn-dropout 0 + preload + crop-aware（amp 在 CPU 上收益不稳，需单独开）。"""
import os, sys, json, argparse, time, math, signal, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_fcn_train_v24 as V
from beat_fcn_train_v22 import collate_fn, density_weighted_loss
from beat_fcn_train_v23 import TokenBudgetSampler, make_eval_set, build_model


class Preloaded(torch.utils.data.Dataset):
    """把 TypedSongs 的每首 (feat, grid, label) 物化到内存（feat 存 fp16，取用时转 fp32）。"""
    def __init__(self, ds, max_frames=None):
        self.fids, self.items = ds.fids, []
        for i in range(len(ds)):
            it = ds[i]
            if it is not None and max_frames and it[0].shape[0] > max_frames: it = None
            self.items.append(None if it is None else (np.ascontiguousarray(it[0], np.float16), it[1], it[2]))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return None if it is None else (it[0].astype(np.float32), it[1], it[2])
    def length(self, i): return 0 if self.items[i] is None else self.items[i][0].shape[0]


class CropAware:
    """先按 epoch 抽每首的裁剪比例(60-100%，同 collate_fn 的分布)，再按裁剪后长度分桶 → padding 近零。
    TokenBudgetSampler 原本按全长分桶、collate 里才随机裁剪，导致 batch 内长短不一（注意力有效填充 ~76%）。"""
    def __init__(self, ds, lens, max_tokens):
        self.ds, self.full, self.max_tokens, self.seed, self.crop = ds, lens, max_tokens, 0, {}
    def set_seed(self, s):
        self.seed = s; rng = random.Random(1000 + s); self.crop = {}
        for i, l in enumerate(self.full):
            if l > 200:
                r = rng.uniform(0.6, 1.0); T = int(l * r); self.crop[i] = (rng.randint(0, l - T), T)
        self.inner = TokenBudgetSampler([self.crop[i][1] if i in self.crop else l for i, l in enumerate(self.full)], self.max_tokens, seed=s)
    def __iter__(self): return iter(self.inner._batches())
    def __len__(self): return len(self.inner)
    def get(self, i):
        it = self.ds[i]
        if it is None or i not in self.crop: return it
        mel, grid, label = it; c0, T = self.crop[i]; mel = mel[c0:c0 + T]
        ts, te = c0 * 256 / 22050, (c0 + T) * 256 / 22050; gm = (grid >= ts) & (grid < te)
        return None if gm.sum() < 10 else (mel, grid[gm] - ts, label[gm])


class _View(torch.utils.data.Dataset):
    def __init__(self, ca): self.ca = ca
    def __len__(self): return len(self.ca.full)
    def __getitem__(self, i): return self.ca.get(i)


def collate40k(b): return collate_fn(b, max_frames=40000)
def collate40k_nocrop(b): return collate_fn(b, max_frames=40000, crop_aug=False)


def set_attn_dropout(model, p):
    for l in model.transformer.layers: l.self_attn.dropout = p


def train_step(model, opt, batch, wt, w_cls, dev, amp):
    M, G, L, mask, _ = [x.to(dev, non_blocking=True) for x in batch]
    typ, dbl, hb = V.decode_t(L); on = (typ > 0); tgt = (on.long() + (dbl > 0).long() * on.long())
    dur = (G.masked_fill(mask == 0, -1e9).max(1).values - G.masked_fill(mask == 0, 1e9).min(1).values).clamp(min=1)
    dens = torch.log1p((on.float() * mask).sum(1) / dur)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        o = model(M, G, None, grid_mask=mask > 0, dens=dens)
    o = o.float()
    loss = density_weighted_loss(o[..., 0], o[..., 1], tgt, w_cls, 1.5, 0.8, mask=mask, density_mask=mask)
    m = on & (mask > 0)
    if m.any():
        loss = loss + 0.5 * F.cross_entropy(o[..., 2:6][m], (typ[m] - 1).long(), weight=wt)
        hm = m & (typ == 2)
        if hm.any(): loss = loss + 0.3 * F.smooth_l1_loss(o[..., 6][hm], torch.log1p(hb[hm]))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.detach()


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(data='grid_npz_clean', pitch_dir='pitch_npz_full', split='split_full128_final.json', cache_dir='feat_cache_clean',
                     typed_dir='typed_clean', tag='v24', init='', guide_dir='', device='cpu', eval_ths='0.3,0.4,0.5,0.6,0.7').items():
        ap.add_argument('--' + k.replace('_', '-'), default=v)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--lr', type=float, default=7e-4); ap.add_argument('--max-tokens', type=int, default=80000)
    ap.add_argument('--time-budget', type=float, default=2700); ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--attn-dropout', type=float, default=0.1); ap.add_argument('--preload', action='store_true')
    ap.add_argument('--amp', action='store_true'); ap.add_argument('--crop-aware', action='store_true'); ap.add_argument('--workers', type=int, default=0); ap.add_argument('--fast', action='store_true')
    ap.add_argument('--max-steps', type=int, default=0, help='>0：只跑这么多步（基准测试用）')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.fast: a.attn_dropout, a.preload, a.crop_aware = 0.0, True, True
    dev = torch.device(a.device)
    kw = dict(pitch_dir=a.pitch_dir, pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=a.cache_dir, typed_dir=a.typed_dir, guide_dir=a.guide_dir or None)
    tr = V.TypedSongs('train', a.data, a.split, **kw); va = V.TypedSongs('val', a.data, a.split, **kw)
    if a.preload:
        t = time.time(); tr = Preloaded(tr, 40000); va = Preloaded(va); print(f'preload {time.time() - t:.0f}s', flush=True)
    lens = [tr.length(i) for i in range(len(tr))]; lens = [l if l <= 40000 else 0 for l in lens]
    ev = make_eval_set(va, None, 42, 40000)
    hp = dict(n_mels=81, n_pitch=48 if a.guide_dir else 46, beat_guide=bool(a.guide_dir), stem_ch=48, d_model=96, n_layers=4, nhead=4, cells_per_token=16, samples_per_token=16,
              stem='sep', arch='v23', cond_density=True, n_out=7)
    model = build_model(hp)
    if a.init:
        sd = torch.load(a.init, map_location='cpu', weights_only=False)['model']; msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and (v.shape == msd[k].shape or k == 'stem.0.weight')}
        if sd['stem.0.weight'].shape != msd['stem.0.weight'].shape:
            w = torch.zeros_like(msd['stem.0.weight']); w[:, :sd['stem.0.weight'].shape[1]] = sd['stem.0.weight']; sd['stem.0.weight'] = w
        print('init', model.load_state_dict(sd, strict=False), flush=True)
    set_attn_dropout(model, a.attn_dropout); model.to(dev)
    cnt = np.zeros(5)
    for f in tr.fids:
        if os.path.exists(f'{a.typed_dir}/{f}.npz'): cnt += np.bincount(np.load(f'{a.typed_dir}/{f}.npz')['typ'], minlength=5)[:5]
    wt = torch.tensor((cnt[1:].sum() / cnt[1:]) ** 0.5, dtype=torch.float32); wt = (wt / wt.mean()).to(dev)
    print(f'train={sum(l > 0 for l in lens)} eval={len(ev)} attn_dropout={a.attn_dropout} preload={a.preload} amp={a.amp} dev={dev}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1, (e + 1) / 2) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(e / a.epochs, 1)))))
    lkw = dict(num_workers=a.workers, pin_memory=dev.type == 'cuda', persistent_workers=False)
    if a.crop_aware:
        sampler = CropAware(tr, lens, a.max_tokens); sampler.set_seed(1)
        loader = torch.utils.data.DataLoader(_View(sampler), batch_sampler=sampler, collate_fn=collate40k_nocrop, **lkw)
    else:
        sampler = TokenBudgetSampler(lens, a.max_tokens)
        loader = torch.utils.data.DataLoader(tr, batch_sampler=sampler, collate_fn=collate40k, **lkw)
    w_cls = torch.tensor([1.0, 3.0, 4.5], device=dev); best = -1; t0 = time.time(); stop = [False]; steps = 0
    ths = tuple(float(x) for x in a.eval_ths.split(','))
    signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__(0, True))
    for ep in range(1, a.epochs + 1):
        model.train(); sampler.set_seed(ep); tot = []; te = time.time()
        for batch in loader:
            if batch is None or stop[0]: continue
            tot.append(train_step(model, opt, batch, wt, w_cls, dev, a.amp)); steps += 1
            if a.max_steps and steps >= a.max_steps: stop[0] = True
        sched.step(); L = torch.stack(tot).mean().item() if tot else float('nan')
        msg = f'[{ep}] L={L:.3f} train {time.time() - te:.0f}s steps={len(tot)}'
        last = stop[0] or time.time() - t0 > a.time_budget or ep == a.epochs
        if ep % a.eval_every == 0 or last:
            model.cpu(); r = V.evaluate(model, va, ev, ths); model.to(dev); msg += ' ' + json.dumps(r, ensure_ascii=False)
            if r['sel'] > best:
                best = r['sel']; torch.save(dict(model=model.state_dict(), hparams=hp, th1=r['th'], eval=r), f'beat_fcn_best_{a.tag}.pt')
        print(msg, flush=True)
        if last: break
    print(f'done best sel(f20+0.5*typeF)={best:.4f}', flush=True)

if __name__ == '__main__':
    main()

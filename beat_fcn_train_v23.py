#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v23 训练脚本 —— 网格 token Transformer（model_v23.BeatTokenNet）。

和 v22 的区别：
- 模型：注意力在节拍 token 上（默认每 16 格=半拍一个 token），不在音频帧上 → 单曲训练 FLOPs 约 1/60~1/200。
- 不用 gradient checkpointing，CPU 上也能整曲训练。
- 数据：首次读取把 (mel+flux+pitch 已标准化) 存成 float16 .npy 缓存（--cache-dir），之后 mmap 读，省掉解压/重算。
- 按 token 预算组 batch（--max-tokens，按帧数计），长度分桶。
- 评估修正：v22 的 inst(0ms) 和 grid(半格容差) 在格点预测上恒等（预测和真值都在格点上，半格容差=精确匹配）。
  v23 改为：strict = 格级精确 F1（选模型用）；loose = ±50ms 容差 F1（参考）；评估集固定（整曲、固定遮罩点），
  并在每次评估时扫阈值 th1，把最优阈值存进 ckpt（推理自动使用）。
- --strict-diff：只用文件名难度标签以 IN/AT 开头的谱面（v22 实际会混入 [4]/[SP]/[DISAPPEAR] 等）。
用法见 README。
"""
import os, sys, json, argparse, time, random, glob, re, math, signal
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_fcn_train_v22 import Songs, collate_fn, density_weighted_loss, f1_inst  # noqa: E402
from model_v23 import BeatTokenNet  # noqa: E402

DIFF_RE = re.compile(r'_\[(IN|AT)[^\]]*\]$', re.I)
TH_GRID = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


class CachedSongs(Songs):
    """Songs + float16 特征缓存（mmap）。缓存内容 = _load 的输出：feat (T,127) / grid / label。"""
    def __init__(self, *a, cache_dir=None, strict_diff=False, **kw):
        super().__init__(*a, **kw)
        if strict_diff:
            n0 = len(self.fids)
            self.fids = [f for f in self.fids if DIFF_RE.search(f)]
            print(f"[strict-diff] {n0} → {len(self.fids)} 个谱面（只留 IN/AT）", flush=True)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _load(self, fid):
        if not self.cache_dir:
            return super()._load(fid)
        fp = os.path.join(self.cache_dir, fid + '.feat.npy')
        lp = os.path.join(self.cache_dir, fid + '.gl.npy')
        if os.path.exists(fp) and os.path.exists(lp):
            feat = np.load(fp, mmap_mode='r')
            gl = np.load(lp)
            return feat, gl[0], gl[1]
        if fid in self.bad:
            return None
        item = super()._load(fid)
        if item is None:
            return None
        mel, grid, label = item
        np.save(fp + '.tmp.npy', mel.astype(np.float16)); os.replace(fp + '.tmp.npy', fp)
        np.save(lp + '.tmp.npy', np.stack([grid.astype(np.float32), label.astype(np.float32)])); os.replace(lp + '.tmp.npy', lp)
        self.cache.clear()  # 父类内存缓存不需要了
        return np.load(fp, mmap_mode='r'), grid, label

    def __getitem__(self, i):
        item = self._load(self.fids[i])
        if item is None:
            return None
        mel, grid, label = item
        return np.asarray(mel, dtype=np.float32), np.asarray(grid, np.float32), np.asarray(label).astype(np.int64)

    def length(self, i):
        fid = self.fids[i]
        if self.cache_dir:
            fp = os.path.join(self.cache_dir, fid + '.feat.npy')
            if os.path.exists(fp):
                return int(np.load(fp, mmap_mode='r').shape[0])
        it = self._load(fid)
        return 0 if it is None else int(it[0].shape[0])


class TokenBudgetSampler(torch.utils.data.Sampler):
    """按帧数预算组 batch：长度排序分桶 → 桶内打乱 → 贪心装箱（batch 内 最长×首数 ≤ max_tokens）。"""
    def __init__(self, lens, max_tokens, crop_frames=None, seed=0):
        self.lens, self.max_tokens, self.crop, self.seed = lens, max_tokens, crop_frames, seed
    def set_seed(self, s): self.seed = s
    def _batches(self):
        rng = random.Random(self.seed)
        idx = [i for i, l in enumerate(self.lens) if l > 0]
        idx.sort(key=lambda i: self.lens[i] + rng.random())
        batches, cur, mx = [], [], 0
        for i in idx:
            L = min(self.lens[i], self.crop) if self.crop else self.lens[i]
            if cur and max(mx, L) * (len(cur) + 1) > self.max_tokens:
                batches.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur: batches.append(cur)
        rng.shuffle(batches)
        return batches
    def __iter__(self): return iter(self._batches())
    def __len__(self): return len(self._batches())


def make_eval_set(songs, max_songs, seed, max_frames):
    """固定评估集：按 seed 抽 max_songs 首（整曲，不裁剪），过滤超长。"""
    fids = list(range(len(songs)))
    if max_songs and max_songs < len(fids):
        fids = random.Random(seed).sample(fids, max_songs)
    out = []
    for i in fids:
        it = songs[i]
        if it is None or it[0].shape[0] > max_frames:
            continue
        out.append(i)
    return out


def song_density(grid, label):
    """难度标量：log1p(音符数/秒)。"""
    dur = max(float(grid[-1] - grid[0]), 1.0) if len(grid) > 1 else 1.0
    return math.log1p(float((np.asarray(label) > 0).sum()) / dur)


def predict_scores(model, mel, grid, notes, device, dens=None):
    M = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0).to(device)
    G = torch.from_numpy(np.asarray(grid, np.float32)).unsqueeze(0).to(device)
    N = torch.from_numpy(notes.astype(np.float32)).unsqueeze(0).to(device)
    D = None if dens is None else torch.tensor([dens], dtype=torch.float32, device=device)
    o1, o2 = model(M, G, N, dens=D)
    return torch.sigmoid(o1)[0].float().cpu().numpy(), torch.sigmoid(o2)[0].float().cpu().numpy()


@torch.no_grad()
def evaluate(model, songs, idxs, device, cond_frac=0.3, th_grid=TH_GRID, th2=0.5, loose_tol=0.050):
    """固定口径评估。每首歌两次：盲采(全遮罩) + 条件采音(前 cond_frac 给真值)。只在遮罩段计分。
    返回 dict：每个 th1 的 strict/loose F1（盲采、条件、平均），以及最优 th1 下的双押 P/R。"""
    model.eval()
    acc = {th: {'s_b': [], 's_c': [], 'l_b': [], 'l_c': [], 'm_b': [], 'm_c': []} for th in th_grid}
    dbl = {th: [0, 0, 0] for th in th_grid}
    for i in idxs:
        mel, grid, label = songs[i]
        gt = label.astype(int)
        Gc = len(grid)
        for mode in ('b', 'c'):
            notes = np.full(Gc, -1.0)
            mpt = 0
            if mode == 'c':
                mpt = int(Gc * cond_frac)
                notes[:mpt] = gt[:mpt]
            s1, s2 = predict_scores(model, mel, grid, notes, device, song_density(grid, gt))
            g = grid[mpt:]; gt_m = gt[mpt:] > 0
            for th in th_grid:
                hit = s1[mpt:] >= th
                tp = int((hit & gt_m).sum()); fp = int((hit & ~gt_m).sum()); fn = int((~hit & gt_m).sum())
                acc[th]['s_' + mode].append(2 * tp / max(2 * tp + fp + fn, 1))
                acc[th]['l_' + mode].append(f1_inst(g[hit], g[gt_m], loose_tol)[0])
                acc[th]['m_' + mode].append(f1_inst(g[hit], g[gt_m], 0.020)[0])
                pd = hit & (s2[mpt:] >= th2); gd = gt[mpt:] == 2
                dbl[th][0] += int((pd & gd).sum()); dbl[th][1] += int((pd & ~gd).sum()); dbl[th][2] += int((~pd & gd).sum())
    res = {}
    for th in th_grid:
        a = {k: float(np.mean(v)) if v else 0.0 for k, v in acc[th].items()}
        a['strict'] = (a['s_b'] + a['s_c']) / 2
        a['loose'] = (a['l_b'] + a['l_c']) / 2
        a['f20'] = (a['m_b'] + a['m_c']) / 2
        tp, fp, fn = dbl[th]
        a['dprec'] = tp / max(tp + fp, 1); a['drec'] = tp / max(tp + fn, 1)
        res[th] = a
    best_th = max(th_grid, key=lambda t: res[t]['f20'])
    return res, best_th


def build_model(hp):
    return BeatTokenNet(n_mels=hp.get('n_mels', 81), n_pitch=hp.get('n_pitch', 46), stem_ch=hp.get('stem_ch', 48),
                        d_model=hp.get('d_model', 96), n_layers=hp.get('n_layers', 4), nhead=hp.get('nhead', 4),
                        cells_per_token=hp.get('cells_per_token', 16), samples_per_token=hp.get('samples_per_token', 16),
                        stem=hp.get('stem', 'sep'), cond_density=hp.get('cond_density', False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default='v23')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--threads', type=int, default=8, help='CPU 线程 / DataLoader worker（CPU 模式不用多进程）')
    ap.add_argument('--data', default='grid_npz_full128')
    ap.add_argument('--pitch-dir', default='pitch_npz_full')
    ap.add_argument('--split', default='split_full128_final.json')
    ap.add_argument('--cache-dir', default='feat_cache_v23', help='float16 特征缓存目录（空字符串=不缓存）')
    ap.add_argument('--strict-diff', action='store_true', help='只用 IN/AT 难度谱面')
    ap.add_argument('--max-songs', type=int, default=None)
    ap.add_argument('--max-tokens', type=int, default=120000, help='每个 batch 的帧数预算（≈首数×最长帧数）')
    ap.add_argument('--max-frames', type=int, default=40000, help='长歌过滤（v23 显存线性，可比 v22 放宽）')
    ap.add_argument('--crop-frames', type=int, default=None, help='随机裁剪片段最大帧数（默认整曲 60%%-100%%）')
    ap.add_argument('--eval-songs', type=int, default=300)
    ap.add_argument('--eval-seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--w-sparse', type=float, default=1.5)
    ap.add_argument('--w-dense', type=float, default=0.8)
    ap.add_argument('--w-double', type=float, default=4.5)
    ap.add_argument('--d-model', type=int, default=96)
    ap.add_argument('--stem-ch', type=int, default=48)
    ap.add_argument('--stem', default='sep', choices=['sep', 'conv'])
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--cells-per-token', type=int, default=16)
    ap.add_argument('--samples-per-token', type=int, default=16)
    ap.add_argument('--cond-density', action='store_true', help='难度/密度条件输入')
    ap.add_argument('--no-epoch-ckpt', action='store_true', help='不写逐轮 ckpt（省 inode）')
    ap.add_argument('--compile', action='store_true', help='torch.compile（GPU 上可再提速）')
    ap.add_argument('--time-budget', type=float, default=None, help='训练总时长上限（秒），到点评估并停止')
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    n_pitch = int(np.load(sorted(glob.glob(f'{args.pitch_dir}/*.npz'))[0])['pitch'].shape[1])
    kw = dict(pitch_dir=args.pitch_dir, pitch_mode='chroma', n_pitch=n_pitch, flux=True,
              cache_dir=args.cache_dir or None, strict_diff=args.strict_diff)
    tr = CachedSongs('train', args.data, args.split, **kw)
    va = CachedSongs('val', args.data, args.split, **kw)
    if args.max_songs:
        tr.fids = tr.fids[:args.max_songs]
    t0 = time.time()
    lens = [tr.length(i) for i in range(len(tr))]
    lens = [l if l <= args.max_frames else 0 for l in lens]
    eval_idx = make_eval_set(va, args.eval_songs, args.eval_seed, args.max_frames)
    print(f'v23: device={device} amp={use_amp} train={sum(l > 0 for l in lens)} eval={len(eval_idx)} '
          f'(缓存/长度扫描 {time.time()-t0:.0f}s)', flush=True)

    hp = dict(n_mels=81, n_pitch=n_pitch, stem_ch=args.stem_ch, d_model=args.d_model, n_layers=args.layers,
              nhead=4, cells_per_token=args.cells_per_token, samples_per_token=args.samples_per_token,
              stem=args.stem, arch='v23', cond_density=args.cond_density)
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
    ck = f'ckpt_{args.tag}.pth'
    if args.resume and os.path.exists(ck):
        d = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(d['model']); opt.load_state_dict(d['opt']); sched.load_state_dict(d['sched'])
        start_ep, best, hist = d['ep'] + 1, d['best'], d['hist']
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
        tot, n, t0, t_data, t_step = 0.0, 0, time.time(), 0.0, 0.0
        tb = time.time()
        for batch in loader:
            t_data += time.time() - tb; ts = time.time()
            if stop[0]: break
            if batch is None: tb = time.time(); continue
            M, G, L, mask, _ = batch
            M, G, L, mask = M.to(device, non_blocking=True), G.to(device), L.to(device), mask.to(device)
            B, Gm = L.shape
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
            loss = density_weighted_loss(o1.float(), o2.float(), L, w_cls, args.w_sparse, args.w_dense,
                                         mask=loss_mask, density_mask=mask)
            if not torch.isfinite(loss):
                opt.zero_grad(); tb = time.time(); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
            t_step += time.time() - ts; tb = time.time()
        sched.step()
        t_ep = time.time() - t0
        te = time.time()
        res, th = evaluate(model, va, eval_idx, device)
        r = res[th]
        t_eval = time.time() - te
        improved = r['f20'] > best
        if improved: best = r['f20']
        hist.append(dict(ep=ep, loss=tot / max(n, 1), th1=th, train_s=t_ep, eval_s=t_eval, **{k: r[k] for k in
                    ('strict', 'f20', 'loose', 's_b', 's_c', 'l_b', 'l_c', 'dprec', 'drec')}))
        print(f'[{ep}/{args.epochs}] strict={r["strict"]:.4f} (盲{r["s_b"]:.3f}/条件{r["s_c"]:.3f}) '
              f'f20ms={r["f20"]:.4f} loose50ms={r["loose"]:.4f} dP={r["dprec"]:.3f} dR={r["drec"]:.3f} th1={th} best={best:.4f} '
              f'L={tot/max(n,1):.3f} lr={opt.param_groups[0]["lr"]:.1e} '
              f'train {t_ep:.0f}s (数据{t_data:.0f}s/计算{t_step:.0f}s) eval {t_eval:.0f}s', flush=True)
        state = dict(model=model.state_dict(), hparams=hp, th1=th, ep=ep, best=best, hist=hist)
        if improved:
            torch.save(state, f'beat_fcn_best_{args.tag}.pt')
        torch.save(dict(state, opt=opt.state_dict(), sched=sched.state_dict()), ck)
        if not args.no_epoch_ckpt:
            torch.save(dict(model=model.state_dict(), hparams=hp, th1=th), f'ckpt_{args.tag}_ep{ep}.pt')
        json.dump(dict(best=best, hist=hist), open(f'ckpt_{args.tag}.json', 'w'), indent=1)
        if stop[0] or (args.time_budget and time.time() - t_train0 > args.time_budget):
            print('[停止] 到达时间预算或收到停止信号', flush=True); break
    print(f'done. best f20ms F1={best:.4f}', flush=True)


if __name__ == '__main__':
    main()

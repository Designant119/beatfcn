#!/usr/bin/env python3
"""BeatFCN v24：多类型选音（无/tap/hold/flick/drag + 双押 + 长条时长），密度条件，网格同 v23。
输出 7 维：[onset, double, type×4(tap,hold,flick,drag), log1p(hold 拍数)]。onset 含 drag（v23 标签不含 drag）。
标签来自 gen/typed_labels.py（--typed-dir）。可从 v23dens 热启动（--init，head 末层除外）。"""
import os, sys, json, argparse, time, random, math, signal
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_fcn_train_v22 import collate_fn, density_weighted_loss, f1_inst
from beat_fcn_train_v23 import CachedSongs, TokenBudgetSampler, make_eval_set, build_model
TYPES = {1: 'tap', 2: 'hold', 3: 'flick', 4: 'drag'}

def encode(typ, cnt, hb):
    return typ.astype(np.int64) + 10 * (cnt >= 2) + 100 * np.minimum(np.round(hb * 8), 4000).astype(np.int64)

def decode_t(L):
    typ = L % 10; dbl = (L // 10) % 10; hb = (L // 100).float() / 8
    return typ, dbl, hb

def beat_guide(T, beats, downs, lag=0.0133):
    """GenéLive 式节拍引导通道：下拍=2，其他拍=1（±1 帧），其余 0。"""
    g = np.zeros(T, np.float32)
    for arr, v in ((beats, 1.0), (downs, 2.0)):
        f = np.round((np.asarray(arr) - lag) * 22050 / 256).astype(int)
        for d in (-1, 0, 1):
            k = np.clip(f + d, 0, T - 1); g[k] = np.maximum(g[k], v)
    return g

def guide_channels(mel, beats, downs):
    from gen.accent import accent_curve
    return np.concatenate([mel, beat_guide(len(mel), beats, downs)[:, None], np.clip(accent_curve(mel), -2, 6)[:, None] / 2], 1).astype(np.float32)

class TypedSongs(CachedSongs):
    def __init__(self, *a, typed_dir=None, guide_dir=None, **kw):
        super().__init__(*a, **kw); self.typed_dir = typed_dir; self.guide_dir = guide_dir
        self.fids = [f for f in self.fids if os.path.exists(f'{typed_dir}/{f}.npz') and (not guide_dir or os.path.exists(f'{guide_dir}/{f}.npz'))]
    def __getitem__(self, i):
        it = super().__getitem__(i)
        if it is None: return None
        mel, grid, _ = it; z = np.load(f'{self.typed_dir}/{self.fids[i]}.npz'); n = len(grid)
        if self.guide_dir:
            b = np.load(f'{self.guide_dir}/{self.fids[i]}.npz'); mel = guide_channels(mel, b['beats'], b['downs'])
        return mel, grid, encode(z['typ'][:n], z['dbl'][:n], z['hold_beats'][:n])

@torch.no_grad()
def predict(model, mel, grid, dens):
    o = model(torch.from_numpy(np.ascontiguousarray(mel, np.float32))[None], torch.from_numpy(grid.astype(np.float32))[None],
              None, dens=torch.tensor([dens], dtype=torch.float32))[0]
    return torch.sigmoid(o[:, 0]).numpy(), torch.sigmoid(o[:, 1]).numpy(), o[:, 2:6].argmax(-1).numpy() + 1, torch.expm1(o[:, 6].clamp(0, 5)).numpy()

def nms(s, g, th, gap=0.040):
    idx = np.where(s >= th)[0]; blk = np.zeros(len(g), bool); keep = []
    lo = np.searchsorted(g, g - gap + 1e-6); hi = np.searchsorted(g, g + gap - 1e-6)
    for i in idx[np.argsort(-s[idx])]:
        if not blk[i]: keep.append(i); blk[lo[i]:hi[i]] = True
    return np.sort(np.array(keep, int))

def song_dens(g, on):
    return math.log1p(on.sum() / max(float(g[-1] - g[0]), 1.0))

@torch.no_grad()
def evaluate(model, ds, idxs, ths=(0.3, 0.4, 0.5, 0.6, 0.7)):
    model.eval(); acc = {th: [] for th in ths}; typ_pr = {th: {k: [0, 0, 0] for k in TYPES} for th in ths}; herr = []
    for i in idxs:
        mel, g, L = ds[i]; typ = L % 10; on = typ > 0; hb = (L // 100) / 8
        s1, s2, ty, hd = predict(model, mel, g, song_dens(g, on))
        for th in ths:
            sel = nms(s1, g, th); acc[th].append(f1_inst(g[sel], g[on], .02)[0])
            for k in TYPES:
                p = g[sel[ty[sel] == k]]; t = g[typ == k]; _, pr, rc = f1_inst(p, t, .02)
                tp = round(rc * len(t)); typ_pr[th][k][0] += tp; typ_pr[th][k][1] += len(p); typ_pr[th][k][2] += len(t)
        m = typ == 2
        if m.any(): herr.append(np.median(abs(hd[m] - hb[m])))
    best = max(ths, key=lambda t: np.mean(acc[t]))
    per = {TYPES[k]: dict(P=round(v[0] / max(v[1], 1), 3), R=round(v[0] / max(v[2], 1), 3), n_gt=v[2]) for k, v in typ_pr[best].items()}
    mf = float(np.mean([2 * v['P'] * v['R'] / max(v['P'] + v['R'], 1e-9) for v in per.values()]))
    return dict(f20=float(np.mean(acc[best])), type_macroF=mf, sel=float(np.mean(acc[best])) + 0.5 * mf, th=best, types=per, hold_mae_beats=float(np.median(herr)) if herr else None)

def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(data='grid_npz_clean', pitch_dir='pitch_npz_full', split='split_full128_final.json', cache_dir='feat_cache_clean',
                     typed_dir='typed_clean', tag='v24', init='', guide_dir='').items(): ap.add_argument('--' + k.replace('_', '-'), default=v)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--lr', type=float, default=7e-4); ap.add_argument('--max-tokens', type=int, default=80000)
    ap.add_argument('--time-budget', type=float, default=2700); ap.add_argument('--eval-every', type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    kw = dict(pitch_dir=a.pitch_dir, pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=a.cache_dir, typed_dir=a.typed_dir, guide_dir=a.guide_dir or None)
    tr = TypedSongs('train', a.data, a.split, **kw); va = TypedSongs('val', a.data, a.split, **kw)
    lens = [tr.length(i) for i in range(len(tr))]; lens = [l if l <= 40000 else 0 for l in lens]
    ev = make_eval_set(va, None, 42, 40000)
    hp = dict(n_mels=81, n_pitch=48 if a.guide_dir else 46, beat_guide=bool(a.guide_dir), stem_ch=48, d_model=96, n_layers=4, nhead=4, cells_per_token=16, samples_per_token=16,
              stem='sep', arch='v23', cond_density=True, n_out=7)
    model = build_model(hp); build_model.__globals__  # noqa
    if a.init:
        sd = torch.load(a.init, map_location='cpu', weights_only=False)['model']
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and (v.shape == msd[k].shape or k == 'stem.0.weight')}
        if sd['stem.0.weight'].shape != msd['stem.0.weight'].shape:  # 新增输入通道：零初始化
            w = torch.zeros_like(msd['stem.0.weight']); w[:, :sd['stem.0.weight'].shape[1]] = sd['stem.0.weight']; sd['stem.0.weight'] = w
        print('init', model.load_state_dict(sd, strict=False), flush=True)
    # 类型频率 → 类权重（1/sqrt）
    cnt = np.zeros(5)
    for i in range(len(tr)):
        z = np.load(f'{a.typed_dir}/{tr.fids[i]}.npz'); cnt += np.bincount(z['typ'], minlength=5)[:5]
    wt = torch.tensor((cnt[1:].sum() / cnt[1:]) ** 0.5, dtype=torch.float32); wt = wt / wt.mean()
    print(f'train={sum(l > 0 for l in lens)} eval={len(ev)} type counts={cnt[1:].astype(int).tolist()} w={wt.numpy().round(2).tolist()}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1, (e + 1) / 2) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(e / a.epochs, 1)))))
    sampler = TokenBudgetSampler(lens, a.max_tokens)
    loader = torch.utils.data.DataLoader(tr, batch_sampler=sampler, num_workers=0, collate_fn=lambda b: collate_fn(b, max_frames=40000))
    w_cls = torch.tensor([1.0, 3.0, 4.5]); best = -1; t0 = time.time(); stop = [False]
    signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__(0, True))
    for ep in range(1, a.epochs + 1):
        model.train(); sampler.set_seed(ep); tot = n = 0; te = time.time()
        for batch in loader:
            if batch is None or stop[0]: continue
            M, G, L, mask, _ = batch
            typ, dbl, hb = decode_t(L); on = (typ > 0); tgt = (on.long() + (dbl > 0).long() * on.long())
            dur = (G.masked_fill(mask == 0, -1e9).max(1).values - G.masked_fill(mask == 0, 1e9).min(1).values).clamp(min=1)
            dens = torch.log1p((on.float() * mask).sum(1) / dur)
            o = model(M, G, None, grid_mask=mask > 0, dens=dens)
            loss = density_weighted_loss(o[..., 0], o[..., 1], tgt, w_cls, 1.5, 0.8, mask=mask, density_mask=mask)
            m = on & (mask > 0)
            if m.any():
                loss = loss + 0.5 * F.cross_entropy(o[..., 2:6][m], (typ[m] - 1).long(), weight=wt)
                hm = m & (typ == 2)
                if hm.any(): loss = loss + 0.3 * F.smooth_l1_loss(o[..., 6][hm], torch.log1p(hb[hm]))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        sched.step(); msg = f'[{ep}] L={tot / max(n, 1):.3f} train {time.time() - te:.0f}s'
        last = stop[0] or time.time() - t0 > a.time_budget or ep == a.epochs
        if ep % a.eval_every == 0 or last:
            r = evaluate(model, va, ev); msg += ' ' + json.dumps(r, ensure_ascii=False)
            if r['sel'] > best:
                best = r['sel']; torch.save(dict(model=model.state_dict(), hparams=hp, th1=r['th'], eval=r), f'beat_fcn_best_{a.tag}.pt')
        print(msg, flush=True)
        if last: break
    print(f'done best sel(f20+0.5*typeF)={best:.4f}', flush=True)

if __name__ == '__main__':
    main()

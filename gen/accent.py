"""重音/重要度先验 + 漏音分析。
accent(feat, t)：t 时刻（±1 帧取最大）的 全频带正向 mel 差分 + 低频(0-15 bin)差分，按曲 z-score。
  python gen/accent.py analyze <bench_root> <ckpt> <layer1_dir>   # 用 v23/v24 在自动网格上 top-K，统计漏音
"""
import sys, os, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HR = 22050 / 256

def accent_curve(feat):
    m = np.asarray(feat[:, :80], np.float32); d = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0)
    full = d.sum(1); low = d[:, :16].sum(1)
    z = lambda x: (x - np.median(x)) / (x.std() + 1e-6)
    a = 0.5 * z(full) + 0.5 * z(low)
    return np.maximum.reduce([np.roll(a, -1), a, np.roll(a, 1)])

def accent_at(curve, t):
    return np.interp(t * HR, np.arange(len(curve)), curve)

def boosted(s1, curve, grid, downs=None, lam=0.15, down_bonus=0.05):
    """模型分 + λ·clip(重音z,0,4)/4 + 下拍加分。只改排序，不改时间。"""
    acc = np.clip(accent_at(curve, grid), 0, 4) / 4
    b = s1 + lam * acc
    if downs is not None and len(downs):
        j = np.clip(np.searchsorted(downs, grid), 1, len(downs) - 1)
        b = b + down_bonus * (np.minimum(abs(downs[j] - grid), abs(downs[j - 1] - grid)) < 0.02)
    return b

def topk(score, grid, k, nms_fn):
    sel = nms_fn(score, grid, -1e9)
    return np.sort(sel[np.argsort(-score[sel])][:k])

def analyze(root, ckpt, l1, lams=(0.0, 0.1, 0.2, 0.3)):
    from model_v23 import load_any_model
    from beat_fcn_train_v23 import CachedSongs
    from beat_fcn_train_v22 import f1_inst
    from beat_fcn_train_v24 import nms
    from gen.selector import auto_grid
    torch.set_num_threads(3); model, hp, _ = load_any_model(ckpt); v24 = hp.get('n_out', 2) != 2
    lb = json.load(open(f'{l1}/../layer1_metrics.json'))['summary']['lag_beat_ms'] / 1000
    ds = CachedSongs('test', f'{root}/grid_npz_clean', f'{root}/split_full128_final.json', pitch_dir=f'{root}/pitch_npz_full',
                     pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=None)
    res = {lam: dict(f20=[], rec_hi=[0, 0], rec_all=[0, 0]) for lam in lams}; miss = dict(drag=0, other=0, hi=0, n=0); per_song = {}
    for i, fid in enumerate(ds.fids):
        it = ds[i]
        if it is None or not os.path.exists(f'{l1}/{fid}.npz'): continue
        feat, g, l = it; z = np.load(f'{l1}/{fid}.npz'); beats = z['beats'] - lb; downs = z['downs'] - lb
        tz = np.load(f'{root}/typed_clean/{fid}.npz'); typ = tz['typ'][:len(g)]
        on = typ > 0 if v24 else l > 0; nt = g[on]; span = max(float(g[-1] - g[0]), 1.0)
        ag = auto_grid(beats, feat.shape[0] / HR); dens = math.log1p(on.sum() / span)
        with torch.no_grad():
            o = model(torch.from_numpy(feat)[None], torch.from_numpy(ag.astype(np.float32))[None], None, dens=torch.tensor([dens]))
        s1 = torch.sigmoid(o[0, :, 0] if v24 else o[0][0]).numpy()
        cur = accent_curve(feat); a_nt = accent_at(cur, nt); hi = a_nt >= np.percentile(a_nt, 75); K = int(on.sum())
        for lam in lams:
            sel = topk(boosted(s1, cur, ag, downs, lam, 0.05 if lam else 0), ag, K, nms); p = np.sort(ag[sel])
            res[lam]['f20'].append(f1_inst(p, nt, .02)[0])
            j = np.clip(np.searchsorted(p, nt), 1, max(len(p) - 1, 1)); hit = np.minimum(abs(p[j] - nt), abs(p[j - 1] - nt)) <= .02
            res[lam]['rec_hi'][0] += int(hit[hi].sum()); res[lam]['rec_hi'][1] += int(hi.sum())
            res[lam]['rec_all'][0] += int(hit.sum()); res[lam]['rec_all'][1] += len(nt)
            if lam == 0:
                ty = typ[on]; miss['n'] += int((~hit).sum()); miss['drag'] += int((~hit & (ty == 4)).sum()); miss['hi'] += int((~hit & hi).sum())
                per_song[fid] = dict(recall=round(float(hit.mean()), 3), miss_drag_frac=round(float((~hit & (ty == 4)).sum() / max((~hit).sum(), 1)), 3),
                                     recall_drag=round(float(hit[ty == 4].mean()) if (ty == 4).any() else -1, 3), recall_hi_accent=round(float(hit[hi].mean()), 3))
    out = {str(lam): dict(f20=round(float(np.mean(r['f20'])), 4), recall=round(r['rec_all'][0] / r['rec_all'][1], 4),
                          recall_hi_accent=round(r['rec_hi'][0] / r['rec_hi'][1], 4)) for lam, r in res.items()}
    out['missed_at_lam0'] = dict(total=miss['n'], drag_frac=round(miss['drag'] / max(miss['n'], 1), 3), hi_accent_frac=round(miss['hi'] / max(miss['n'], 1), 3))
    out['demo'] = {k: v for k, v in per_song.items() if k.split('_')[0] in ('74587', '10096', '4988')}
    print(json.dumps(out, ensure_ascii=False, indent=1)); return out

if __name__ == '__main__':
    o = analyze(*sys.argv[2:5]); json.dump(o, open(sys.argv[5] if len(sys.argv) > 5 else 'accent_analysis.json', 'w'), ensure_ascii=False, indent=1)

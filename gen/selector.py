"""Layer 2：在自动节拍网格（beat_this 节拍 32 等分）上用 v23(条件密度) 选音，并评估。
  python gen/selector.py eval <bench_root> <ckpt> <layer1_dir>
"""
import sys, os, json, glob, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v23 import load_any_model
from beat_fcn_train_v22 import f1_inst, compute_flux
from gen.layer1 import subdiv, signed_lag

def features(y):
    """mel80 + flux + pitch46（各自标准化），与训练 Songs._load 一致。"""
    from chart_utils import mel_of
    from extract_pitch import extract_pitch
    mel = mel_of(y)[:, :80]; mel = np.concatenate([mel, compute_flux(mel)], 1)
    p = extract_pitch(y).astype(np.float32); n = min(len(p), len(mel))
    p = (p[:n] - p[:n].mean(0)) / (p[:n].std(0) + 1e-5)
    return np.concatenate([mel[:n], p], 1).astype(np.float32)

@torch.no_grad()
def score(model, feat, grid, dens):
    M = torch.from_numpy(feat).unsqueeze(0); G = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0)
    o1, o2 = model(M, G, None, dens=torch.tensor([dens], dtype=torch.float32))
    return torch.sigmoid(o1)[0].numpy(), torch.sigmoid(o2)[0].numpy()

def select(s1, grid, th, min_gap=0.040):
    """阈值 + 时间 NMS（min_gap 内只留最高分格），避免同一起音选两个格。"""
    idx = np.where(s1 >= th)[0]; blocked = np.zeros(len(grid), bool); keep = []
    lo = np.searchsorted(grid, grid - min_gap + 1e-6); hi = np.searchsorted(grid, grid + min_gap - 1e-6)
    for i in idx[np.argsort(-s1[idx])]:
        if blocked[i]: continue
        keep.append(i); blocked[lo[i]:hi[i]] = True
    return np.sort(np.array(keep, int))

def auto_grid(beats, dur):
    g = subdiv(beats, 32); return g[(g >= 0) & (g < dur)]

def run_eval(root, ckpt, l1):
    torch.set_num_threads(3)
    model, hp, _ = load_any_model(ckpt)
    split = json.load(open(f'{root}/split_full128_final.json'))
    from beat_fcn_train_v23 import CachedSongs
    kw = dict(pitch_dir=f'{root}/pitch_npz_full', pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=None)
    S = {}
    for sp in ('val', 'test'):
        ds = CachedSongs(sp, f'{root}/grid_npz_clean', f'{root}/split_full128_final.json', **kw)
        for i, fid in enumerate(ds.fids):
            it = ds[i]
            if it is None or not os.path.exists(f'{l1}/{fid}.npz'): continue
            S[fid] = (sp,) + it
    lb = json.load(open(f'{l1}/../layer1_metrics.json'))['summary']['lag_beat_ms'] / 1000
    TH = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; rows = []
    for fid, (sp, feat, g, l) in S.items():
        nt = g[l > 0]; dur = feat.shape[0] * 256 / 22050; span = max(float(g[-1] - g[0]), 1.0)
        d_true = math.log1p(len(nt) / span)
        beats = np.load(f'{l1}/{fid}.npz')['beats'] - lb; ag = auto_grid(beats, dur)
        r = dict(fid=fid, split=sp, nps_true=len(nt) / span)
        s_chart, _ = score(model, feat, g, d_true)
        s_auto, s2 = score(model, feat, ag, d_true)
        for th in TH:
            r[f'f20_chart_{th}'] = f1_inst(g[select(s_chart, g, th)], nt, .02)[0]
            r[f'f20_auto_{th}'] = f1_inst(ag[select(s_auto, ag, th)], nt, .02)[0]
            r[f'f50_auto_{th}'] = f1_inst(ag[select(s_auto, ag, th)], nt, .05)[0]
        r['s_auto'] = s_auto; r['ag'] = ag; r['feat'] = feat; r['d_true'] = d_true
        rows.append(r)
    best = {k: max(TH, key=lambda t: np.mean([r[f'f20_{k}_{t}'] for r in rows if r['split'] == 'val'])) for k in ('chart', 'auto')}
    out = dict(th=best)
    for sp in ('val', 'test'):
        R = [r for r in rows if r['split'] == sp]
        out[sp] = dict(n=len(R), f20_chartgrid=float(np.mean([r[f"f20_chart_{best['chart']}"] for r in R])),
                       f20_autogrid=float(np.mean([r[f"f20_auto_{best['auto']}"] for r in R])),
                       f50_autogrid=float(np.mean([r[f"f50_auto_{best['auto']}"] for r in R])))
    # 密度可控性（test）：请求 nps = 真值 × {0.5,0.75,1,1.5,2}，看产出 nps
    req, prod = [], []
    for r in [r for r in rows if r['split'] == 'test']:
        span = max(float(r['ag'][-1] - r['ag'][0]), 1.0)
        for m in (0.5, 0.75, 1.0, 1.5, 2.0):
            q = r['nps_true'] * m; s, _ = score(model, r['feat'], r['ag'], math.log1p(q))
            req.append(q); prod.append(len(select(s, r['ag'], best['auto'])) / span)
    req, prod = np.array(req), np.array(prod)
    out['density'] = dict(pearson=float(np.corrcoef(req, prod)[0, 1]),
                          spearman=float(np.corrcoef(np.argsort(np.argsort(req)), np.argsort(np.argsort(prod)))[0, 1]),
                          median_ratio_prod_over_req=float(np.median(prod / req)))
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{root}/layer2_metrics.json', 'w'), indent=1)

if __name__ == '__main__':
    run_eval(*sys.argv[2:5])

"""类型分配评估（自动网格 + top-K + 重音先验之后）：比较 模型 argmax / logit 偏置(val 上调) / +连打→drag 启发式。
各类型 P/R 都按 ±20ms 时间匹配到同类型谱面音符。  python gen/types_eval.py <root> <v24.pt> <layer1_dir>"""
import sys, os, json, math, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v23 import load_any_model
from beat_fcn_train_v23 import CachedSongs
from beat_fcn_train_v22 import f1_inst
from beat_fcn_train_v24 import nms, guide_channels
from gen.selector import auto_grid
from gen.accent import accent_curve, boosted, topk
HR = 22050 / 256; T = {1: 'tap', 2: 'hold', 3: 'flick', 4: 'drag'}

def run_heur(t, types, gap=0.11, min_len=4):
    """连打段（相邻间隔 < gap 且长度 ≥ min_len，且不含长条）→ 除首音外改为 drag。"""
    out = types.copy(); i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] - t[j] < gap: j += 1
        if j - i + 1 >= min_len: out[i + 1:j + 1] = np.where(out[i + 1:j + 1] == 2, 2, 4)
        i = j + 1
    return out

def main(root, ckpt, l1, lam=0.3):
    torch.set_num_threads(3); model, hp, _ = load_any_model(ckpt)
    lb = json.load(open(f'{l1}/../layer1_metrics.json'))['summary']['lag_beat_ms'] / 1000
    data = {}
    for sp in ('val', 'test'):
        ds = CachedSongs(sp, f'{root}/grid_npz_clean', f'{root}/split_full128_final.json', pitch_dir=f'{root}/pitch_npz_full',
                         pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=None)
        rows = []
        for i, fid in enumerate(ds.fids):
            it = ds[i]
            if it is None or not os.path.exists(f'{l1}/{fid}.npz') or not os.path.exists(f'{root}/typed_clean/{fid}.npz'): continue
            feat, g, _ = it; z = np.load(f'{l1}/{fid}.npz')
            if hp.get('beat_guide'): feat = guide_channels(feat, z['beats'], z['downs'])
            typ = np.load(f'{root}/typed_clean/{fid}.npz')['typ'][:len(g)]; on = typ > 0
            span = max(float(g[-1] - g[0]), 1.0); ag = auto_grid(z['beats'] - lb, feat.shape[0] / HR)
            with torch.no_grad():
                o = model(torch.from_numpy(feat)[None], torch.from_numpy(ag.astype(np.float32))[None], None, dens=torch.tensor([math.log1p(on.sum() / span)]))[0]
            s1 = torch.sigmoid(o[:, 0]).numpy(); lg = o[:, 2:6].numpy()
            sel = topk(boosted(s1, accent_curve(feat), ag, z['downs'] - lb, lam, 0.05), ag, int(on.sum()), nms)
            rows.append((ag[sel], lg[sel], {k: g[typ == k] for k in T}, g[on]))
        data[sp] = rows
    def score(rows, bias, heur):
        acc = {k: [0, 0, 0] for k in T}; f = []
        for t, lg, gt, gall in rows:
            ty = (lg + bias).argmax(1) + 1
            if heur: ty = run_heur(t, ty)
            f.append(f1_inst(t, gall, .02)[0])
            for k in T:
                p = t[ty == k]; _, pr, rc = f1_inst(p, gt[k], .02); acc[k][0] += rc * len(gt[k]); acc[k][1] += len(p); acc[k][2] += len(gt[k])
        per = {T[k]: dict(P=round(v[0] / max(v[1], 1), 3), R=round(v[0] / max(v[2], 1), 3)) for k, v in acc.items()}
        mf = np.mean([2 * v['P'] * v['R'] / max(v['P'] + v['R'], 1e-9) for v in per.values()])
        return per, float(mf), float(np.mean(f))
    grid = [np.array([0, a, b, c]) for a, b, c in itertools.product((0, .5, 1, 1.5), (0, .5, 1, 1.5), (0, .5, 1, 1.5, 2))]
    out = {}
    for heur in (False, True):
        best = max(grid, key=lambda b: score(data['val'], b, heur)[1])
        out[f'heur={heur}'] = dict(bias=best.tolist(), test=score(data['test'], best, heur)[0], test_macroF=score(data['test'], best, heur)[1])
    out['argmax_test'] = score(data['test'], np.zeros(4), False)[0]
    out['onset_f20_test'] = score(data['test'], np.zeros(4), False)[2]
    print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open(f'{root}/types_eval.json', 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main(*sys.argv[1:4])

"""±20ms 上限分析：网格量化 oracle、同曲不同难度谱面一致性、现有 ckpt 的 ±20ms 评估。"""
import sys, os, glob, json, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beat_fcn_train_v22 import f1_inst
root = sys.argv[1]; torch.set_num_threads(2)
split = json.load(open(f'{root}/split_full128_final.json'))
files = sorted(glob.glob(f'{root}/grid_npz_full128/*.npz'))
half = []; merged = 0; ntap = 0
by = collections.defaultdict(list)
for f in files:
    d = np.load(f); g = d['grid']; l = d['label']
    half.append(np.median(np.diff(g)) / 2 * 1000)
    merged += int((l == 2).sum()); ntap += int(l.sum())
    by[os.path.basename(f).split('_')[0]].append(g[l > 0])
half = np.array(half)
print(f'grid half-step ms: median {np.median(half):.1f} max {half.max():.1f} (>20ms: {(half>20).sum()} charts)')
print(f'label==2 cells {merged}, taps {ntap}')
agree = []
for k, v in by.items():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            agree.append(f1_inst(v[i], v[j], 0.020)[0])
print(f'同曲不同难度 onset F1@20ms: n={len(agree)} mean {np.mean(agree):.3f} median {np.median(agree):.3f}')
if len(sys.argv) > 2:
    import beat_fcn_train_v23 as V
    kw = dict(pitch_dir=f'{root}/pitch_npz_full', pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=f'{root}/feat_cache_v23')
    va = V.CachedSongs('val', f'{root}/grid_npz_full128', f'{root}/split_full128_final.json', **kw)
    idx = V.make_eval_set(va, 300, 42, 40000)
    for ck in sys.argv[2:]:
        d = torch.load(ck, map_location='cpu', weights_only=False)
        m = V.build_model(d['hparams']); m.load_state_dict(d['model'])
        res, th = V.evaluate(m, va, idx, torch.device('cpu'))
        r = res[th]; print(ck, f"th={th} strict={r['strict']:.4f} f20={r['f20']:.4f} loose50={r['loose']:.4f} dP={r['dprec']:.3f} dR={r['drec']:.3f}", flush=True)

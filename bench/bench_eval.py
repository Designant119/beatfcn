#!/usr/bin/env python3
"""v22 / v23 同口径精度对比：固定 val 选阈值 th1 → 固定 test 报分（整曲、盲采 + 前30%条件采音）。
strict = 格级精确 F1；loose = ±50ms F1。
用法: python bench/bench_eval.py --root DATA_ROOT --ckpt 名字=路径 [名字=路径 ...]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from beat_fcn_train_v23 import CachedSongs, make_eval_set, evaluate
from model_v23 import load_any_model

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--ckpt', nargs='+', required=True)
ap.add_argument('--max-frames', type=int, default=14000, help='v22 整曲 CPU 推理内存上限，两者用同一批歌')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--out', default='bench_eval_result.json')
a = ap.parse_args()
torch.set_num_threads(a.threads)
R = a.root
kw = dict(pitch_dir=f'{R}/pitch_npz_full', pitch_mode='chroma', n_pitch=46, flux=True, cache_dir=f'{R}/feat_cache_v23')
va = CachedSongs('val', f'{R}/grid_npz_full128', f'{R}/split_full128_final.json', **kw)
te = CachedSongs('test', f'{R}/grid_npz_full128', f'{R}/split_full128_final.json', **kw)
vi = make_eval_set(va, None, 0, a.max_frames); ti = make_eval_set(te, None, 0, a.max_frames)
print(f'val {len(vi)} 首 / test {len(ti)} 首（≤{a.max_frames} 帧）', flush=True)
rows = []
for spec in a.ckpt:
    name, path = spec.split('=', 1)
    m, hp, _ = load_any_model(path)
    rv, th = evaluate(m, va, vi, 'cpu')
    t = time.time()
    rt, _ = evaluate(m, te, ti, 'cpu', th_grid=[th])
    dt = time.time() - t
    r = rt[th]
    row = dict(name=name, th1=th, val_strict=rv[th]['strict'], test_strict=r['strict'], test_loose=r['loose'],
               blind_strict=r['s_b'], cond_strict=r['s_c'], dprec=r['dprec'], drec=r['drec'],
               infer_s_per_song=dt / max(len(ti), 1) / 2, params=sum(p.numel() for p in m.parameters()))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.append(row)
json.dump(rows, open(a.out, 'w'), indent=1, ensure_ascii=False)

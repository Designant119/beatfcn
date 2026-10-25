"""从原始 RPE/PEC 谱面提取音符类型标签，对齐到清洗后的网格（grid_npz_clean）。
输出 <dst>/<fid>.npz: typ (G,) int8 0无/1tap/2hold/3flick/4drag（同格多音取 hold>flick>drag>tap），
hold_beats (G,) float32 长条时长（拍），dbl (G,) 同格音符数。并打印类型统计。
  python gen/typed_labels.py <bench_root> <dst>
"""
import sys, os, glob, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_full import load_zip
from chart_utils import parse_chart, parse_pec, chart_sec, beat_to_sec, beats_of
from beat_fcn_train_v22 import f1_inst
PRI = {2: 4, 3: 3, 4: 2, 1: 1}

def chart_notes(chart, info_offset):
    mo = chart.get('META', {}).get('offset')
    off = float(mo) / 1000.0 if mo is not None else info_offset
    segs = chart_sec(chart); out = []
    for jl in chart.get('judgeLineList', []):
        for n in jl.get('notes', []):
            if n.get('isFake'): continue
            t = n.get('type')
            if t not in (1, 2, 3, 4): continue
            b0 = beats_of(n['startTime']); b1 = beats_of(n.get('endTime', n['startTime']))
            out.append((beat_to_sec(b0, segs) + off, t, max(b1 - b0, 0.0), beat_to_sec(b1, segs) + off))
    return out

def main(root, dst):
    os.makedirs(dst, exist_ok=True)
    rep = json.load(open(f'{root}/offset_report.json'))
    files = sorted(glob.glob(f'{root}/grid_npz_clean/*.npz'))
    byid = collections.defaultdict(list)
    for f in files: byid[os.path.basename(f).split('_')[0]].append(f)
    stat = collections.Counter(); per_song = []
    for sid, fl in byid.items():
        got = load_zip(f'{root}/charts/{sid}.zip')
        if got is None: continue
        jbs, pbs, _, info_off = got[:4]
        charts = []
        for _, b in jbs + pbs:
            txt = b.decode('utf-8-sig', errors='replace'); c = parse_chart(txt) or parse_pec(txt)
            if c: charts.append(chart_notes(c, info_off))
        for f in fl:
            fid = os.path.basename(f)[:-4]; d = np.load(f); g, l = d['grid'], d['label']
            orig = np.load(f'{root}/grid_npz_full128/{fid}.npz'); og = orig['grid'][orig['label'] > 0]
            best = max(charts, key=lambda c: f1_inst(np.sort([x[0] for x in c if x[1] != 4]), og, .003)[0], default=None)
            if best is None: continue
            s = rep[fid]['offset_ms'] / 1000.0
            typ = np.zeros(len(g), np.int8); hb = np.zeros(len(g), np.float32); cnt = np.zeros(len(g), np.int8)
            half = float(np.median(np.diff(g))) / 2 + 1e-4
            for t, ty, dur, _ in best:
                t += s; j = int(np.clip(np.searchsorted(g, t), 1, len(g) - 1)); j = j if abs(g[j] - t) < abs(g[j - 1] - t) else j - 1
                if abs(g[j] - t) > half: continue
                stat[ty] += 1; cnt[j] += 1
                if PRI[ty] > PRI.get(int(typ[j]), 0): typ[j] = ty
                if ty == 2: hb[j] = max(hb[j], dur)
            np.savez(f'{dst}/{fid}.npz', typ=typ, hold_beats=hb, dbl=cnt)
            per_song.append({k: int((typ == k).sum()) for k in (1, 2, 3, 4)})
    tot = sum(stat.values())
    print('notes by type:', {k: (v, round(v / tot, 3)) for k, v in sorted(stat.items())})
    print('songs with any hold/flick/drag:', {k: sum(p[k] > 0 for p in per_song) for k in (2, 3, 4)}, 'of', len(per_song))

if __name__ == '__main__':
    main(*sys.argv[1:3])

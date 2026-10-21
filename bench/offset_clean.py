"""每谱面全局 offset 估计（音符时刻 vs 音频 onset 强度互相关，±200ms，2ms 步），质量标记，输出修正后 grid npz。
用法: python bench/offset_clean.py <src_npz_dir> <dst_npz_dir> <report.json>"""
import sys, os, glob, json, numpy as np
HR = 22050 / 256
src, dst, rep = sys.argv[1:4]; os.makedirs(dst, exist_ok=True)
SH = np.arange(-0.200, 0.2001, 0.002)
def env_of(mel):
    m = mel.astype(np.float32)
    d = np.maximum(m[1:] - m[:-1], 0).sum(1); d = np.concatenate([[0], d])
    k = np.array([.25, .5, .25]); d = np.convolve(d, k, 'same')
    loc = np.convolve(d, np.ones(43) / 43, 'same')          # 0.5s 局部均值，去掉响度
    return (d - loc) / (d.std() + 1e-6)
def peaks(e):
    return np.where((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > 0.5))[0] + 1
out = {}
for f in sorted(glob.glob(f'{src}/*.npz')):
    fid = os.path.basename(f)[:-4]; d = np.load(f); mel, g, l = d['mel'], d['grid'], d['label']
    T = len(mel); e = env_of(mel); fr = np.arange(T) / HR
    nt = g[l > 0]; dur = T / HR
    if len(nt) < 20:
        out[fid] = dict(n=int(len(nt)), drop='too_few_notes'); continue
    sc = np.array([np.interp(nt + s, fr, e).mean() for s in SH])
    i = int(sc.argmax()); s = float(SH[i])
    conf = float((sc[i] - np.median(sc)) / (sc.std() + 1e-6))
    pk = peaks(e) / HR
    def agree(off):
        t = nt + off; j = np.clip(np.searchsorted(pk, t), 1, len(pk) - 1)
        return float((np.minimum(abs(pk[j] - t), abs(pk[j - 1] - t)) <= 0.020).mean())
    dens = len(nt) / dur
    r = dict(n=int(len(nt)), offset_ms=round(s * 1000, 1), conf=round(conf, 2), edge=bool(i in (0, len(SH) - 1)),
             agree20_before=round(agree(0), 3), agree20_after=round(agree(s), 3), nps=round(dens, 2))
    reasons = []
    if r['edge']: reasons.append('offset_at_search_edge')
    if conf < 2.0: reasons.append('low_conf')
    if r['agree20_after'] < 0.25: reasons.append('low_agree')
    if dens > 20 or dens < 0.5: reasons.append('extreme_density')
    r['drop'] = ','.join(reasons) or None
    out[fid] = r
    g2 = g + s; keep = (g2 >= 0) & (g2 < dur)
    np.savez(f'{dst}/{fid}.npz', mel=mel, grid=g2[keep].astype(np.float32), label=l[keep])
json.dump(out, open(rep, 'w'), indent=1, ensure_ascii=False)
v = [r for r in out.values() if 'offset_ms' in r]; o = np.array([r['offset_ms'] for r in v])
print(f'charts {len(out)}  offset ms: median {np.median(o):.1f}  |o| p50 {np.median(abs(o)):.0f} p90 {np.percentile(abs(o),90):.0f}  |o|>20ms {(abs(o)>20).mean():.2f}')
print('agree20 before %.3f after %.3f' % (np.mean([r['agree20_before'] for r in v]), np.mean([r['agree20_after'] for r in v])))
from collections import Counter; print('drop:', Counter(r['drop'] for r in out.values()))

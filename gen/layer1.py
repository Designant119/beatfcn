"""Layer 1：自动节拍/下拍跟踪（beat_this）+ onset 候选（librosa），并对清洗后的谱面评估。
需在 venv_bt（torch 2.11 + torchaudio + beat_this + librosa）里运行。
  python gen/layer1.py detect <audio_file> <out.npz>
  python gen/layer1.py eval <bench_root> <out_dir>      # val+test，输出 metrics json
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 22050

def detect(y, sr=SR):
    import torch, librosa
    torch.set_num_threads(3)
    from beat_this.inference import Audio2Beats
    a2b = Audio2Beats(checkpoint_path='final0', device='cpu', dbn=False)
    beats, downs = a2b(y, sr)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=256, units='time', delta=0.02, wait=2)
    return np.asarray(beats, float), np.asarray(downs, float), np.asarray(on, float)

def smooth_beats(b, w=4):
    """beat_this 输出量化到 20ms 帧；用 ±w 拍局部线性回归平滑（保留变速），离群拍(>40ms)不动。"""
    b = np.asarray(b, float); out = b.copy()
    for k in range(len(b)):
        lo, hi = max(0, k - w), min(len(b), k + w + 1)
        if hi - lo < 4: continue
        x = np.arange(lo, hi); p = np.polyfit(x, b[lo:hi], 1); v = np.polyval(p, k)
        if abs(v - b[k]) < 0.040: out[k] = v
    return out

def subdiv(beats, n):
    """节拍间 n 等分（含首尾外推一拍）。"""
    if len(beats) < 2: return beats
    b = np.concatenate([[2 * beats[0] - beats[1]], beats, [2 * beats[-1] - beats[-2]]])
    return np.concatenate([b[:-1, None] + (b[1:] - b[:-1])[:, None] * np.arange(n) / n]).ravel()

def cover(c, t, tol=0.020):
    c = np.sort(c)
    if not len(c) or not len(t): return 0.0
    j = np.clip(np.searchsorted(c, t), 1, len(c) - 1)
    return float((np.minimum(abs(c[j] - t), abs(c[j - 1] - t)) <= tol).mean())

def snap_onsets(beats, on, n=(4, 3), tol=0.035):
    """细分网格点里，±tol 内有 onset 的保留（+ 所有 1/2 拍点），onset 原样也加入。"""
    g = np.unique(np.concatenate([subdiv(beats, k) for k in n]))
    j = np.clip(np.searchsorted(on, g), 1, max(len(on) - 1, 1))
    d = np.minimum(abs(on[j] - g), abs(on[j - 1] - g)) if len(on) > 1 else np.full(len(g), 9.)
    return np.unique(np.concatenate([g[d <= tol], subdiv(beats, 2)]))

def beat_f(est, ref, tol):
    from beat_fcn_train_v22 import f1_inst
    return f1_inst(est, ref, tol)[0]

def chart_beats(grid, label, div=32):
    """chart 的整拍时刻：选使音符最多落在 idx%div==k 的相位 k。"""
    idx = np.where(label > 0)[0]
    k = int(np.bincount(idx % div, minlength=div).argmax()) if len(idx) else 0
    return grid[k::div]

def signed_lag(c, t, w=0.06):
    j = np.clip(np.searchsorted(c, t), 1, len(c) - 1)
    d = np.where(abs(c[j] - t) < abs(c[j - 1] - t), c[j] - t, c[j - 1] - t)
    return d[abs(d) < w]

def candidates(beats, on, lag_b=0.0, lag_o=0.0):
    b, o = beats - lag_b, on - lag_o
    g16 = subdiv(b, 4); g12 = subdiv(b, 3)
    return {'onset': o, 'grid16': g16, 'grid16+12': np.unique(np.concatenate([g16, g12])),
            'snap': snap_onsets(b, o), 'onset+grid16': np.unique(np.concatenate([o, g16])),
            'onset+grid16+12': np.unique(np.concatenate([o, g16, g12]))}

def run_eval(root, out):
    from extract_full import load_zip
    from chart_utils import decode_audio
    split = json.load(open(f'{root}/split_full128_final.json'))
    os.makedirs(out, exist_ok=True); songs = []
    for f in sorted(glob.glob(f'{root}/grid_npz_clean/*.npz')):
        fid = os.path.basename(f)[:-4]; sid = fid.split('_')[0]
        if split.get(sid) not in ('val', 'test'): continue
        cache = f'{out}/{fid}.npz'
        if os.path.exists(cache):
            z = np.load(cache); beats, downs, on = z['beats'], z['downs'], z['on']
        else:
            y = decode_audio(load_zip(f'{root}/charts/{sid}.zip')[2])
            beats, downs, on = detect(y)
            np.savez(cache, beats=beats, downs=downs, on=on)
        d = np.load(f); songs.append((fid, split[sid], beats, on, d['grid'], d['label']))
    # 常数延迟在 val 上标定（检测器峰值相对真实起音的系统滞后），test 只用
    lb = float(np.median(np.concatenate([signed_lag(s[2], s[4][s[5] > 0]) for s in songs if s[1] == 'val'])))
    lo = float(np.median(np.concatenate([signed_lag(s[3], s[4][s[5] > 0]) for s in songs if s[1] == 'val'])))
    rows = []
    for fid, sp, beats, on, g, l in songs:
        nt = g[l > 0]; cb = chart_beats(g, l); b = beats - lb
        step = np.diff(g); var_bpm = bool(step.max() / max(step.min(), 1e-6) > 1.02)
        r = dict(fid=fid, split=sp, n_notes=int(len(nt)), var_bpm=var_bpm,
                 beatF70=beat_f(b, cb, .07), beatF20=beat_f(b, cb, .02),
                 beatF70_any=max(beat_f(b, cb, .07), beat_f(subdiv(b, 2), cb, .07), beat_f(b[::2], cb, .07)))
        for k, c in candidates(beats, on, lb, lo).items():
            r[f'rec_{k}'] = cover(c, nt); r[f'cpn_{k}'] = len(c) / max(len(nt), 1)
        rows.append(r)
    keys = [k for k in rows[0] if k.startswith(('rec_', 'cpn_', 'beatF'))]
    summ = dict(lag_beat_ms=lb * 1000, lag_onset_ms=lo * 1000)
    for sp in ('val', 'test'):
        R = [r for r in rows if r['split'] == sp]
        summ[sp] = {k: float(np.median([r[k] for r in R])) for k in keys}
        summ[sp]['n'] = len(R); V = [r['beatF70_any'] for r in R if r['var_bpm']]
        summ[sp]['var_bpm_n'] = len(V); summ[sp]['beatF70_any_varbpm'] = float(np.median(V)) if V else None
    json.dump(dict(summary=summ, rows=rows), open(f'{out}/../layer1_metrics.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))

if __name__ == '__main__':
    if sys.argv[1] == 'detect':
        import librosa
        y, _ = librosa.load(sys.argv[2], sr=SR, mono=True)
        b, dn, on = detect(y); np.savez(sys.argv[3], beats=b, downs=dn, on=on)
    else:
        run_eval(sys.argv[2], sys.argv[3])

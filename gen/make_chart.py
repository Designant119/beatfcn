"""端到端：音频 → 自动节拍网格 → v23 选音（按目标密度）→ Phira 可导入的 RPE 谱面 zip。
需 venv_bt（beat_this + torchaudio + librosa）。
  python gen/make_chart.py <audio> <ckpt> <out_dir> [--nps 4.0] [--name 曲名] [--no-audio-in-zip]
谱面形态（骨架）：单条静止判定线（y=-250，速度 10），只有 Tap；音符左右交替 ±220（双押=±300 两个）。
BPM 表：每个检测到的节拍一段（变速自然支持）；第 0 拍固定在 t=0，第 1 拍=首个检测拍，offset=0。
"""
import sys, os, json, math, argparse, zipfile, subprocess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen.layer1 import detect, subdiv
from gen.selector import features, score, select
from model_v23 import load_any_model
SR = 22050; DIV = 32
LAG_BEAT = 0.0133  # beat_this 相对谱面起音的系统滞后，val 上标定（layer1_metrics.json）

def decode(path):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)

def frac(b):
    k = int(math.floor(b + 1e-9)); n = int(round((b - k) * DIV))
    if n == DIV: k, n = k + 1, 0
    g = math.gcd(n, DIV); return [k, n // g, DIV // g]

def ev(start, end, t0=(0, 0, 1), t1=(1, 0, 1)):
    return dict(bezier=0, bezierPoints=[0.0] * 4, easingLeft=0.0, easingRight=1.0, easingType=1,
                start=start, end=end, startTime=list(t0), endTime=list(t1), linkgroup=0)

def build_rpe(beats_t, note_beats, doubles, name, level):
    # 拍 0 在 t=0；拍 k(k>=1) 在 beats_t[k-1]
    bt = np.concatenate([[0.0], beats_t])
    bpm = [dict(bpm=float(60.0 / max(bt[k + 1] - bt[k], 1e-3)), startTime=[k, 0, 1]) for k in range(len(bt) - 1)]
    notes, side = [], 1
    for b, dbl in zip(note_beats, doubles):
        xs = [-300.0, 300.0] if dbl else [220.0 * side]; side = -side
        for x in xs:
            t = frac(b); notes.append(dict(above=1, alpha=255, startTime=t, endTime=t, isFake=0, positionX=x,
                                            size=1.0, speed=1.0, type=1, visibleTime=999999.0, yOffset=0.0))
    layer = dict(alphaEvents=[ev(255, 255)], moveXEvents=[ev(0.0, 0.0)], moveYEvents=[ev(-250.0, -250.0)],
                 rotateEvents=[ev(0.0, 0.0)], speedEvents=[dict(start=10.0, end=10.0, startTime=[0, 0, 1], endTime=[1, 0, 1], linkgroup=0)])
    line = dict(Group=0, Name='BeatFCN', Texture='line.png', bpmfactor=1.0, eventLayers=[layer], father=-1, isCover=1,
                notes=notes, numOfNotes=len(notes), zOrder=0, alphaControl=[dict(alpha=1.0, easing=1, x=0.0)],
                posControl=[dict(easing=1, pos=1.0, x=0.0)], sizeControl=[dict(easing=1, size=1.0, x=0.0)],
                skewControl=[dict(easing=1, skew=0.0, x=0.0)], yControl=[dict(easing=1, x=0.0, y=1.0)])
    meta = dict(RPEVersion=130, background='bg.png', charter='BeatFCN (auto)', composer='', id='beatfcn', level=level,
                name=name, offset=0, song='song')
    return dict(BPMList=bpm, META=meta, judgeLineGroup=['Default'], judgeLineList=[line])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio'); ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--nps', type=float, default=4.0, help='目标密度（音符/秒）')
    ap.add_argument('--th', type=float, default=None); ap.add_argument('--name', default=None)
    ap.add_argument('--no-audio-in-zip', action='store_true')
    ap.add_argument('--save-times', action='store_true')
    ap.add_argument('--th2', type=float, default=0.6, help='双押阈值（test: 0.6 → P0.46/R0.29）')
    ap.add_argument('--exact-density', type=int, default=1, help='1=按目标密度取 top-K；0=用阈值（密度条件只做软控制）')
    a = ap.parse_args(); torch.set_num_threads(3)
    name = a.name or os.path.splitext(os.path.basename(a.audio))[0]
    y = decode(a.audio); dur = len(y) / SR
    beats, downs, onsets = detect(y)
    beats = beats - LAG_BEAT; beats = beats[beats > 0.05]
    feat = features(y)
    model, hp, th_ck = load_any_model(a.ckpt)
    bidx = np.arange(len(beats) + 2) - 1.0
    gb = subdiv(np.arange(len(beats), dtype=float), DIV) + 1       # 网格点的拍号（第 1 拍=beats[0]）
    gt = subdiv(beats, DIV)
    m = (gt >= 0) & (gt < dur); gb, gt = gb[m], gt[m]
    s1, s2 = score(model, feat, gt, math.log1p(a.nps))
    th = a.th if a.th is not None else (th_ck or 0.5)
    if a.exact_density:  # 按分数取前 K 个（NMS 后），K = 目标密度×时长：密度精确可控
        sel = select(s1, gt, 0.0); sel = np.sort(sel[np.argsort(-s1[sel])][:int(round(a.nps * dur))])
    else:
        sel = select(s1, gt, th)
    nb = gb[sel]; dbl = s2[sel] >= a.th2
    # 拍号 < 1 的音（首拍前）按首段 bpm：拍 0..1 映射 t∈[0, beats[0]]，与 BPM 表一致
    nb = np.where(gt[sel] < beats[0], gt[sel] / beats[0], nb)
    os.makedirs(a.out, exist_ok=True)
    level = f'AUTO nps{a.nps:g}'
    chart = build_rpe(beats, nb, dbl, name, level)
    ext = os.path.splitext(a.audio)[1] or '.mp3'; chart['META']['song'] = 'song' + ext
    json.dump(chart, open(f'{a.out}/chart.json', 'w'), ensure_ascii=False)
    info = f'name: {name}\nlevel: {level}\ncharter: BeatFCN (auto)\ncomposer: ""\nillustrator: ""\nchart: chart.json\nformat: rpe\nmusic: song{ext}\nillustration: bg.png\noffset: 0.0\n'
    open(f'{a.out}/info.yml', 'w').write(info)
    from PIL import Image
    Image.new('RGB', (64, 36), (20, 20, 30)).save(f'{a.out}/bg.png')
    stats = dict(name=name, dur_s=round(dur, 1), beats=int(len(beats)), bpm_median=round(float(60 / np.median(np.diff(beats))), 1),
                 notes=int(len(chart['judgeLineList'][0]['notes'])), taps=int(len(sel)), doubles=int(dbl.sum()),
                 nps_req=a.nps, nps_out=round(len(sel) / dur, 2), th=th)
    json.dump(stats, open(f'{a.out}/stats.json', 'w'), ensure_ascii=False, indent=1)
    if not a.no_audio_in_zip:
        with zipfile.ZipFile(f'{a.out}/{name}.pez', 'w') as z:
            for f in ('chart.json', 'info.yml', 'bg.png'): z.write(f'{a.out}/{f}', f)
            z.write(a.audio, 'song' + ext)
    np.save(f'{a.out}/note_times.npy', gt[sel]) if a.save_times else None
    print(json.dumps(stats, ensure_ascii=False))

if __name__ == '__main__':
    main()

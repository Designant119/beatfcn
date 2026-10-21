"""v24 端到端：音频 → 节拍(beat_this) → v24 多类型选音 + 重音先验 → Phira RPE（Tap1/Hold2/Flick3/Drag4）与 osu!mania 4K .osz。
需 venv_bt。  python gen/make_chart2.py <audio> <v24.pt> <out_dir> --nps 4 --name 曲名 [--lam 0.3] [--no-audio]
摆位：单条静止判定线。Drag 连段沿频谱质心轮廓连续滑动；Tap/Flick 在上一个位置附近左右摆动；双押 ±300。
4K：车道按频谱质心分 4 档 + 防连打（同道 <120ms 换邻道）+ 长条占道；双押占两道。
"""
import sys, os, json, math, argparse, zipfile, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen.layer1 import detect, subdiv
from gen.selector import features
from gen.accent import accent_curve, boosted, topk
from gen.make_chart import decode, frac, build_rpe, DIV, LAG_BEAT
from beat_fcn_train_v24 import nms
from model_v23 import load_any_model
SR = 22050; HR = SR / 256

def centroid_z(feat, t):
    m = np.exp(np.asarray(feat[:, :80], np.float32)); c = (m * np.arange(80)).sum(1) / (m.sum(1) + 1e-6)
    c = np.convolve(c, np.ones(5) / 5, 'same'); c = (c - np.median(c)) / (c.std() + 1e-6)
    return np.interp(t * HR, np.arange(len(c)), c)

def place_x(times, types, dbl, cz):
    xs, x, side = [], 0.0, 1
    for i, (t, ty) in enumerate(zip(times, types)):
        if dbl[i]: xs.append([-300.0, 300.0]); continue
        prev_drag = i > 0 and types[i - 1] == 4 and t - times[i - 1] < 0.3
        if ty == 4 and prev_drag:
            x = float(np.clip(x + np.clip((cz[i] - cz[i - 1]) * 120, -90, 90) + 25 * side * (abs(cz[i] - cz[i - 1]) < 0.05), -500, 500))
        else:
            side = -side; x = float(np.clip(0.6 * x + side * 220 + 40 * cz[i], -500, 500))
        xs.append([x])
    return xs

def write_rpe(beats, nb, types, hold_b, xs, name, level):
    ch = build_rpe(beats, nb, np.zeros(len(nb), bool), name, level); notes = []
    for b, ty, hb, xl in zip(nb, types, hold_b, xs):
        for x in xl:
            st = frac(b); en = frac(b + hb) if ty == 2 else st
            notes.append(dict(above=1, alpha=255, startTime=st, endTime=en, isFake=0, positionX=x, size=1.0, speed=1.0,
                              type=int(ty), visibleTime=999999.0, yOffset=0.0))
    ch['judgeLineList'][0]['notes'] = notes; ch['judgeLineList'][0]['numOfNotes'] = len(notes)
    return ch

def lanes_4k(times, ends, dbl, cz):
    free = np.zeros(4); last = np.full(4, -9.0); out = []
    q = np.clip(np.searchsorted(np.quantile(cz, [.25, .5, .75]), cz), 0, 3)
    for t, e, d, l0 in zip(times, ends, dbl, q):
        order = sorted(range(4), key=lambda l: (free[l] > t - 0.03, t - last[l] < 0.12, abs(l - l0)))
        ls = order[:2] if d else order[:1]
        for l in ls: last[l] = t; free[l] = max(e, t) + 0.03
        out.append(ls)
    return out

def write_osu(path_osu, name, audio_name, beats, times, ends, lanes, version):
    tp = []; prev = None
    for k in range(len(beats) - 1):
        bl = (beats[k + 1] - beats[k]) * 1000
        if prev is None or abs(bl - prev) / prev > 0.01:
            tp.append(f'{int(round(beats[k] * 1000))},{bl:.4f},4,2,0,60,1,0'); prev = bl
    ho = []
    for t, e, ls in zip(times, ends, lanes):
        for l in ls:
            x = int(l * 512 / 4 + 64); ms = int(round(t * 1000))
            ho.append((ms, f'{x},192,{ms},128,0,{int(round(e * 1000))}:0:0:0:0:' if e > t + 0.05 else f'{x},192,{ms},1,0,0:0:0:0:'))
    ho.sort()
    safe = name.replace(',', ' ')
    txt = f'''osu file format v14

[General]
AudioFilename: {audio_name}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
SpecialStyle: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{safe}
TitleUnicode:{safe}
Artist:Unknown
ArtistUnicode:Unknown
Creator:BeatFCN
Version:{version}
Source:
Tags:beatfcn auto
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:7
CircleSize:4
OverallDifficulty:7
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
0,0,"bg.png",0,0

[TimingPoints]
''' + '\n'.join(tp) + '\n\n\n[HitObjects]\n' + '\n'.join(h for _, h in ho) + '\n'
    open(path_osu, 'w', encoding='utf-8').write(txt)
    return len(ho)

def validate_osu(path):
    s = open(path, encoding='utf-8').read(); sec = s.split('[HitObjects]')[1].strip().splitlines()
    assert s.startswith('osu file format v14') and 'Mode: 3' in s and 'CircleSize:4' in s
    busy = {}; prev = -1
    for ln in sec:
        p = ln.split(','); x, t, typ = int(p[0]), int(p[2]), int(p[3]); lane = x * 4 // 512
        assert 0 <= lane < 4 and t >= prev and typ in (1, 128); prev = t
        end = int(p[5].split(':')[0]) if typ == 128 else t
        assert end >= t and busy.get(lane, -1) < t, f'lane overlap at {t}'
        busy[lane] = end
    tps = s.split('[TimingPoints]')[1].split('[HitObjects]')[0].strip().splitlines()
    assert all(float(l.split(',')[1]) > 0 for l in tps)
    return dict(objects=len(sec), timing_points=len(tps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio'); ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--nps', type=float, default=4.0); ap.add_argument('--name', default=None)
    ap.add_argument('--lam', type=float, default=0.3); ap.add_argument('--th2', type=float, default=0.6)
    ap.add_argument('--no-audio', action='store_true'); ap.add_argument('--save-times', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(3)
    name = a.name or os.path.splitext(os.path.basename(a.audio))[0]
    y = decode(a.audio); dur = len(y) / SR
    beats, downs, _ = detect(y); beats = beats - LAG_BEAT; downs = downs - LAG_BEAT; beats = beats[beats > 0.05]
    feat = features(y); model, hp, _ = load_any_model(a.ckpt)
    gb = subdiv(np.arange(len(beats), dtype=float), DIV) + 1; gt = subdiv(beats, DIV)
    m = (gt >= 0) & (gt < dur); gb, gt = gb[m], gt[m]
    with torch.no_grad():
        o = model(torch.from_numpy(feat)[None], torch.from_numpy(gt.astype(np.float32))[None], None,
                  dens=torch.tensor([math.log1p(a.nps)]))[0]
    s1 = torch.sigmoid(o[:, 0]).numpy(); s2 = torch.sigmoid(o[:, 1]).numpy()
    ty = o[:, 2:6].argmax(-1).numpy() + 1; hd = torch.expm1(o[:, 6].clamp(0, 5)).numpy()
    sc = boosted(s1, accent_curve(feat), gt, downs, a.lam, 0.05 if a.lam else 0)
    sel = topk(sc, gt, int(round(a.nps * dur)), nms)
    t = gt[sel]; nb = np.where(t < beats[0], t / beats[0], gb[sel]); types = ty[sel].copy(); dbl = (s2[sel] >= a.th2) & (types != 4)
    # 长条：拍数取 1/4 拍，[0.5, 4] 拍，且不越过下下个音
    hb = np.clip(np.round(hd[sel] * 4) / 4, 0.5, 4.0)
    for i in range(len(nb)):
        if types[i] == 2 and i + 2 < len(nb): hb[i] = min(hb[i], max(nb[i + 2] - nb[i] - 0.25, 0.25))
    hb = np.where(types == 2, hb, 0.0)
    cz = centroid_z(feat, t); xs = place_x(t, types, dbl, cz)
    os.makedirs(a.out, exist_ok=True); level = f'AUTO nps{a.nps:g}'; ext = os.path.splitext(a.audio)[1] or '.mp3'
    ch = write_rpe(beats, nb, types, hb, xs, name, level); ch['META']['song'] = 'song' + ext
    json.dump(ch, open(f'{a.out}/chart.json', 'w'), ensure_ascii=False)
    open(f'{a.out}/info.yml', 'w').write(f'name: {name}\nlevel: {level}\ncharter: BeatFCN (auto)\ncomposer: ""\nillustrator: ""\nchart: chart.json\nformat: rpe\nmusic: song{ext}\nillustration: bg.png\noffset: 0.0\n')
    from PIL import Image
    Image.new('RGB', (64, 36), (20, 20, 30)).save(f'{a.out}/bg.png')
    # 4K
    bt_all = np.concatenate([[0.0], beats])
    def b2t(b):
        k = np.clip(np.floor(b).astype(int), 0, len(bt_all) - 2); return bt_all[k] + (b - k) * (bt_all[k + 1] - bt_all[k])
    ends = np.where(types == 2, b2t(nb + hb), t)
    lanes = lanes_4k(t, ends, dbl, cz)
    osu = f'{a.out}/{name} (BeatFCN) [4K nps{a.nps:g}].osu'.replace('/', '_') if False else f'{a.out}/beatfcn_4k.osu'
    n4 = write_osu(osu, name, 'song' + ext, beats, t, ends, lanes, f'4K AUTO nps{a.nps:g}'); v = validate_osu(osu)
    st = dict(name=name, dur_s=round(dur, 1), notes=len(ch['judgeLineList'][0]['notes']), selected=int(len(sel)),
              by_type={k: int((types == c).sum()) for c, k in ((1, 'tap'), (2, 'hold'), (3, 'flick'), (4, 'drag'))}, doubles=int(dbl.sum()),
              nps_req=a.nps, nps_out=round(len(sel) / dur, 2), lam=a.lam, osu=v, bpm_median=round(float(60 / np.median(np.diff(beats))), 1))
    json.dump(st, open(f'{a.out}/stats.json', 'w'), ensure_ascii=False, indent=1)
    if a.save_times: np.savez(f'{a.out}/sel.npz', t=t, types=types)
    if not a.no_audio:
        with zipfile.ZipFile(f'{a.out}/{name}_phira.zip', 'w') as z:
            for f in ('chart.json', 'info.yml', 'bg.png'): z.write(f'{a.out}/{f}', f)
            z.write(a.audio, 'song' + ext)
        with zipfile.ZipFile(f'{a.out}/{name}_4k.osz', 'w') as z:
            z.write(osu, os.path.basename(osu)); z.write(f'{a.out}/bg.png', 'bg.png'); z.write(a.audio, 'song' + ext)
    print(json.dumps(st, ensure_ascii=False))

if __name__ == '__main__':
    main()

"""v25 端到端：音频 → beat_this 节拍(平滑) → v25 选音(onset+双押) → Phira 编辑器可导入的 RPE 谱面 zip。

与 gen/make_chart.py 的区别：
- 特征 130 维（feat_v25.append_channels：+节拍引导 +重音 +起音包络），与训练严格同源
- 节拍平滑（gen/layer1.smooth_beats，±4 拍局部线性回归，保变速；实测节拍 F@20 0.618→0.644）
- NMS 最小间距随目标密度缩放（低密度不出 40ms 连打）
- --accent-gate：双押只允许落在重音较强的候选上（按已选候选的重音分位数门控，0=关）

用法（需 venv_bt：beat_this + librosa）：
  python gen/make_chart_v25.py song.mp3 beat_fcn_best_v25.pt out_dir --nps 5 --name 曲名 [--th2 0.55] [--accent-gate 40]
"""
import sys, os, json, math, argparse, zipfile, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen.layer1 import detect, subdiv, smooth_beats
from gen.selector import features, score, select
from gen.make_chart import decode, build_rpe, DIV, LAG_BEAT
from gen.accent import accent_curve, accent_at
from model_v23 import load_any_model
import feat_v25
SR = 22050


def min_gap_of(nps):
    """目标密度 → NMS 最小间距：nps=2 → 160ms，4 → 87ms，8 → 44ms，≥10 → 35ms 下限。"""
    return float(np.clip(0.35 / max(nps, 0.5), 0.035, 0.16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio'); ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--nps', type=float, default=4.0, help='目标密度（音符/秒），top-K 精确命中')
    ap.add_argument('--th2', type=float, default=0.55, help='双押阈值（test 实测 0.5→P0.40/R0.41，0.6→P0.46/R0.29，草稿宁多标取偏低）')
    ap.add_argument('--accent-gate', type=int, default=40, help='双押重音门控：重音低于已选候选第 N 百分位的禁止双押（0=关）')
    ap.add_argument('--name', default=None)
    ap.add_argument('--no-audio-in-zip', action='store_true')
    ap.add_argument('--no-smooth', action='store_true')
    ap.add_argument('--save-times', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(3)
    name = a.name or os.path.splitext(os.path.basename(a.audio))[0]
    y = decode(a.audio); dur = len(y) / SR
    beats, downs, _ = detect(y)
    # 通道构造用 beat_this 原始输出（与训练的 detect_all 缓存一致）；出谱网格用滞后校正+平滑后的节拍
    feat = features(y)
    feat25 = feat_v25.append_channels(feat, beats, downs)
    beats_g = smooth_beats(beats - LAG_BEAT) if not a.no_smooth else (beats - LAG_BEAT)
    beats_g = np.asarray(beats_g, float); beats_g = beats_g[beats_g > 0.05]

    model, hp, th_ck = load_any_model(a.ckpt)
    cin = hp.get('n_mels', 81) + hp.get('n_pitch', 46)
    if cin != feat25.shape[1]:
        raise ValueError(f'ckpt 特征维度 {cin} ≠ v25 特征 {feat25.shape[1]}：请用 beat_fcn_train_v25.py 训出的 ckpt（或旧版用 gen/make_chart.py）')

    gb = subdiv(np.arange(len(beats_g), dtype=float), DIV) + 1
    gt = subdiv(beats_g, DIV)
    m = (gt >= 0) & (gt < dur); gb, gt = gb[m], gt[m]
    s1, s2 = score(model, feat25, gt, math.log1p(a.nps))
    sel = select(s1, gt, 0.0, min_gap=min_gap_of(a.nps))
    sel = np.sort(sel[np.argsort(-s1[sel])][:int(round(a.nps * dur))])  # 密度精确 top-K
    nb = gb[sel]; dbl = s2[sel] >= a.th2
    if a.accent_gate > 0 and len(sel):
        acc = accent_at(accent_curve(feat), gt[sel])
        dbl = dbl & (acc >= np.percentile(acc, a.accent_gate))
    nb = np.where(gt[sel] < beats_g[0], gt[sel] / beats_g[0], nb)  # 首拍前的音按首段 bpm 折算

    os.makedirs(a.out, exist_ok=True)
    level = f'AUTO nps{a.nps:g}'
    chart = build_rpe(beats_g, nb, dbl, name, level)
    ext = os.path.splitext(a.audio)[1] or '.mp3'; chart['META']['song'] = 'song' + ext
    json.dump(chart, open(f'{a.out}/chart.json', 'w'), ensure_ascii=False)
    info = f'name: {name}\nlevel: {level}\ncharter: BeatFCN v25 (auto)\ncomposer: ""\nillustrator: ""\nchart: chart.json\nformat: rpe\nmusic: song{ext}\nillustration: bg.png\noffset: 0.0\n'
    open(f'{a.out}/info.yml', 'w').write(info)
    from PIL import Image
    Image.new('RGB', (64, 36), (20, 20, 30)).save(f'{a.out}/bg.png')
    stats = dict(name=name, dur_s=round(dur, 1), beats=int(len(beats_g)), bpm_median=round(float(60 / np.median(np.diff(beats_g))), 1),
                 notes=int(len(chart['judgeLineList'][0]['notes'])), taps=int(len(sel)), doubles=int(dbl.sum()),
                 nps_req=a.nps, nps_out=round(len(sel) / dur, 2), th2=a.th2, accent_gate=a.accent_gate, ckpt_th1=th_ck)
    json.dump(stats, open(f'{a.out}/stats.json', 'w'), ensure_ascii=False, indent=1)
    if not a.no_audio_in_zip:
        with zipfile.ZipFile(f'{a.out}/{name}.pez', 'w') as z:
            for f in ('chart.json', 'info.yml', 'bg.png'): z.write(f'{a.out}/{f}', f)
            z.write(a.audio, 'song' + ext)
    np.save(f'{a.out}/note_times.npy', gt[sel]) if a.save_times else None
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()

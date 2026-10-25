#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""谱面解析与音频特征公共工具库。
被 extract_full.py（数据提取）和 infer_v22.py（推理）共用。
"""
import os, io, re, unicodedata
import numpy as np

SR = 22050; N_FFT = 1024; HOP = 256; N_MELS = 80; FMIN = 20; FMAX = 8000
TOL = 0.030; HR = SR / HOP


def normalize_name(name):
    """歌曲名归一化：全角转半角(NFKC)、去空格、去标点、小写，尽量合并同一首歌的不同写法。
    所有split相关脚本必须用这个函数，保证归一化逻辑一致，避免漏检同曲泄漏。"""
    if not name:
        return ''
    # 全角转半角（NFKC）—— 关键！否则全角括号和半角括号会被当成不同歌
    name = unicodedata.normalize('NFKC', name)
    # 转小写
    name = name.lower()
    # 去掉所有空白字符
    name = re.sub(r'\s+', '', name)
    # 去掉常见标点和特殊符号（保留中日韩文字和字母数字）
    name = re.sub(r'[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', name)
    return name

_fb = None
def get_fb():
    """自写 mel 滤波器组（与 librosa slaney 三角滤波等价）"""
    global _fb
    if _fb is not None:
        return _fb
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2)
    hzs = mel2hz(mels)
    bins = np.floor((N_FFT + 1) * hzs / SR).astype(int)
    bins = np.clip(bins, 0, N_FFT // 2)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for i in range(N_MELS):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    _fb = fb
    return fb

def _stft_power(wav):
    """自写分帧 + hanning + rfft -> 功率谱 (n, N_FFT//2+1)
    帧数与 extract_pitch._stft_power 完全一致：n = 1 + len//HOP，
    中心对齐（帧 i 窗口 [i*HOP-pad, i*HOP+pad)），消除 librosa 帧数差 1-2 的 bug。"""
    n = 1 + len(wav) // HOP
    if n < 1:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.float32)
    pad = N_FFT // 2
    wav_p = np.concatenate([np.zeros(pad, wav.dtype), wav, np.zeros(pad, wav.dtype)])
    win = np.hanning(N_FFT).astype(np.float32)
    idx = np.arange(n)[:, None] * HOP + (np.arange(N_FFT) - pad)[None, :]
    frames = wav_p[idx + pad] * win
    return (np.abs(np.fft.rfft(frames, N_FFT)) ** 2).astype(np.float32)

def mel_of(wav):
    """自写 mel 提取（帧数与 pitch 一致 1+len//HOP）+ log + 标准化"""
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    spec = _stft_power(wav)          # (n, 513)
    fb = get_fb()                     # (80, 513)
    mel = spec @ fb.T                 # (n, 80)
    mel = np.log(mel + 1e-7)
    # 每通道独立标准化（axis=0）：每个mel bin跨时间独立计算均值和标准差
    # 低频通道能量高、高频通道能量低，全局标量标准化会抹掉通道间相对尺度
    return ((mel - mel.mean(axis=0, keepdims=True)) / (mel.std(axis=0, keepdims=True) + 1e-5)).astype(np.float32)

def decode_audio(data):
    """ffmpeg 解码为 float32 单声道（subprocess + 30s 超时，防 C 层死锁）
    统一用 ffmpeg 解码，保证 mel 和 pitch 帧数一致"""
    import subprocess, tempfile, os
    if not data:
        return None
    with tempfile.NamedTemporaryFile(suffix='.aud', delete=False) as f:
        f.write(data); tmp = f.name
    try:
        r = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', tmp, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
            capture_output=True, timeout=30)
        if r.returncode != 0 or not r.stdout:
            return None
        return np.frombuffer(r.stdout, dtype=np.float32)
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def beats_of(t):
    return t[0] + t[1] / t[2]

def chart_sec(chart):
    """BPMList → [(起始beat, bpm, 起始秒)]"""
    segs = []; prev_bpm = None; prev_b = 0; prev_s = 0
    for node in chart['BPMList']:
        bpm = node['bpm']; sb = beats_of(node['startTime'])
        if prev_bpm is None:
            segs.append((sb, bpm, 0.0)); prev_b = sb; prev_s = 0.0; prev_bpm = bpm
        else:
            ds = (sb - prev_b) * 60. / prev_bpm
            segs.append((sb, bpm, prev_s + ds)); prev_b, prev_s = sb, prev_s + ds; prev_bpm = bpm
    return segs

def beat_to_sec(b, segs):
    for i in range(len(segs) - 1, -1, -1):
        sb, bpm, ss = segs[i]
        if b >= sb: return ss + (b - sb) * 60. / bpm
    # BPMList 起点之前的 beat：用首段 bpm 反推（与 beat_grid 补段同公式，保持单调）
    sb0, bpm0, ss0 = segs[0]
    return ss0 + (b - sb0) * 60. / bpm0

def beat_grid(chart, segs, step=0.25):
    """网格：默认 16分（0.25 beat 步长），step=0.03125 为 128分，末尾 +8 beat
    起点补全：若 BPMList 起始 beat >0，先用首段 BPM 补 0~sb0（避免起点前音符漏标）"""
    max_b = 0
    for jl in chart.get('judgeLineList', []):
        for n in jl.get('notes', []):
            b = beats_of(n['startTime'])
            if b > max_b: max_b = b
    max_b += 8
    grid = []
    # 起点补全：第一段起始 beat 之前，用首段 bpm 反推（负秒单调；与 beat_to_sec 同公式）
    sb0, bpm0, ss0 = segs[0]
    if sb0 > 0:
        b = 0.0
        while b < sb0:
            grid.append(ss0 + (b - sb0) * 60. / bpm0); b += step
    for i in range(len(segs) - 1):
        sb, bpm, ss = segs[i]; eb = segs[i + 1][0]
        b = sb
        while b < eb:
            grid.append(ss + (b - sb) * 60. / bpm); b += step
    sb, bpm, ss = segs[-1]; b = sb
    while b <= max_b:
        grid.append(ss + (b - sb) * 60. / bpm); b += step
    return np.array(grid, dtype=np.float32)

def parse_chart(txt):
    """多 JSON 拼接谱面文件里提取主谱面（含 BPMList + judgeLineList）"""
    import json
    dec = json.JSONDecoder(); pos = 0
    while pos < len(txt):
        s = txt[pos:].lstrip()
        if not s: break
        off = len(txt[pos:]) - len(s)
        try:
            obj, end = dec.raw_decode(s)
            if isinstance(obj, dict) and 'BPMList' in obj and 'judgeLineList' in obj:
                return obj
            pos += off + end
        except Exception:
            break
    return None

def _pec_frac(b):
    """拍小数 → [b, frac, 32]（32分音符精度；RPE 分母 32——避免 16 分量化丢失 24/32 分音符）"""
    bi = int(b)
    frac = int(round((b - bi) * 32))
    if frac >= 32:
        bi += 1; frac = 0
    return [bi, frac, 32]

def parse_pec(txt):
    """PEC 文本谱面 → RPE chart dict。

    PEC 行格式：
      <offset ms，负数；也可能是 <行数>（正数）>
      bp <beat> <bpm>                  # BPM 事件
      n1 <line> <beat> <x> <above> <alpha>   # Tap（n2=Hold n3=Flick n4=Drag）
      # <size>  /  & <alpha>           # 音符属性（跟随上一个 n）
      cp/cm/cv/cd/ca/cr/cf ...         # 判定线事件（训练不需要）

    兼容两种实际存在的"PEC"：
      1) 内容为 JSON（RPE 结构或 Phigros formatVersion 3，被命名为 .pec）→ 按 JSON 解析；
         Phigros 格式 offset 单位为秒，×1000 换算为 RPE ms。
      2) 真 PEC 文本：首行若为负数 → 该值即谱面 offset（ms），解析从下一行开始；
         首行若为正数 → 行数（旧格式），offset 视为 0。
    """
    import json
    st = txt.lstrip()
    if st.startswith('{'):
        try:
            d = json.loads(txt)
            if 'BPMList' in d and 'judgeLineList' in d:
                # RPE 结构 JSON（被命名为 .pec）
                return d
            if d.get('formatVersion') == 3 and 'judgeLineList' in d:
                # Phigros 原生 JSON：offset 单位秒；音符字段(time/holdTime)需专用转换，暂不支持
                return None
        except Exception:
            pass
    lines = txt.replace('\r', '').split('\n')
    bpm_list = []
    notes = []
    offset = 0.0
    start = 0
    first = lines[0].strip() if lines else ''
    if first and first.lstrip('-').replace('.', '', 1).isdigit() and float(first) < 0:
        offset = float(first)  # ms
        start = 1
    for l in lines[start:]:
        l = l.strip()
        if not l: continue
        parts = l.split()
        k = parts[0]
        if k == 'bp':
            bpm_list.append({'bpm': float(parts[2]), 'startTime': _pec_frac(float(parts[1]))})
        elif k in ('n1', 'n2', 'n3', 'n4'):
            t = {'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4}[k]
            notes.append({'type': t, 'startTime': _pec_frac(float(parts[2])),
                          'positionX': float(parts[3]), 'isFake': False})
    if not bpm_list or not notes:
        return None
    return {'BPMList': bpm_list,
            'judgeLineList': [{'numOfNotes': len(notes), 'notes': notes}],
            'META': {'offset': offset}}

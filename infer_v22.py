#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v22 推理脚本——Transformer + 条件采音。

用法:
  # 纯音频预测（不给音符）
  python3 infer_v22.py --wav song.wav --ckpt beat_fcn_best_v22.pt --out output.json

  # 条件采音（喂已采的半分钟音符）
  python3 infer_v22.py --wav song.wav --ckpt beat_fcn_best_v22.pt --out output.json \
      --notes existing_notes.json --notes-end 30.0

  # 条件采音（喂前30秒已采音符，后面让模型预测）
  # existing_notes.json 格式: {"time": [0.0, 0.5, 1.0, ...], "type": [1, 2, 1, ...]}
  # type: 1=单音, 2=双押

输出: JSON，包含每个音符的时间(秒)和类型(1/2)
"""
import os, sys, json, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_utils import mel_of, SR, HOP, HR, N_FFT, decode_audio
from extract_pitch import extract_pitch
from model import BeatFCNContext, compute_flux

DIV = 32  # 每拍32格 = 128分
TH1 = 0.4  # 有音阈值
TH2 = 0.5  # 双押阈值


def load_audio(wav_path):
    """加载音频为 22050 单声道 float32。强制用ffmpeg，和训练/提取完全一致"""
    with open(wav_path, 'rb') as f:
        data = f.read()
    y = decode_audio(data)
    if y is None:
        raise RuntimeError(f'音频解码失败: {wav_path}（请确认已安装ffmpeg）')
    return y


def make_grid(bpm, duration):
    """根据恒定 BPM 和时长生成 128分网格时间点"""
    beat_dur = 60.0 / bpm
    grid_dur = beat_dur / DIV
    n = int(duration / grid_dur) + 1
    return np.arange(n) * grid_dur


def make_grid_variable(bpm_list, duration):
    """根据变速 BPM 列表生成 128分网格时间点。
    bpm_list: [(change_t, bpm), ...] 在change_t时刻BPM变为bpm，按时间排序。
    第一个时间段从0开始（若第一个变速点>0，前面也用第一个BPM）。
    """
    grid = []
    prev_t = 0.0
    for i, (change_t, bpm) in enumerate(bpm_list):
        next_t = bpm_list[i+1][0] if i+1 < len(bpm_list) else duration
        beat_dur = 60.0 / bpm
        grid_dur = beat_dur / DIV
        # 用np.arange精确生成，避免逐格累加的浮点误差
        n = int((next_t - prev_t) / grid_dur) + 1
        seg = np.arange(n) * grid_dur + prev_t
        grid.extend(seg[seg < next_t].tolist())
        prev_t = next_t
    return np.array(grid, dtype=np.float32)


def infer(model, mel, pitch, grid, notes=None, notes_end_sec=None, device='cpu', notes_complete=True):
    """v22 推理。

    Args:
        mel: (T, 80) float32（纯mel，不含相位）
        pitch: (T, 46) float32
        grid: (G,) 时间点(秒)
        notes: dict {"time": [...], "type": [...]} 已采音符，None=纯音频预测
        notes_end_sec: 已采音符的结束时间(秒)，之后的让模型预测
    Returns:
        pred_times: 预测音符时间列表
        pred_types: 预测音符类型列表(1/2)
    """
    T = mel.shape[0]
    # 拼 flux（80 mel + 1 flux = 81）
    flux = compute_flux(mel[:, :80])
    mel_in = np.concatenate([mel, flux], axis=1)  # (T, 81)
    # 拼 pitch（帧数对齐）
    minT = min(T, pitch.shape[0])
    mel_in = mel_in[:minT]
    pitch = pitch[:minT]
    feat = np.concatenate([mel_in, pitch], axis=1)  # (T, 127)

    M = torch.from_numpy(feat).unsqueeze(0).float().to(device)
    G = torch.from_numpy(grid).float().to(device)
    Gc = len(grid)

    # 构造 notes 输入
    notes_in = torch.full((1, Gc), -1, dtype=torch.float, device=device)
    if notes is not None and notes_end_sec is not None:
        ref_end = int(np.searchsorted(grid, notes_end_sec))
        if notes_complete:
            # 参考段全部填0（确认空格），再覆盖有音符的格
            # 和训练时一致：训练时前半段是完整ground truth（空格=0，有音=1/2）
            notes_in[0, :ref_end] = 0.0
        # else: 参考段保持-1（未知），只放已确认的音符，没标到的位置让模型自己猜
        # 把已采音符放到对应格上（用searchsorted，支持变速歌非等间距grid）
        for t, typ in zip(notes['time'], notes['type']):
            if t >= notes_end_sec:
                continue
            # 找最近的格（searchsorted支持非等间距grid）
            idx = int(np.searchsorted(grid, t))
            if idx > 0 and (idx >= Gc or abs(t - grid[idx-1]) < abs(t - grid[idx])):
                idx = idx - 1
            if 0 <= idx < Gc:
                # 双押：同格已有音符(>0)且新音符也是有音(>0)，升级为双押(=2)
                cur = notes_in[0, idx].item()
                if cur > 0 and typ > 0:
                    notes_in[0, idx] = 2.0  # 两个单押同格=双押
                elif cur <= 0:
                    notes_in[0, idx] = float(typ)
        # notes_end_sec 之后的格保持 -1（遮罩）

    with torch.no_grad():
        o1, o2 = model(M, G, notes_in)
        s1 = torch.sigmoid(o1)[0].cpu().numpy()
        s2 = torch.sigmoid(o2)[0].cpu().numpy()

    # 转音符
    pred_times, pred_types = [], []
    for i in range(Gc):
        if s1[i] >= TH1:
            pred_times.append(float(grid[i]))
            pred_types.append(2 if s2[i] >= TH2 else 1)
    return pred_times, pred_types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True, help='输入音频文件')
    ap.add_argument('--ckpt', required=True, help='模型权重')
    ap.add_argument('--out', required=True, help='输出 JSON')
    ap.add_argument('--bpm', type=float, default=120.0, help='歌曲 BPM（默认120，恒定BPM用这个）')
    ap.add_argument('--bpm-list', default=None, help='变速BPM列表，格式: "0:120,30:140,60:120"（时间秒:BPM）')
    ap.add_argument('--offset', type=float, default=0.0, help='谱面offset(秒)，网格整体平移校正')
    ap.add_argument('--ch', type=int, default=96, help='模型通道数（默认96，从ckpt自动读取）')
    ap.add_argument('--n-pitch', type=int, default=46, help='pitch特征维度（默认46，训练时自动探测）')
    ap.add_argument('--th1', type=float, default=None, help='有音阈值（默认：v23 用 ckpt 里存的最优阈值，v22 用 0.4）')
    ap.add_argument('--th2', type=float, default=0.5, help='双押阈值')
    ap.add_argument('--notes', default=None, help='已采音符 JSON（条件采音用）')
    ap.add_argument('--notes-end', type=float, default=None, help='已采音符结束时间(秒)')
    ap.add_argument('--notes-complete', action='store_true', default=True,
                    help='已采音符是完整的（参考段空格=确认没音，默认开）。加--no-notes-complete则参考段未知位置让模型自己猜')
    ap.add_argument('--no-notes-complete', dest='notes_complete', action='store_false',
                    help='已采音符不完整，参考段没标到的位置填-1（未知），让模型自己猜')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    args = ap.parse_args()

    global TH1, TH2
    TH1, TH2 = (args.th1 if args.th1 is not None else 0.4), args.th2

    # 加载音频
    print(f'加载音频: {args.wav}', flush=True)
    y = load_audio(args.wav)
    duration = len(y) / SR
    print(f'时长: {duration:.1f}s, 采样率: {SR}', flush=True)

    # 提取特征
    print('提取 mel...', flush=True)
    mel = mel_of(y)
    print(f'mel: {mel.shape}', flush=True)

    print('提取 pitch...', flush=True)
    pitch = extract_pitch(y)
    if pitch is None:
        raise RuntimeError('pitch提取失败！训练时pitch缺失的歌是直接跳过的，模型没见过全0 pitch的输入。请检查音频文件是否正常。')
    # pitch每通道标准化（跨时间轴），和训练时Songs._load里的口径完全一致
    pitch_mean = pitch.mean(axis=0, keepdims=True)
    pitch_std = pitch.std(axis=0, keepdims=True) + 1e-5
    pitch = (pitch - pitch_mean) / pitch_std
    print(f'pitch: {pitch.shape}（已标准化）', flush=True)

    # 生成网格（支持变速）
    is_variable = args.bpm_list is not None
    if args.bpm_list:
        bpm_list = []
        for part in args.bpm_list.split(','):
            t, b = part.split(':')
            bpm_list.append((float(t), float(b)))
        grid = make_grid_variable(sorted(bpm_list), duration)
        print(f'网格(变速): {len(grid)} 格', flush=True)
    else:
        grid = make_grid(args.bpm, duration)
        print(f'网格: {len(grid)} 格, 格距 {grid[1]-grid[0]:.4f}s', flush=True)
    # offset校正：网格整体平移
    if args.offset != 0.0:
        grid = grid + args.offset
        print(f'offset校正: +{args.offset:.3f}s', flush=True)
        # 过滤负时间格（负offset会让开头的格子变成负时间，对应音频开始前，无意义）
        n_before = len(grid)
        grid = grid[grid >= 0]
        if len(grid) < n_before:
            print(f'过滤负时间格: {n_before} -> {len(grid)} 格（负offset导致开头格子时间为负）', flush=True)

    # 裁剪grid末尾：去掉落在静音段/超出mel范围的格子，避免歌尾出假音符
    mel_T = mel.shape[0]
    mel_end_time = (mel_T - 1) / HR  # mel最后一帧对应的时间
    n_before = len(grid)
    grid = grid[grid < mel_end_time - 0.1]  # 留0.1s余量
    if len(grid) < n_before:
        print(f'裁剪grid末尾: {n_before} -> {len(grid)} 格（去掉超出mel范围的静音段）', flush=True)

    # 加载模型（自动从checkpoint读取超参数）
    print(f'加载模型: {args.ckpt}', flush=True)
    from model_v23 import load_any_model  # 自动识别 v22 / v23 ckpt
    model, hp, ck_th1 = load_any_model(args.ckpt, device='cpu', ch=args.ch, n_pitch=args.n_pitch)
    print(f'模型结构: {hp.get("arch", "v22")}  hparams={hp}', flush=True)
    if args.th1 is None:
        TH1 = ck_th1 if ck_th1 is not None else 0.4
        print(f'th1 = {TH1}（{"ckpt 里存的验证集最优阈值" if ck_th1 is not None else "默认"}）', flush=True)
    model = model.to(args.device)
    model.eval()

    # 加载已采音符（条件采音）
    notes = None
    if args.notes:
        notes = json.load(open(args.notes))
        print(f'条件采音: 已加载 {len(notes["time"])} 个音符，结束于 {args.notes_end}s', flush=True)

    # 推理
    print('推理中...', flush=True)
    pred_times, pred_types = infer(model, mel, pitch, grid, notes, args.notes_end, args.device, notes_complete=args.notes_complete)
    print(f'预测: {len(pred_times)} 个音符', flush=True)

    # 输出
    result = {
        'bpm': args.bpm if not is_variable else None,
        'bpm_list': args.bpm_list if is_variable else None,
        'offset': args.offset,
        'duration': duration,
        'num_notes': len(pred_times),
        'time': pred_times,
        'type': pred_types,
    }
    json.dump(result, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f'输出: {args.out}', flush=True)


if __name__ == '__main__':
    main()

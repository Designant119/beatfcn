#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 共享特征通道 —— 训练与推理必须逐位一致（唯一的通道构造入口）。

在 v23 的 127 维特征（mel80 + flux1 + pitch46）上追加 3 维：
  +1 节拍引导（v24b：下拍=2 / 拍=1，±1 帧扩散；beat_this 系统滞后 13.3ms 在此扣除）
  +1 重音曲线（gen/accent.accent_curve，clip[-2,6] / 2）
  +1 起音包络（本文件 onset_env：mel 正差分 → ±1 帧取 max → 曲内 z-score → clip）

共 130 维。模型侧只需 n_pitch 46 → 49（cin = n_mels 81 + n_pitch 49），架构不变。
滞后常数 LAG_BEAT 与 gen/make_chart.py 保持同源，改动需两处同步。
"""
import numpy as np

HR = 22050.0 / 256.0
LAG_BEAT = 0.0133  # beat_this 相对谱面起音的系统滞后（bench/layer1_metrics.json 标定）


def onset_env(mel80):
    """mel80 (T,80) → (T,) 起音强度包络。

    全频带正向 mel 差分求和（升调能量=起音线索）→ ±1 帧取 max（容忍 ±1 帧对齐误差）
    → 曲内 z-score → clip[-3,4]。与 compute_flux 的区别：flux 是逐帧均值进入主干，
    本通道是峰值化包络，主要给逐格头提供"这一点附近有没有爆发"的形状信息。
    """
    m = np.asarray(mel80, np.float32)
    d = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0.0).sum(1)
    a = np.maximum.reduce([np.roll(d, -1), d, np.roll(d, 1)])
    z = (a - np.median(a)) / (a.std() + 1e-6)
    return np.clip(z, -3.0, 4.0).astype(np.float32)


def beat_guide(T, beats, downs, lag=LAG_BEAT):
    """(T,) 节拍引导通道：下拍=2 / 其他拍=1（写入 ±1 帧，取 max）。beats/downs 用 beat_this 原始输出。"""
    g = np.zeros(T, np.float32)
    for arr, v in ((beats, 1.0), (downs, 2.0)):
        f = np.round((np.asarray(arr, np.float64) - lag) * HR).astype(np.int64)
        for d in (-1, 0, 1):
            k = np.clip(f + d, 0, T - 1)
            g[k] = np.maximum(g[k], v)
    return g


def accent_ch(feat):
    """重音通道：复用 gen/accent.accent_curve（只读 feat 前 80 列 mel），与 v24b 完全同源。"""
    from gen.accent import accent_curve
    return (np.clip(accent_curve(feat), -2.0, 6.0) / 2.0).astype(np.float32)


def append_channels(feat, beats, downs):
    """feat (T,127) + beat_this 原始 beats/downs → (T,130)。训练（GuideSongs）与推理（make_chart_v25）共用。"""
    T = feat.shape[0]
    ch = np.stack([beat_guide(T, beats, downs), accent_ch(feat), onset_env(feat[:, :80])], 1)
    return np.concatenate([np.asarray(feat, np.float32), ch], 1)

## 14:05 interim (executor)
- logs: /workspace/bench/v22run/train.log, /workspace/bench/v23b/train.log
- v22b: ~900s/epoch (crop3000, CPU 6 threads), ep1-3 strict F1 0.221/0.209/0.210 (eval 4 songs)
- v23b: ~82s/epoch full songs (2 threads), ep17 strict 0.236 loose50ms 0.529 (eval 38 songs); occasional stalls (CPU contention)
- bench: train step T=3000 v22 3.40s vs v23 0.027s (127x), GFLOPs 69 vs 2.46 (28x), RSS 2459 vs 371MB; v22 OOM at T=6000

## 2026-10-21 16:00 ±20ms 分析（executor）
- 注意：box 重启后 /workspace/venv 丢失，已重装 CPU torch 2.14.1；数据实际在 /workspace/bench/{grid_npz_full128,pitch_npz_full,feat_cache_v23,split_full128_final.json}
- 口径：网格步长 1/32 拍，半步中位数 5.4ms、最大 15.6ms → "strict"=精确格（≈±5ms），比 ±20ms 更严。新增 f20（±20ms onset F1，mir_eval 式一对一匹配），v23 训练现按 f20 选模型/阈值。
- v23b best（38 首 val）：strict 0.234 / **f20 0.355** / ±50ms 0.546，双押 P 0.133 R 0.223
- 量化 oracle：标签吸附误差 ≤15.6ms <20ms → 完美格分类器 ±20ms F1 = 1.0，网格不是瓶颈，不需要亚格偏移回归。
- 标签噪声（盘上同名歌、不同谱师/ID 共 11 对，bench/analyze_f20.py 及内联脚本）：±20ms F1 原始 0.007–0.99，中位约 0.44；
  扫全局偏移后中位约 0.67（谱师间 offset 差 6–154ms）。即人类谱师之间 ±20ms 一致度大约 0.6–0.8，0.9 远高于人际一致性。
- 数据规模：盘上 307 曲（train 232/val 38/test 37），全量 split train 11373/val 1577/test 209。320 曲下载 2.1GB → 全量约 75GB。
- v23c 训练中：d_model 128，3 线程，50ep/80min 预算，日志 /workspace/bench/v23c/train.log

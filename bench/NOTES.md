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

## 16:45 v22 vs v23 同口径对比（bench/bench_eval.py，/workspace/bench/eval_cmp.json）
同一 train 子集（232 首）、同一评估歌（val 23 首选阈值 → test 21 首报分，≤14000 帧，盲采+条件平均）：
| ckpt | 训练 | test strict | test ±20ms | test ±50ms | val ±20ms |
| v22 ep4 | 4 ep ≈ 60 min (6 线程, crop3000) | 0.177 | 0.277 | 0.457 | 0.354 |
| v23b ep4 | 4 ep ≈ 6 min | 0.163 | 0.250 | 0.371 | 0.301 |
| v23b ep19 | ≈ 同墙钟 (~27 min 计算，2 线程) | 0.237 | 0.333 | 0.554 | 0.405 |
结论：同 epoch 数 v23 略低（±20ms −0.03），同墙钟 v23 明显更好（+0.06）；推理 0.08s vs 4.7s/首。
- v23c (d128) 第一次因 prlimit --as=10G 在 ep7 OOM（虚拟内存上限过紧），已 --resume，as=16G、max-tokens 80000。systemd-run --user 在此 box 不可用（无 user bus）。

## 17:12 状态（executor 收尾）
- v23c (d128) 仍在后台跑（pid 120979，timeout 到约 17:45 自停），ep15 f20 0.326 / strict 0.228 / ±50ms 0.498（38 首 val），未超过 v23b（f20 0.355）→ 加大 d_model 在 232 首上没收益，瓶颈是数据量和标签噪声。
- 吞吐：d96/2 线程 ≈0.35 s/首/epoch，d128/3 线程 ≈0.50 s/首/epoch → 全量 11373 首 ≈66–95 min/epoch，30 epoch ≈ 33–48 h（CPU，限 3 线程）。
- 全量数据：约 75GB 下载（磁盘剩 72GB，不够，需清理/扩容）；按交接文档提取 mel+pitch 8 进程 5–7h，限 3 进程约 15–20h。
- 未做（时间不够）：子格偏移头（oracle 证明不需要）、focal loss、onset 峰值特征。

## 17:45 offset 清洗（bench/offset_clean.py，报告 bench/offset_report.json）
- 方法：每谱面音符时刻 vs 音频 onset 包络（mel 正向差分，去局部均值）互相关，±200ms，2ms 步；修正 grid=grid+offset（标签不变），输出 /workspace/bench/grid_npz_clean（原 grid_npz_full128 保留）。
- 分布（307 谱面）：中位 0ms，|offset| p50 20ms、p90 71ms，48% 谱面 |offset|>20ms。音符落在音频 onset 峰 ±20ms 内比例 0.403 → 0.695（注意：该指标是优化目标，偏乐观）。
- 丢弃 11 个（low_conf 5 / low_agree 4 / 密度极端 2），剩 296（文件改名 .npz.dropped）。无谱师 ID 字段，未做按谱师统计；无精确重复。
- 标签噪声上限重测（11 对同名不同谱师）：不同上传的音频本身有 0–650ms 前导差，先用音频包络互相关对齐音频，再比：
  ±20ms F1 原 offset 中位 0.615/均值 0.531 → 修正 offset 后中位 0.703/均值 0.711。同音频同 offset 的对（Collapsing World 0.77、cyanine 0.83）说明剩下的是"放哪些音"的内容分歧。

## 18:40 v23clean 结果（清洗数据重训，d96，3 线程，45 min 预算，23 ep，best f20 0.598@val36）
| 模型 | 评估标签 | val ±20ms | test strict | test ±20ms | test ±50ms |
| v23b | 原标签(val38/test37) | 0.355 | 0.167 | 0.264 | 0.499 |
| v23clean | 原标签 | 0.382 | 0.181 | 0.280 | 0.548 |
| v23b | 修正标签(val36/test36) | 0.528 | 0.450 | 0.540 | 0.581 |
| v23clean | 修正标签 | 0.598 | 0.513 | 0.609 | 0.651 |
（整曲 ≤40000 帧；test 阈值取自 val。）
- 解读：仅修正评估标签 offset，v23b ±20ms 就从 0.26→0.54 → 原来大半"错误"是谱面 offset 噪声；在清洗数据上训练再 +0.07。
- 注意：offset 用真值音符+音频估计；推理时网格 offset 需由用户/谱面或节拍跟踪提供，需对齐音频。
- box 磁盘 inode 用满（其他工作区 node_modules 占用 /workspace/qa、/workspace/amp），feat_cache_clean 的 test 缓存写不进，评估改为不缓存。

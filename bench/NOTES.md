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

## 20:15 重设计：Layer 1 定时（gen/layer1.py，bench/layer1_metrics.json；venv_bt = torch2.11+torchaudio+beat_this+librosa）
- beat_this(final0, CPU, 无 DBN) 节拍 + librosa onset。常数滞后在 val 上标定：节拍 +13ms、onset +26ms（相对修正后谱面音符），test 只套用。
- 候选召回 @±20ms（中位数，test 36 首 / val 36 首）：
  onset 0.71（1.27 个/音符）；1/16 网格 0.73（1.78）；1/16+1/12 网格 0.79（2.68）；onset+1/16 0.87（2.99）；
  **onset+1/16+1/12 0.903（3.87 个/音符）** ← 达到 ≥0.9 目标。val 同口径 0.895（4.19）。
- 节拍 vs 谱面整拍（谱面拍相位按音符最多的 idx%32 选）：F@70ms test 0.71 / val 0.77；允许半/倍速 0.86 / 0.81；F@20ms 0.62/0.60。
  变速歌（每段步长比>1.02）test 3 首、val 3 首，F@70(含半倍速) 0.74 / 0.78。
- Layer 2 训练中：v23dens = v23clean + 密度条件（log1p 音符/秒 → token 偏置），--no-epoch-ckpt。
- 端到端 gen/make_chart.py：已通过往返测试（生成 chart.json 用 chart_utils.parse_chart 解析回秒，557 音最大误差 0.005ms）。未在 Phira App 内实测加载。

## 20:50 Layer 2 选音 + 端到端（gen/selector.py, gen/make_chart.py, bench/layer2_metrics.json）
- v23dens：v23clean + 密度条件，清洗数据，d96，3 线程，约 25 ep（被 timeout 3300s 截断），best val f20 0.601（谱面网格口径）。
- 选音评估（给定真值密度，NMS 40ms，阈值在 val 上选）：test ±20ms F1 谱面网格 0.673 / **自动网格(beat_this 1/32) 0.644**，自动网格 ±50ms 0.711；val 0.647/0.635。
- 密度可控性（test，请求 = 真值×0.5…2）：Pearson 0.83 / Spearman 0.85，但主要来自歌间差异；歌内 产出/真值 中位 1.00/1.13/1.25/1.49/1.65 → 软条件很弱。
  所以 make_chart 默认 --exact-density 1：NMS 后按分数取 top-K（K=目标音符/秒×时长），密度精确命中。
- 双押阈值 th2（test）：0.5→P0.40/R0.41，0.6→0.46/0.29，0.7→0.55/0.18；默认 0.6。
- demo（bench/demo_charts/<id>，不含音频，版权原因；用原曲音频放进 zip 命名 song.<ext> 即可）：
  74587 シャルル ±20ms 0.653；10096 Crisscross 0.638；4988 冬雨 0.670（请求密度=原谱密度）。
- 未验证：Phira App 实际导入/游玩手感；重复段落一致性未测；只有 Tap，无 Hold/Drag/Flick。

## 21:00 v24 多类型（gen/typed_labels.py, beat_fcn_train_v24.py）
- 原始谱面类型统计（296 张清洗谱面，对齐到清洗网格）：tap 277398 (63.6%) / hold 48779 (11.2%) / flick 25278 (5.8%) / drag 84577 (19.4%)；
  含 hold/flick/drag 的谱面 291/242/255。数据不稀疏 → 直接训练多类型模型。
- 重要：v22/v23 的 grid 标签只含 type 1/2/3，**drag 从未作为音符**，这可能就是"连打/滑音段没打击感"的原因之一。v24 onset 包含 drag。
- v24 = v23dens 热启动 + 7 维输出 [onset, 双押, 类型4, log1p(长条拍数)]，类型 CE 按 1/sqrt 频率加权。训练中：/workspace/bench/v24/train.log

## 21:40 v24 结果 / v24b 启动
- v24（无节拍引导）30 ep：onset ±20ms（谱面网格，val36）~0.648–0.649 基本持平 v23dens；类型（ep22，val）tap P0.48/R0.65，hold 0.17/0.22，flick 0.11/0.02，drag 0.11/0.03，hold 时长中位误差 0.35 拍。
  失误：按 f20 存最佳 → 存下的是 ep1（类型头几乎未训练）。已改为按 f20+0.5×类型宏F 选。
- 漏音分析（v23dens，自动网格 top-K，test，accent_v23dens.json）：重音 top25% 音符召回 0.878；λ=0.3 重音先验 → 0.931，整体 F1 0.644→0.652、召回 0.683→0.689。
- v24b：+GenéLive 式节拍引导输入（beat_this 下拍=2/拍=1，±1 帧）+ 重音曲线通道，stem 新通道零初始化，从 v23dens 热启动。train 全部 296 首已跑 beat_this（gen/detect_all.py）。

## 22:05 v24b 多类型 + 重音 + 4K 出谱（demo 在 /workspace/demo_out/v24_<id>/）
- v24b（节拍引导+重音通道）中途 ckpt（约 ep10，训练仍在跑）：自动网格 top-K + λ0.3 重音先验，test onset ±20ms F1 0.657。
- 类型 P/R（test，±20ms 同类型匹配，bench/types_eval_v24b.json）：
  argmax：tap 0.454/0.640，hold 0.194/0.211，flick 0.120/0.002，drag 0.051/0.007
  val 调 logit 偏置 [0,0,.5,1]：tap 0.505/0.420，hold 0.209/0.147，flick 0.097/0.108，drag 0.092/0.139（宏F 0.211）
  +连打→drag 启发式：drag 0.109/0.198，宏F 0.199（没更好）。类型仍弱，主要受 onset F1≈0.65 限制，类型本身也难。
- 连打补 Drag（onset 间隔<120ms 连续≥4、离已选音>60ms）：test F1 0.653→0.652，drag 音符召回 0.409→0.416，重音 top25% 召回 0.911→0.919，提升很小。
- 节拍平滑（beat_this 20ms 量化，±4 拍线性回归）：test 节拍 F@20 0.618→0.644；Crisscross 中位 BPM 130.4→128.0（原曲 128）。
- demo（请求密度=原谱含 drag 的密度）：シャルル F1@20 0.743，Crisscross 0.649，冬雨 0.753。Phira zip + 4K .osz 都带音频，仅在 /workspace/demo_out（不进 git）；谱面文件拷贝在 bench/demo_charts_v24。
- 4K .osu：自检通过（v14 头、Mode 3、CircleSize 4、车道 0-3、时间单调、同道无重叠、长条 end>start、红线 beatLength>0）；未在 osu! 客户端实测。

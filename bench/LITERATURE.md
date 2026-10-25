# BeatFCN 文献调研（2026-10-21）

所有链接已于 2026-10-21 打开确认（GitHub 许可证来自 GitHub API）。标「未核」的数字是凭记忆写的，没在原文里查到，用之前先核对。

## A. 谱面/beatmap 生成

| 工作 | 架构 | 数据 / 条件 | 音符类型 | 评估 | 代码 / 权重 / 许可 | CPU |
|---|---|---|---|---|---|---|
| Dance Dance Convolution（Donahue 2017）[arXiv 1703.06891](https://arxiv.org/abs/1703.06891)，[代码](https://github.com/chrisdonahue/ddc) | 两阶段：①放置（CNN+LSTM 读频谱，按难度条件逐帧给 onset 概率）②选择（条件 LSTM 选步型） | StepMania 包 Fraxtil / ITG；条件=难度 | DDR 4 方向步 | 放置 F-score、选择 perplexity | MIT，代码+数据构建脚本 | 是（模型小） |
| GenéLive!（KLab，AAAI-23）[arXiv 2202.12823](https://arxiv.org/abs/2202.12823)，[代码](https://github.com/KLab/AAAI-23.6040) | DDC 改进：**beat guide**（小节首拍=2 / 其余拍=1 / 非拍=0 作为输入通道）+ **多尺度卷积栈** + BiLSTM；20s 片段、229 维 mel | 自家生产谱 + 公开数据；条件=难度 | 论文只讲 onset 模型；另有 sym 子模型（tap/flick） | 客观 F 值；生产中出初稿，**谱面交付成本减半** | MIT，代码+模型公开 | 推理可以 |
| TaikoNation（FDG'21）[arXiv 2107.12506](https://arxiv.org/abs/2107.12506)，[代码](https://github.com/emily-halina/TaikoNationV1) | LSTM **一次预测多步**（时间与类型联合），偏向长程 pattern | 约 100 张精选 Taiko 谱 | don / ka 等 | 自定义 patterning 指标，对比 DDC | 无许可证声明 | 是 |
| osuT5 [代码](https://github.com/gyataro/osuT5) | 仿 MT3 的 T5 编码器-解码器：mel 帧输入 → **稀疏事件 token**（时间 / 类型 / 坐标） | osu! 谱 | osu!std 物件 | 演示为主 | MIT，有 release 权重 | 推理勉强 |
| Mapperatorinator（OliBomby）[代码](https://github.com/OliBomby/Mapperatorinator)，[V32 更新日志](https://github.com/OliBomby/Mapperatorinator/releases) | 改版 Whisper 编码器-解码器，**219M** 参数，事件 token；16s 窗口，**后一窗用前一窗一半的 token 预填**；时间事件加随机偏移防漂移；**难度/年份/谱师/风格描述符/键数/LN 比例做成解码器前缀 token + CFG**；sustain token 表示 hold；坐标再用 diffusion 精修 | HF 数据集 [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps)（全部 ranked 谱），700k step × bs32；作者自述约 5700 GPU 小时 | std/taiko/ctb/**mania**（含 hold） | 社区试玩为主 | MIT，有权重、Colab | 配置有 device=cpu，但 219M 自回归在 CPU 上慢 |
| Mug Diffusion [代码](https://github.com/keytoyze/mug-diffusion) | 改版 Stable Diffusion（潜空间 diffusion），接入音频 | osu!mania / Malody 4K（[数据集列表](https://mugdiffusion.keytoix.vip/dataset.html)）；条件=星级 / Etterna MSD、ranked 风格、**LN 比例**、**pattern 类型**（jack/stream 等） | 4K note + LN | 社区反馈；[issue #3](https://github.com/Keytoyze/Mug-Diffusion/issues/3) 指出连续音被吸到 1/16、简单难度断流 | MIT，打包 exe 含权重 | 作者说 3050Ti 上 3 分钟歌出 4 张谱约 30s；CPU 未知 |
| BeatLearning [代码](https://github.com/sedthh/BeatLearning) | Transformer beatmap 生成 | osu! | osu! 物件（taiko/mania 计划中） | 示例谱 | Apache-2.0 | — |
| maiChartGen [GitCode](https://gitcode.com/Goldgom/maiChartGen)，[HF](https://huggingface.co/Goldgom/maiChartGen/tree/main) | EnCodec token + Transformer | maimai | maimai 音符 | 未见正式评估 | 见仓库 | — |
| PhiGen [代码](https://github.com/CMDHL/PhiGen) | 用 MoveNet 姿态 + OpenCV 从视频生成 Phira 谱，**不是学谱模型** | — | — | — | GPL-3.0 | — |

没找到公开的 Arcaea 或 Phigros 学习型生成器（[Aff2Preview](https://github.com/Arcaea-Infinity/Aff2Preview) 只做渲染）。

## B. 音乐基础编码器

| 模型 | 要点 | 帧率 / 规模 | 许可 | 对 BeatFCN |
|---|---|---|---|---|
| MERT [arXiv 2306.00107](https://arxiv.org/abs/2306.00107)，[代码](https://github.com/yizhilll/MERT) | MLM 自监督；声学老师 RVQ-VAE + 音乐老师 CQT | 95M / 330M；约 75Hz（≈13ms，未核） | 代码 Apache-2.0；HF 权重 v1 为 CC-BY-NC（未核） | 帧率接近 ±20ms，**最适合做冻结特征**；95M 在 CPU 上可预先算好缓存 |
| MusicFM [arXiv 2311.03318](https://arxiv.org/abs/2311.03318)，[代码](https://github.com/minzwon/musicfm) | BEST-RQ 随机投影量化 | 约 330M；约 25Hz（40ms，未核） | 代码 MIT；权重见仓库 | 帧太粗，只能当上下文 / 段落特征 |
| MuQ [arXiv 2501.01108](https://arxiv.org/abs/2501.01108)，[代码](https://github.com/tencent-ailab/MuQ) | Mel-RVQ 目标，MIR 基准上超过 MERT / MusicFM | 约 300M，开源版只用 MSD 训练（仓库说明开源版可能达不到论文成绩） | 代码 MIT，**权重 CC-BY-NC 4.0** | 同上；只能非商用 |
| BEATs [arXiv 2212.09058](https://arxiv.org/abs/2212.09058)，[代码](https://github.com/microsoft/unilm) | 通用音频（AudioSet）+ 声学 tokenizer | 约 90M；patch 级，时间分辨率粗 | MIT | 偏音效，对节奏帮助存疑 |
| CLAP（LAION）[arXiv 2211.06687](https://arxiv.org/abs/2211.06687)，[代码](https://github.com/LAION-AI/CLAP) | 音频-文本对比学习，**片段级向量** | — | CC0 | 只能做全曲 / 段落风格条件（情绪），不能定时 |
| Jukebox 特征（Castellon 2021）[arXiv 2107.05677](https://arxiv.org/abs/2107.05677) | 用 5B 级语言模型的中间层做 MIR 特征，效果强 | 极大 | — | CPU 上不可行 |

## C. 节拍与结构

- **beat_this**（ISMIR 2024）[arXiv 2407.21658](https://arxiv.org/abs/2407.21658)，[代码](https://github.com/CPJKU/beat_this)：输出拍与强拍，不需要 DBN 后处理。MIT。有 small 版（约 8MB），CPU 可跑。项目在用，但目前**没用强拍输出**。
- **All-In-One**（mir-aidj）[arXiv 2307.16425](https://arxiv.org/abs/2307.16425)，[代码](https://github.com/mir-aidj/all-in-one)：先用 Demucs 分轨，再用邻域注意力联合输出 BPM / 拍 / 强拍 / 段落边界和功能标签（intro/verse/chorus…）。在 Harmonix 上 8 折训练。MIT。能跑 CPU，但 Demucs 慢，适合离线缓存。

## D. 什么让这些工作成功（和"打击感"的关系）

1. **显式节拍 / 小节先验**：GenéLive! 的 beat guide 在简单难度上提升最明显。BeatFCN 已有网格 token，但强拍层级可以更明确。
2. **长程 pattern 一致性**：TaikoNation 多步联合预测；Mapperatorinator 用自回归事件 token、上一窗口预填。逐候选独立分类的方案正好缺这一块。滑音（glissando）是连续的 pattern，独立判每个点就容易断。
3. **条件 token + CFG**：Mapperatorinator 和 Mug Diffusion 都把难度 / LN 比例 / 风格当可控条件。
4. **数据量**：Mapperatorinator 用了全部 ranked 谱。BeatFCN 只有 232 首训练曲，项目笔记里加大 d_model 没收益，说明瓶颈在数据和标签噪声，不在架构。
5. 架构大小不是关键：DDC 和 GenéLive! 都是小 CNN+LSTM，就投入了实际生产。

## E. 可迁移思路排名（预期收益 / 成本）

1. **强拍 + 段落特征通道**（beat_this 已有的强拍输出、All-In-One 的段落标签、局部能量对比度做重音特征）。来源：GenéLive! 的 beat guide。成本：零训练，改特征后重训 v23 一次，CPU 不到 1 小时。针对"重音没有打击感"。
2. **冻结的 MERT-95M 特征离线缓存**，降维后拼进现有特征。CPU 上每首约分钟级（未实测），只算一次。预期能改善音色 / 乐器层面"该不该放音"的判断。风险是权重许可证，以及 75Hz 帧要对齐到网格。
3. **局部自回归 / 多步联合解码**：在候选序列上让模型看到已选的音符（teacher forcing，或两遍"选完再细化"），也可以加一个 CRF / 短语级损失。来源：TaikoNation、osuT5。针对滑音断裂。
4. **条件 token + CFG**：难度、hold 比例、风格做成前缀 token，训练时随机丢弃条件，推理时 cfg_scale。来源：Mapperatorinator、Mug Diffusion。成本低，主要提升可控性。
5. **扩数据：利用 osu!mania 4K 数据**（如 HF project-riz/osu-beatmaps）预训练"放置"头，再在 Phigros 数据上微调。收益可能最大，但磁盘 / 下载成本高，另外有分布偏移。

## F. v25 提案

- 编码：现有 log-mel / flux / pitch，加 MERT-95M 冻结层（取中间层加权，PCA 到约 64 维，以 .npy 缓存），再加强拍 / 段落 / 重音通道。
- 主干：保留 v23 的候选 / 网格 token 小 transformer（约 1–3M 参数）。**不改成纯 token 事件序列 seq2seq**：Layer 1 召回已有 0.90，逐候选分类更省，seq2seq 在 CPU、几百首歌的条件下不划算。
- 加一个第二遍细化器（可选多轮）：输入第一遍的选择结果加音频特征，输出最终选择和音符类型（Tap/Hold/Drag/Flick 用多头；Hold 给出时长回归或结束 token）。这样以很低成本得到"看见上下文"的效果，类似 mask-predict。
- **不用 diffusion**：Mug Diffusion 的优势在多样性和 pattern 控制，但训练 / 推理都更依赖 GPU，评测上也不比自回归方法更准。
- 算力估计：MERT 特征提取一次性约 300 首 ×（CPU 上 1–3 分钟/首，未实测），约 5–15 小时，可 3 进程并行；训练与 v23 同量级，加细化器约 1.5–2 倍，单次实验 CPU 1–2 小时。**CPU 足够**。osu 数据预训练要上 GPU，或者 CPU 跑数十小时。

## G. 注意事项

- 人际一致度约 0.70，当前约 0.64，±20ms F1 能涨的空间很小。"打击感"需要另做主观评测或按重音 / 滑音片段单独评测，不能只看 F1。
- MERT / MuQ 的权重多为 CC-BY-NC，商用不可行。
- 本文所有收益都是推测，没有实验验证。表中标「未核」的帧率、许可数字要再查一遍。
- 已有工作的评估大多是社区试玩，可比的客观基准很少。

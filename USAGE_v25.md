# v25 使用手册：tap+双押 草稿生成（面向 Phira 编辑器导入）

任务定义：音频 → 音符时间点 + 双押标记，全部输出 Tap，进编辑器人工加工。
不预测类型（hold/flick/drag）、不预测位置——那部分留给编辑器里的人。

## 方案构成

| 项 | 内容 |
|---|---|
| 模型 | `BeatTokenNet`（v23 骨干），输入 130 维（127 + 节拍引导/重音/起音包络），输出 2 头（onset + 双押），密度条件，约 60 万参数 |
| 新增文件 | `feat_v25.py`（特征唯一入口）、`beat_fcn_train_v25.py`（训练）、`gen/make_chart_v25.py`（出谱） |
| 修改文件 | `model_v23.py`（stem 新增 `sep3` 空洞卷积选项，默认仍 `sep`，旧 ckpt 不受影响）、`extract_full.py`（`--drag` 开关） |
| 数据规则 | 纯 Phira、只收 IN/AT（训练加 `--strict-diff`）、rating/人数不限；同刻 ≥2 音折叠为双押（`extract_full` 的 `min(2,·)` 天然实现） |

## 全流程命令

```bash
# 0) 环境：主 venv（torch + requirements.txt）+ venv_bt（beat_this/librosa/pillow，出谱用）
python3 -m venv venv && source venv/bin/activate
pip install torch -r requirements.txt
python3 -m venv venv_bt && source venv_bt/bin/activate && pip install torch torchaudio beat_this librosa pillow

# 1) 下载谱面 zip（用户自备语料时跳过，直接把 zip 放进 charts/）
python download_shard.py --jsonl phira_all_charts.jsonl --outdir charts --manifest manifest.csv --workers 8

# 2) 提特征到原始目录（--drag：drag onset 计入标签；同刻≥2 自动折叠为双押）
python extract_full.py charts grid_npz_raw grid 8 "" 32 --drag
python extract_pitch.py charts pitch_npz_full "" 8

# 3) offset 清洗（必做；不洗的历史实测 F1 只有 0.26，洗完 0.54）
#    用法：python bench/offset_clean.py <src_npz_dir> <dst_npz_dir> <report.json>
python bench/offset_clean.py grid_npz_raw grid_npz_clean offset_report.json

# 4) beat_this 节拍缓存（训练通道 + 推理网格共用；CPU 每首几十秒，一次性）
#    root 目录下需有 charts/（原始 zip）和 grid_npz_clean/（清洗后的 npz）
source venv_bt/bin/activate
python gen/detect_all.py . beats_clean

# 5) 训练（4090：几分钟/轮；建议 100+ 轮看曲线平了再停）
source venv/bin/activate
python beat_fcn_train_v25.py --data ./grid_npz_clean --pitch-dir ./pitch_npz_full \
    --split ./split_full128_final.json --guide-dir ./beats_clean \
    --strict-diff --epochs 100 --device cuda --tag v25

# 6) 出谱（.pez 直接拖进 Phira 编辑器）
source venv_bt/bin/activate
python gen/make_chart_v25.py song.mp3 beat_fcn_best_v25.pt out_dir --nps 5 --name 曲名
```

## 训练参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--jitter-ms` | 12 | 时间抖动增强：grid 相对特征随机平移 ±12ms，模拟谱师 offset 噪声（0=关） |
| `--ema` | 0.999 | EMA 权重平均；评估与最优 ckpt 都用 EMA（0=关） |
| `--focal` | 0.0 | o1 focal gamma（建议 A/B 一次 1.5） |
| `--soft-nb` | 0.0 | 邻近格软标签（建议 A/B 一次 0.3） |
| `--stem` | sep3 | sep3=带 ±130ms 空洞卷积局部上下文；sep=旧行为 |
| `--w-double` | 4.5 | 双押类别权重（双押召回低可加到 6–8 再 A/B） |
| `--strict-diff` | 关 | 只用文件名难度 [IN]/[AT] 的谱面（本方案的默认数据口径） |

## 出谱参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--nps` | 4.0 | 目标密度（音符/秒），top-K 精确命中 |
| `--th2` | 0.55 | 双押阈值。草稿用途宁多标：调低→多标让人删，调高→漏标人补 |
| `--accent-gate` | 40 | 双押重音门控：重音低于已选候选第 40 百分位的禁止双押（0=关） |
| `--no-smooth` | 关 | 关掉节拍平滑（平滑实测节拍 F@20 0.618→0.644，一般别关） |

## 冲效果的顺序（USAGE 之外的杠杆）

1. 数据用满（IN/AT 全量 ≈ 数千张）+ offset 清洗（两件事占收益大头）；
2. 训到 val 平（历史上从没训到饱和过）；
3. `--d-model 160 --layers 6` 放大 A/B（数据上来了才值得）；
4. 训 4–5 个种子做 ensemble（推理端 sigmoid 平均）；
5. `--focal 1.5`、`--soft-nb 0.3` 各 A/B 一次。

## 纪律

- val 固定 300 首（默认 `--eval-songs 300 --eval-seed 42`），实验次数有限，别在 val 上扫海；
- test 只在最终模型上开一次；
- 训练/推理的特征必须都走 `feat_v25.append_channels`——这是唯一入口，别在两边各自拼通道。

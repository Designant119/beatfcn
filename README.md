# BeatFCN

给一首歌，自动生成 Phigros / Phira 风格谱面的**按键时间点**（单押 / 双押）。
只做“节拍骨架”：不预测左右位置、不预测判定线运动，输出需要再加工才能成为可玩谱面。


## 当前状态

| 项目 | 状态 |
|---|---|
| 代码（下载 / 提取 / 训练 / 重评 / 推理） | ✅ 全流程在 CPU 上实测跑通（`demo_cpu.sh`） |
| 数据划分 `split_full128_final.json` | ✅ train 11373 / val 1577 / test 209，按歌名分组无泄漏（`test_core.py` 验证） |
| 正式训练好的模型 | ❌ 还没有（需要 GPU，RTX 3090 约 70-130 小时，约 ¥90-170） |
| v22 阈值模板 | ❌ 训完后才能生成（目前推理用固定 th1=0.4 / th2=0.5） |

仓库**不含**音频、训练数据（npz，约 95GB）和模型权重；它们按下面的流程从 Phira 下载并生成。


## v23（推荐）：网格 token 架构，训练提速约 100 倍

v22 在每个音频帧上做全局注意力（3 分钟歌≈15k 帧，O(T²)）。v23 (`model_v23.py`) 改成：
帧级轻量卷积 → 按节拍网格每 16 格（半拍）收拢成一个 token → Transformer 只看约 1k 个 token
→ 逐格小头（token 上下文 + 该格帧特征）输出 o1/o2。输入输出接口不变，`infer_v22.py` / `reval_v22.py` 自动识别 v23 ckpt。

```bash
python beat_fcn_train_v23.py --data ./grid_npz_full128 --pitch-dir ./pitch_npz_full \
    --split ./split_full128_final.json --epochs 60 --device cuda   # 首轮自动建 float16 特征缓存 feat_cache_v23/
python infer_v22.py --wav song.mp3 --ckpt beat_fcn_best_v23.pt --out out.json --bpm 128   # th1 自动用 ckpt 里的最优值
```
- 评估口径已修正：`strict` = 格级精确 F1（选模型），`loose` = ±50ms F1；v22 的 inst/grid 两个数恒等。
- `--strict-diff` 只用 IN/AT 难度谱面（v22 实际会混入 SP/EX/数字难度等，约 30%）。
- 基准脚本：`bench/bench_speed.py`（单曲 FLOPs/耗时/内存）、`bench/bench_eval.py`（同口径精度对比），结果见 `bench/RESULTS.md`。

## 环境

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch            # GPU 机器装对应 CUDA 版本
pip install -r requirements.txt
sudo apt install -y ffmpeg curl   # 音频解码 / 下载
python test_core.py               # 单元测试（CPU 即可）
```

## 一键冒烟测试（CPU，约 5-10 分钟）

```bash
bash demo_cpu.sh
```
下载 18 个谱面 → 提取特征 → 训练 3 轮小模型 → 重评 → 对一首 test 歌推理，产物在 `demo_cpu/`。
只证明流程通，结果没有参考价值。

## 正式流程

```bash
# 1. 下载全部谱面 zip（20-30GB）
python download_shard.py --jsonl phira_all_charts.jsonl --outdir charts --manifest manifest.csv --workers 8
# 2. 提取特征（CPU 多进程，几小时）
python extract_full.py charts grid_npz_full128 grid 8 "" 32
python extract_pitch.py charts pitch_npz_full "" 8
# 3. 训练（GPU）
python beat_fcn_train_v22.py --epochs 200 --tag v22 --device cuda --batch-size 4 --threads 8 \
    --data ./grid_npz_full128 --split ./split_full128_final.json --pitch-dir ./pitch_npz_full
#    中断后加 --resume 续训；或用网页控制台：python train_web.py（浏览器开 http://localhost:8000）
# 4. 全量重评选冠军，并在 test 上报一次无偏分数
python reval_v22.py
# 5. 推理
python infer_v22.py --wav song.mp3 --ckpt beat_fcn_best_v22.pt --out out.json --bpm 128 [--offset 0.0]
#    条件采音：--notes 已采.json --notes-end 30 ；变速：--bpm-list "0:120,30:140"
```

输出：`{"time": [秒...], "type": [1=单押, 2=双押...], ...}`

CPU / 小显存调试可加 `--crop-frames 3000 --ch 32`（限制训练片段长度和模型宽度）。

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | 模型（CNN + Transformer，约 65.9 万参数） |
| `beat_fcn_train_v22.py` | 训练 |
| `reval_v22.py` | 全量重评 / 选冠军 / test 打分 |
| `infer_v22.py` | 推理 |
| `chart_utils.py` | 谱面解析、mel、常量 |
| `extract_full.py` / `extract_pitch.py` | 特征提取 |
| `download_shard.py` + `phira_all_charts.jsonl` | 谱面下载 |
| `train_web.py` / `train_helper.py` | 网页 / 命令行训练控制台 |
| `split_full128_final.json` | **唯一有效**的数据划分 |
| `archive/` | 历史文件（旧 split 等），不要用 |

#!/usr/bin/env bash
# CPU 冒烟测试：下载少量谱面 → 提取特征 → 小模型训练 3 轮 → reval → 推理一首 test 歌
# 只证明全流程能跑通，不是正式训练（正式训练需要 GPU，见 技术交接文档_AutoDL版.md）
# 用法: bash demo_cpu.sh   （需要 python3、ffmpeg、curl；约 5-10 分钟，内存 ≥8GB）
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
W="$SRC/demo_cpu"; mkdir -p "$W"; cd "$W"
cp "$SRC/split_full128_final.json" .

# 1) 选 12 train + 4 val + 2 test 的谱面 id
"$PY" - <<'P'
import json
s = json.load(open('split_full128_final.json'))
pick = lambda n, k: [i for i, v in s.items() if v == n][:k]
open('ids.txt', 'w').write('\n'.join(pick('train', 12) + pick('val', 4) + pick('test', 2)) + '\n')
P
"$PY" "$SRC/download_shard.py" --jsonl "$SRC/phira_all_charts.jsonl" --outdir charts \
    --manifest manifest.csv --workers 6 --only-ids ids.txt

# 2) 提取特征
"$PY" "$SRC/extract_full.py" charts grid_npz_full128 grid 4 "" 32
"$PY" "$SRC/extract_pitch.py" charts pitch_npz_full "" 4

# 3) 小模型训练（ch=32，片段≤3000帧≈35秒，CPU 内存友好）
"$PY" "$SRC/beat_fcn_train_v22.py" --epochs 3 --tag demo --device cpu --batch-size 2 --threads 2 \
    --data grid_npz_full128 --split split_full128_final.json --pitch-dir pitch_npz_full \
    --ch 32 --eval-songs 4 --eval-seed 42 --crop-frames 3000

# 4) 重评选冠军 + test 分数
"$PY" "$SRC/reval_v22.py" --data-root . --device cpu --tag demo

# 5) 推理：取第一个 test 谱面的音频
TID=$("$PY" -c "import json;s=json.load(open('split_full128_final.json'));print(next(i for i,v in s.items() if v=='test'))")
rm -rf song && mkdir song && (cd song && unzip -q -o "../charts/$TID.zip")
AUDIO=$(find song -maxdepth 1 -type f \( -iname "*.mp3" -o -iname "*.ogg" -o -iname "*.wav" \) | head -1)
"$PY" "$SRC/infer_v22.py" --wav "$AUDIO" --ckpt beat_fcn_best_demo.pt --out "out_$TID.json" --device cpu
echo "完成：输出在 $W/out_$TID.json（3轮小模型，结果只用于验证流程，不代表效果）"

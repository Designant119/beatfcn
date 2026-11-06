#!/bin/bash
cd /root/beatfcn; export BEATFCN_ALLOW_MISSING_MERT=1 PY=/root/envs/bf/bin/python P=/data/evolve/plans; O=/data/evolve/g21; mkdir -p $O; T=/data/full-database-v1/chart-timing/chart_timing-v2.json; C=R1-peer
K='{"kind": 1.0, "head_block": 0.7, "head_cell": 0.7, "head_ms": 0.7}'
echo "=== $(date +%H:%M:%S) train"; timeout 14000 $PY -u -m scripts.evolve_finetune --plan $P/$C.json --updates ${1:-2000} --no-compile --deadline-unix ${2} > /root/data/evolve-$C.log 2>&1; echo "train_rc=$? $(date +%H:%M:%S)"; tail -n 1 /root/data/evolve-$C.log | cut -c1-150
CK=$(ls -t /data/evolve/$C/step-*.pt | head -n 1); echo "ck $CK"; n=E_final
timeout 1500 $PY -u -m scripts.run_workflow_benchmark --plan /data/round5-level/plan.json --checkpoint $CK --weights raw --tape /data/full-database-v1/native-v2-tape/tape.json --root / --raw-root /data/raw --device cuda --human-baseline /data/stage0/reports/goal-100usd-20261101/human-floor-full/workflow-benchmark.json --max-songs 2000 --panel-split val --only-audio /data/evolve/screen60_only.json --temperature 1 --seed 0 --max-wall-s 1400 --flush-reserve-s 40 --out-dir $O/$n --hold-release-ms 40 --save-samples --parallel-samples --batch-size 320 --scheduler continuous --no-score --consensus-min-jack-ms 100 --consensus 8 --field-temperature "$K" --charter-timing $T --history-reference > $O/$n.log 2>&1; echo "rc=$? $n $(date +%H:%M:%S)"
bash /root/onbox_score.sh $O/$n 8 > /dev/null 2>&1; rm -f $O/$n/samples.jsonl; OMP_NUM_THREADS=1 $PY /root/onbox_single.py $O/$n > $O/$n.single.log 2>&1
echo "=== $(date +%H:%M:%S) g21_done"

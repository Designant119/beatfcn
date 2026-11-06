#!/bin/bash
cd /root/beatfcn; export BEATFCN_ALLOW_MISSING_MERT=1 PY=/root/envs/bf/bin/python P=/data/evolve/plans; O=/data/evolve/g20; mkdir -p $O; T=/data/full-database-v1/chart-timing/chart_timing-v2.json; C=C9-long
rm -rf /data/evolve/C8-song-style /data/evolve/g5/P_r5_product/samples.jsonl
K='{"kind": 1.0, "head_block": 0.7, "head_cell": 0.7, "head_ms": 0.7}'
ev(){ n=$1; ck=$2; shift 2; echo "=== $(date +%H:%M:%S) $n"; timeout 1500 $PY -u -m scripts.run_workflow_benchmark --plan /data/round5-level/plan.json --checkpoint $ck --weights raw --tape /data/full-database-v1/native-v2-tape/tape.json --root / --raw-root /data/raw --device cuda --human-baseline /data/stage0/reports/goal-100usd-20261101/human-floor-full/workflow-benchmark.json --max-songs 2000 --panel-split val --only-audio /data/evolve/screen60_only.json --temperature 1 --seed 0 --max-wall-s 1400 --flush-reserve-s 40 --out-dir $O/$n --hold-release-ms 40 --save-samples --parallel-samples --batch-size 320 --scheduler continuous --no-score --consensus-min-jack-ms 100 --consensus 8 --field-temperature "$K" --charter-timing $T --history-reference "$@" > $O/$n.log 2>&1; echo "rc=$? $n $(date +%H:%M:%S)"; (bash /root/onbox_score.sh $O/$n 8 >/dev/null; rm -f $O/$n/samples.jsonl) & }
for U in 6000 12000; do
  echo "=== $(date +%H:%M:%S) train to $U"; A=""; [ $U != 6000 ] && A="--resume"; timeout 13500 $PY -u -m scripts.evolve_finetune --plan $P/$C.json --updates $U $A --deadline-unix $(( $(date +%s)+13400 )) > /root/data/evolve-$C-$U.log 2>&1; echo "train_rc=$? $U $(date +%H:%M:%S)"; tail -n 1 /root/data/evolve-$C-$U.log | cut -c1-150
  CK=$(printf "/data/evolve/$C/step-%06d.pt" $U); [ -f $CK ] && cp $CK /data/evolve/$C-keep-$U.pt && ev E_$U $CK
  ls -t /data/evolve/$C/step-*.pt 2>/dev/null | tail -n +3 | xargs -r rm -f
done
wait; echo "=== $(date +%H:%M:%S) g20_done"

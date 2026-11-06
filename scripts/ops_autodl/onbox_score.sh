#!/bin/bash
# onbox_score.sh ARMDIR N: rebuild consensus rules on the instance and score benchmark v3 with audio features. Writes small JSONs only.
D=$1; N=$2; cd /root/beatfcn; export OMP_NUM_THREADS=1 BEATFCN_ALLOW_MISSING_MERT=1; PY=/root/envs/bf/bin/python; T=/data/full-database-v1/native-v2-tape
V="--floor /data/evolve/floor.json --tape $T --timing /data/full-database-v1/chart-timing/chart_timing-v2.json --features-dir /root/data/feat_new/features"
rule(){ tag=$1; shift; o=$D/t0_$tag; mkdir -p $o; [ -f $o/drafts-consensus-$N-jackfree.json ] || $PY -u -m scripts.run_workflow_benchmark --score-from $D --tape $T/tape.json --root / --raw-root /data/raw --consensus $N --consensus-min-jack-ms 100 --hold-release-ms 40 --human-baseline /data/stage0/reports/goal-100usd-20261101/human-floor-full/workflow-benchmark.json --out-dir $o "$@" > $o/score.log 2>&1; $PY scripts/benchmark_v3.py score --drafts $o/drafts-consensus-$N-jackfree.json --panel $D/run-meta.json --out $o/v3.json --label $(basename $D)_$tag $V > /dev/null 2> $o/v3.err || tail -n 2 $o/v3.err; rm -f $o/drafts-samples.json $o/drafts.json; echo "done $(basename $D) $tag"; }
rule maj &
rule anchor --consensus-pattern anchor &
[ "$N" = 8 ] && rule anchor_v3 --consensus-pattern anchor --consensus-votes 3 &
[ "$N" = 4 ] && rule anchor_v1 --consensus-pattern anchor --consensus-votes 1 &
wait

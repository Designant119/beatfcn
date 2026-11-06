#!/bin/bash
# Tokenless guard. Stops at epoch in /root/guard_deadline, or after 15 idle minutes with no known job.
PAT='run_workflow_benchmark|evolve_finetune|teacher_ce_probe|bench_sched|evolve_surgery|batch_extract|g[0-9]+_.*\.sh|rsync|scp'
idle=0
while true; do
  [ -f /root/guard.stop ] && exit 0
  now=$(date +%s); dl=$(cat /root/guard_deadline 2>/dev/null || echo $((now+3600)))
  if pgrep -f "$PAT" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  echo "$(date +%H:%M:%S) idle=$idle deadline_in=$((dl-now)) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)" >> /root/guard.log
  if [ "$now" -ge "$dl" ] || [ "$idle" -ge 15 ]; then echo "$(date) guard shutdown" >> /root/guard.log; sync; nohup /usr/bin/shutdown >/dev/null 2>&1; fi
  sleep 60
done

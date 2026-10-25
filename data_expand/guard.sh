#!/bin/bash
# kill $1 pattern if free disk <15GB or inode use >85%
while sleep 60; do
  av=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); iu=$(df --output=ipcent / | tail -1 | tr -dc 0-9)
  echo "$(date +%T) avail=${av}G inode=${iu}%"
  if [ "$av" -lt 15 ] || [ "$iu" -gt 85 ]; then echo GUARD_STOP; pkill -f "$1"; exit 1; fi
done

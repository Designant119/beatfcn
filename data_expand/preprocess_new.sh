#!/bin/bash
# Extract newly downloaded charts into a staging dir, offset-clean them, then merge into
# grid_npz_full128 / grid_npz_clean without touching existing files. Run from /workspace/bench.
set -e
P=/workspace/phira; PY=/workspace/venv/bin/python
awk -F, '$9=="success"{print $1}' manifest_expand.csv > ids_expand_ok.txt
timeout 10h nice -n 10 $PY $P/extract_full.py charts grid_npz_new grid 3 ids_expand_ok.txt 32
timeout 4h nice -n 10 $PY $P/bench/offset_clean.py grid_npz_new grid_npz_new_clean offset_report_expand.json
$PY - <<'P'
import json,os,shutil
r=json.load(open('offset_report_expand.json'))
drop={k for k,v in r.items() if v.get('drop')}
n=0
for f in os.listdir('grid_npz_new'):
    if not os.path.exists('grid_npz_full128/'+f): shutil.move('grid_npz_new/'+f,'grid_npz_full128/'+f)
for f in os.listdir('grid_npz_new_clean'):
    k=f[:-4]
    if k in drop or os.path.exists('grid_npz_clean/'+f): continue
    shutil.move('grid_npz_new_clean/'+f,'grid_npz_clean/'+f); n+=1
print('new extracted',len(r),'kept',n,'dropped',len(drop))
P

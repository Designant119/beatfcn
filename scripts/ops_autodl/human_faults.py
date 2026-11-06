"""Playability faults per minute of human Phira charts on the SCREEN-60 panel, same rules as onbox_single."""
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ['/root/beatfcn', '/root/beatfcn/scripts']
import run_workflow_benchmark as W
from beatfcn.events.phira_postprocess import playable
from beatfcn.model.full_event_codec import TYPES
meta = json.load(open('/data/evolve/base_S5/run-meta.json'))
KEYS = ('fragments_dropped', 'chords_cleaned', 'holds_shortened', 'holds_to_tap', 'dropped', 'shortened', 'to_tap', 'types_cleaned')
total = {k: 0 for k in KEYS}; minutes = notes = 0
for s in (x for x in meta['panel']['songs'] if x['mode'] == 'phira'):
    r = W.primary_reference(s)
    with np.load(Path('/')/r['target_path'].lstrip('/') if Path('/'+r['target_path'].lstrip('/')).exists() else Path('/data/full-database-v1/native-v2-tape')/Path(r['target_path']).name, allow_pickle=False) as z:
        rows = [dict(onset_us=int(a), end_us=int(b), type=TYPES[int(k)], lane=-1) for a, b, k in zip(z['onset_us'], z['end_us'], z['kind'])]
    _, rep = playable(rows, 'standard')
    notes += len(rows); minutes += s['duration_us']/60e6
    for k in KEYS: total[k] += rep.get(k, 0)
print('human notes/min', round(notes/minutes, 1), 'faults/min', round(sum(total.values())/minutes, 2), {k: round(v/minutes, 2) for k, v in total.items()})

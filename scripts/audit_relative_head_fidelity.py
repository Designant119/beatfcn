"""Compare relative head coding to frozen real requests, without model execution."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.model.full_event_codec import EventState
from beatfcn.model.relative_event_codec import RelativeHeadState, encode_events, teacher_sequence
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    pp = ROOT/'bench/rhythm-v2/d88-relative-fidelity-v1.json'
    plan = json.loads(pp.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Prepared fidelity input changed: '+name)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposed = {i for s in json.loads((ROOT/plan['exposure']).read_text())['steps'] for i in s['indices']}
    groups = defaultdict(list)
    for i, row in enumerate(panel.records):
        if i in exposed or row['split'] == 'val':
            groups[row['chart_index']].append(i)
    indices = sorted({i for group in groups.values() for i in (group[0], group[-1])})
    if len(indices) > plan['maximum_requests']:
        raise ValueError('Frozen request limit exceeded')
    start = time.monotonic()
    rows, sources = [], dict(plan['inputs_sha256'])
    for i in indices:
        if time.monotonic()-start > plan['maximum_wall_s']:
            raise TimeoutError('Fidelity deadline reached; no training allowed')
        row = panel.records[i]
        chart = panel.charts[row['chart_index']]
        for key in ('token', 'target'):
            path = chart[key+'_path']
            if path not in sources:
                if sha(ROOT/path) != chart[key+'_sha256']:
                    raise ValueError('Frozen target or original token array changed')
                sources[path] = chart[key+'_sha256']
        request = event_request(row['request'])
        events = panel.targets(i)
        original, _ = panel._arrays(row['chart_index'])
        original = original[row['token_start']:row['token_stop']].tolist()
        tokens = encode_events(events, request)
        if len(tokens) != len(original):
            raise ValueError('Changed token exposure')
        old, new = EventState(request), RelativeHeadState(request)
        changed = Counter()
        for a, b in zip(original, tokens, strict=True):
            field, value = old.vocab.decision(a)
            if new.vocab.decision(b)[0] != field:
                raise ValueError('Changed prediction field or ordering')
            if a != b:
                changed[field] += 1
                if not field.startswith('head_') or field == 'head_block' and value >= old.vocab.blocks:
                    raise ValueError('Changed a tail/member/type/key/COPY token')
            old.apply(a)
            new.apply(b)
            if old.events != new.events or old.last_onset != new.last_onset or old.lane_ends != new.lane_ends:
                raise ValueError('Physical event or occupancy state differs')
        if not old.done or not new.done or sorted(new.events) != sorted(events):
            raise ValueError('Incomplete or non-fidelitous target reconstruction')
        seq = teacher_sequence(tokens, request)
        if (not np.isfinite(seq['times']).all() or
                not seq['allowed'][np.arange(len(tokens)), seq['targets']].all()):
            raise ValueError('Teacher clock or target legality mismatch')
        rows.append(dict(index=i, chart_id=row['chart_id'], split=row['split'], mode=row['mode'],
            stratum=row['stratum'], events=len(events), tokens=len(tokens), changed_fields=dict(changed),
            carry=len(request.known_carry), copy=len(request.copy_times),
            cross_window_hold_ends=sum(e.type == 'hold' and e.end_us > request.end_us for e in events),
            types=dict(Counter(e.type for e in events)), teacher_shape=list(seq['allowed'].shape),
            teacher_query_range_s=[float(seq['times'].min()), float(seq['times'].max())]))
    result = dict(id=plan['id'], complete=True, passed=True, source_sha256=sources,
        plan_sha256=sha(pp), selection='First and last exposed train request / first and last val request per chart',
        rows=rows, requests=len(rows), charts=len({r['chart_id'] for r in rows}),
        events=sum(r['events'] for r in rows), tokens=sum(r['tokens'] for r in rows),
        empty_requests=sum(r['events']==0 for r in rows),
        requests_with_carry=sum(r['carry']>0 for r in rows),
        cross_window_hold_ends=sum(r['cross_window_hold_ends'] for r in rows),
        requests_with_copy=sum(r['copy']>0 for r in rows),
        by_mode=dict(Counter(r['mode'] for r in rows)),
        by_split=dict(Counter(r['split'] for r in rows)),
        optimizer_updates=0, model_forward_calls=0, new_cloud_compute_calls=0,
        quality_passed=False, training_activated=False, wall_s=time.monotonic()-start,
        scope='Real-target fidelity only. No audio tensor, learned prediction, cache parity or generation quality claim.')
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))


if __name__ == '__main__':
    main()

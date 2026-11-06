"""Compare actual exposed chart intervals with saved D87 free outputs; no model calls."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    types = Counter()
    gaps, tap_gaps, first_delays = [], [], []
    for row in rows:
        events = row['events']
        types.update(e['type'] for e in events)
        heads = sorted({e['onset_us'] for e in events})
        taps = sorted({e['onset_us'] for e in events if e['type'] == 'tap'})
        gaps.extend(np.diff(heads).tolist())
        tap_gaps.extend(np.diff(taps).tolist())
        if heads:
            first_delays.append(heads[0] - row['start_us'])

    def describe(values):
        a = np.asarray(values, dtype=np.int64)
        if len(a) and (a < 0).any():
            raise ValueError('Negative interval or first onset before request')
        return dict(n=len(a), quantiles_ms=np.quantile(a/1000, [0, .25, .5, .75, .95, 1]).tolist() if len(a) else None,
                    under10ms_fraction=float((a < 10000).mean()) if len(a) else None,
                    exactly1ms_count=int((a == 1000).sum()),
                    histogram=dict(Counter('lt10' if t < 10000 else '10to40' if t < 40000 else
                                           '40to100' if t < 100000 else '100to250' if t < 250000 else
                                           '250to500' if t < 500000 else 'ge500' for t in a)))

    return dict(requests=len(rows), empty_requests=sum(not r['events'] for r in rows),
                events=sum(types.values()), types=dict(types), onset_group_intervals=describe(gaps),
                tap_intervals=describe(tap_gaps), first_delay=describe(first_delays))


def main():
    output = ROOT/'reports/d87_interval_distribution_v1.json'
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    plan_path = 'bench/rhythm-v2/d87-sequence-candidate-v1.json'
    plan = json.loads((ROOT/plan_path).read_text())
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen source changed: '+source)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposure = json.loads((ROOT/plan['exposure']).read_text())
    required = {i for s in exposure['steps'] for i in s['indices']}
    records, sources = {}, {p: sha(ROOT/p) for p in [plan_path, plan['tape'], plan['exposure'],
        'scripts/audit_sequence_interval_distribution.py', 'beatfcn/train/local_request_panel.py']}
    for i in sorted(required):
        row = panel.records[i]
        if row['split'] != 'train':
            raise ValueError('Training exposure contains validation')
        chart = panel.charts[row['chart_index']]
        path = chart['target_path']
        if path not in sources:
            if sha(ROOT/path) != chart['target_sha256']:
                raise ValueError('Training target changed')
            sources[path] = chart['target_sha256']
        events = panel.targets(i)
        if len(events) != row['event_count']:
            raise ValueError('Target count changed')
        records[i] = dict(mode=row['mode'], start_us=row['request']['start_us'],
                          events=[dict(onset_us=e.onset_us, type=e.type) for e in events])
    groups = defaultdict(list)
    for limit in (250, 500, 1000):
        for s in exposure['steps'][:limit]:
            for i in s['indices']:
                groups[f'train_to{limit}/{records[i]["mode"]}'].append(records[i])
    for step in (250, 500):
        for arm in ('control', 'candidate'):
            path = f'reports/d87_{arm}_stage_{step}_v1.json'
            if not (ROOT/path).exists():
                continue
            report = json.loads((ROOT/path).read_text())
            if not report.get('complete') or report.get('error'):
                continue
            sources[path] = sha(ROOT/path)
            for r in report['rows']:
                groups[f'{arm}_{step}/{r["role"]}'].append(dict(start_us=r['scope_us'][0], events=r['prediction']['events']))
                if arm == 'control':
                    groups[f'val_reference_{step}/{r["role"]}'].append(dict(start_us=r['scope_us'][0], events=r['truth_events']))
    result = dict(id='D87-interval-distribution-v1', complete=True, optimizer_updates=0, model_forward_calls=0,
                  new_cloud_compute_calls=0, elapsed_s=time.monotonic()-start,
                  summaries={k:summarize(v) for k,v in groups.items()}, source_sha256=sources,
                  caveats=['Counts retain actual request-exposure multiplicity, not independent songs.',
                           '1000-step target distribution includes future exposure if that arm is not yet at1000.',
                           'Distinct head gaps exclude same-time members; all real chart types remain intact.',
                           'Distribution mismatch is not a causal mechanism or an instruction to match one chart exactly.'])
    write_json(output, result)
    print(json.dumps({k:dict(events=v['events'], gaps=v['onset_group_intervals']) for k,v in result['summaries'].items()}))


if __name__ == '__main__':
    main()

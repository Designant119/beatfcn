"""Independent replay, partition and report-identity checks; no training."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.model.full_event_codec import EventState, FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.reference_completion import TASKS, completion_row

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    plan = read('bench/rhythm-v2/root-cause-v2.json')
    report = read(plan['output'])
    assert report['complete'] and report['checkpoint_unchanged']
    assert len(report['rows']) == len(plan['records'])
    old = read('reports/d98_teacher_decisions_v1.json')
    by_index = {r['index']: r for r in old['rows']}
    comparisons = 0
    for row in report['rows']:
        previous = by_index[row['index']]
        for name, values in row['variants']['correct'].items():
            assert values['count'] == previous['fields'][name]['count']
            assert values['correct'] == previous['fields'][name]['correct']
            assert abs(np.mean(values['nll']) - previous['fields'][name]['nll_sum'] / values['count']) < 1e-5
            comparisons += 1
    trajectories = 0
    for row in report['rollouts']:
        for arm in row['outputs'].values():
            artifact = read(arm['artifact'])
            pred = artifact['prediction']
            state = EventState(event_request(artifact['request']))
            assert len(pred['trace']) == len(pred['tokens'])
            for token, trace in zip(pred['tokens'], pred['trace']):
                assert token == trace['token']
                assert FIELDS[state.query()[0]] == trace['field']
                if not trace['forced_oracle']:
                    assert token == trace['argmax']
                state.apply(token)
            assert state.done == pred['complete']
            sig = lambda e: (e.onset_us, e.end_us, e.type, e.lane)
            assert Counter(map(sig, state.events)) == Counter((e['onset_us'], e['end_us'], e['type'], e['lane']) for e in pred['events'])
            trajectories += 1
    health_plan = read('bench/rhythm-v2/reference-completion-v1.json')
    health = read(health_plan['report'])
    assert health['complete'] and health['passed'] and health['optimizer_updates'] == 100
    panel = LocalRequestPanel(ROOT, health_plan['tape'], health_plan['features'])
    masks = 0
    for case in health_plan['cases']:
        for task in TASKS:
            a, req, meta = completion_row(panel, case['index'], task, reference=True, seed=health_plan['seed'])
            b, req2, _ = completion_row(panel, case['index'], task, reference=False, seed=health_plan['seed'])
            assert req == req2
            assert panel.records[case['index']]['split'] == 'train'
            for name, array in a['inputs'].items():
                assert np.isfinite(array).all()
                if not name.startswith('reference'):
                    assert np.array_equal(array, b['inputs'][name])
            for name in a['sequence']:
                assert np.array_equal(a['sequence'][name], b['sequence'][name])
            sig = lambda e: (e['onset_us'], e['end_us'], e['type'], e['lane'])
            assert Counter(map(sig, meta['original'])) == Counter(map(sig, meta['visible'] + meta['hidden']))
            masks += 1
    for case in health['generation']:
        artifact = read(case['artifact'])
        assert artifact['exact'] and artifact['prediction']['complete']
        state = EventState(event_request(artifact['metadata']['request']))
        for token in artifact['prediction']['tokens']:
            state.apply(token)
        assert state.done
        signature = lambda e: (e.onset_us, e.end_us, e.type, e.lane)
        assert Counter(map(signature, state.events)) == Counter((e['onset_us'], e['end_us'], e['type'], e['lane']) for e in artifact['metadata']['hidden'])
    for key in (plan['checkpoint'], health_plan['parent']):
        expected = plan['source_sha256'].get(key, health_plan['source_sha256'].get(key))
        assert sha(key) == expected
    result = dict(passed=True, teacher_field_reconstructions=comparisons,
                  real_causal_trajectory_replays=trajectories, matched_mask_cases=masks,
                  health_exact_replays=len(health['generation']), parent_weights_unchanged=True,
                  optimizer_updates=0, cloud_calls=0)
    path = ROOT / 'reports/causal_completion_verification_v1.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

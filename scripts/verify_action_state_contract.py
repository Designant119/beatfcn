"""Verify D49 saved transitions against source events without importing its codec."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def event_key(event):
    return (event['onset_us'], event['end_us'], event['type'], event['lane'])


def verify():
    plan_path = 'bench/rhythm-v2/d49-action-state-contract-v1.json'
    report_path = 'reports/d49_action_state_contract.json'
    plan = json.loads((ROOT / plan_path).read_text())
    report = json.loads((ROOT / report_path).read_text())
    assert report['plan_sha256'] == sha(plan_path)
    for path, digest in plan['input_sha256'].items():
        assert sha(path) == digest, path
    sources = {}
    panel = json.loads((ROOT / plan['whole_song_panel']).read_text())
    for row in panel['rows']:
        events = json.loads((ROOT / row['canonical']).read_text())['events']
        for name, shift in [('native', 0), ('frozen_model', row['frozen_model_shift_us'])]:
            sources[row['chart_id'] + '/' + name] = (row['mode'], [
                dict(onset_us=e['onset_us'] + shift, end_us=e['end_us'] + shift,
                     type=e['type'], lane=e['location']['lane'] if row['mode'] == 'mania_4k' else -1)
                for e in events])
    cores = json.loads((ROOT / plan['fixed_core_labels']).read_text())
    for row in cores['rows']:
        identity = row['identity']
        label = identity['panel_id'] + ('/translated' if identity['translated'] else '/original')
        sources[label] = identity['mode'], row['target_events']
    assert len(report['rows']) == len(sources) == 40
    checks = 0
    verified = []
    for row in report['rows']:
        mode, expected = sources[row['label']]
        assert sha(row['trace']) == row['trace_sha256']
        trace = json.loads((ROOT / row['trace']).read_text())
        active, completed, previous = {}, [], None
        for step in trace['traces']:
            t = step['action']['time_us']
            assert previous is None or previous < t
            boundary_completed = []
            if mode == 'mania_4k':
                assert step['before']['lane_starts'] == [active.get(k) for k in range(4)]
                assert step['before']['hold_groups'] == []
                code = step['action']['code']
                assert 0 < code < 6 ** 4
                for k in range(4):
                    digit = code // 6 ** k % 6
                    assert digit in ((0, 3, 4, 5) if k in active else (0, 1, 2))
                    if digit in (3, 4, 5):
                        boundary_completed.append((active.pop(k), t, 'hold', k))
                    if digit in (1, 4):
                        boundary_completed.append((t, t, 'tap', k))
                    elif digit in (2, 5):
                        active[k] = t
                assert step['after']['lane_starts'] == [active.get(k) for k in range(4)]
            else:
                assert step['before']['lane_starts'] == [None] * 4
                assert step['before']['hold_groups'] == [list(p) for p in sorted(active.items())]
                for start, count in step['action']['releases']:
                    assert 0 < count <= active[start] and start < t
                    boundary_completed.extend((start, t, 'hold', -1) for _ in range(count))
                    active[start] -= count
                    if not active[start]:
                        del active[start]
                for kind, count in zip(('tap', 'hold', 'flick', 'drag'), step['action']['starts']):
                    assert type(count) is int and count >= 0
                    if kind == 'hold' and count:
                        active[t] = count
                    elif kind != 'hold':
                        boundary_completed.extend((t, t, kind, -1) for _ in range(count))
                assert step['after']['hold_groups'] == [list(p) for p in sorted(active.items())]
            assert step['before']['last_time_us'] == previous
            assert step['after']['last_time_us'] == t
            assert Counter(map(event_key, step['completed'])) == Counter(boundary_completed)
            completed.extend(boundary_completed)
            previous = t
            checks += 1
        assert not active
        assert Counter(completed) == Counter(map(event_key, expected)), row['label']
        assert row['actions'] == len(trace['traces']) and row['events'] == len(expected)
        verified.append(dict(label=row['label'], events=len(expected), boundaries=len(trace['traces'])))
    old = json.loads((ROOT / plan['fixed_bad_predictions']).read_text())
    negatives = {(x['view'], x['panel_id'], x['round']): x for x in report['negative_controls']}
    for view in ('original', 'translated'):
        for core in old['generation'][view]['cores']:
            if core['identity']['mode'] != 'mania_4k':
                continue
            for source in (core['rounds'][0], core['rounds'][-1]):
                row = negatives[view, core['identity']['panel_id'], source['round']]
                assert row['prior_lane_conflicts'] == source['mania_lane_conflicts']
                assert row['rejected'] == (source['mania_lane_conflicts'] > 0)
    assert report['complete'] and report['passed'] and not report['model_constructed']
    assert report['optimizer_updates'] == report['new_cloud_spend_usd'] == 0
    assert sha('beatfcn/model/p1.py') == 'c86c6fafa5b9c1fc911903d95c3bc22ed1c56b7c4c8f425ab969c4e35bf99364'
    result = dict(verified=True, verified_at_unix=time.time(),
        plan_sha256=sha(plan_path), result_sha256=sha(report_path),
        verifier_sha256=sha('scripts/verify_action_state_contract.py'),
        codec_imported=False, frozen_input_identities=len(plan['input_sha256']),
        independently_replayed_boundaries=checks, verified_views=verified,
        saved_negative_controls_verified=len(negatives),
        rejected_negative_controls=sum(x['rejected'] for x in negatives.values()),
        production_p1_sha256=sha('beatfcn/model/p1.py'))
    (ROOT / 'reports/d49_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'verified_views'}))


if __name__ == '__main__':
    verify()

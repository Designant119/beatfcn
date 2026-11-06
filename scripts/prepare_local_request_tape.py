"""Prepare complete-song event requests, without changing model or loss."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import build_model_view, TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events
from beatfcn.train.data import chart_from_canon, model_shift_us
from beatfcn.train.event_conditions import encode_controls, training_labels
from beatfcn.train.local_canonical_audit import rhythm_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requests_for_chart(events, mode, duration_us, split):
    events = sorted(events, key=lambda e: (e.onset_us, e.event_id))
    if any(e.onset_us < 0 or e.onset_us >= duration_us or e.end_us > duration_us for e in events):
        raise ValueError('Model-view event exceeds decoded sample clock; never clip it')
    heads = np.array([e.onset_us for e in events], np.int64)
    tokens, rows = [], []
    token_offset = 0
    for start in range(0, duration_us, 8_000_000):
        end = min(start+8_000_000, duration_us)
        lo, hi = map(int, np.searchsorted(heads, [start, end]))
        carry = tuple(RhythmEvent.from_event(e, mode) for e in events
                      if e.is_hold and e.onset_us < start < e.end_us)
        request = EventRequest(mode, start, end, duration_us, known_carry=carry)
        target = [RhythmEvent.from_event(e, mode) for e in events[lo:hi]]
        # encode_events independently replays and checks the exact multiset.
        encoded = encode_events(target, request)
        bands, actual = training_labels(mode, events, (start, end))
        controls = encode_controls(mode, bands if split == 'train' else {}, (start, end), origin_us=start)
        rows.append(dict(request=asdict(request), target_start=lo, target_stop=hi,
            token_start=token_offset, token_stop=token_offset+len(encoded),
            tokens=len(encoded), event_count=hi-lo,
            types=dict(Counter(e.type for e in events[lo:hi])), incoming_holds=len(carry),
            crossing_outgoing_holds=sum(e.is_hold and e.end_us > end for e in events[lo:hi]),
            controls=controls.tolist(), bands=bands if split == 'train' else None,
            condition_source='training target scope' if split == 'train' else 'unspecified; no hidden target attributes',
            reference='unknown_region',
            scope_role='training_or_teacher_forced_diagnostic; whole-song generation builds its own carries'))
        tokens.extend(encoded)
        token_offset += len(encoded)
    if sum(r['event_count'] for r in rows) != len(events):
        raise ValueError('Window partition lost event heads')
    return rows, np.asarray(tokens, np.int32), dict(
        onset_us=np.asarray([e.onset_us for e in events], np.int64),
        end_us=np.asarray([e.end_us for e in events], np.int64),
        kind=np.asarray([TYPES.index(e.type) for e in events], np.int8),
        lane=np.asarray([e.location.lane if mode == 'mania_4k' else -1 for e in events], np.int8))


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen request source changed: {name}')
    output = ROOT/plan['report']
    code = {p: sha(ROOT/p) for p in ['scripts/prepare_local_request_tape.py',
        'beatfcn/model/full_event_codec.py', 'beatfcn/events/canonical.py',
        'beatfcn/train/event_conditions.py', 'beatfcn/events/controls.py', 'beatfcn/train/data.py']}
    result = json.loads(output.read_text()) if output.exists() else dict(id=plan['id'],
        plan_sha256=sha(plan_path), code_sha256=code, charts=[], failures=[], attempts=[], complete=False,
        training_ready=False, optimizer_updates=0, cloud_compute_calls=0)
    if result['plan_sha256'] != sha(plan_path) or result['code_sha256'] != code or result['failures']:
        raise ValueError('Request identity changed or unresolved failure remains')
    if result['complete']:
        raise ValueError('Request tape already prepared')
    panel = json.loads((ROOT/plan['panel']).read_text())
    policy = json.loads((ROOT/plan['policy']).read_text())
    if not policy['all_prepared_panels_policy_pass']:
        raise ValueError('Latest Phira rating/vote gate has not passed')
    clock = {r['audio_sha256']: r for r in json.loads((ROOT/plan['clock']).read_text())['audio']}
    manifest = {r['chart_id']: r for r in map(json.loads, (ROOT/plan['manifest']).read_text().splitlines())}
    files = {r['path']: r for r in json.loads((ROOT/plan['canonical_receipt']).read_text())['files']}
    wanted = set(panel['tiers'][str(plan['tier'])]) | set(panel['validation'])
    charts = [r for r in panel['charts'] if r['component'] in wanted]
    seen = {r['chart_id'] for r in result['charts']}
    started = time.monotonic()
    attempt = dict(started_at_unix=time.time(), new_charts=0)
    result['attempts'].append(attempt)
    result['tier'] = plan['tier']
    result['feature_panel'] = plan['panel']
    result['train_components'] = panel['tiers'][str(plan['tier'])]
    result['validation_components'] = panel['validation']

    def save():
        result.update(updated_at_unix=time.time(), prepared_charts=len(result['charts']),
            expected_charts=len(charts), complete=len(result['charts']) == len(charts) and not result['failures'])
        result['summary'] = {}
        for split in ('train', 'val'):
            subset = [r for r in result['charts'] if r['split'] == split]
            requests = [q for r in subset for q in r['requests']]
            result['summary'][split] = dict(charts=len(subset), requests=len(requests),
                songs=len({r['component'] for r in subset}),
                modes=dict(Counter(r['mode'] for r in subset)),
                events=sum(q['event_count'] for q in requests), tokens=sum(q['tokens'] for q in requests),
                empty_requests=sum(q['event_count'] == 0 for q in requests),
                partial_end_requests=sum(q['request']['end_us']-q['request']['start_us'] < 8_000_000 for q in requests),
                incoming_hold_requests=sum(q['incoming_holds'] > 0 for q in requests),
                outgoing_hold_requests=sum(q['crossing_outgoing_holds'] > 0 for q in requests))
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        tmp.replace(output)

    save()
    for chart in charts:
        if chart['chart_id'] in seen:
            continue
        if time.monotonic()-started > plan['max_wall_s']:
            break
        try:
            row = manifest[chart['chart_id']]
            receipt = files[f"canon/{row['source']}/{row['native_id']}.json"]
            path = ROOT/receipt['local']
            if sha(path) != receipt['sha256']:
                raise ValueError('Canonical cache changed')
            source = chart_from_canon(json.loads(path.read_text()), model_shift_us(row, True))
            view = build_model_view(row['mode'], (), source.events)
            events = tuple(e for group in view.target_groups for e in group.events)
            if view.quarantine or len(events) != len(source.events) or rhythm_fingerprint(row['mode'], events) != chart['model_rhythm_sha256']:
                raise ValueError('Model target differs from source gate')
            duration = clock[chart['audio_sha256']]['decoded_samples']*1_000_000//24000
            requests, tokens, targets = requests_for_chart(events, row['mode'], duration, chart['split'])
            folder = ROOT/plan['cache_root']
            folder.mkdir(parents=True, exist_ok=True)
            stem = chart['chart_id'].replace(':', '-')
            target_path, token_path = folder/(stem+'-targets.npz'), folder/(stem+'-tokens.npy')
            if target_path.exists() or token_path.exists():
                raise FileExistsError('Unregistered request cache exists; inspect before replacing')
            np.savez(target_path, **targets)
            np.save(token_path, tokens, allow_pickle=False)
            result['charts'].append(dict(chart_id=chart['chart_id'], component=chart['component'],
                split=chart['split'], mode=chart['mode'], stratum=chart['stratum'],
                audio_sha256=chart['audio_sha256'], canonical_sha256=receipt['sha256'],
                model_rhythm_sha256=chart['model_rhythm_sha256'], duration_us=duration,
                target_path=str(target_path.relative_to(ROOT)), token_path=str(token_path.relative_to(ROOT)),
                target_sha256=sha(target_path), token_sha256=sha(token_path), requests=requests))
            attempt['new_charts'] += 1
            save()
        except Exception as exc:
            result['failures'].append(dict(chart_id=chart['chart_id'], error=f'{type(exc).__name__}: {exc}'))
            save()
            raise
    attempt.update(ended_at_unix=time.time(), wall_s=time.monotonic()-started)
    save()
    print(json.dumps(dict(complete=result['complete'], summary=result['summary'], wall_s=attempt['wall_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

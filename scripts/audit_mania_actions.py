"""D43: lossless action-label audit, with no model or data-admission changes."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

from beatfcn.events.canonical import build_model_view
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.mania_actions import ACTION_COUNT, action_size, action_states, encode_action, decode_action
from beatfcn.train.data import chart_from_canon, model_shift_us

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_sha(events):
    return hashlib.sha256(json.dumps([asdict(e) for e in sorted(events)],
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_events(events, mode):
    groups = defaultdict(list)
    for event in events:
        groups[event.onset_us].append(event)
    codes, restored, errors, examples = Counter(), [], [], []
    if mode == 'mania_4k':
        for onset, members in sorted(groups.items()):
            try:
                action = encode_action(members, onset_us=onset)
                decoded = decode_action(action, onset_us=onset)
                if Counter(decoded) != Counter(members):
                    raise ValueError('event multiset differs after action decoding')
                if encode_action(members[::-1], onset_us=onset) != action:
                    raise ValueError('member storage order changed binding')
                restored.extend(decoded)
                codes[action.code] += 1
                if action.code not in {x['code'] for x in examples}:
                    examples.append(dict(code=action.code, onset_us=onset,
                                         events=[asdict(e) for e in sorted(members)]))
            except (ValueError, TypeError) as exc:
                errors.append(dict(onset_us=onset, error=str(exc),
                                   events=[asdict(e) for e in sorted(members)]))
    elif mode == 'phira':
        # Preserve multiplicity exactly. The mania codec is never called here.
        restored = list(events)
    else:
        raise ValueError('unknown mode')
    overlapping = []
    if mode == 'mania_4k':
        for lane in range(4):
            prior = None
            for event in sorted(e for e in events if e.lane == lane):
                if prior is not None and event.onset_us < prior.end_us:
                    overlapping.append(dict(previous=asdict(prior), next=asdict(event)))
                if event.type == 'hold' and (prior is None or event.end_us > prior.end_us):
                    prior = event
    holds = [e for e in events if e.type == 'hold']
    return dict(events=len(events), groups=len(groups), types=dict(Counter(e.type for e in events)),
                event_sha256=event_sha(events), restored_sha256=event_sha(restored),
                action_roundtrip=(not errors and Counter(events) == Counter(restored)) if mode == 'mania_4k' else None,
                phira_passthrough_unchanged=(Counter(events) == Counter(restored)) if mode == 'phira' else None,
                action_codes=dict(sorted(codes.items())), group_sizes=dict(sorted(Counter(map(len, groups.values())).items())),
                errors=errors, code_examples=examples, overlapping_hold_starts=overlapping,
                holds_crossing_8s=sum(e.onset_us // 8_000_000 < (e.end_us - 1) // 8_000_000 for e in holds),
                holds_at_least_8s=sum(e.end_us - e.onset_us >= 8_000_000 for e in holds),
                non_millisecond_notes=sum(e.onset_us % 1000 != 0 or e.end_us % 1000 != 0 for e in events))


def support_summary(records):
    support = []
    for code in range(1, ACTION_COUNT):
        rows = [r for r in records if r['audit']['action_codes'].get(code, 0)]
        support.append(dict(code=code, states=list(action_states(code)), size=action_size(code),
                            groups=sum(r['audit']['action_codes'][code] for r in rows),
                            chart_ids=sorted({r['chart_id'] for r in rows}),
                            audio_sha256=sorted({r['audio_sha256'] for r in rows}),
                            song_groups=sorted({r['song_group'] for r in rows}),
                            records=sorted({r['record_id'] for r in rows})))
    seen = [r for r in support if r['groups']]
    return dict(nonempty_possible=80, nonempty_seen=len(seen),
                unseen_codes=[r['code'] for r in support if not r['groups']],
                seen_codes_with_at_most_2_groups=sum(r['groups'] <= 2 for r in seen),
                seen_codes_on_one_chart=sum(len(r['chart_ids']) == 1 for r in seen),
                seen_codes_on_one_song=sum(len(r['song_groups']) == 1 for r in seen),
                groups=sum(r['groups'] for r in support),
                group_sizes={str(n):sum(r['groups'] for r in support if r['size'] == n) for n in range(1, 5)},
                codes=support)


def main():
    started = time.perf_counter()
    plan_path = ROOT / 'bench/rhythm-v2/d43-mania-actions-v1.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'frozen input changed: {name}')
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve previous audit')
    sources = json.loads((ROOT / plan['source_plan']).read_text())
    old = json.loads((ROOT / plan['prior_capacity_audit']).read_text())
    old_charts = {r['chart_id']:r for r in old['charts']}
    full, source_targets, source_rows = [], {}, {}
    for row in sources['train_rows']:
        if row['split'] != 'train' or row['chart_id'] in source_rows:
            raise ValueError('duplicate chart or non-training split')
        path = ROOT / sources['train_canonical_root'] / row['source'] / f"{row['native_id']}.json"
        if sha(path) != sources['train_canonical_sha256'][row['chart_id']]:
            raise ValueError('canonical identity changed')
        chart = chart_from_canon(json.loads(path.read_text()), model_shift_us(row, True))
        view = build_model_view(row['mode'], (), chart.events)
        raw = [RhythmEvent.from_event(e, row['mode']) for e in chart.events]
        events = [RhythmEvent.from_event(e, row['mode']) for g in view.target_groups for e in g.events]
        audit, raw_audit = audit_events(events, row['mode']), audit_events(raw, row['mode'])
        previous = old_charts[row['chart_id']]
        if len(events) != previous['quantized_events'] or sha(path) != previous['source_sha256']:
            raise ValueError('does not reproduce prior complete-chart cohort')
        full.append(dict(record_id=row['chart_id'], chart_id=row['chart_id'], mode=row['mode'],
                         audio_sha256=row['audio_sha256'], song_group=row['song_group'], split=row['split'],
                         source_sha256=sha(path), source_path=str(path.relative_to(ROOT)),
                         model_shift_us=model_shift_us(row, True),
                         model_view_issues=[i.to_json() for i in view.quarantine],
                         raw_canonical_audit=raw_audit, audit=audit))
        source_targets[row['chart_id']], source_rows[row['chart_id']] = events, row
    evidence = json.loads((ROOT / plan['panel_evidence']).read_text())
    panels = {}
    for variant in ('original', 'translated'):
        cores = []
        for core in evidence['generation'][variant]['cores']:
            identity = core['identity']
            row = source_rows[identity['chart_id']]
            if identity['audio_sha256'] != row['audio_sha256'] or identity['mode'] != row['mode']:
                raise ValueError('panel identity differs from source cohort')
            events = [RhythmEvent(**e) for e in core['target_events']]
            start = identity['source_clip_start_us']
            absolute = [RhythmEvent(e.onset_us + start, e.end_us + start, e.type, e.lane) for e in events]
            lo, hi = identity['source_core_us'], identity['source_core_us'] + 8_000_000
            expected = [e for e in source_targets[identity['chart_id']] if lo <= e.onset_us < hi]
            if Counter(absolute) != Counter(expected):
                raise ValueError('panel target differs from exact original source interval')
            cores.append(dict(record_id=identity['panel_id'], chart_id=row['chart_id'], mode=row['mode'],
                              audio_sha256=row['audio_sha256'], song_group=row['song_group'],
                              identity=identity, source_interval_exact=True, audit=audit_events(events, row['mode'])))
        panels[variant] = dict(cores=cores, mania=support_summary([r for r in cores if r['mode'] == 'mania_4k']))
    summary = {}
    for mode in ('mania_4k', 'phira'):
        rows = [r for r in full if r['mode'] == mode]
        summary[mode] = dict(charts=len(rows), songs=len({r['song_group'] for r in rows}),
                             audio_files=len({r['audio_sha256'] for r in rows}),
                             events=sum(r['audit']['events'] for r in rows),
                             groups=sum(r['audit']['groups'] for r in rows),
                             types=dict(sum((Counter(r['audit']['types']) for r in rows), Counter())),
                             model_view_issue_charts=sum(bool(r['model_view_issues']) for r in rows),
                             encoding_errors=sum(len(r['audit']['errors']) + len(r['raw_canonical_audit']['errors']) for r in rows),
                             overlapping_hold_starts=sum(len(r['audit']['overlapping_hold_starts']) for r in rows),
                             holds_crossing_8s=sum(r['audit']['holds_crossing_8s'] for r in rows),
                             holds_at_least_8s=sum(r['audit']['holds_at_least_8s'] for r in rows))
    mania = [r for r in full if r['mode'] == 'mania_4k']
    valid = all(r['audit']['action_roundtrip'] and r['raw_canonical_audit']['action_roundtrip'] for r in mania)
    valid = valid and all(not r['model_view_issues'] for r in full)
    valid = valid and all(r['audit']['action_roundtrip'] for p in panels.values() for r in p['cores'] if r['mode'] == 'mania_4k')
    if summary['mania_4k']['charts'] != 26 or summary['phira']['charts'] != 18:
        raise ValueError('complete cohort changed')
    result = dict(id='D43', complete=True, representation_audit_passed=valid,
                  scope='44 existing train charts plus D32 overlapping train views; no quota or quality claim',
                  zero_means='No new start on this key; does not release a previous hold',
                  unknown_code=-1, inactive_group_code=0,
                  model_view_policy='Existing 1ms target quantization unchanged; raw canonical and model-view codec audited separately',
                  plan_sha256=sha(plan_path), input_sha256=plan['input_sha256'],
                  summary=summary, full_chart_support=support_summary(mania), charts=full, panels=panels,
                  optimizer_updates=0, model_forwards=0, model_promotion=False, new_paid_compute_usd=0,
                  wall_s=time.perf_counter()-started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('representation_audit_passed','summary','wall_s','peak_rss_gib')}, ensure_ascii=False))
    print(json.dumps({'full_seen_codes':result['full_chart_support']['nonempty_seen'],
                      'panels':{k:v['mania']['nonempty_seen'] for k,v in panels.items()}}))


if __name__ == '__main__':
    main()

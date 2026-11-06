"""Verify fixed group/cardinality targets on real windows and full train sources."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event, build_model_view
from beatfcn.model.event_canvas import ClockLayout, RhythmEvent
from beatfcn.model.group_canvas import encode_group_targets, decode_group_targets, masked_group_view
from beatfcn.train.data import chart_from_canon, model_shift_us


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT / 'bench/rhythm-v2/p2c-group-representation-v1.json')
    args = parser.parse_args()
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text())
    path = ROOT / plan['catalog']
    if sha(path) != plan['catalog_sha256']:
        raise ValueError('catalog identity changed')
    catalog = torch.load(path, map_location='cpu', weights_only=False)
    entries = {}
    for pair in catalog['train_pairs']:
        for entry in pair['entries']:
            if entry['row']['split'] != 'train':
                raise ValueError('non-training window')
            entries[(entry['row']['chart_id'], entry['example'].c0_us, entry['target_sha256'])] = entry
    samples = []
    for (chart_id, c0, target_sha), entry in entries.items():
        mode = entry['row']['mode']
        samples.append(dict(source='window', chart_id=chart_id, mode=mode, start_us=c0,
            end_us=c0 + 8_000_000, target_sha256=target_sha,
            copy_times=tuple(map(int, entry['example'].copy_times + c0)),
            events=[RhythmEvent.from_event(Event.from_json(e), mode) for e in entry['truth_events']]))
    source_plan_path = ROOT / plan['full_training_source_plan']
    source_plan = json.loads(source_plan_path.read_text())
    for row in source_plan['train_rows']:
        if row['split'] != 'train':
            raise ValueError('non-training full source')
        path = ROOT / source_plan['train_canonical_root'] / row['source'] / f'{row["native_id"]}.json'
        if sha(path) != source_plan['train_canonical_sha256'][row['chart_id']]:
            raise ValueError('full source identity changed')
        chart = chart_from_canon(json.loads(path.read_text()), model_shift_us(row, True))
        view = build_model_view(row['mode'], (), chart.events)
        if view.quarantine:
            raise ValueError('existing model view has quarantine; no target deletion allowed')
        samples.append(dict(source='full_chart', chart_id=row['chart_id'], mode=row['mode'],
            start_us=0, end_us=round(row['audio_sec'] * 1e6), target_sha256=sha(path), copy_times=(),
            events=[RhythmEvent.from_event(e, row['mode']) for g in view.target_groups for e in g.events]))
    rows, errors = [], []
    counts_histogram, members_histogram = defaultdict(Counter), defaultdict(Counter)
    for sample in samples:
        events = sample['events']
        for phase in plan['phases_us']:
            record = {k: v for k, v in sample.items() if k not in ('events', 'copy_times')}
            record['phase_us'] = phase
            try:
                layout = ClockLayout(sample['start_us'], sample['end_us'], plan['block_us'], phase)
                labels = encode_group_targets(events, layout, sample['mode'],
                    groups=plan['group_slots_per_block'], members=plan['notes_per_group'], copy_times=sample['copy_times'])
                restored = decode_group_targets(labels)
                if Counter(restored) != Counter(events):
                    raise AssertionError('group rhythm multiset changed')
                blank = masked_group_view(labels, np.zeros_like(labels.multiplicity, dtype=bool))
                if not ((blank.counts == -1).all() and (blank.multiplicity == -1).all() and (blank.heads == -1).all()):
                    raise AssertionError('unknown count or head leakage')
                group_count = len({e.onset_us for e in events})
                if labels.counts.sum() != group_count or labels.multiplicity.sum() != len(events):
                    raise AssertionError('cardinality or multiplicity changed')
                scope = sample['source'] + '/' + sample['mode']
                counts_histogram[scope].update(map(int, labels.counts))
                members_histogram[scope].update(map(int, labels.multiplicity[labels.multiplicity > 0]))
                record.update(passed=True, events=len(events), onset_groups=group_count,
                    max_groups_per_block=int(labels.counts.max()),
                    max_notes_per_group=int(labels.multiplicity.max()),
                    holds_crossing_8s=sum(e.type == 'hold' and e.onset_us // 8_000_000 < (e.end_us - 1) // 8_000_000 for e in events),
                    counts_shape=list(labels.counts.shape), group_shape=list(labels.heads.shape),
                    member_shape=list(labels.notes.shape), all_arrays_finite=True,
                    empty_blocks=int((labels.counts == 0).sum()),
                    phira_rhythm_multiplicity_excess=sum(n - 1 for n in Counter(events).values()) if sample['mode'] == 'phira' else 0)
            except (ValueError, AssertionError) as exc:
                record.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                errors.append(record)
            rows.append(record)
    summary = {}
    for scope in counts_histogram:
        selected = [r for r in rows if r['passed'] and f'{r["source"]}/{r["mode"]}' == scope]
        summary[scope] = dict(attempts=len(selected), event_instances=sum(r['events'] for r in selected),
            onset_groups=sum(r['onset_groups'] for r in selected),
            max_groups_per_block=max(r['max_groups_per_block'] for r in selected),
            max_notes_per_group=max(r['max_notes_per_group'] for r in selected),
            block_group_count_histogram=dict(sorted(counts_histogram[scope].items())),
            group_member_count_histogram=dict(sorted(members_histogram[scope].items())))
    report = dict(schema='beatfcn.group_canvas_probe.v1', complete=True,
        passed=not errors and len(rows) == 2 * (290 + 44),
        train_windows=len(entries), full_charts=len(source_plan['train_rows']), attempts=len(rows),
        summary=summary, errors=errors, rows=rows, wall_s=time.perf_counter() - started,
        model_updates=0, new_paid_compute_usd=0, model_promotion=False,
        plan_sha256=sha(args.plan), source_plan_sha256=sha(source_plan_path), catalog_sha256=plan['catalog_sha256'],
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'beatfcn/model/group_canvas.py']},
        limits=['Existing 44 training charts and paired windows only; expanded corpus capacity still unverified.',
                'This is target fidelity, not learned cardinality or held-out quality.',
                'Native protected object identity and exact COPY representation have separate contracts; product export not integrated.'])
    (ROOT / 'reports/p2c_group_representation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'train_windows', 'full_charts', 'attempts', 'summary', 'errors', 'wall_s')}))
    if not report['passed']:
        raise SystemExit('group representation gate failed')


if __name__ == '__main__':
    main()

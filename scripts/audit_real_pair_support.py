"""Local CPU census of legal, aligned, same-audio alternative chart windows.

This is a convenience cache, not training exposure or held-out evaluation.
Target-derived bands are training-label diagnostics only.
"""
from collections import Counter, defaultdict
import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time

from beatfcn.model import features as F
from beatfcn.model.batch import MAX_NOTES
from beatfcn.train.condition_pairs import blank_context_excerpt
from beatfcn.train.data import chart_from_canon, model_shift_us, selected_for_training, dup_only, check_stacks

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'inputs/training_audit_20261023/canon')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/real_pair_support_2026-10-25.json')
    args = parser.parse_args()
    manifest = ROOT / 'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    cache = args.cache
    selection = json.loads(args.selection.read_text()) if args.selection else None
    if selection and hashlib.sha256(manifest.read_bytes()).hexdigest() != selection['manifest_sha256']:
        raise ValueError('selection manifest identity changed')
    selected_ids = {r['chart_id'] for g in selection['groups'] for r in g['rows']} if selection else None
    heldout = {r['audio_sha256'] for r in json.loads(
        (ROOT / 'bench/capability-v1/manifest.json').read_text())['rows']}
    grouped = defaultdict(list)
    for row in map(json.loads, manifest.read_text().splitlines()):
        if selected_ids is not None and row['chart_id'] not in selected_ids:
            continue
        if not (row['split'] == 'train' and selected_for_training(row) and (row.get('kept') or dup_only(row))):
            continue
        if row['audio_sha256'] in heldout:
            raise ValueError('held-out audio in eligible training manifest')
        if (cache / row['source'] / f'{row["native_id"]}.json').exists():
            grouped[(row['mode'], row['audio_sha256'])].append(row)
    if selected_ids is not None and {r['chart_id'] for rows in grouped.values() for r in rows} != selected_ids:
        raise ValueError('selected training rows are missing or ineligible; not a complete audit')
    sources, records, rejected, chart_failures = {}, [], [], {}
    pairs_per_mode = Counter()
    for (mode, audio), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        charts = {}
        for row in rows:
            path = cache / row['source'] / f'{row["native_id"]}.json'
            raw = path.read_bytes(); doc = json.loads(raw)
            if any(doc[k] != row[k] for k in ('chart_id', 'audio_sha256', 'mode')):
                raise ValueError('cached canonical identity differs')
            sources[row['chart_id']] = hashlib.sha256(raw).hexdigest()
            try:
                chart = chart_from_canon(doc, model_shift_us(row, True))
                if dup_only(row):
                    check_stacks(chart.events)
                charts[row['chart_id']] = chart
            except (ValueError, AssertionError) as exc:
                chart_failures[row['chart_id']] = str(exc)
        for left, right in combinations(sorted(rows, key=lambda r: r['chart_id']), 2):
            ids = [left['chart_id'], right['chart_id']]
            pairs_per_mode[mode] += 1
            invalid = {cid: chart_failures[cid] for cid in ids if cid in chart_failures}
            if invalid:
                rejected.append(dict(mode=mode, chart_ids=ids, reason='source chart rejected by production loader',
                    scope='pair', chart_failures=invalid))
                continue
            durations = [round(row['audio_sec'] * F.SEC) for row in (left, right)]
            if durations[0] != durations[1]:
                rejected.append(dict(mode=mode, chart_ids=ids, reason='audio durations differ', scope='pair'))
                continue
            for c0 in range(0, durations[0] - F.CORE_US + 1, F.CORE_US):
                items = []
                try:
                    for row in (left, right):
                        chart = charts[row['chart_id']]
                        sample = blank_context_excerpt(chart, c0, durations[0], audio, 20261025)
                        ex = F.build_core(sample, chart.events, c0, durations[0], stage_a_arm='legacy')
                        errors = F.check_targets(ex)
                        if errors:
                            raise ValueError('illegal target: ' + ','.join(errors))
                        if any(len(g['notes']) > MAX_NOTES for g in ex.groups):
                            raise ValueError('group exceeds model note capacity')
                        rhythm = [(g['t_rel_us'], sorted((n['type'],
                            n['lane'] if mode == 'mania_4k' else -1, n.get('end_rel_us', 0))
                            for n in g['notes'])) for g in ex.groups]
                        items.append(dict(chart_id=row['chart_id'], groups=len(ex.groups),
                            notes=len(sample.hidden), types=dict(Counter(e.type for e in sample.hidden)),
                            rhythm_target_sha256=hashlib.sha256(json.dumps(rhythm).encode()).hexdigest(),
                            controls=sample.request['controls'], raw_controls=sample.raw_controls))
                    changed = [field for field in F.CONTROL_FIELDS
                        if items[0]['controls'].get(field) != items[1]['controls'].get(field)]
                    records.append(dict(mode=mode, audio_sha256=audio, chart_ids=ids,
                        c0_us=c0, duration_us=F.CORE_US, changed_fields=changed, items=items,
                        rhythm_targets_equal=items[0]['rhythm_target_sha256'] == items[1]['rhythm_target_sha256']))
                except (ValueError, AssertionError) as exc:
                    rejected.append(dict(mode=mode, chart_ids=ids, c0_us=c0, reason=str(exc), scope='window'))
    summary = {}
    for mode in ('mania_4k', 'phira'):
        cases = [r for r in records if r['mode'] == mode]
        contrastive = [r for r in cases if r['changed_fields']]
        summary[mode] = dict(cached_same_audio_songs=sum(m == mode and len(rs) > 1 for (m, _), rs in grouped.items()),
            chart_pairs=pairs_per_mode[mode], valid_pair_windows=len(cases),
            differing_band_windows=len(contrastive),
            distinct_audio_with_band_contrast=len({r['audio_sha256'] for r in contrastive}),
            single_field_only_windows=sum(len(r['changed_fields']) == 1 for r in contrastive),
            identical_rhythm_target_windows=sum(r['rhythm_targets_equal'] for r in cases),
            different_rhythm_same_bands_windows=sum(not r['rhythm_targets_equal'] and not r['changed_fields'] for r in cases),
            field_independent_audio_support={f: len({r['audio_sha256'] for r in contrastive if f in r['changed_fields']})
                for f in F.CONTROL_FIELDS if any(f in r['changed_fields'] for r in contrastive)},
            isolated_field_independent_audio_support={f: len({r['audio_sha256'] for r in contrastive if r['changed_fields'] == [f]})
                for f in F.CONTROL_FIELDS if any(r['changed_fields'] == [f] for r in contrastive)},
            changed_field_counts=dict(Counter(f for r in contrastive for f in r['changed_fields'])),
            isolated_field_counts=dict(Counter(r['changed_fields'][0] for r in contrastive if len(r['changed_fields']) == 1)))
    result = dict(schema='beatfcn.real_pair_support.v1',
        scope='Fixed hash-selected multiple-chart training audio groups; CPU label-support audit, not training exposure.'
            if selection else __doc__.strip(),
        selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest() if args.selection else None,
        cached_eligible_charts=sum(len(rows) for rows in grouped.values()),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), canonical_sha256=sources,
        summary=summary, records=records, rejected=rejected, chart_failures=chart_failures,
        rejection_counts=dict(Counter(r['reason'] for r in rejected)),
        existing_policy_changed=False, training_updates=0, new_paid_dispatches=0,
        product_promotion=False, wall_s=time.perf_counter()-started,
        limitations=['Fixed selected training audio groups; not full-corpus or test-set evidence.' if selection else
            'Convenience cache; not representative of the full corpus.',
            'Equal audio identity and model offsets align coordinates, but do not prove artistic equivalence.',
            'A single-field band contrast can still contain differences in other attributes within the same bands.',
            'Bands derive from training targets. They are not product evaluation requests.',
            'Boundary holds, invalid targets and capacity overflow reject a whole window; no note thinning.'])
    output = args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summary=summary, rejected=len(rejected), wall_s=result['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Audit cached training charts and current request policy, entirely on local CPU.

This is a deterministic reconstruction on a convenience cache, not the exact
32,000-window training tape. Never use these counts as actual training exposure.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from beatfcn.events.requests import sample_request, KINDS
from beatfcn.model import features as F
from beatfcn.model.batch import MAX_NOTES
from beatfcn.train.data import chart_from_canon, model_shift_us, selected_for_training, dup_only, check_stacks
from beatfcn.train.density import complete_window
from beatfcn.train.condition_audit import inspect_sample, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl')
    parser.add_argument('--cache', type=Path, default=ROOT / 'inputs/training_audit_20261023/canon')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/condition_coverage_2026-10-25.json')
    parser.add_argument('--seeds', type=int, default=2)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError('seeds must be positive')
    raw = args.manifest.read_bytes()
    rows = [json.loads(l) for l in raw.splitlines()]
    eligible = [r for r in rows if r['split'] == 'train' and selected_for_training(r)
                and (r.get('kept') or dup_only(r))]
    heldout = {r['audio_sha256'] for r in json.loads((ROOT / 'bench/capability-v1/manifest.json').read_text())['rows']}
    if heldout & {r['audio_sha256'] for r in eligible}:
        raise ValueError('development audio occurs in eligible training manifest')
    pairs = defaultdict(list)
    for r in eligible:
        pairs[(r['mode'], r['audio_sha256'])].append(r['chart_id'])
    pair_counts = {}
    for mode in ('mania_4k', 'phira'):
        groups = [v for (m, _), v in pairs.items() if m == mode]
        pair_counts[mode] = dict(songs=len(groups), songs_with_multiple_charts=sum(len(g) > 1 for g in groups),
                                chart_counts=dict(Counter(len(g) for g in groups)),
                                scope='manifest eligibility before uncached canonical/stack/target validation')
    records, failures, sources = [], [], []
    started = time.perf_counter()
    for row in sorted(eligible, key=lambda r: r['chart_id']):
        path = args.cache / row['source'] / (str(row['native_id']) + '.json')
        if not path.exists():
            continue
        payload = path.read_bytes()
        doc = json.loads(payload)
        source = dict(chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], mode=row['mode'],
                      canon_sha256=hashlib.sha256(payload).hexdigest())
        sources.append(source)
        try:
            if any(doc[k] != row[k] for k in ('chart_id', 'audio_sha256', 'mode')):
                raise ValueError('cached chart identity differs from manifest')
            chart = chart_from_canon(doc, model_shift_us(row, True))
            if dup_only(row):
                check_stacks(chart.events)
        except (ValueError, AssertionError) as exc:
            failures.append(dict(chart_id=row['chart_id'], stage='chart', reason=str(exc)))
            continue
        duration = int(round(row['audio_sec'] * F.SEC))
        for kind in KINDS:
            for i in range(args.seeds):
                seed = 20261025 + i
                context = dict(chart_id=row['chart_id'], song=row['audio_sha256'], seed=seed, kind=kind)
                try:
                    sample = sample_request(chart, seed=seed, audio_id=row['audio_sha256'], audio_duration_us=duration,
                                            mix={kind: 1.}, control_drop_all_p=.2)
                    if not sample.ok:
                        raise F.SkipExample('model view quarantine')
                    c0 = F.choose_core(sample, np.random.default_rng([seed, 1]), duration)
                except (ValueError, AssertionError) as exc:
                    failures.append(dict(**context, stage='request', reason=str(exc)))
                    continue
                for variant in ('current_request', 'density_window'):
                    try:
                        view = sample if variant == 'current_request' else complete_window(sample, chart, c0, duration, seed)
                        ex = F.build_core(view, chart.events, c0, duration, stage_a_arm='legacy')
                        bad = F.check_targets(ex)
                        if bad:
                            raise F.SkipExample('illegal_target:' + ','.join(bad))
                        if any(len(g['notes']) > MAX_NOTES for g in ex.groups):
                            raise F.SkipExample('too many notes in group')
                        record = inspect_sample(view, c0, duration)
                        record.update(chart_id=row['chart_id'], song=row['audio_sha256'], seed=seed, variant=variant,
                                      original_kind=kind)
                        records.append(record)
                    except (ValueError, AssertionError) as exc:
                        failures.append(dict(**context, variant=variant, stage='core', reason=str(exc)))
        if len(sources) % 40 == 0:
            print(json.dumps(dict(charts=len(sources), records=len(records), wall_s=round(time.perf_counter()-started, 2))), flush=True)
    result = dict(schema='beatfcn.condition_coverage.v1', scope='local cached training charts; reconstructed requests, NOT exact training tape',
                  manifest_sha256=hashlib.sha256(raw).hexdigest(), source_files=sources, seed_count=args.seeds,
                  eligible_rows=len(eligible), available_cached_charts=len(sources), pairing=pair_counts,
                  sampling='each cached chart x five request kinds x fixed seeds; unweighted, not original mixture or accepted exposure',
                  variants='current request and density-window reconstruction; no new training policy applied',
                  records=records, summary=summarize(records), failures=failures,
                  local_label_difference_is_error=False,
                  limitations=['request-wide conditions can legitimately differ from a local crop',
                               'cached charts are a convenience sample, not representative of all training',
                               'matching audio hash does not prove paired charts agree in timeline or other attributes'],
                  new_paid_dispatches=0, wall_s=time.perf_counter()-started)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), charts=len(sources), records=len(records), failures=len(failures),
                          wall_s=result['wall_s'])), flush=True)


if __name__ == '__main__':
    main()

"""Local-only preparation. No Modal import, optimizer, cloud dispatch, or GPU work.

Writes a versioned development benchmark, three arm configs, capacity audits and
geometry/drag fixtures. Existing v1 results and manifests are never rewritten.
"""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.phira import parse_phira
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.spatial import compile_rpe
from beatfcn.spatial.rpe import digest
from beatfcn.train.data import chart_from_canon, check_stacks, model_shift_us
from beatfcn.train.stage_a import ARMS, build_triplet, prepare_sample, protocol


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def benchmark_v2():
    old = json.loads((ROOT / 'bench/capability-v1/manifest.json').read_text())
    new = {k: v for k, v in old.items() if k != 'manifest_sha256'}
    new.update(schema='beatfcn.capability_benchmark.v2', parent_manifest_sha256=old['manifest_sha256'],
               purpose='development_only_previously_inspected_songs', status='prepared_no_model_scores',
               request_policy='positive_preserving_reference_v2', score_policy='event_scores_v2',
               permission_quantization='crossing_events_observed_whole_v1',
               stage_a_arms=list(ARMS), input_policy='authorized_rhythm_prefix_v1',
               requires_request_regeneration=True, sealed_test_used=False,
               comparability='v1 is historical; all new arms must regenerate the same v2 requests',
               spatial_source_policy='native source hashes must match canon; missing geometry reported, never fabricated')
    new['manifest_sha256'] = digest(new)
    return new


def capacity_audit(manifest):
    records = []
    for cid in (12870, 56053):
        row = next(r for r in manifest['rows'] if r['chart_id'] == f'phira:{cid}')
        doc = json.loads((ROOT / f'reports/phira_{cid}_canon_audit.json').read_text())
        shift = (int(row.get('offset_decision', {}).get('shift_ms', 0)) +
                 int(row.get('offset_decision', {}).get('c_ms', -2))) * 1000
        chart = chart_from_canon(doc, shift)
        for kind in ('middle_holes', 'reference'):
            sample = build_request(chart, kind=kind, seed=row['benchmark_seed'], audio_id=row['audio_sha256'],
                                   audio_duration_us=round(row['audio_sec'] * 1e6), control_dropout=False)
            sample, ctx = prepare_sample(sample)
            arms = {arm: {'cores': 0, 'target_events': 0, 'legacy_overflow_max': 0} for arm in ARMS}
            for c0 in range(0, sample.request['audio_duration_us'], F.CORE_US):
                for arm, ex in build_triplet(sample, chart.events, c0, ctx).items():
                    arms[arm]['cores'] += 1
                    arms[arm]['target_events'] += sum(len(g['notes']) for g in ex.groups)
                    arms[arm]['legacy_overflow_max'] = max(arms[arm]['legacy_overflow_max'], ex.provenance['legacy_line_overflow'])
            records.append({'chart_id': row['chart_id'], 'kind': kind, 'shift_us': shift,
                            'positive_events_preserved': sum(len(r['selected']) for r in sample.evidence),
                            'reference_budget': sample.recipe['reference_budget'], 'arms': arms,
                            'geometry_status': 'unavailable_in_local_canon; explicitly masked'})
    return records


def native_fixtures():
    base = ROOT / 'inputs/phira_spatial_20261024'
    records, drag_cases = [], []
    for cid in (56172, 64140, 9761, 44233, 52792):
        raw = (base / f'{cid}.chart.json').read_bytes()
        info = json.loads((base / f'{cid}.info.json').read_text())
        chart = parse_phira(raw, info_offset_sec=info.get('offset', 0))
        # Existing corpus rule: intentional small duplicate stacks are retained.
        check_stacks(chart.events)
        chart = replace(chart, issues=tuple(i for i in chart.issues if i.code != 'duplicate_object'))
        scene = compile_rpe(raw, info_offset_s=info.get('offset', 0))
        duration = max(e.end_us for e in chart.events) + 2_000_000
        for kind in ('from_scratch', 'middle_holes', 'reference'):
            sample = build_request(chart, kind=kind, seed=20261024, audio_id=f'geometry-fixture:{cid}',
                                   audio_duration_us=duration, control_dropout=False)
            sample, ctx = prepare_sample(sample, scene, memory_overflow=protocol()['common']['memory_overflow'],
                                         reference_sources={r['reference_id']: scene for r in sample.evidence})
            c0 = F.choose_core(sample, np.random.default_rng(20261024), duration)
            triplet = build_triplet(sample, chart.events, c0, ctx)
            records.append({'chart_id': cid, 'purpose': 'engineering_fixture_not_quality_cohort', 'kind': kind,
                            'source_sha256': scene.source_sha256, 'compiler_cache_key': scene.cache_key,
                            'request_sha256': digest(sample.request), 'c0_us': c0, 'projection': ctx.audit,
                            'arms': {arm: {'known': len(ex.known), 'refs': len(ex.refs), 'groups': len(ex.groups),
                                           'rhythm_targets_sha256': ex.provenance['stage_a_common']['rhythm_targets_sha256']}
                                     for arm, ex in triplet.items()}})
        # Candidate windows for review, not inferred musical/gesture annotations.
        onsets = sorted({e.onset_us for e in chart.events if e.type == 'drag'})
        clusters, current = [], []
        for t in onsets:
            if current and t - current[-1] > 160_000:
                clusters.append(current); current = []
            current.append(t)
        if current:
            clusters.append(current)
        candidates = [g for g in clusters if len(g) >= 3]
        if candidates:
            g = max(candidates, key=len)
            drag_cases.append({'case_id': f'{cid}:drag:{g[0]}', 'chart_id': cid,
                               'source_sha256': scene.source_sha256, 'span_us': [g[0], g[-1] + 1],
                               'event_ids': [e.event_id for e in chart.events if e.type == 'drag' and g[0] <= e.onset_us <= g[-1]],
                               'distinct_onsets': len(g), 'semantic_status': 'temporal_candidate',
                               'selection': 'largest 160ms-gap cluster per engineering fixture; not a gesture label',
                               'human_review_needed': ['music relation', 'operation continuity', 'visual expression']})
    return records, drag_cases


def development_preflight(manifest, source_root):
    """Rebuild all Phira development requests, with no audio encoder or model execution."""
    records, sources = [], {}
    for row in manifest['rows']:
        if row['mode'] != 'phira':
            continue
        cid = str(row['native_id'])
        meta_path = source_root / f'{cid}.source.json'
        if not meta_path.exists():
            records.append({'chart_id': row['chart_id'], 'status': 'source_missing'})
            continue
        meta = json.loads(meta_path.read_text())
        doc = json.loads((source_root / f'{cid}.canon.json').read_text())
        raw = (source_root / f'{cid}.chart.json').read_bytes()
        scene = compile_rpe(raw, info_offset_s=meta['info_offset_s'])
        if scene.source_sha256 != doc['source_sha256'] or meta['audio_sha256'] != row['audio_sha256']:
            raise ValueError('development source identity mismatch')
        sources[row['chart_id']] = meta
        chart = chart_from_canon(doc, model_shift_us(row, True))
        for kind in ('from_scratch', 'middle_holes', 'reference', 'retained_scene_holes'):
            record = {'chart_id': row['chart_id'], 'kind': kind, 'status': 'pending'}
            try:
                sample = build_request(chart, kind='middle_holes' if kind == 'retained_scene_holes' else kind,
                                       seed=row['benchmark_seed'], audio_id=row['audio_sha256'],
                                       audio_duration_us=round(row['audio_sec'] * 1e6), control_dropout=False)
                grants = []
                if kind == 'retained_scene_holes':
                    # Explicit simulated editor permission, derived only from O line IDs.
                    # This is reported separately from normal no-scene requests.
                    lines = sorted({e['location']['line_id'] for e in sample.request['observed_events']})
                    grants = [{'kind': 'retained_line_scene', 'source_sha256': scene.source_sha256,
                               'line_ids': lines, 'span_us': [0, sample.request['audio_duration_us']]}]
                sample, ctx = prepare_sample(sample, scene, shift_us=model_shift_us(row, True), grants=grants,
                                             reference_sources={r['reference_id']: scene for r in sample.evidence})
                totals = {arm: {'cores': 0, 'target_events': 0} for arm in ARMS}
                for c0 in range(0, sample.request['audio_duration_us'], F.CORE_US):
                    if not any(r['state'] == 'editable' and r['start_us'] < c0 + F.CORE_US and c0 < r['end_us']
                               for r in sample.request['regions']):
                        continue
                    for arm, ex in build_triplet(sample, chart.events, c0, ctx).items():
                        totals[arm]['cores'] += 1
                        totals[arm]['target_events'] += sum(len(g['notes']) for g in ex.groups)
                expected = sum(len(g.events) for g in sample.model_view.target_groups)
                if any(t['target_events'] != expected for t in totals.values()):
                    raise AssertionError('development target multiplicity changed')
                record.update(status='ready', source_sha256=scene.source_sha256,
                              request_sha256=digest(sample.request), evidence_sha256=digest(sample.evidence),
                              reference_budget=sample.recipe['reference_budget'], projection=ctx.audit,
                              target_events=expected, arms=totals,
                              permission_promotions=sample.recipe.get('permission_provenance', []),
                              scene_permission='simulated_editor_retains_observed_lines' if grants else 'observed_event_spans_only')
            except (ValueError, AssertionError) as exc:
                record.update(status='failed', error=f'{type(exc).__name__}:{exc}')
            records.append(record)
    return {'expected_requests': 80, 'ready_requests': sum(r['status'] == 'ready' for r in records),
            'records': records, 'sources': sources, 'model_inference': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'bench/phira-stage-a-v1')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'inputs/phira_stage_a_development')
    args = parser.parse_args()
    plan = protocol()
    source_files = ['beatfcn/events/requests.py', 'beatfcn/events/reference_policy.py', 'beatfcn/train/data.py',
                    'beatfcn/train/stage_a.py', 'beatfcn/model/features.py', 'beatfcn/model/batch.py',
                    'beatfcn/model/p1.py', 'beatfcn/model/sample.py', 'beatfcn/model/space.py',
                    'beatfcn/model/evaluation.py', 'beatfcn/model/benchmark.py', 'beatfcn/model/drag_benchmark.py',
                    'beatfcn/model/input_diagnostics.py', 'beatfcn/spatial/rpe.py', 'beatfcn/spatial/context.py',
                    'scripts/prepare_phira_stage_a.py']
    plan['source_fingerprints'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    plan['source_bundle_sha256'] = digest(plan['source_fingerprints'])
    manifest = benchmark_v2()
    capacities = capacity_audit(manifest)
    native, drag = native_fixtures()
    development = development_preflight(manifest, args.source_root)
    manifest['spatial_sources'] = development['sources']
    manifest['manifest_sha256'] = digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    report = {'schema': 'phira_stage_a_preparation_v1', 'training_updates': 0, 'cloud_jobs_started': 0,
              'paid_compute_jobs_started': 0, 'campaign_cap_usd': 10, 'training_remains_paused': True,
              'source_bundle_sha256': plan['source_bundle_sha256'],
              'capacity_audit': capacities, 'native_fixtures': native, 'drag_cases': drag,
              'development_preflight': development,
              'quality_result': 'not_measured',
              'remaining_before_paid_execution': ['user resumes training', 'official bill reconciliation and per-arm reservation',
                                                  'native geometry availability audit for training corpus and frozen training-core schedule',
                                                  'mapper calibration of drag cases and provisional quality margins']}
    dump(args.output / 'protocol.json', plan)
    for arm in ARMS:
        dump(args.output / f'{arm}.json', {**plan, 'selected_arm': arm, 'model_variant': {'phira_context': arm,
             'selection': False, 'lane_history_groups': 0, 'offset_sigma_ms': 0.0, 'phira_location': False}})
    dump(args.output / 'benchmark-v2.json', manifest)
    retained = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    retained['scene_condition'] = 'retain_observed_lines_full_audio'
    retained['rows'] = [r for r in manifest['rows'] if r['mode'] == 'phira']
    retained['rows_sha256'] = digest(retained['rows'])
    retained['request_kinds'] = ['middle_holes']
    retained['requests'] = [r for r in manifest['requests'] if r['mode'] == 'phira' and r['kind'] == 'middle_holes']
    retained['manifest_sha256'] = digest(retained)
    dump(args.output / 'retained-scene-benchmark.json', retained)
    dump(args.output / 'drag-cases.json', drag)
    dump(args.output / 'preflight.json', report)
    print(json.dumps({'output': str(args.output), 'capacity_cases': len(capacities), 'native_requests': len(native),
                      'drag_candidate_cases': len(drag), 'development_ready': development['ready_requests'],
                      'training_updates': 0, 'cloud_jobs_started': 0}))


if __name__ == '__main__':
    main()

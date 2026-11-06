"""Read-only D98 audit of actual conditional contrasts in the frozen exposure.

This audits training opportunities, not causality or generated controllability.
No model is loaded and no timestamp, label, split, or sampler is changed.
"""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.train.event_conditions import CONTROL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d98-condition-exposure-audit-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def identity(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def flatten(tape):
    return [dict(q, chart_index=i, chart_id=c['chart_id'],
                 audio_sha256=c['audio_sha256'], component=c['component'],
                 split=c['split'], mode=c['mode'], stratum=c['stratum'])
            for i, c in enumerate(tape['charts']) for q in c['requests']]


def aggregate_pairs(pairs):
    result = {}
    flags = ('available', 'both_targets_seen', 'both_field_specified',
             'both_single_field', 'same_view_only_this_field_differs',
             'same_update_both_field_specified', 'same_update_single_field')
    for mode in ('phira', 'mania_4k'):
        result[mode] = {}
        for field in CONTROL_FIELDS:
            rows = [r for r in pairs if r['mode'] == mode and r['field'] == field]
            if not rows:
                continue
            result[mode][field] = {
                flag: dict(pair_windows=sum(r[flag] for r in rows),
                           independent_songs=len({r['component'] for r in rows if r[flag]}))
                for flag in flags}
            result[mode][field]['known_package_clock_identical'] = sum(
                r['known_package_clock_identical'] is True for r in rows)
    return result


def run():
    started = time.monotonic()
    plan = read(PLAN)
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen audit source changed: ' + path)
    tape, exposure = read(plan['tape']), read(plan['exposure'])
    if sha(plan['tape']) != exposure['request_tape_sha256']:
        raise ValueError('Exposure tape identity changed')
    records = flatten(tape)
    old_exposure = read(exposure['shared_prefix_exposure'])
    old_records = flatten(read(old_exposure['request_tape']))
    shared = exposure['shared_prefix_updates']
    if exposure['steps'][:shared] != old_exposure['steps'][:shared]:
        raise ValueError('Actual common parent exposure differs')
    for step in exposure['steps'][:shared]:
        for index in step['indices']:
            if records[index] != old_records[index]:
                raise ValueError('Common parent index changed meaning')

    visits = defaultdict(list)
    views = defaultdict(Counter)
    support = defaultdict(lambda: defaultdict(list))
    token_count = 0
    for step in exposure['steps']:
        for index, specified in zip(step['indices'], step['specified_condition_fields'], strict=True):
            r = records[index]
            if r['split'] != 'train':
                raise ValueError('Exposure contains a non-training song')
            fields = tuple(sorted(specified))
            if any(r['controls'][k][2] != 1 or CONTROL_FIELDS[k] not in (r['bands'] or {}) for k in fields):
                raise ValueError('Exposure specifies an undefined label')
            visits[index].append((step['effective_update'], fields))
            token_count += r['tokens']
            available = sum(c[2] == 1 for c in r['controls'])
            category = ('unknown' if not fields else 'single' if len(fields) == 1
                        else 'all_defined' if len(fields) == available else 'subset')
            views[r['mode']][category] += 1
            for k in fields:
                support[r['mode']][CONTROL_FIELDS[k]].append(dict(
                    update=step['effective_update'], index=index, component=r['component'],
                    band=r['bands'][CONTROL_FIELDS[k]], single=len(fields) == 1))
    if token_count != exposure['exposure']['valid_tokens']:
        raise ValueError('Actual token exposure changed')

    groups = defaultdict(list)
    loose_groups = defaultdict(list)
    for i, r in enumerate(records):
        if r['split'] != 'train':
            continue
        if r['reference'] != 'unknown_region':
            raise ValueError('Unexpected visible reference requires a new identity contract')
        non_value_controls = [c[3:] for c in r['controls']]
        # Includes carries, allowed types, clock/duration and generation limits.
        key = identity([r['mode'], r['audio_sha256'], r['request'],
                        r['reference'], non_value_controls])
        groups[key].append(i)
        loose_groups[identity([r['mode'], r['audio_sha256'], r['request']['start_us'],
                               r['request']['end_us']])].append(i)

    native = {r['chart_id']: r for p in plan['native_reports'] for r in read(p)['records']}
    sources = dict(plan['source_sha256'])
    chart_events = {}

    def events(index, shift=0):
        r = records[index]
        chart = tape['charts'][r['chart_index']]
        cid = r['chart_id']
        if cid not in chart_events:
            if sha(chart['target_path']) != chart['target_sha256']:
                raise ValueError('Target file changed')
            sources[chart['target_path']] = chart['target_sha256']
            with np.load(ROOT / chart['target_path'], allow_pickle=False) as data:
                chart_events[cid] = list(zip(*(data[k].tolist() for k in
                                               ('onset_us', 'end_us', 'kind', 'lane'))))
        lo, hi = r['request']['start_us'], r['request']['end_us']
        return Counter((h + shift, t + shift, k, lane) for h, t, k, lane in chart_events[cid]
                       if lo <= h + shift < hi)

    pairs = []
    identical_target_different_band = []
    paired_indices = set()
    for ids in groups.values():
        for a, b in combinations(ids, 2):
            ra, rb = records[a], records[b]
            ba, bb = ra['bands'] or {}, rb['bands'] or {}
            changed = [k for k, name in enumerate(CONTROL_FIELDS)
                       if name in ba and name in bb and ba[name] != bb[name]]
            if not changed:
                continue
            if ra['component'] != rb['component']:
                raise ValueError('Identical audio has inconsistent song components')
            if events(a) == events(b):
                identical_target_different_band.append([a, b])
            clock_identical = None
            offsets = None
            if ra['mode'] == 'phira':
                na, nb = native.get(ra['chart_id']), native.get(rb['chart_id'])
                if na and nb and na['source_identity_passed'] and nb['source_identity_passed']:
                    offsets = [round(float(n.get('package_offset') or 0) * 1e6) for n in (na, nb)]
                    clock_identical = events(a) == events(b, offsets[0] - offsets[1])
            va, vb = visits[a], visits[b]
            for k in changed:
                field = CONTROL_FIELDS[k]
                fa, fb = [(u, f) for u, f in va if k in f], [(u, f) for u, f in vb if k in f]
                sa, sb = [u for u, f in fa if f == (k,)], [u for u, f in fb if f == (k,)]
                same_views = {f for _, f in fa} & {f for _, f in fb}
                isolated = [f for f in same_views if all(
                    ba[CONTROL_FIELDS[j]] == bb[CONTROL_FIELDS[j]] for j in f if j != k)]
                pair = dict(mode=ra['mode'], component=ra['component'],
                    field=field, indices=[a, b], chart_ids=[ra['chart_id'], rb['chart_id']],
                    scope_us=[ra['request']['start_us'], ra['request']['end_us']],
                    strata=[ra['stratum'], rb['stratum']], bands=[ba[field], bb[field]],
                    available=True, both_targets_seen=bool(va and vb),
                    both_field_specified=bool(fa and fb), both_single_field=bool(sa and sb),
                    same_view_only_this_field_differs=bool(isolated),
                    same_update_both_field_specified=bool({u for u, _ in fa} & {u for u, _ in fb}),
                    same_update_single_field=bool(set(sa) & set(sb)),
                    field_visits=[fa, fb], known_package_clock_identical=clock_identical,
                    diagnostic_package_offsets_us=offsets)
                pairs.append(pair)
                if fa and fb:
                    paired_indices.update((a, b))
    if identical_target_different_band:
        raise ValueError('Identical targets have different frozen labels: ' + repr(identical_target_different_band[:5]))
    summaries = {}
    for mode, by_field in support.items():
        summaries[mode] = {}
        for field, rows in by_field.items():
            summaries[mode][field] = dict(requests=len(rows),
                independent_songs=len({r['component'] for r in rows}),
                distinct_windows=len({r['index'] for r in rows}),
                single_field_requests=sum(r['single'] for r in rows),
                bands=dict(Counter(identity(r['band']) for r in rows)),
                requests_by_quarter=dict(Counter(str((r['update'] - 1) // 250 + 1) for r in rows)))
    result = dict(id='D98-condition-pair-exposure-audit-v1', complete=True,
        plan_sha256=sha(PLAN), source_sha256=sources, shared_parent_requests_verified=shared * 4,
        actual_requests=sum(len(v) for v in visits.values()), valid_tokens=token_count,
        distinct_exposed_requests=sum(bool(v) for v in visits.values()),
        exposed_requests_with_bilateral_field_support=len(paired_indices),
        mode_views={k: dict(v) for k, v in views.items()}, field_exposure=summaries,
        potential_same_audio_scope_chart_pairs=sum(len(v) * (len(v) - 1) // 2 for v in loose_groups.values()),
        strict_same_non_condition_input_chart_pairs=sum(len(v) * (len(v) - 1) // 2 for v in groups.values()),
        pair_support=aggregate_pairs(pairs), pairs=pairs,
        optimizer_updates=0, model_forward_calls=0, new_cloud_compute_calls=0,
        elapsed_s=time.monotonic() - started, training_enabled=False,
        limitations=[
            'Pair opportunity is not required for supervised conditional learning and does not prove a cause.',
            'Pairs compare identical initial inputs; teacher-forced prefixes differ after their first target divergence.',
            'Same-field contrast does not hold every unspecified target property fixed.',
            'Same effective batch is diagnostic, not a necessary condition for gradient learning.',
            'Known package offsets are normalized only for duplicate diagnosis; labels are unchanged.',
            'Reference is unknown in every audited request; this is not human-style training support.',
            'No held-out target was used to construct a product condition or training pair.'])
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k in (
        'complete', 'actual_requests', 'valid_tokens', 'mode_views', 'pair_support', 'elapsed_s')}, ensure_ascii=False))


if __name__ == '__main__':
    run()

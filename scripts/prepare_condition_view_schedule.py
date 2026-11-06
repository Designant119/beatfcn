"""CPU-only feasibility of paired condition views at exactly the old exposure.

The output is not a training authorization. It permutes masks only within each
mode, available-field set and 250-update block; common-parent steps are pinned.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
import hashlib
import json
from pathlib import Path

from beatfcn.train.event_conditions import CONTROL_FIELDS
from scripts.audit_condition_pair_exposure import flatten, identity, read, sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d98-view-schedule-preparation-v1.json'


def run():
    plan = read(PLAN)
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Preparation source changed: ' + path)
    paths = [ROOT / plan[k] for k in ('output_exposure', 'output_report')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preparation output already exists')
    old, audit = read(plan['exposure']), read(plan['audit'])
    rows = flatten(read(old['request_tape']))
    locations, positions, groups, original = [], defaultdict(list), defaultdict(list), []
    for step in old['steps']:
        for j, (index, fields) in enumerate(zip(step['indices'], step['specified_condition_fields'], strict=True)):
            r = rows[index]
            available = tuple(k for k, c in enumerate(r['controls']) if c[2] == 1)
            group = (r['mode'], available, (step['effective_update'] - 1) // 250)
            pos = len(locations)
            locations.append(dict(step=step['effective_update'], column=j, index=index, group=group))
            positions[index].append(pos)
            groups[group].append(pos)
            original.append(tuple(fields))
    current = original.copy()
    pinned = {p for p, loc in enumerate(locations) if loc['step'] <= old['shared_prefix_updates']}
    candidates = defaultdict(list)
    for pair in audit['pairs']:
        if pair['both_targets_seen'] and pair['known_package_clock_identical'] is not True:
            candidates[(pair['mode'], pair['field'])].append(pair)
    for key, pairs in candidates.items():
        pairs.sort(key=lambda r: hashlib.sha256(identity(r['indices']).encode()).hexdigest())
    order = sorted(candidates, key=lambda key: (key[0], len({r['component'] for r in candidates[key]}), key[1]))
    covered = defaultdict(set)
    selected = []

    def trial(pair, k):
        for a, b in product(*(positions[i] for i in pair['indices'])):
            if a == b or a in pinned or b in pinned:
                continue
            proposed, swaps = current.copy(), []
            success = True
            for endpoint in (a, b):
                if proposed[endpoint] == (k,):
                    continue
                donors = [p for p in groups[locations[endpoint]['group']]
                          if p not in pinned and p not in (a, b) and proposed[p] == (k,)]
                if not donors:
                    success = False
                    break
                donor = donors[0]
                proposed[endpoint], proposed[donor] = proposed[donor], proposed[endpoint]
                swaps.append([endpoint, donor])
            if success:
                return proposed, (a, b), swaps
        return None

    while True:
        added = 0
        for key in order:
            k = CONTROL_FIELDS.index(key[1])
            for pair in candidates[key]:
                if pair['component'] in covered[key]:
                    continue
                result = trial(pair, k)
                if result is None:
                    continue
                current, endpoints, swaps = result
                pinned.update(endpoints)
                covered[key].add(pair['component'])
                selected.append(dict(mode=key[0], field=key[1], component=pair['component'],
                    indices=pair['indices'], chart_ids=pair['chart_ids'], bands=pair['bands'],
                    scope_us=pair['scope_us'], positions=list(endpoints), swaps=swaps))
                added += 1
                break
        if not added:
            break

    candidate = deepcopy(old)
    for pos, loc in enumerate(locations):
        candidate['steps'][loc['step'] - 1]['specified_condition_fields'][loc['column']] = list(current[pos])
    if candidate['steps'][:old['shared_prefix_updates']] != old['steps'][:old['shared_prefix_updates']]:
        raise ValueError('Parent exposure changed')
    for group, ids in groups.items():
        if Counter(original[i] for i in ids) != Counter(current[i] for i in ids):
            raise ValueError('Condition view histogram changed')
    for p, fields in enumerate(current):
        if not set(fields) <= set(locations[p]['group'][1]):
            raise ValueError('Undefined condition became specified')
    for pair in selected:
        k = CONTROL_FIELDS.index(pair['field'])
        if any(current[p] != (k,) for p in pair['positions']):
            raise ValueError('A pinned contrast was overwritten')

    # Independently enumerate coverage from the final schedule, not from pins.
    final_single = defaultdict(set)
    for p, fields in enumerate(current):
        if len(fields) == 1:
            final_single[locations[p]['index']].add(CONTROL_FIELDS[fields[0]])
    realized = defaultdict(set)
    for pair in audit['pairs']:
        if pair['known_package_clock_identical'] is True:
            continue
        a, b = pair['indices']
        if pair['field'] in final_single[a] & final_single[b]:
            realized[(pair['mode'], pair['field'])].add(pair['component'])
    candidate.update(id='D98-condition-views-only-prepared-v1',
        source_exposure=plan['exposure'], source_exposure_sha256=sha(plan['exposure']),
        preparation_plan_sha256=sha(PLAN), training_started=False, training_ready=False,
        feature_admission_pending=False,
        view_assignment='Deterministic contrast coverage; not IID sampling from historical probabilities.',
        meaning='Prepared feasibility tape only. No new quality or probe dispatch is authorized by this file.')
    counts = {mode: {field: len(realized[(mode, field)]) for m, field in candidates if m == mode}
              for mode in ('phira', 'mania_4k')}
    report = dict(id='D98-view-schedule-feasibility-v1', complete=True,
        training_ready=False, selected_contrasts=selected,
        contrast_songs_by_field=counts,
        changed_view_requests=sum(a != b for a, b in zip(original, current, strict=True)),
        actual_requests=len(locations), valid_tokens=old['exposure']['valid_tokens'],
        request_indices_and_order_unchanged=all(a['indices'] == b['indices'] for a, b in zip(old['steps'], candidate['steps'], strict=True)),
        common_parent_unchanged=True,
        exact_mask_histogram_preserved_per_mode_availability_quarter=True,
        every_applicable_field_at_least_eight_songs=all(n >= 8 for d in counts.values() for n in d.values()),
        source_sha256={**plan['source_sha256'], PLAN: sha(PLAN)},
        optimizer_updates=0, model_forward_calls=0, new_cloud_compute_calls=0,
        limitations=['This is feasible supervision exposure, not evidence that the unchanged decoder can learn it.',
            'Historical original-CE real paired fitting already failed; do not rerun that probe or auto-dispatch this tape.',
            'Condition-band frequencies and their association with targets can change; total view/field frequencies do not.',
            'No optimization-optimality claim is made for the deterministic greedy assignment.',
            'Package-offset-only duplicate comparisons are excluded, not charts; no labels or audio clocks change.'])
    for path, value in zip(paths, (candidate, report), strict=True):
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('selected_contrasts', 'source_sha256', 'limitations')}, ensure_ascii=False))


if __name__ == '__main__':
    run()

"""D40 same-target retrospective; compare raw predictions, never unlike losses."""
import json
from pathlib import Path

from scripts.acoustic_probe_metrics import event_metrics
from beatfcn.model.lane_draft import make_lane_draft, maximum_overlap
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT / 'bench/rhythm-v2/d40-branch-review-v1.json'
    plan = json.loads(plan_path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve previous review')
    docs = {name:json.loads((ROOT / path).read_text()) for name, path in plan['reports'].items()}
    reference = docs['R13']['generation']['original']['cores']
    baseline_features = docs['R13']['input_sha256']
    required = {key:value for key, value in baseline_features.items() if
        key.startswith('inputs/r09_positive_pairs/taps/') or
        key in ('inputs/d32_acoustic_panel/manifest.json', 'inputs/r08_local_real_probe/pca/i8_9-16-24.npz')}
    if len(required) < 17:
        raise ValueError('incomplete baseline feature identities')
    results = {}
    for name, doc in docs.items():
        if not doc['complete'] or doc.get('error') or doc['optimizer_updates'] != 100:
            raise ValueError('not a complete100 report')
        identities = json.loads((ROOT / plan['training_plans'][name]).read_text())['input_sha256']
        if any(identities.get(key) != value for key, value in required.items()):
            raise ValueError('real acoustic feature identity mismatch: ' + name)
        original = doc['generation']['original']
        if len(original['cores']) != len(reference):
            raise ValueError('core count mismatch')
        cases = []
        for core, old in zip(original['cores'], reference):
            if core['identity'] != old['identity'] or core['target_events'] != old['target_events']:
                raise ValueError('raw target or clock differs: ' + name)
            mode, truth = core['identity']['mode'], core['target_events']
            rounds = []
            for saved in core['rounds']:
                pred = saved['events']
                metric = event_metrics(pred, truth, mode)
                for key in ('pred', 'true', 'matched', 'full_note_f1', 'mania_lane_conflicts'):
                    if abs(metric[key] - saved[key]) > 1e-12:
                        raise ValueError('saved raw-event metric mismatch: ' + name + '/' + key)
                no_lane = event_metrics([dict(e, lane=-1) for e in pred], [dict(e, lane=-1) for e in truth], 'phira')
                time_only = event_metrics([dict(e, lane=-1, type='tap', end_us=e['onset_us']) for e in pred],
                    [dict(e, lane=-1, type='tap', end_us=e['onset_us']) for e in truth], 'phira')
                rounds.append(dict(round=saved['round'], **metric, no_lane_f1=no_lane['full_note_f1'],
                    time_only_f1=time_only['full_note_f1'],
                    maximum_required_keys=maximum_overlap(make_lane_draft(pred)) if mode == 'mania_4k' and pred else 0,
                    returned_complete=saved.get('complete', True) and not saved.get('revision_capacity_error')))
            cases.append(dict(identity=core['identity'], rounds=rounds))
        modes = {}
        for mode in ('mania_4k', 'phira'):
            selected = [c for c in cases if c['identity']['mode'] == mode]
            complete = all([r['round'] for r in c['rounds']] == [1, 2, 3, 4] and
                all(r['returned_complete'] for r in c['rounds']) for c in selected)
            row = dict(first_full_f1=sum(c['rounds'][0]['full_note_f1'] for c in selected)/len(selected),
                fourth_complete=complete, fourth_full_f1=None)
            if complete:
                final = [c['rounds'][3] for c in selected]
                row.update(fourth_full_f1=sum(r['full_note_f1'] for r in final)/len(final),
                    fourth_no_lane_f1=sum(r['no_lane_f1'] for r in final)/len(final),
                    fourth_time_only_f1=sum(r['time_only_f1'] for r in final)/len(final),
                    fourth_lane_conflicts=sum(r['mania_lane_conflicts'] for r in final),
                    fourth_maximum_required_keys=max(r['maximum_required_keys'] for r in final),
                    fourth_types={kind:{key:sum(r['per_type'][kind][key] for r in final)
                        for key in ('pred', 'true', 'matched')} for kind in ('tap', 'hold', 'flick', 'drag')})
            modes[mode] = row
        results[name] = dict(modes=modes, cases=cases, initial_state_sha256=doc['initial_state_sha256'],
            recorded_original_generation_wall_s=original['wall_s'])
    result = dict(id='D40', complete=True, same_core_clocks_targets_verified=True,
        same_acoustic_feature_identities_verified=len(required), arms=results,
        causal_architecture_comparison=False, held_out_quality_proven=False, model_promotion=False,
        optimizer_updates=0, new_paid_compute_usd=0, plan_sha256=file_sha(plan_path),
        limitation='Same train panel and100 updates; architectures/objective decompositions/initializations differ. No causal attribution to one component; no cross-loss ranking. Recorded generation times are not a dedicated latency benchmark.')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name:dict(modes=row['modes'], recorded_original_generation_wall_s=row['recorded_original_generation_wall_s'])
        for name, row in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Recompute R21 raw-event outcomes and decide the frozen paired hypothesis."""
from collections import Counter
import json
from pathlib import Path

from beatfcn.model.lane_draft import make_lane_draft, maximum_overlap
from scripts.acoustic_probe_metrics import event_metrics, physical_events, in_interval
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def exact_counter(events):
    return Counter((e['onset_us'], e['end_us'], e['type'], e['lane']) for e in events)


def audit_arm(raw, baseline):
    if not raw['complete'] or raw.get('error') or raw['optimizer_updates'] != 100:
        raise ValueError('not a completed fixed100 run')
    cases = []
    for core, old in zip(raw['generation']['original']['cores'], baseline['generation']['original']['cores']):
        if core['identity'] != old['identity'] or core['target_events'] != old['target_events']:
            raise ValueError('changed target or source clock')
        mode = core['identity']['mode']
        truth = core['target_events']
        interval = [core['identity']['source_core_us'] + 2_000_000, core['identity']['source_core_us'] + 8_000_000]
        rounds = []
        if [r['round'] for r in core['rounds']] != [1, 2, 3, 4]:
            raise ValueError('four-round generation incomplete')
        for saved in core['rounds']:
            pred = saved['events']
            metric = event_metrics(pred, truth, mode)
            for key in ('pred', 'true', 'matched', 'full_note_f1', 'exact_onset_group_recall', 'mania_lane_conflicts'):
                if abs(metric[key] - saved[key]) > 1e-12:
                    raise ValueError('raw-event mismatch: ' + key)
            shared = event_metrics(in_interval(physical_events(pred, core['identity']), interval),
                in_interval(physical_events(truth, core['identity']), interval), mode)
            rounds.append(dict(round=saved['round'], **metric,
                shared_interval_full_f1=shared['full_note_f1'],
                maximum_required_keys=maximum_overlap(make_lane_draft(pred)) if mode == 'mania_4k' and pred else 0))
        exact_first = exact_counter(core['rounds'][0]['events']) & exact_counter(truth)
        exact_last = exact_counter(core['rounds'][3]['events']) & exact_counter(truth)
        cases.append(dict(identity=core['identity'], rounds=rounds,
            exact_correct_events_added=sum((exact_last - exact_first).values()),
            exact_correct_events_removed=sum((exact_first - exact_last).values())))
    if len(cases) != 8:
        raise ValueError('panel is incomplete')
    modes = {}
    for mode in ('mania_4k', 'phira'):
        rows = [c for c in cases if c['identity']['mode'] == mode]
        rounds = []
        for index in range(4):
            current = [c['rounds'][index] for c in rows]
            rounds.append(dict(round=index + 1,
                full_f1=sum(r['full_note_f1'] for r in current) / len(current),
                group_recall=sum(r['exact_onset_group_recall'] for r in current) / len(current),
                shared_f1=sum(r['shared_interval_full_f1'] for r in current) / len(current),
                conflicts=sum(r['mania_lane_conflicts'] for r in current),
                peak_keys=max(r['maximum_required_keys'] for r in current),
                types={kind:{key:sum(r['per_type'][kind][key] for r in current)
                    for key in ('pred', 'true', 'matched')} for kind in ('tap', 'hold', 'flick', 'drag')}))
        modes[mode] = dict(rounds=rounds, first_full_f1=rounds[0]['full_f1'], fourth_full_f1=rounds[3]['full_f1'],
            revision_gain=rounds[3]['full_f1'] - rounds[0]['full_f1'],
            exact_correct_added=sum(c['exact_correct_events_added'] for c in rows),
            exact_correct_removed=sum(c['exact_correct_events_removed'] for c in rows))
    original_gates = dict(loss=raw['final']['mean_loss'] <= .1 and
        1 - raw['final']['mean_loss'] / raw['initial']['mean_loss'] >= .95,
        heads=all(v >= .95 for c in raw['final']['cores'] for v in c['accuracy'].values()),
        original=all(m['rounds'][3]['full_f1'] >= .9 and m['rounds'][3]['group_recall'] >= .9 and
            m['rounds'][3]['shared_f1'] >= .9 for m in modes.values()),
        original_legality=all(m['rounds'][3]['conflicts'] == 0 for m in modes.values()))
    return dict(modes=modes, cases=cases, original_necessary_gates=original_gates,
        original_necessary_passed=all(original_gates.values()),
        initial_loss=raw['initial']['mean_loss'], final_loss=raw['final']['mean_loss'],
        fit_s=raw['fit_s'], generation_s=raw['generation']['original']['wall_s'],
        wall_s=raw['wall_s'], peak_rss_gib=raw['peak_rss_gib'])


def main():
    path = ROOT / 'bench/rhythm-v2/r21-binding-review-v1.json'
    plan = json.loads(path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve prior review')
    docs = {arm:json.loads((ROOT / source).read_text()) for arm, source in plan['reports'].items()}
    baseline = json.loads((ROOT / plan['baseline']).read_text())
    if docs['A']['initial_state_sha256'] != docs['B']['initial_state_sha256'] or docs['A']['parameter_count'] != docs['B']['parameter_count']:
        raise ValueError('capacity or initialization differs')
    for key, value in docs['A']['input_sha256'].items():
        if docs['B']['input_sha256'].get(key) != value:
            raise ValueError('paired source identity differs: ' + key)
    arms = {arm:audit_arm(doc, baseline) for arm, doc in docs.items()}
    a, b = arms['A']['modes'], arms['B']['modes']
    limits = json.loads((ROOT / plan['frozen_fit_plan']).read_text())['acceptance']
    mechanism = dict(
        mania_fourth_relative=b['mania_4k']['fourth_full_f1'] >= a['mania_4k']['fourth_full_f1'] + limits['mania_fourth_gain_over_A_at_least'],
        mania_fourth_absolute=b['mania_4k']['fourth_full_f1'] >= limits['mania_fourth_at_least'],
        mania_revision_gain=b['mania_4k']['revision_gain'] >= a['mania_4k']['revision_gain'] + limits['mania_revision_gain_over_A_at_least'],
        first_guard=all(b[mode]['first_full_f1'] >= a[mode]['first_full_f1'] - limits['first_regression_per_mode_at_most'] for mode in a),
        phira_fourth_relative=b['phira']['fourth_full_f1'] >= a['phira']['fourth_full_f1'] - limits['phira_fourth_regression_over_A_at_most'],
        phira_fourth_absolute=b['phira']['fourth_full_f1'] >= limits['phira_fourth_at_least'])
    passed = all(mechanism.values()) and arms['B']['original_necessary_passed']
    result = dict(id='R21', complete=True, arms=arms,
        paired_init_and_input_identities_verified=True, mechanism_gates=mechanism,
        mechanism_passed=all(mechanism.values()), original_necessary_passed=arms['B']['original_necessary_passed'],
        expanded_evaluation_required=passed, full_joint_quality_passed=False,
        decision='requires_original_translated_and_audio_tests_before_any_acceptance' if passed else 'reject_and_isolate',
        model_promotion=False, held_out_quality_proven=False,
        optimizer_updates=200, new_paid_compute_usd=0,
        b_over_a_fit_time_ratio=arms['B']['fit_s'] / arms['A']['fit_s'],
        plan_sha256=file_sha(path), input_sha256=plan['input_sha256'])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(decision=result['decision'], mechanism=mechanism,
        quality={arm:r['original_necessary_gates'] for arm,r in arms.items()},
        metrics={arm:dict(loss=r['final_loss'],fit_s=r['fit_s'],modes={mode:{k:v[k] for k in
           ('first_full_f1','fourth_full_f1','revision_gain','exact_correct_added','exact_correct_removed')} for mode,v in r['modes'].items()}) for arm,r in arms.items()})))


if __name__ == '__main__':
    main()

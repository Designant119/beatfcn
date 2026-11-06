"""Independent R18 event/termination audit; no neural imports or output repair."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

from beatfcn.model.event_canvas import ClockLayout
from beatfcn.model.event_sequence import Decision, EventSequenceState, SequenceSpec
from scripts.review_complete_event_network import audit

ROOT = Path(__file__).resolve().parents[1]


def termination(doc):
    rows = []
    for core in doc['generation']['original']['cores']:
        rounds = []
        truth = core['target_events']
        for raw in core['rounds']:
            state = EventSequenceState(SequenceSpec(ClockLayout(8_000_000, 16_000_000),
                core['identity']['mode'], 32_000_000))
            ends = []
            for d in raw['decisions']:
                token = Decision(**d)
                if token.field == 'group' and token.value == 0:
                    ends.append(dict(domain_forced=state.allowed_values() == (0,),
                        last_onset_us=state.last_onset_us, empty=not state.events))
                state.apply(token)
            if [asdict(e) for e in state.events] != raw['events'] or state.done != raw['complete']:
                raise ValueError('independent decision replay mismatch')
            last = max((e.onset_us for e in state.events), default=None)
            later = [e for e in truth if last is None or e['onset_us'] > last + 20_000]
            rounds.append(dict(round=raw['round'], ends=ends,
                target_events_after_last_prediction_tolerance=len(later),
                target_groups_after_last_prediction_tolerance=len({e['onset_us'] for e in later}),
                pred_groups=len({e.onset_us for e in state.events}), true_groups=len({e['onset_us'] for e in truth}),
                last_prediction_us=last, last_target_us=max((e['onset_us'] for e in truth), default=None)))
        rows.append(dict(identity=core['identity'], rounds=rounds))
    first = [end for c in rows for end in c['rounds'][0]['ends']]
    all_ends = [end for c in rows for r in c['rounds'] for end in r['ends']]
    return dict(cores=rows, first_nonforced=sum(not e['domain_forced'] for e in first),
        first_empty=sum(e['empty'] for e in first),
        total_observed_ends=len(all_ends), total_forced=sum(e['domain_forced'] for e in all_ends))


def main():
    destination = ROOT / 'reports/r18_review.json'
    if destination.exists():
        raise FileExistsError('preserve previous review')
    paths = {'baseline': ROOT / 'reports/r17_b_fit.json', 'candidate': ROOT / 'reports/r18_fit.json'}
    docs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    a, b = docs['baseline'], docs['candidate']
    if a['initial_state_sha256'] != b['initial_state_sha256'] or a['parameter_count'] != b['parameter_count']:
        raise ValueError('initialization or capacity mismatch')
    for x, y in zip(a['generation']['original']['cores'], b['generation']['original']['cores']):
        if x['identity'] != y['identity'] or x['target_events'] != y['target_events']:
            raise ValueError('data or target mismatch')
    if len(a['history']) != 100 or len(b['history']) != 100:
        raise ValueError('step count mismatch')
    if any(x['views'] != y['views'] for x, y in zip(a['history'], b['history'])):
        raise ValueError('corruption schedule mismatch')
    verified = {name: audit(doc) for name, doc in docs.items()}
    ends = {name: termination(doc) for name, doc in docs.items()}
    if ends['baseline']['total_forced'] != 24 or ends['baseline']['total_observed_ends'] != 24:
        raise ValueError('baseline replay differs from D36')
    if ends['candidate']['first_nonforced'] != b['termination']['first_round_nonforced_ends']:
        raise ValueError('candidate stop count mismatch')
    teacher = {}
    for field in ('group', 'member', 'kind'):
        entries = [core['classes'][field] for core in b['final']['cores']]
        width = len(entries[0]['support'])
        support = [sum(e['support'][i] for e in entries) for i in range(width)]
        correct = [sum(e['confusion_true_rows'][i][i] for e in entries) for i in range(width)]
        predicted = [sum(e['predicted'][i] for e in entries) for i in range(width)]
        teacher[field] = dict(support=support, correct=correct, predicted=predicted,
            pooled_recall=[correct[i] / support[i] if support[i] else None for i in range(width)],
            mean_core_recall=[sum(e['recall'][i] for e in entries if e['recall'][i] is not None) /
                sum(e['recall'][i] is not None for e in entries) if support[i] else None for i in range(width)])
    baseline_teacher = {name: sum(c['accuracy'][name] for c in a['final']['cores'] if name in c['accuracy']) /
        sum(name in c['accuracy'] for c in a['final']['cores']) for name in ('group', 'member', 'kind', 'kind_1', 'kind_2')}
    plan = json.loads((ROOT / 'bench/rhythm-v2/r18-group-end-fit-v1.json').read_text())
    objective_gates = dict(first_round_autonomous_end=ends['candidate']['first_nonforced'] >= 6,
        no_empty_original=ends['candidate']['first_empty'] == 0,
        group_class_recall=all(v is not None and v >= .95 for c in b['final']['cores'] for v in c['classes']['group']['recall']),
        first_round_quality_nonregression=all(verified['candidate']['modes'][mode]['first_f1'] >= floor
            for mode, floor in plan['first_f1_floors'].items()))
    if objective_gates != b['objective_gates']:
        raise ValueError('objective gate mismatch')
    result = dict(id='R18-review', complete=True, independent_raw_event_recompute=True,
        same_data_initialization_capacity_corruption_verified=True, arms=verified,
        termination=ends, candidate_teacher=teacher, baseline_teacher_aggregate=baseline_teacher,
        baseline_teacher_group_classes=None, objective_gates=objective_gates,
        mechanism_passed=all(objective_gates.values()), joint_quality_passed=b['probe_passed'],
        original_unweighted_loss=dict(baseline=a['final']['mean_loss'], candidate=b['final']['mean_loss']),
        optimized_candidate_objective_loss=b['final']['mean_objective_loss'],
        first_round_gains={mode:verified['candidate']['modes'][mode]['first_f1'] - verified['baseline']['modes'][mode]['first_f1']
            for mode in verified['baseline']['modes']},
        evidence_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        new_paid_compute_usd=0, model_promotion=False, held_out_quality_proven=False, formal_training_ready=False)
    if result['joint_quality_passed'] != bool(all(b['gates'].values()) and result['mechanism_passed']):
        raise ValueError('joint decision mismatch')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('mechanism_passed', 'joint_quality_passed', 'objective_gates',
        'first_round_gains', 'candidate_teacher', 'original_unweighted_loss')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

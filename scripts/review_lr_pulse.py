"""Independent R20 trajectory identity and raw-event quality review."""
import hashlib
import json
from pathlib import Path

from scripts.review_complete_event_network import audit
from scripts.review_group_end_objective import termination

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'reports/r20_review.json'
    if destination.exists():
        raise FileExistsError('preserve previous review')
    docs = {name:json.loads((ROOT / path).read_text()) for name, path in
        [('baseline', 'reports/r19_fit.json'), ('candidate', 'reports/r20_fit.json')]}
    a, b = docs['baseline'], docs['candidate']
    plan = json.loads((ROOT / 'bench/rhythm-v2/r20-early-pulse-fit-v1.json').read_text())
    for key in ('initial_state_sha256', 'parameter_count', 'initial'):
        if a[key] != b[key]:
            raise ValueError('baseline mismatch: ' + key)
    if not b['complete'] or b.get('error') or b['optimizer_updates'] != 100 or len(b['history']) != 100:
        raise ValueError('not a complete100 experiment')
    for name, sha in plan['input_sha256'].items():
        source = ROOT / name
        if name in ('scripts/probe_lr_pulse.py', 'tests/test_lr_pulse.py', 'design/R20_EARLY_LR_PULSE_v1.md'):
            archived = ROOT / 'inputs/code_baselines/r20-early-lr-pulse/rejected' / name
            if archived.exists():
                source = archived
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError('frozen input changed: ' + name)
    if len(a['generation']['original']['cores']) != len(b['generation']['original']['cores']):
        raise ValueError('core count mismatch')
    for x, y in zip(a['generation']['original']['cores'], b['generation']['original']['cores']):
        if x['identity'] != y['identity'] or x['target_events'] != y['target_events']:
            raise ValueError('target identity mismatch')
    rates, errors = [], []
    for row, old in zip(b['history'], a['history']):
        t = row['step']
        expected = .0003 * (1 + 2 * (50 - t) / 24) if 26 <= t <= 50 else .0003
        if row['learning_rate'] != expected or row['views'] != old['views']:
            raise ValueError('LR schedule or corruption changed')
        rates.append(expected)
        if t <= 25:
            errors.extend(abs(x - y) for x, y in zip(row['core_losses'], old['core_losses']))
    if max(errors) > 1e-6 or abs(sum(rates) - .0375) > 1e-12 or b['diagnostic_forwards'] != 24:
        raise ValueError('trajectory or diagnostic accounting mismatch')
    verified = {name:audit(doc) for name, doc in docs.items()}
    ends = {name:termination(doc) for name, doc in docs.items()}
    gains = {mode:verified['candidate']['modes'][mode]['final_f1'] - row['final_f1']
        if verified['candidate']['modes'][mode]['final_f1'] is not None else None
        for mode, row in verified['baseline']['modes'].items()}
    training_gates = dict(loss_reduction=b['final']['mean_loss'] <= plan['mechanism_loss_ceiling'],
        fourth_round_quality_gain=all(g is not None and verified['candidate']['modes'][mode]['final_f1'] >= plan['fourth_f1_floors'][mode]
            for mode, g in gains.items()))
    member = dict(mean_loss=sum(c['terms']['member'] for c in b['final']['cores']) / 8,
        mean_accuracy=sum(c['accuracy']['member'] for c in b['final']['cores']) / 8)
    protection = dict(member_loss=member['mean_loss'] <= .35, member_accuracy=member['mean_accuracy'] >= .85,
        first_round_quality_nonregression=all(verified['candidate']['modes'][mode]['first_f1'] >= floor
            for mode, floor in plan['first_f1_floors'].items()))
    if training_gates != b['training_gates'] or protection != b['data_gates'] or member != b['member_metrics']:
        raise ValueError('registered gate mismatch')
    joint = all(training_gates.values()) and all(protection.values()) and all(b['gates'].values())
    if joint != b['probe_passed']:
        raise ValueError('joint decision mismatch')
    teacher = {}
    for field in ('group', 'member', 'kind'):
        rows = [c['classes'][field] for c in b['final']['cores']]
        n = len(rows[0]['support'])
        support = [sum(c['support'][i] for c in rows) for i in range(n)]
        correct = [sum(c['confusion_true_rows'][i][i] for c in rows) for i in range(n)]
        teacher[field] = dict(support=support, correct=correct,
            pooled_recall=[c / s if s else None for s, c in zip(support, correct)])
    result = dict(id='R20-review', complete=True, independent_raw_event_recompute=True,
        same_data_initialization_capacity_corruption_verified=True, exact_schedule_verified=True,
        first25_maximum_loss_error=max(errors), cumulative_lr=sum(rates), diagnostic_forwards=24,
        arms=verified, termination=ends, training_gates=training_gates, protection_gates=protection,
        fourth_round_gains=gains, candidate_teacher=teacher, member_metrics=member,
        mechanism_passed=all(training_gates.values()) and all(protection.values()), joint_quality_passed=joint,
        original_loss={name:doc['final']['mean_loss'] for name, doc in docs.items()},
        evidence_sha256={name:hashlib.sha256((ROOT / 'reports' / filename).read_bytes()).hexdigest()
            for name, filename in [('baseline', 'r19_fit.json'), ('candidate', 'r20_fit.json')]},
        model_promotion=False, held_out_quality_proven=False, formal_training_ready=False, new_paid_compute_usd=0)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('mechanism_passed', 'joint_quality_passed', 'fourth_round_gains',
        'original_loss', 'candidate_teacher', 'member_metrics')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

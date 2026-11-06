"""Independent R19 raw-event and member-class audit; never edits predictions."""
from pathlib import Path
import hashlib
import json

from scripts.review_complete_event_network import audit
from scripts.review_group_end_objective import termination

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/r19_review.json'
    if output.exists():
        raise FileExistsError('preserve previous review')
    paths = {arm: ROOT / path for arm, path in
             [('baseline', 'reports/r17_b_fit.json'), ('candidate', 'reports/r19_fit.json')]}
    docs = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    a, b = docs['baseline'], docs['candidate']
    plan = json.loads((ROOT / 'bench/rhythm-v2/r19-count-unit-fit-v1.json').read_text())
    if a['initial_state_sha256'] != b['initial_state_sha256'] or a['parameter_count'] != b['parameter_count']:
        raise ValueError('initialization or capacity mismatch')
    for x, y in zip(a['generation']['original']['cores'], b['generation']['original']['cores']):
        if x['identity'] != y['identity'] or x['target_events'] != y['target_events']:
            raise ValueError('data or target mismatch')
    if len(a['history']) != 100 or len(b['history']) != 100:
        raise ValueError('step count mismatch')
    if any(x['views'] != y['views'] for x, y in zip(a['history'], b['history'])):
        raise ValueError('corruption schedule mismatch')
    original_source = (ROOT / 'inputs/code_baselines/r17-complete-event-state/rejected/scripts/r17_sequence_candidate.py').read_text()
    candidate_path = ROOT / 'scripts/r19_count_candidate.py'
    if not candidate_path.exists():
        candidate_path = ROOT / 'inputs/code_baselines/r19-member-count-unit/rejected/scripts/r19_count_candidate.py'
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != plan['input_sha256']['scripts/r19_count_candidate.py']:
        raise ValueError('candidate source identity mismatch')
    candidate_source = candidate_path.read_text()
    if candidate_source != original_source.replace('s[:, 9:10].float() / 8', 's[:, 9:10].float()'):
        raise ValueError('intervention is not the one registered scalar')
    verified = {arm: audit(doc) for arm, doc in docs.items()}
    ends = {arm: termination(doc) for arm, doc in docs.items()}
    teacher = {}
    for field in ('group', 'member', 'kind'):
        cells = [core['classes'][field] for core in b['final']['cores']]
        n = len(cells[0]['support'])
        support = [sum(c['support'][i] for c in cells) for i in range(n)]
        correct = [sum(c['confusion_true_rows'][i][i] for c in cells) for i in range(n)]
        teacher[field] = dict(support=support, correct=correct,
            pooled_recall=[correct[i] / support[i] if support[i] else None for i in range(n)])
    members = {arm:dict(mean_loss=sum(c['terms']['member'] for c in doc['final']['cores']) / 8,
        mean_accuracy=sum(c['accuracy']['member'] for c in doc['final']['cores']) / 8) for arm, doc in docs.items()}
    gates = dict(member_loss=members['candidate']['mean_loss'] <= .35,
        member_accuracy=members['candidate']['mean_accuracy'] >= .85,
        first_round_quality_nonregression=all(verified['candidate']['modes'][mode]['first_f1'] >= floor
            for mode, floor in plan['first_f1_floors'].items()))
    if gates != b['data_gates'] or members['candidate'] != b['member_metrics']:
        raise ValueError('registered data gate mismatch')
    joint = all(gates.values()) and all(b['gates'].values())
    if joint != b['probe_passed']:
        raise ValueError('joint decision mismatch')
    result = dict(id='R19-review', complete=True, independent_raw_event_recompute=True,
        same_data_initialization_capacity_corruption_verified=True, single_source_delta_verified=True,
        arms=verified, termination=ends, member_metrics=members, candidate_teacher=teacher,
        data_gates=gates, mechanism_passed=all(gates.values()), joint_quality_passed=joint,
        original_loss={arm:doc['final']['mean_loss'] for arm, doc in docs.items()},
        first_round_gains={mode:verified['candidate']['modes'][mode]['first_f1'] - verified['baseline']['modes'][mode]['first_f1']
            for mode in verified['baseline']['modes']},
        evidence_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        new_paid_compute_usd=0, model_promotion=False, held_out_quality_proven=False, formal_training_ready=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('mechanism_passed', 'joint_quality_passed', 'member_metrics',
        'candidate_teacher', 'first_round_gains', 'original_loss')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

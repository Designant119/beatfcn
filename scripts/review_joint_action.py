"""R22 independent raw-event review; common marginal CE, not incomparable losses."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.review_member_binding import audit_arm, exact_counter
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def action_errors(core):
    truth_groups = defaultdict(list)
    for event in core['target_events']:
        truth_groups[event['onset_us']].append(event)
    rounds = []
    for saved in core['rounds']:
        pred_groups = defaultdict(list)
        for event in saved['events']:
            pred_groups[event['onset_us']].append(event)
        strata = {}
        for size in range(1,5):
            groups = [g for g in truth_groups.values() if len(g) == size]
            exact_events = sum(sum((exact_counter(g) & exact_counter(pred_groups[g[0]['onset_us']])).values()) for g in groups)
            complete = sum(exact_counter(g) == exact_counter(pred_groups[g[0]['onset_us']]) for g in groups)
            same_size = [(g,pred_groups[g[0]['onset_us']]) for g in groups if len(pred_groups[g[0]['onset_us']]) == size]
            strata[str(size)] = dict(target_groups=len(groups), target_notes=size*len(groups),
                exact_correct_events=exact_events, exact_complete_groups=complete,
                predicted_same_exact_head_and_size=len(same_size),
                lane_vector_correct_at_same_head_size=sum(sorted(e['lane'] for e in a) == sorted(e['lane'] for e in b) for a,b in same_size),
                action_correct_at_same_head_size=sum(Counter((e['type'],e['lane']) for e in a) == Counter((e['type'],e['lane']) for e in b) for a,b in same_size))
        rounds.append(dict(round=saved['round'], strata=strata,
                           scope='exact-time target recall and conditional action diagnostics; not standalone stratum F1'))
    return dict(identity=core['identity'], rounds=rounds)


def main():
    path = ROOT / 'bench/rhythm-v2/r22-joint-action-review-v1.json'
    plan = json.loads(path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve prior review')
    docs = {arm:json.loads((ROOT/source).read_text()) for arm,source in plan['reports'].items()}
    baseline = json.loads((ROOT/plan['baseline']).read_text())
    for key in ('initial_state_sha256','common_initial_state_sha256','parameter_count','trainable_parameter_count','input_sha256'):
        if docs['A'][key] != docs['B'][key]:
            raise ValueError('paired identity mismatch: '+key)
    arms = {arm:audit_arm(doc,baseline) for arm,doc in docs.items()}
    for arm,doc in docs.items():
        arms[arm]['objective_final_loss'] = doc['objective_final']['mean_loss']
        arms[arm]['joint_action_teacher'] = [dict(identity=c['identity'], accuracy=c['accuracy']['joint_action'])
            for c in doc['objective_final']['cores'] if c['identity']['mode'] == 'mania_4k']
        arms[arm]['action_error_strata'] = [action_errors(c) for c in doc['generation']['original']['cores'] if c['identity']['mode'] == 'mania_4k']
    limits = json.loads((ROOT/plan['frozen_fit_plan']).read_text())['acceptance']
    a,b = arms['A']['modes'],arms['B']['modes']
    gates = dict(mania_fourth_relative=b['mania_4k']['fourth_full_f1'] >= a['mania_4k']['fourth_full_f1']+limits['mania_fourth_gain_over_A_at_least'],
        mania_fourth_absolute=b['mania_4k']['fourth_full_f1'] >= limits['mania_fourth_at_least'],
        first_guard=all(b[m]['first_full_f1'] >= a[m]['first_full_f1']-limits['first_regression_per_mode_at_most'] for m in a),
        phira_fourth_relative=b['phira']['fourth_full_f1'] >= a['phira']['fourth_full_f1']-limits['phira_fourth_regression_over_A_at_most'],
        phira_fourth_absolute=b['phira']['fourth_full_f1'] >= limits['phira_fourth_at_least'],
        joint_action_teacher=all(c['accuracy'] >= limits['per_core_joint_action_at_least'] for c in arms['B']['joint_action_teacher']))
    required = all(gates.values()) and arms['B']['original_necessary_passed']
    result = dict(id='R22', complete=True, arms=arms, mechanism_gates=gates, mechanism_passed=all(gates.values()),
        paired_init_capacity_data_verified=True, common_loss_scope='both arms recomputed from member marginal fields; training objective losses reported separately',
        decision='requires_frozen_expanded_validation' if required else 'reject_and_isolate',
        expanded_evaluation_required=required, model_promotion=False, full_joint_quality_passed=False,
        optimizer_updates=200, new_paid_compute_usd=0,
        plan_sha256=file_sha(path), input_sha256=plan['input_sha256'])
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(decision=result['decision'],mechanism=gates,
        arms={arm:dict(common_loss=x['final_loss'],objective_loss=x['objective_final_loss'],
            necessary=x['original_necessary_gates'],joint=x['joint_action_teacher'],
            modes={m:{k:y[k] for k in ('first_full_f1','fourth_full_f1','revision_gain')} for m,y in x['modes'].items()}) for arm,x in arms.items()})))


if __name__ == '__main__':
    main()

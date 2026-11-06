"""Independently review R23 events, payload provenance, and frozen quality gates."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from beatfcn.model.event_canvas import ClockLayout, RhythmEvent
from beatfcn.model.group_canvas import new_group_canvas, encode_group_targets, decode_group_targets
from scripts.audit_draft_exposure import canvas_payload, saved_payload
from scripts.audit_edit_support import transition_record
from scripts.review_member_binding import audit_arm
from scripts.r09_local_data import file_sha, verify_files

ROOT=Path(__file__).resolve().parents[1]


def transition_audit(raw, seed):
    counts=defaultdict(Counter);records=[]
    for core in raw['generation']['original']['cores']:
        identity=core['identity'];mode=identity['mode']
        layout=ClockLayout(identity['c0_us'],identity['c0_us']+8_000_000)
        labels=encode_group_targets([RhythmEvent(**e) for e in core['target_events']],layout,mode)
        truth=canvas_payload(labels);rng=np.random.default_rng(seed)
        for index,threshold in enumerate((.5,.25,.125)):
            previous,following=core['rounds'][index:index+2]
            sizes,_,payload,_,_=saved_payload(previous,layout)
            next_sizes,_,next_payload,_,_=saved_payload(following,layout)
            visible=rng.random(sizes.shape)>=threshold
            for pos in zip(*np.nonzero(visible & (sizes>0))):
                row=transition_record(payload.get(pos),truth.get(pos,[]),next_payload.get(pos),mode,bool(next_sizes[pos]>0))
                counts[mode]['visible_positive']+=1
                counts[mode]['source_'+row['label']]+=1
                if row['copy_eligible_in_saved_transition']:
                    counts[mode]['eligible_'+row['label']]+=1
                    counts[mode]['correct_damaged']+=int(row['correct_group_damaged'])
                    counts[mode]['wrong_repaired']+=int(row['wrong_group_repaired'])
                if row['correct_group_damaged'] or row['wrong_group_repaired']:
                    records.append(dict(identity=identity,from_round=index+1,position=[int(x) for x in pos],
                        **row,source_group=payload.get(pos),following_group=next_payload.get(pos),target_group=truth.get(pos,[])))
    return dict(summary={k:dict(v) for k,v in counts.items()},records=records,
        scope='same partial-ownership reconstruction as D45 for descriptive comparison, not a causal rate or free-output substitution')


def main():
    plan_path=ROOT/'bench/rhythm-v2/r23-local-action-review-v1.json'
    plan=json.loads(plan_path.read_text());verify_files(plan)
    destination=ROOT/plan['result']
    if destination.exists():raise FileExistsError('preserve previous review')
    raw=json.loads((ROOT/plan['raw']).read_text());baseline=json.loads((ROOT/plan['baseline']).read_text())
    audited,old=audit_arm(raw,baseline),audit_arm(baseline,baseline)
    if raw['initial_state_sha256']!=baseline['initial_state_sha256'] or raw['initial']!=baseline['initial']:
        raise ValueError('model or initial data identity changed')
    for i in range(3):
        if raw['history'][i]['core_losses']!=baseline['history'][i]['core_losses']:raise ValueError('prefix trajectory differs')
    for core in raw['generation']['original']['cores']:
        layout=ClockLayout(core['identity']['c0_us'],core['identity']['c0_us']+8_000_000)
        for row in core['rounds']:
            canvas=new_group_canvas(layout,core['identity']['mode'],unknown=False)
            canvas.counts=np.asarray(row['block_counts'],np.int64)
            canvas.heads=np.asarray(row['raw_group_heads'],np.int64)
            canvas.multiplicity=np.asarray(row['member_counts'],np.int64)
            canvas.notes=np.asarray(row['raw_group_notes'],np.int64)
            if [asdict(e) for e in decode_group_targets(canvas)]!=row['events']:raise ValueError('complete raw payload differs')
    modes,control=audited['modes'],old['modes']
    gates=dict(mania_fourth_gain=modes['mania_4k']['fourth_full_f1']>=control['mania_4k']['fourth_full_f1']+.05,
        mania_revision_gain=modes['mania_4k']['revision_gain']>=control['mania_4k']['revision_gain']+.03,
        first_pass_protection=all(modes[m]['first_full_f1']>=control[m]['first_full_f1']-.01 for m in modes),
        phira_fourth_protection=modes['phira']['fourth_full_f1']>=control['phira']['fourth_full_f1']-.01,
        phira_rare_types=all(modes['phira']['rounds'][3]['types'][k]['matched']==modes['phira']['rounds'][3]['types'][k]['true'] for k in ('hold','flick')))
    if gates!=raw['mechanism_gates']:raise ValueError('independent gate mismatch')
    passed=all(gates.values()) and audited['original_necessary_passed']
    result=dict(id='R23-review',complete=True,plan_sha256=file_sha(plan_path),input_sha256=plan['input_sha256'],
        scope=raw['scope'],decision='original_necessary_passed_pending_extended' if passed else 'reject_and_isolate',
        quality_promotion=False,mechanism_gates=gates,baseline=old,candidate=audited,
        transitions=dict(baseline=transition_audit(baseline,plan['seed']),candidate=transition_audit(raw,plan['seed'])),
        initial_model_and_outputs_exact=True,first_three_updates_exact=True,all32_raw_group_payloads_lossless=True,
        model_forwards=0,optimizer_updates=0,new_paid_compute_usd=0,
        next_dimension_if_rejected='architecture or objective or training; no local-noise grid or extra steps')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(decision=result['decision'],gates=gates,necessary=audited['original_necessary_gates'],
        modes={m:{k:v[k] for k in ('first_full_f1','fourth_full_f1','revision_gain','exact_correct_added','exact_correct_removed')} for m,v in modes.items()},
        transitions={k:v['summary'] for k,v in result['transitions'].items()}),ensure_ascii=False))


if __name__=='__main__':main()

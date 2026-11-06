"""Independent R17 raw-event, sequence-error, and paired-decision audit."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.event_sequence import SequenceSpec, encode_sequence
from beatfcn.model.event_canvas import ClockLayout
from beatfcn.model.lane_draft import make_lane_draft, maximum_overlap
from scripts.acoustic_probe_metrics import event_metrics

ROOT=Path(__file__).resolve().parents[1]


def summary(generation):
    """Reporting-only completeness, with no fallback to the last available round."""
    result={}
    for mode in ('mania_4k','phira'):
        cases=[c for c in generation['cores'] if c['identity']['mode']==mode]
        if not cases:continue
        complete=all(len(c['rounds'])==4 and all(r['complete'] and not r.get('revision_capacity_error') for r in c['rounds']) for c in cases)
        result[mode]=dict(first_f1=sum(c['rounds'][0]['full_note_f1'] for c in cases)/len(cases),
            last_available_f1=sum(c['rounds'][-1]['full_note_f1'] for c in cases)/len(cases),
            final_f1=sum(c['rounds'][3]['full_note_f1'] for c in cases)/len(cases) if complete else None,
            final_group_recall=sum(c['rounds'][3]['exact_onset_group_recall'] for c in cases)/len(cases) if complete else None,
            all_four_rounds=complete)
    return result


def audit(doc):
    if not doc['complete'] or doc.get('error') or doc['optimizer_updates']!=100:
        raise ValueError('not a complete100 result')
    rows=[]
    for case in doc['generation']['original']['cores']:
        mode=case['identity']['mode'];truth=case['target_events'];rounds=[]
        tokens=[dict(field=d.field,value=d.value) for d in encode_sequence(
            [RhythmEvent(**e) for e in truth],SequenceSpec(ClockLayout(8_000_000,16_000_000),mode,32_000_000))]
        for raw in case['rounds']:
            pred=raw['events'];m=event_metrics(pred,truth,mode)
            for key in ('pred','true','matched','full_note_f1','exact_onset_group_recall','mania_lane_conflicts'):
                if m[key]!=raw[key]:raise ValueError('saved event metric mismatch: '+key)
            peak=maximum_overlap(make_lane_draft(pred)) if mode=='mania_4k' and pred else 0
            if peak!=raw['peak_required_keys']:raise ValueError('saved occupancy mismatch')
            no_lane=event_metrics([dict(e,lane=-1) for e in pred],[dict(e,lane=-1) for e in truth],'phira')
            time_only=event_metrics([dict(e,lane=-1,type='tap',end_us=e['onset_us']) for e in pred],
                                    [dict(e,lane=-1,type='tap',end_us=e['onset_us']) for e in truth],'phira')
            mismatch=next((i for i,(a,b) in enumerate(zip(tokens,raw['decisions'])) if a!=b),None)
            if mismatch is None and len(tokens)!=len(raw['decisions']):mismatch=min(len(tokens),len(raw['decisions']))
            first=dict(index=mismatch,target=tokens[mismatch] if mismatch is not None and mismatch<len(tokens) else None,
                       generated=raw['decisions'][mismatch] if mismatch is not None and mismatch<len(raw['decisions']) else None)
            heads=sorted(set(e['onset_us'] for e in pred));gaps=np.diff([8_000_000,*heads,16_000_000])/1000
            rounds.append(dict(round=raw['round'],**m,peak_required_keys=peak,ignore_lane_full_f1=no_lane['full_note_f1'],
                               time_only_f1=time_only['full_note_f1'],first_sequence_mismatch=first,
                               prediction_count_ratio=len(pred)/len(truth) if truth else None,
                               largest_empty_gap_ms=float(max(gaps)) if len(gaps) else None,
                               complete=raw['complete'],revision_capacity_error=raw.get('revision_capacity_error'),
                               first_events=pred[:10],lane_counts=dict(Counter(e['lane'] for e in pred))))
        rows.append(dict(identity=case['identity'],rounds=rounds))
    modes=summary(doc['generation']['original'])
    for mode in ('mania_4k','phira'):
        selected=[r for r in rows if r['identity']['mode']==mode]
        modes[mode]['last_available_diagnostics']={key:sum(c['rounds'][-1][key] for c in selected)/len(selected) for key in
                     ('full_note_f1','ignore_lane_full_f1','time_only_f1','prediction_count_ratio')}
        if modes[mode]['first_f1']!=doc['summary'][mode]['first_f1']:raise ValueError('first-round macro mismatch')
        modes[mode]['original_report_final_was_invalid_fallback']=modes[mode]['final_f1']!=doc['summary'][mode]['final_f1']
        if 'last_available_f1' in doc['summary'][mode] and modes[mode]['original_report_final_was_invalid_fallback']:
            raise ValueError('corrected fourth-round metric mismatch')
    return dict(modes=modes,cases=rows)


def main():
    paths={arm:ROOT/f'reports/r17_{arm.lower()}_fit.json' for arm in ('A','B')}
    docs={a:json.loads(p.read_text()) for a,p in paths.items()}
    if docs['A']['initial_state_sha256']!=docs['B']['initial_state_sha256']:raise ValueError('paired init mismatch')
    if docs['A']['parameter_count']!=docs['B']['parameter_count']:raise ValueError('paired capacity mismatch')
    for a,b in zip(docs['A']['generation']['original']['cores'],docs['B']['generation']['original']['cores']):
        if a['identity']!=b['identity'] or a['target_events']!=b['target_events']:raise ValueError('paired target mismatch')
    verified={arm:audit(doc) for arm,doc in docs.items()}
    gains={mode:verified['B']['modes'][mode]['final_f1']-verified['A']['modes'][mode]['final_f1']
           if verified['A']['modes'][mode]['final_f1'] is not None and verified['B']['modes'][mode]['final_f1'] is not None else None
           for mode in verified['A']['modes']}
    mechanism=(gains['mania_4k']>=.1 and gains['phira']>=-.01) if all(v is not None for v in gains.values()) else None
    if gains!=docs['B']['paired_gain'] or mechanism!=docs['B']['gates']['mechanism_gain']:raise ValueError('paired gate mismatch')
    joint=bool(mechanism and all(docs['B']['gates'].values()))
    result=dict(id='R17-review',complete=True,independent_raw_event_recompute=True,mechanism_passed=mechanism,
                joint_quality_passed=joint,gains=gains,arms=verified,
                auxiliary_first_round_gains={mode:verified['B']['modes'][mode]['first_f1']-verified['A']['modes'][mode]['first_f1'] for mode in verified['A']['modes']},
                evidence_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
                model_promotion=False,held_out_quality_proven=False,formal_training_ready=False,new_paid_compute_usd=0)
    (ROOT/'reports/r17_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','mechanism_passed','joint_quality_passed','gains')},ensure_ascii=False))


if __name__=='__main__':main()

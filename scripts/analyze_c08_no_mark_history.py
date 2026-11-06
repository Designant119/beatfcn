"""Saved-output-only I/Q/M and mechanism analysis, no forward or training."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from scripts.run_local_portfolio_scale import sha,write


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c08-no-mark-history-v1'
OLD=ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1'
FOLDER=ROOT/'reports/local-evolution-v4'
KINDS=('tap','hold','flick','drag')


def read(path):return json.loads(path.read_text())


def components(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(r['scores'][field]['f1'] if field!='full_note_f1' else r['scores'][field])
    return {k:float(np.mean(v)) for k,v in groups.items()}


def output_summary(paths):
    draws=[];component_draws=[];support=[]
    for path in paths:
        rows=read(path);cm=components(rows,'note_50ms');component_draws.append(cm)
        counts={k:{f:sum(r['scores']['per_type'][k][f] for r in rows) for f in ('tp','pred','true')} for k in KINDS}
        rare={f:sum(counts[k][f] for k in KINDS[1:]) for f in ('tp','pred','true')}
        draws.append(dict(typed50=float(np.mean(list(cm.values()))),typed10=float(np.mean(list(components(rows,'note_10ms').values()))),
            full50=float(np.mean(list(components(rows,'full_note_f1').values()))),non_tap=2*rare['tp']/max(rare['pred']+rare['true'],1),
            complete=all(r['prediction']['complete'] for r in rows)))
        support.append(counts)
    mean={k:float(np.mean([d[k] for d in draws])) for k in ('typed50','typed10','full50','non_tap')}
    mean['components']={c:float(np.mean([d[c] for d in component_draws])) for c in component_draws[0]}
    mean['per_type']={k:{f:float(np.mean([d[k][f] for d in support])) for f in ('tp','pred','true')} for k in KINDS}
    for v in mean['per_type'].values():
        v.update(f1=2*v['tp']/max(v['pred']+v['true'],1),precision=v['tp']/max(v['pred'],1),recall=v['tp']/max(v['true'],1))
    return dict(mean=mean,draws=draws)


def probability_summary(rows,prior):
    all_nll=[];all_labels=[];same=[];groups=defaultdict(list)
    confusion=np.zeros((4,4),np.float64)
    for r in rows:
        v=np.array(r['node_nll']);labels=np.array(r['truth'],np.int64);heads=r['heads']
        n=len(labels);groups[r['component']].append((n,float(v.mean()),float(-np.log(prior[labels]).mean())))
        all_nll.append(v);all_labels.append(labels)
        same.extend([j>0 and h==heads[j-1] for j,h in enumerate(heads)])
        chosen=np.array(r['probabilities']).argmax(-1)
        for d in range(chosen.shape[0]):np.add.at(confusion,(labels,chosen[d]),1/chosen.shape[0])
    nll=np.concatenate(all_nll,axis=1);labels=np.concatenate(all_labels);same=np.array(same,bool)
    return dict(nll=float(nll.mean()),prior=float(-np.log(prior[labels]).mean()),events=len(labels),
        components={c:dict(events=sum(v[0] for v in a),nll=sum(v[0]*v[1] for v in a)/sum(v[0] for v in a),prior=sum(v[0]*v[2] for v in a)/sum(v[0] for v in a)) for c,a in groups.items()},
        per_type={k:dict(n=int((labels==j).sum()),nll=float(nll[:,labels==j].mean()),node_correct=float(confusion[j,j])) for j,k in enumerate(KINDS)},
        confusion=confusion.tolist(),node_accuracy=float(np.trace(confusion)/confusion.sum()),
        same_head_subsequent=dict(n=int(same.sum()),nll=float(nll[:,same].mean())),
        first_at_head=dict(n=int((~same).sum()),nll=float(nll[:,~same].mean())))


def match_heads(predicted,truth,tolerance=50000):
    """Time-ordered maximum-cardinality/minimum-total-error DP.

    Equal count/error paths use lexicographically smallest original index pairs.
    Sorted one-dimensional L1 admits an optimum with no crossing pairs.
    """
    empty=(0,0,())
    previous=[empty]*(len(truth)+1)
    def key(v):return (-v[0],v[1],v[2])
    for i,h in enumerate(predicted):
        current=[empty]
        for j,t in enumerate(truth):
            options=[previous[j+1],current[j]]
            if abs(h-t)<=tolerance:
                n,error,pairs=previous[j];options.append((n+1,error+abs(h-t),pairs+((i,j),)))
            current.append(min(options,key=key))
        previous=current
    return previous[-1][2]


def matching_ledger(rows):
    ledger=[]
    for r in rows:
        pred=r['prediction']['events'];truth=r['target'];pairs=match_heads([e['onset_us'] for e in pred],[e['onset_us'] for e in truth])
        marks=[];tail=[]
        for i,j in pairs:
            p,t=pred[i],truth[j]
            typed=p['type']==t['type'];full=typed and (t['type']!='hold' or abs(p['end_us']-t['end_us'])<=50000)
            marks.append(dict(pred_index=i,true_index=j,head_error_us=p['onset_us']-t['onset_us'],typed=typed,full=full))
            if typed and t['type']=='hold':tail.append(dict(head_error_us=p['onset_us']-t['onset_us'],end_error_us=p['end_us']-t['end_us'],duration_error_us=(p['end_us']-p['onset_us'])-(t['end_us']-t['onset_us'])))
        ledger.append(dict(index=r['index'],component=r['component'],pred=len(pred),true=len(truth),matched=len(pairs),missing=len(truth)-len(pairs),spurious=len(pred)-len(pairs),
            type_errors=sum(not v['typed'] for v in marks),typed_tail_errors=sum(v['typed'] and not v['full'] for v in marks),pairs=marks,matched_hold_errors=tail))
    return ledger


def main():
    started=time.monotonic();report=read(FOLDER/'no-mark-history.json');plan=read(ROOT/'bench/rhythm-v2/c08-no-mark-history-v1.json')
    if report['status']!='terminal_pending_mechanism_review':raise ValueError('Only complete original terminal is analyzed here')
    assert report['quality_updates']==1000 and report['effective_quality_updates']==996 and report['health_updates']==100
    assert sha(ART/'quality-1000.pt')==report['final_weight_sha256']
    assert read(ART/'data-identity.json')==read(OLD/'data-identity.json')
    old_summary={};new_summary={}
    for track in ('machine','oracle'):
        sub='real' if track=='machine' else 'oracle'
        old_summary[track]=output_summary([OLD/'evaluation'/sub/f'draw{d}-draft.json' for d in range(4)])
        new_summary[track]=output_summary([ART/'evaluation'/sub/f'draw{d}-draft.json' for d in range(4)])
    prior=np.array(read(ART/'data-identity.json')['prior']);base_free=read(ART/'B-self-prefix-T1-baseline.json')
    old_prob=probability_summary(base_free['traces'],prior)
    probability={c:probability_summary(read(ART/'free-probability'/f'{c}.json'),prior) for c in ('real','local_roll','song_foreign','audio_foreign')}
    real=probability['real'];g=plan['gate'];n=real['nll'];prior_wins=sum(v['nll']<v['prior'] for v in real['components'].values())
    audio={c:dict(gap=probability[c]['nll']-n,components=sum(probability[c]['components'][k]['nll']>v['nll'] for k,v in real['components'].items())) for c in ('local_roll','song_foreign','audio_foreign')}
    wins={t:sum(v>old_summary[t]['mean']['components'][c] for c,v in new_summary[t]['mean']['components'].items()) for t in ('oracle','machine')}
    oracle,machine=new_summary['oracle']['mean'],new_summary['machine']['mean']
    tests=dict(I=dict(prior_nll=n<=g['I']['nll_max'],prior_components=prior_wins>=g['I']['prior_components'],B_free_gain=old_prob['nll']-n>=g['I']['B_free_gain'],
            local_roll=audio['local_roll']['gap']>=.05 and audio['local_roll']['components']>=9,
            all_audio=audio['audio_foreign']['gap']>=.05 and audio['audio_foreign']['components']>=9),
        Q=dict(typed=oracle['typed50']>=g['Q']['typed50'],components=wins['oracle']>=9,full=oracle['full50']>=g['Q']['full50'],non_tap=oracle['non_tap']>=g['Q']['non_tap']),
        M=dict(typed=machine['typed50']>=g['M']['typed50'],components=wins['machine']>=9,full=machine['full50']>=g['M']['full50'],non_tap=machine['non_tap']>=g['M']['non_tap'],complete=all(d['complete'] for d in new_summary['machine']['draws'])))
    gates={k:all(v.values()) for k,v in tests.items()};gates['default']=all(gates.values()) and machine['typed50']>=g['default']['typed50'] and machine['full50']>=g['default']['full50']
    teacher=report['evaluation']['oracle_probability'];tnll=sum(v['events']*v['type_nll'] for v in teacher if v['task']=='draft' and v['condition']=='real')/real['events']
    ledgers=[]
    for d in range(4):
        rows=read(ART/'evaluation/real'/f'draw{d}-draft.json');ledgers.append(matching_ledger(rows))
        old=read(OLD/'evaluation/real'/f'draw{d}-draft.json')
        for a,b in zip(rows,old):assert [v['onset_us'] for v in a['prediction']['events']]==[v['onset_us'] for v in b['prediction']['events']]
    write(ART/'analysis-only-head-matching.json',dict(policy='50ms maximum cardinality,minimum total absolute error,lexicographic original index tie; no type used to match; no inference/metric replacement',draws=ledgers))
    tail=[v for rows in ledgers for r in rows for v in r['matched_hold_errors']]
    empty={}
    for d in range(4):
        a=read(ART/'evaluation/real'/f'draw{d}-empty.json');b=read(OLD/'evaluation/real'/f'draw{d}-empty.json')
        assert [[v['onset_us'] for v in r['prediction']['events']] for r in a]==[[v['onset_us'] for v in r['prediction']['events']] for r in b]
        empty[str(d)]=[len(r['prediction']['events']) for r in a]
    controls=report['evaluation']['control_failures'];oldcontrols=read(ROOT/'reports/local-evolution-v3/anchor-marks.json')['evaluation']['control_failures']
    assert controls==oldcontrols
    result=dict(candidate=plan['id'],status='analyzed',gates=gates,gate_details=tests,B=old_summary,candidate_outputs=new_summary,
        B_self_T1=old_prob,free_probability=probability,audio_controls=audio,prior_component_wins=prior_wins,component_wins=wins,
        teacher_free_nll_difference=tnll-n,empty_counts_interface_only=empty,reference_control_failures=controls,
        head_matching_summary={k:float(np.mean([sum(r[k] for r in draw) for draw in ledgers])) for k in ('pred','true','matched','missing','spurious','type_errors','typed_tail_errors')},
        emitted_matched_hold_tail=dict(n=len(tail),end_mae_ms=float(np.mean([abs(v['end_error_us'])/1000 for v in tail]))) if tail else None,
        resources={k:report[k] for k in ('parameters','health_wall_s','quality_wall_s','evaluation_wall_s','process_wall_s','effective_quality_updates')},
        source_identity={str(p.relative_to(ROOT)):sha(p) for p in (FOLDER/'no-mark-history.json',ROOT/'bench/rhythm-v2/c08-no-mark-history-v1.json',ART/'data-identity.json',ART/'quality-1000.pt')},
        neural_calls=0,optimizer_updates=0,cloud_calls=0,analysis_wall_s=time.monotonic()-started)
    write(FOLDER/'terminal-analysis.json',result)
    print(json.dumps(dict(gates=gates,gate_details=tests,new={t:v['mean'] for t,v in new_summary.items()},nll=n,B_T1=old_prob['nll'],audio=audio,matching=result['head_matching_summary']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

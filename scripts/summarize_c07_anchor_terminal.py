"""Recompute fixed mark-chain terminal and diagnose teacher/self-prefix gap."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain,KINDS
from beatfcn.train.anchor_marks import teacher_example
from scripts.run_c07_anchor_marks import device
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1'


def component_values(rows,key):
    groups=defaultdict(list)
    for row in rows:groups[row['component']].append(row['scores'][key]['f1'])
    return {c:float(np.mean(v)) for c,v in groups.items()}


def oracle_components(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['component']].append(row)
    return {c:dict(events=sum(x['events'] for x in v),nll=sum(x['events']*x['type_nll'] for x in v)/sum(x['events'] for x in v),
        prior=sum(x['events']*x['prior_nll'] for x in v)/sum(x['events'] for x in v)) for c,v in groups.items()}


def mean_score(rows,key):
    return float(np.mean(list(component_values(rows,key).values())))


def main():
    started=time.monotonic();report=ROOT/'reports/local-evolution-v3/anchor-marks.json';r=json.loads(report.read_text())
    p=json.loads((ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json').read_text());e=r['evaluation']
    assert r['quality_updates']==1000 and r['effective_quality_updates']==996 and r['health_updates']==100
    assert not r['health_weights_used'] and r['health_passed'] and not r.get('error')
    assert sha(ART/'quality-1000.pt')==r['final_weight_sha256']
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
    baseline=json.loads((ROOT/p['baseline']/'quality-1000-draft.json').read_text());b_components=component_values(baseline,'note_50ms')
    machine={};all_files={}
    for condition in ('real','local_roll','song_foreign','audio_foreign'):
        draws=[];components=[]
        for draw in range(4):
            path=ART/'evaluation'/condition/f'draw{draw}-draft.json';rows=json.loads(path.read_text());all_files[str(path.relative_to(ROOT))]=sha(path)
            assert len(rows)==32
            for row,old in zip(rows,baseline):
                assert sorted(v['onset_us'] for v in row['prediction']['events'])==sorted(v['onset_us'] for v in old['prediction']['events'])
            cm=component_values(rows,'note_50ms');components.append(cm)
            totals={key:sum(row['scores']['per_type'][k][key] for row in rows for k in KINDS[1:]) for key in ('tp','pred','true')}
            draws.append(dict(typed50=float(np.mean(list(cm.values()))),full50=e['machine'][condition]['draft'][draw]['full_note_f1'],
                non_tap_pooled_f1=2*totals['tp']/max(totals['pred']+totals['true'],1),non_tap_counts=totals,
                typed10=mean_score(rows,'note_10ms')))
        mean={k:float(np.mean([v[k] for v in draws])) for k in ('typed50','typed10','full50','non_tap_pooled_f1')}
        mean['component_typed50']={c:float(np.mean([v[c] for v in components])) for c in b_components}
        machine[condition]=dict(mean=mean,draws=draws)
    oracle={}
    for condition in ('real','local_roll','song_foreign','audio_foreign'):
        rows=[v for v in e['oracle_probability'] if v['task']=='draft' and v['condition']==condition]
        oracle[condition]=dict(event_weighted_nll=sum(v['events']*v['type_nll'] for v in rows)/sum(v['events'] for v in rows),
            event_weighted_prior=sum(v['events']*v['prior_nll'] for v in rows)/sum(v['events'] for v in rows),components=oracle_components(rows))
    real=oracle['real'];audio=oracle['audio_foreign'];g=p['gate']
    component_wins=sum(v>b_components[c] for c,v in machine['real']['mean']['component_typed50'].items())
    oracle_wins=sum(v['prior']>v['nll'] for v in real['components'].values())
    audio_wins=sum(audio['components'][c]['nll']>v['nll'] for c,v in real['components'].items())
    gates=dict(machine_typed=machine['real']['mean']['typed50']>=g['typed50'],machine_full=machine['real']['mean']['full50']>=g['full50'],
        non_tap=machine['real']['mean']['non_tap_pooled_f1']>=g['non_tap_pooled'],machine_components=component_wins>=9,
        oracle_prior=real['event_weighted_prior']-real['event_weighted_nll']>=.1 and oracle_wins>=9,
        conditional_audio=audio['event_weighted_nll']-real['event_weighted_nll']>=.05 and audio_wins>=9 and
            machine['real']['mean']['typed50']>machine['audio_foreign']['mean']['typed50'],
        reference=False)
    deltas=[machine['real']['mean']['component_typed50'][c]-b_components[c] for c in sorted(b_components)]
    rng=np.random.default_rng(2026102905);resampled=np.asarray(deltas)[rng.integers(0,16,size=(10000,16))].mean(-1)
    greedy=json.loads((ROOT/'reports/local-evolution-v3/mark-decision-diagnostic.json').read_text())
    prefix=greedy['oracle_free_prefix'];free_nll=sum(v['events']*v['nll'] for v in prefix)/sum(v['events'] for v in prefix)
    trace=json.loads((ROOT/'artifacts/c07-mark-decision-diagnostic-v1/oracle-trace.json').read_text())
    trace_rows=json.loads((ROOT/'artifacts/c07-mark-decision-diagnostic-v1/oracle-events.json').read_text());free_conf=np.zeros((4,4),np.int64)
    for row,t in zip(trace_rows,trace):
        target=sorted(row['target'],key=lambda v:(v['onset_us'],KINDS.index(v['type']),v['end_us']))
        for gt,pred in zip(target,t['type_probability_under_generated_prefix']):free_conf[KINDS.index(gt['type']),int(np.argmax(pred))]+=1
    # Numerical control: batched teacher recurrence and stepwise teacher recurrence.
    panel,row=load_data();model=AnchorMarkChain().to('mps');model.load_state_dict(torch.load(ART/'quality-1000.pt',map_location='cpu',weights_only=False)['model']);model.eval();parity=[]
    with torch.no_grad():
        for index in p['indices']['dev'][:2]:
            x,y,heads=teacher_example(row(index,'draft'));x,y=device(x),device(y)
            expected=model.teacher(x,y['kind'])['kind'];features,h=model.encode(x);outputs=[]
            for j in range(len(heads)):
                previous=torch.full((1,1),4,dtype=torch.long,device='mps') if j==0 else y['kind'][:,j-1:j]
                z,h=model.gru(torch.cat((features[:,j:j+1],model.types(previous)),-1),h);outputs.append(model.kind(z))
            actual=torch.cat(outputs,1);error=float((actual-expected).abs().max());assert error<=1e-4
            parity.append(dict(index=index,max_logit_error=error))
    reference_rows=[v for v in e['oracle_probability'] if v['task']=='completion' and v['condition']=='real']
    masked=[v for v in e['oracle_probability'] if v['task']=='reference_masked' and v['condition']=='real']
    weighted=lambda values:sum(v['events']*v['type_nll'] for v in values)/sum(v['events'] for v in values)
    result=dict(id='c07-anchor-terminal-analysis-v1',status='completed',primary_unchanged=True,
        identity={str(v.relative_to(ROOT)):sha(v) for v in (report,ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json',ART/'quality-1000.pt',ART/'data-identity.json',ROOT/'reports/local-evolution-v3/mark-decision-diagnostic.json')},
        machine=machine,oracle=oracle,gates=gates,component_wins=dict(machine=component_wins,oracle_prior=oracle_wins,audio=audio_wins),
        baseline=dict(typed50=float(np.mean(list(b_components.values()))),full50=.3160745598850721,non_tap=24/653),
        generated_prefix=dict(oracle_teacher_nll=real['event_weighted_nll'],oracle_self_greedy_nll=free_nll,
            oracle_self_greedy_confusion=free_conf.tolist(),oracle_self_greedy_node_accuracy=float(np.trace(free_conf)/free_conf.sum()),
            step_teacher_numerical_replays=parity,interpretation='Same anchors/features/weights; changing previous true marks to self-generated marks increases conditional loss.Oracle truth used only for diagnostic scoring.'),
        greedy_secondary={track:{k:v[k] for k in ('full_note_f1','typed50','onset10')} for track,v in greedy['summary'].items()},
        reference=dict(true_teacher_nll=weighted(reference_rows),masked_teacher_nll=weighted(masked),
            control_failures=e['control_failures'],valid_mismatched_requests=2,total_requested=16,
            disposition='Reference claim blocked by14/16unmatched controls; do not compare2request wrong-reference average with16request real.'),
        machine_typed50_component_difference=deltas,component_bootstrap_descriptive95=np.quantile(resampled,[.025,.975]).tolist(),
        per_draw_true_anchor_full50=[v['full_note_f1'] for v in e['oracle_free'].values()],
        resource={k:r[k] for k in ('parameters','health_wall_s','quality_wall_s','evaluation_wall_s','process_wall_s','effective_quality_updates')},
        prediction_sha256=all_files,analysis_neural_calls='Two teacher-recurrence numerical controls only; no new samples or training.',wall_s=time.monotonic()-started)
    write(ROOT/'reports/local-evolution-v3/terminal-analysis-v1.json',result)
    print(json.dumps(dict(gates=gates,wins=result['component_wins'],machine=machine['real']['mean'],oracle={c:{k:v for k,v in x.items() if k!='components'} for c,x in oracle.items()},prefix=result['generated_prefix'],greedy=result['greedy_secondary']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    import os
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5);main()

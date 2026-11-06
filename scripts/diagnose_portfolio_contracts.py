"""Targeted frozen-model falsification of three observed generator failures."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_code_flow import EventCodeFlow, encode_event_code, decode_event_code
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_lattice import expectation
from beatfcn.model.noisy_proposal_refiner import generate_refined
from beatfcn.train.canonical_probability import explicit_row, probability_batch
from beatfcn.train.lattice_targets import lattice_batch
from beatfcn.train.noisy_proposals import proposal_targets, proposal_inputs
from beatfcn.train.whole_song_mel import WholeMelStore
from scripts.diagnose_whole_song_arrangement import first_components
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_lattice_experiment import fresh as lattice_model
from scripts.run_local_portfolio_scale import row_factory, write
from scripts.run_noisy_proposal_refiner_v2 import fresh as refiner_model, cached, augmented_times
from scripts.run_local_typed_canvas import score


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local_portfolio_contract_diagnostics_v2.json'
PLAN=ROOT/'bench/rhythm-v2/local-portfolio-contract-diagnostics-v2.json'
ART=ROOT/'artifacts/local-portfolio-contract-diagnostics-v2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(ident,model):
    path=ROOT/f'artifacts/{ident}/quality-1000.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['updates']!=1000:raise ValueError('Wrong terminal')
    model.load_state_dict(saved['model']);model.eval();return sha(path)


def rank_metrics(scores,labels):
    positive=np.asarray(scores)[labels];negative=np.asarray(scores)[~labels]
    auc=float(np.mean([(v>negative).mean()+.5*(v==negative).mean() for v in positive])) if len(positive) and len(negative) else None
    order=np.argsort(-np.asarray(scores),kind='stable');p=np.asarray(labels)[order];precision=np.cumsum(p)/np.arange(1,len(p)+1)
    return dict(auroc=auc,average_precision=float(precision[p].mean()) if p.any() else None,prevalence=float(labels.mean()))


def main():
    if REPORT.exists() or PLAN.exists():raise FileExistsError('No automatic repeat')
    torch.set_num_threads(2)
    plans={name:json.loads((ROOT/f'bench/rhythm-v2/{ident}.json').read_text()) for name,ident in
        [('lattice','local-lattice-complete-v1'),('flow','local-event-code-flow-v2'),('refiner','local-noisy-proposal-refiner-v2')]}
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));base=row_factory(panel,features)
    def row(*args):return explicit_row(base(*args))
    indices=first_components(panel,plans['flow']['indices']['dev'],4)
    empty=plans['flow']['indices']['empty_dev']
    plan=dict(id='local-portfolio-contract-diagnostics-v2',created_at=datetime.now(timezone.utc).isoformat(),device='cpu',threads=2,
        optimizer_updates=0,dev_indices=indices,empty_indices=empty,flow_steps=[16,64],refiner_train_prior_requests=128,
        refiner_dev_indices=plans['refiner']['indices']['dev'],wall_s=900,
        hypotheses=['Lattice all-zero hard output despite nonzero expected counts/ranking','Flow count coordinate unresolved relative to its .0625 nearest-code boundary; attribute/background dominated MSE',
                    'Balanced refiner action logits overassign holds; train-only frequency correction is a diagnostic, not a calibrated posterior or new official readout'],
        flow_oracle='Replace only terminal count-coordinate by hidden truecounts; all remaining code values unchanged. Excluded from lawful quality.',
        refiner_bias='Logfrequency from first128actual scheduled augmented trainingrequests, frozen before readingdev results; no threshold/temperature sweep. Per-request class balancing makes this only a prior-proxy intervention.',
        primary_results_unchanged=True,cloud_compute_calls=0,source_sha256=sha(__file__),
        original_plan_sha256={k:sha(ROOT/f'bench/rhythm-v2/{v["id"]}.json') for k,v in plans.items()})
    write(PLAN,plan);ART.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    result=dict(status='running',plan_sha256=sha(PLAN),lattice=[],flow=[],refiner={},optimizer_updates=0,cloud_compute_calls=0)
    write(REPORT,result)
    def bound():
        if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Contract diagnostic wall')
    try:
        p=plans['lattice'];model=lattice_model(p,'cpu');result['lattice_weight']=load(p['id'],model)
        with torch.no_grad():
            for i in indices+empty:
                bound();e=row(i,'draft');inp,tgt=lattice_batch(e,device='cpu');out=model(inp);valid=inp['lattice_valid'][0];prob=out['presence'][0].sigmoid()[valid].numpy();labels=(tgt['count'][0,valid]>0).numpy();total,_=expectation(out,inp)
                result['lattice'].append(dict(index=i,positive_frames=int(labels.sum()),true_events=int(tgt['count'].sum()),
                    min_presence=float(prob.min()),mean_presence=float(prob.mean()),max_presence=float(prob.max()),above_half=int((prob>=.5).sum()),
                    expected_events=float(total),mean_positive_presence=float(prob[labels].mean()) if labels.any() else None,
                    mean_negative_presence=float(prob[~labels].mean()),**rank_metrics(prob,labels)))
        write(REPORT,result);print('Lattice probability audit complete',flush=True)
        p=plans['flow'];model=EventCodeFlow(FullEventConfig(**p['model']),density_reference=p['density_reference']).eval();result['flow_weight']=load(p['id'],model)
        raw=[]
        with torch.no_grad():
            for i in indices+empty:
                bound();e=row(i,'draft');inp,tgt=probability_batch([e],'cpu');true=encode_event_code(tgt,e['request']);audio=model.encode_audio(inp)
                encoded_roundtrip=decode_event_code(true,inp,e['request'])
                if encoded_roundtrip!=sorted(e['targets']):raise ValueError('Reversible code no longer exact')
                seed=p['seed']+17*i;noise=torch.from_numpy(np.random.default_rng(seed).standard_normal(tuple(true.shape)).astype(np.float32))
                active=torch.arange(16)<tgt['count'][...,None]
                diagnostic=dict(index=i,true_events=len(e['targets']),active_slot_fraction=float(active.float().mean()),count_coordinate_weight=1/81,
                    count_coordinate_half_bin=.0625,exact_code_roundtrip=True,velocity=[])
                for t in (.25,.5,.75,.95):
                    value=torch.tensor([t]);state=(1-t)*noise+t*true;v=model(inp,state,value,audio=audio);squared=(v-(true-noise)).square()
                    diagnostic['velocity'].append(dict(t=t,count_mse=float(squared[...,0].mean()),attribute_mse=float(squared[...,1:].mean()),
                        count_share_of_total_error=float(squared[...,0].sum()/squared.sum())))
                for steps in plan['flow_steps']:
                    state=noise.clone()
                    for j in range(steps):state=state+model(inp,state,torch.tensor([j/steps]),audio=audio)/steps
                    pred=decode_event_code(state,inp,e['request']);field=state[...,0]
                    zero=field[tgt['count']==0];oracle=state.clone();oracle[...,0]=true[...,0]
                    op=decode_event_code(oracle,inp,e['request'])
                    diagnostic[str(steps)]=dict(count_code_mse=float((field-true[...,0]).square().mean()),
                        zero_group_code_error_mean=float((zero+1).mean()),zero_group_code_error_std=float((zero+1).std()),
                        zero_group_crosses_first_count_boundary=float((zero+1>.0625).float().mean()),
                        predicted=len(pred),scores=score(pred,e['targets']),oracle_count_predicted=len(op),oracle_count_scores=score(op,e['targets']))
                    raw.append(dict(index=i,steps=steps,oracle=False,events=[asdict(v) for v in pred]))
                    raw.append(dict(index=i,steps=steps,oracle=True,events=[asdict(v) for v in op]))
                result['flow'].append(diagnostic);write(REPORT,result);write(ART/'flow-events.json',raw)
        print('Flow endpoint/coordinate audit complete',flush=True)
        p=plans['refiner'];cache=json.loads((ROOT/'reports/local_noisy_proposal_cache_v1.json').read_text());counts=Counter();requests=0
        for step,specs in enumerate(p['sample_stream'][:64],1):
            for j,spec in enumerate(specs):
                bound();e=row(spec['index'],spec['view'],spec['field_index']);times,_=augmented_times(e,cache,p['seed']+17*step+j)
                if times:
                    target,_=proposal_targets(times,e['targets'],e['request']);counts.update(target['kind'].flatten().tolist())
                requests+=1
        frequency=np.array([counts[k]+1 for k in range(5)],float);frequency/=frequency.sum()
        model=refiner_model(p,'cpu');result['refiner_weight']=load(p['id'],model)
        store=WholeMelStore(ROOT,panel.features.records,ROOT/'artifacts/local-noisy-proposal-refiner-v2/whole-mel')
        old_raw={r['index']:r for r in json.loads((ROOT/'artifacts/local-noisy-proposal-refiner-v2/draft-final.json').read_text())}
        diagnostic=[];raw=[]
        with torch.no_grad():
            for i in plan['refiner_dev_indices']:
                bound();e=row(i,'draft');source,times=cached(e,cache);inp,_=store.batch(e,panel.records[i]['audio_sha256'],device='cpu');proposal=proposal_inputs(times,e['request'],'cpu')
                output=model(inp,proposal);baseline=generate_refined(model,inp,proposal,e['request']);class0=output['kind'][0].argmax(-1)
                bias=output['kind'].new_tensor(np.log(frequency));hook=model.kind.register_forward_hook(lambda m,a,o:o+bias)
                try:corrected=generate_refined(model,inp,proposal,e['request'])
                finally:hook.remove()
                target,_=proposal_targets(times,e['targets'],e['request'])
                truth=target['kind'][0];class1=(output['kind'][0]+bias).argmax(-1)
                matrix0=np.zeros((5,5),int);matrix1=matrix0.copy()
                for t,a,b in zip(truth.tolist(),class0.tolist(),class1.tolist()):matrix0[t,a]+=1;matrix1[t,b]+=1
                diagnostic.append(dict(index=i,cpu_exact=[asdict(v) for v in baseline]==old_raw[i]['prediction']['events'],
                    source_proposals=len(times),source_scores=score([__import__('beatfcn.model.event_canvas',fromlist=['RhythmEvent']).RhythmEvent(**v) for v in source['events']],e['targets']),
                    baseline_scores=score(baseline,e['targets']),corrected_scores=score(corrected,e['targets']),
                    typeblind_matched_proposal_confusion_original=matrix0.tolist(),typeblind_matched_proposal_confusion_trainbias=matrix1.tolist()))
                raw.append(dict(index=i,baseline=[asdict(v) for v in baseline],train_frequency_bias=[asdict(v) for v in corrected]))
        result['refiner']=dict(training_requests=requests,training_label_counts=dict(counts),frequency=frequency.tolist(),
            correction_is_diagnostic_not_calibrated=True,records=diagnostic)
        write(ART/'refiner-events.json',raw)
        result.update(status='completed',wall_s=time.monotonic()-started,raw_sha256={p.name:sha(p) for p in ART.glob('*.json')});write(REPORT,result)
        print(json.dumps(dict(status='completed',wall_s=result['wall_s'],refiner_cpu_exact=sum(r['cpu_exact'] for r in diagnostic))),flush=True)
    except BaseException as exc:
        result.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}');write(REPORT,result);raise


if __name__=='__main__':main()

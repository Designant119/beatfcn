"""Matched raw free continuations for one parent-bound output adapter."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.conditional_rl import hold_request_axes, tail_prior_feasible, paired_tail_feasible
from beatfcn.train.consensus_rl import event_f1, FIELD_TEMPERATURE
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from beatfcn.train.process_memory import physical_footprint
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_output_policy_health import device_rows
from scripts.run_conditional_output_rl import tuples, onset_medoid


def metrics(samples, reference, band):
    scored=[]
    for s in samples:
        events=s['events']; axes=hold_request_axes(tuples(events),tuples(reference),band or [0.,1.])
        holds=[e for e in events if e['type']=='hold']
        scored.append(dict(**axes,typed_agreement=event_f1(tuples(events),tuples(reference)),
            count=len(events),hold_count=len(holds),one_ms_holds=sum(e['end_us']-e['onset_us']<=1000 for e in holds),
            type_counts={k:sum(e['type']==k for e in events) for k in ('tap','hold','flick','drag')}))
    return scored


def compare(baseline, candidate, references, criteria):
    """Paired same-song screen. This is not an independent quality gate."""
    rows=[]
    for cid,reference in references.items():
        arms={}
        for label,band in [('native_unspecified',None),('native_low_hold',[.02,.05]),('native_high_hold',[.3,.4]),('zero_audio_high_hold',[.3,.4])]:
            b=metrics(baseline[cid,label]['samples'],reference,band)
            c=metrics(candidate[cid,label]['samples'],reference,band)
            avg=lambda a,k:float(np.mean([s[k] for s in a]))
            arms[label]=dict(baseline=b,candidate=c,baseline_distance=-avg(b,'request_fit'),candidate_distance=-avg(c,'request_fit'),
                onset_delta=avg(c,'onset_agreement')-avg(b,'onset_agreement'),typed_delta=avg(c,'typed_agreement')-avg(b,'typed_agreement'),
                count_ratio=avg(c,'count')/max(avg(b,'count'),1.))
        rows.append(dict(chart_id=cid,mode=candidate[cid,'native_high_hold']['mode'],arms=arms))
    failures=[];mode_results={}
    for mode in sorted({r['mode'] for r in rows}):
        selected=[r for r in rows if r['mode']==mode]
        avg=lambda label,key:float(np.mean([r['arms'][label][key] for r in selected]))
        base=avg('native_high_hold','baseline_distance');cand=avg('native_high_hold','candidate_distance')
        mode_results[mode]=dict(high_baseline_distance=base,high_candidate_distance=cand)
        if cand>base*(1-criteria['minimum_high_distance_reduction_fraction'])+1e-12:failures.append(mode+':high_request_gain')
        for label in ('native_unspecified','native_low_hold','native_high_hold'):
            if avg(label,'onset_delta') < -criteria['maximum_onset_agreement_drop']:failures.append(mode+':'+label+':onset_drop')
        if avg('native_low_hold','candidate_distance')-avg('native_low_hold','baseline_distance')>criteria['maximum_low_distance_increase']:failures.append(mode+':low_request_harm')
        if avg('native_unspecified','typed_delta') < -criteria['maximum_unspecified_typed_f1_drop']:failures.append(mode+':unspecified_type_harm')
    return dict(scope='same training-song native-prefix screen; no independent quality claim',passed=not failures,failures=failures,by_mode=mode_results,rows=rows)


def run(plan):
    p=json.loads(Path(plan).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,optimizer_updates=0,plan_sha256=file_sha(plan),cloud_cost_yuan=0);resources=[]
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
    def check(stage):
        gc.collect();torch.mps.empty_cache();torch.mps.synchronize();mem=physical_footprint();resources.append(dict(stage=stage,at=time.time(),**mem));save('resources.json',resources)
        if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']:raise MemoryError('Physical footprint cap')
        if time.time()>p['deadline_unix']:raise TimeoutError('Absolute evaluation deadline')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted':raise RuntimeError('Storage admission failed')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest:raise ValueError('Identity changed: '+path)
        torch.set_num_threads(4);model=load_checkpoint_model(p['checkpoint'],'mps');policy=OutputOnlyPolicy(model,p['checkpoint_sha256']);policy.load(p['adapter'])
        provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        baseline={(r['chart_id'],r['condition']):r for r in map(json.loads,Path(p['raw_outputs']).read_text().splitlines())};candidate={}
        references={item['chart_id']:baseline[item['chart_id'],'native_unspecified']['samples'][onset_medoid(baseline[item['chart_id'],'native_unspecified']['samples'])]['events']for item in p['rows']}
        limits={r['chart_id']:r['threshold'] for r in json.loads(Path(p['calibration']).read_text())}
        for item in p['rows']:
            for label,band in p['conditions'].items():
                check(item['chart_id']+':'+label);inputs,req,provenance=provider.build(item,band)
                if label=='zero_audio_high_hold':inputs={k:np.zeros_like(v) if k in ('local','song','mel') else v for k,v in inputs.items()}
                started=time.monotonic();samples=[];tails=[];short_tails=[]
                for seed in p['seeds']:
                    inp,empty=device_rows([dict(inputs=inputs,sequence=provider.sequence([],req))]);del empty
                    with torch.no_grad():
                        sample=generate_batch(model,inp,[req],p['max_tokens'],deadline=time.monotonic()+min(p.get('sample_timeout_s',60.),max(1.,p['deadline_unix']-time.time()-10)),
                            temperature=1.,field_temperature=FIELD_TEMPERATURE,generator=[torch.Generator(device='mps').manual_seed(seed)],cuda_graph=False)[0]
                    del inp;samples.append(sample)
                    with (out/'raw-samples.jsonl').open('a') as f:f.write(json.dumps(dict(chart_id=item['chart_id'],condition=label,seed=seed,sample=sample))+'\n');f.flush()
                    check('sample_generated')
                    if not sample['complete']:raise RuntimeError('Incomplete candidate output')
                    inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(sample['events'],req))]);h,logits=policy.features(inp,seq)
                    tails.append(tail_nll(policy.prior(h),seq)[0]);del inp,seq,h,logits;check('sample_replayed')
                    if p.get('tail_guard')=='paired':
                        short=[dict(e,end_us=e['onset_us']+1000)if e['type']=='hold'else e for e in sample['events']]
                        inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(short,req))]);h,logits=policy.features(inp,seq)
                        short_tails.append(tail_nll(policy.prior(h),seq)[0]);del inp,seq,h,logits;check('tail_counterfactual_replayed')
                record=dict(chart_id=item['chart_id'],mode=item['mode'],condition=label,band=band,seeds=p['seeds'],samples=samples,provenance=provenance,wall_s=time.monotonic()-started)
                # Preserve partial raw outputs before any completeness or replay gate raises.
                with (out/'outputs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
                if not all(s['complete'] for s in samples):raise RuntimeError('Incomplete candidate output')
                record['tail_nll']=tails
                if p.get('tail_guard')=='paired':
                    record['short_tail_nll']=short_tails
                    record['tail_feasible']=[paired_tail_feasible(v,u,sum(e['type']=='hold'for e in s['events']),p['tail_minimum_margin'])for v,u,s in zip(tails,short_tails,samples)]
                else:
                    record['tail_feasible']=[tail_prior_feasible(v,sum(e['type']=='hold'for e in s['events']),limits[item['chart_id']])for v,s in zip(tails,samples)]
                candidate[item['chart_id'],label]=record
                print(json.dumps(dict(chart_id=item['chart_id'],condition=label,complete=len(samples),wall_s=record['wall_s'],tail_feasible=record['tail_feasible'])),flush=True)
        scored=compare(baseline,candidate,references,p['criteria'])
        for (cid,label),rec in candidate.items():
            if label=='native_high_hold' and not all(rec['tail_feasible']):scored['failures'].append(cid+':tail_guard_harm')
        scored['passed']=not scored['failures'];save('scores.json',scored);save('tail-scores.json',[dict(chart_id=cid,condition=label,tail_nll=r['tail_nll'],tail_feasible=r['tail_feasible'])for (cid,label),r in candidate.items()])
        result.update(completed=True,samples=sum(len(r['samples'])for r in candidate.values()),screen_passed=scored['passed'],failures=scored['failures'],raw_sha256=file_sha(out/'outputs.jsonl'),scores_sha256=file_sha(out/'scores.json'))
        check('finished');policy.close()
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}');raise
    finally:
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

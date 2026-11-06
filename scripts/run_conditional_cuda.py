"""Portable batched conditional RL with frozen-prior guards and paired validation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.cloud.gpu_util import GpuUtil
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch, _EagerStep
from beatfcn.train.conditional_rl import hold_request_axes, paired_tail_feasible, pareto_preferences
from beatfcn.train.consensus_rl import FIELD_TEMPERATURE, policy_terms
from beatfcn.train.counterfactual_exposure import condition_inputs
from beatfcn.train.local_panel import collate
from beatfcn.train.output_only_policy import OutputOnlyPolicy, save_interrupted_output
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evaluate_conditional_output import compare
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_conditional_output_rl import onset_medoid, tuples
from scripts.run_output_policy_health import ce


CONDITIONS = [('native_unspecified', None), ('native_low_hold', [.02,.05]),
              ('native_high_hold', [.3,.4]), ('zero_audio_high_hold', [.3,.4])]


def cpu_cuda_parity(policy, hidden, sequence, learning_rate):
    records = {}
    for device in ('cpu', 'cuda'):
        weight = policy.base.detach().to(device).clone().requires_grad_(True)
        h = hidden[:2,:128].detach().to(device)
        seq = {k:v[:2,:128].to(device) for k,v in sequence.items()}
        optimizer = torch.optim.AdamW([weight], lr=learning_rate, weight_decay=0., foreach=False)
        prior = F.linear(h, weight.detach())
        pg, kl = policy_terms(F.linear(h,weight), prior, seq, torch.tensor([1.,-1.],device=device))
        loss = pg+.2*kl;loss.backward()
        gradient=weight.grad.detach().cpu().clone()
        torch.nn.utils.clip_grad_norm_([weight],1.,error_if_nonfinite=True)
        optimizer.step()
        records[device]=(float(loss.detach()),gradient,weight.detach().cpu().clone())
    a,b=records['cpu'],records['cuda']
    result=dict(loss_error=abs(a[0]-b[0]), gradient_max_error=float((a[1]-b[1]).abs().max()),
                adam_max_error=float((a[2]-b[2]).abs().max()))
    result['passed']=max(result.values())<=1e-4
    return result


def run(plan_path):
    p=json.loads(Path(plan_path).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,updates=0,plan_sha256=file_sha(plan_path),quality_eligible=False)
    policy=None;monitor=None;intervals=[];started=None;last_sample=0.;stable_started=False
    def save(name, value):
        temp=out/(name+'.partial');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(out/name)
    def append(name, value):
        with (out/name).open('a') as f:f.write(json.dumps(value)+'\n');f.flush()
    def check(*,final=False):
        nonlocal last_sample, stable_started
        now=time.time()
        if now>=p['deadline_unix']:raise TimeoutError('Absolute lifecycle deadline')
        if monitor is None or (not final and now-last_sample<30):return
        interval=monitor.take();last_sample=now
        # Discard only the preregistered warmup; all later pipeline gaps count.
        if now-started<480:return
        if not stable_started:
            stable_started=True;return
        if interval['util_mean'] is None:raise RuntimeError('GPU telemetry missing')
        interval['at']=now;intervals.append(interval);append('gpu-util.jsonl',interval)
        if len(intervals)>=8 and np.mean([v['util_mean'] for v in intervals[-8:]])<p['minimum_gpu_util']:
            raise RuntimeError('GPU last-eight utilization below owner gate')
    try:
        storage=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',storage)
        if storage['status']!='admitted':raise RuntimeError('Storage not admitted')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest:raise ValueError('Changed identity: '+path)
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=load_checkpoint_model(p['checkpoint'],'cuda')
        policy=OutputOnlyPolicy(model,p['checkpoint_sha256'])
        manifest_path=Path(p['inputs_manifest']);manifest=json.loads(manifest_path.read_text())
        rows=[]
        for row in manifest['rows']:
            path=manifest_path.parent/row['inputs']
            if file_sha(path)!=row['inputs_sha256']:raise ValueError('Input identity changed')
            req=dict(row['request']);req['known_carry']=tuple(RhythmEvent(**e)for e in req['known_carry'])
            req['copy_times']=tuple(req['copy_times'])
            if req['allowed_types'] is not None:req['allowed_types']=tuple(req['allowed_types'])
            with np.load(path,allow_pickle=False)as a:inputs={k:a[k].copy()for k in a.files}
            if not all(np.isfinite(v).all()for v in inputs.values()):raise ValueError('Nonfinite input')
            rows.append(dict(row,request=EventRequest(**req),arrays=inputs))
        train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
        if len(train)!=16 or len(val)!=8:raise ValueError('Frozen panel size changed')
        if {r['item']['component'] for r in train}&{r['item']['component'] for r in val}:raise ValueError('Split leak')
        def inputs(row, label, band):
            req=row['request'];v=dict(row['arrays'],**condition_inputs(req.mode,req,bands={}if band is None else {'hold_head_ratio':band}))
            if label=='zero_audio_high_hold':
                v=dict(v,**{k:np.zeros_like(v[k])for k in ('local','song','mel')})
            return v
        def sequence(events,req):
            return teacher_sequence(encode_events([RhythmEvent(**e)for e in events],req,model.vocab),req,model.vocab)
        def batch(specs):
            inp,seq=collate([dict(inputs=inputs(r,label,band),sequence=sequence(events,r['request']))
                             for r,label,band,events in specs])
            return {k:v.cuda()for k,v in inp.items()},{k:v.cuda()for k,v in seq.items()}
        def generate(panel,label,band,seed,tag):
            check();inp,_=batch([(r,label,band,[])for r in panel])
            requests=[r['request']for r in panel for _ in range(4)]
            generators=[torch.Generator(device='cuda').manual_seed(seed+j)for r in panel for j in range(4)]
            with torch.no_grad():
                samples=generate_batch(model,inp,requests,p['max_tokens'],
                    deadline=time.monotonic()+min(p['generation_timeout_s'],max(1,p['deadline_unix']-time.time()-15)),
                    temperature=1.,field_temperature=FIELD_TEMPERATURE,generator=generators,
                    memory_index=np.repeat(np.arange(len(panel)),4),cuda_graph=True,on_step=check)
            records=[]
            for i,r in enumerate(panel):
                record=dict(chart_id=r['item']['chart_id'],mode=r['item']['mode'],condition=label,
                    samples=samples[i*4:i*4+4],seed=seed,tag=tag,updates=result['updates'])
                append('raw-outputs.jsonl',record);records.append(record)
            if not all(s['complete']for s in samples):raise RuntimeError('Incomplete raw generation')
            return records
        started=time.time();last_sample=started;monitor=GpuUtil()
        references={};train_base=generate(train,'native_unspecified',None,0,'train-reference')
        for r in train_base:references[r['chart_id']]=r['samples'][onset_medoid(r['samples'])]['events']
        # Real CUDA sampling/replay and actual objective/optimizer parity precede training.
        sample=train_base[0]['samples'];specs=[(train[0],'native_unspecified',None,s['events'])for s in sample[:2]]
        inp,sq=batch(specs);hidden,base_logits=policy.features(inp,sq)
        parity=cpu_cuda_parity(policy,hidden,sq,p['learning_rate'])
        parity['shared_prior_max_error']=float((policy.prior(hidden)-base_logits).abs().max())
        stepper=_EagerStep(model,model.encode(inp),128);prob_error=0.
        with torch.no_grad():
            for j in range(min(128,sq['tokens'].shape[1])):
                ints=np.stack([sq['tokens'][:,j].cpu().numpy(),sq['fields'][:,j].cpu().numpy()])
                logits=stepper.step(ints,sq['times'][:,j].cpu().numpy())
                allowed=sq['allowed'][:,j];valid=sq['valid'][:,j]
                error=(logits.masked_fill(~allowed,-1e9).softmax(-1)-base_logits[:,j].masked_fill(~allowed,-1e9).softmax(-1)).abs()
                if valid.any():prob_error=max(prob_error,float(error[valid].max()))
        parity['sampling_replay_probability_max_error']=prob_error
        parity['passed'] &= parity['shared_prior_max_error']<=1e-4 and prob_error<=1e-4
        save('cuda-parity.json',parity)
        if not parity['passed']:raise RuntimeError('CUDA numerical parity failed')
        health_optimizer=torch.optim.AdamW([policy.weight],lr=.001,weight_decay=0.,foreach=False)
        initial=float(ce(base_logits,sq))
        for _ in range(50):
            check();health_optimizer.zero_grad(set_to_none=True);loss=ce(policy.logits(hidden),sq);loss.backward()
            torch.nn.utils.clip_grad_norm_([policy.weight],1.,error_if_nonfinite=True);health_optimizer.step()
        final=float(ce(policy.logits(hidden),sq));policy.restore()
        save('cuda-health.json',dict(updates=50,initial_ce=initial,final_ce=final,restored=torch.equal(policy.weight,policy.base),passed=final<.02))
        if final>=.02:raise RuntimeError('CUDA output fit health failed')
        del hidden,base_logits,inp,sq,stepper,health_optimizer
        if p.get('recovery_probe'):
            probe=p['recovery_probe'];saved=policy.load(probe['adapter'])
            if saved['updates']!=13:raise ValueError('Unexpected recovery checkpoint update')
            old={r['chart_id']:r for r in map(json.loads,Path(probe['raw_outputs']).read_text().splitlines())
                 if r['tag']=='train'and r['updates']==13 and r['seed']==probe['seed']}
            new=generate(train,'native_high_hold',[.3,.4],probe['seed'],'frozen13-capacity-probe')
            matches=[];extended=[]
            for r in new:
                before=old[r['chart_id']]
                for index,(a,b)in enumerate(zip(before['samples'],r['samples'])):
                    matches.append(a['tokens']==b['tokens'][:len(a['tokens'])])
                    if not a['complete']:
                        extended.append(dict(chart_id=r['chart_id'],seed=probe['seed']+index,
                            original_tokens=len(a['tokens']),completed_tokens=len(b['tokens']),
                            events=len(b['events']),complete=b['complete']))
            receipt=dict(adapter_sha256=file_sha(probe['adapter']),prefixes_checked=len(matches),
                all_prefixes_identical=all(matches),extended=extended,optimizer_updates=0)
            save('recovery-probe.json',receipt)
            if len(matches)!=64 or not all(matches) or len(extended)!=1:
                raise RuntimeError('Frozen13 capacity isolation failed')
            policy.restore()
            if p.get('probe_only'):
                result.update(completed=True,capacity_probe_passed=True,optimizer_updates=0)
                return
        # Validation never enters a training objective or reference for a training song.
        baseline={}
        for label,band in CONDITIONS:
            for record in generate(val,label,band,0,'validation-baseline'):
                baseline[record['chart_id'],label]=record
                if band is None:references[record['chart_id']]=record['samples'][onset_medoid(record['samples'])]['events']
        save('baseline.json',list(baseline.values()))
        optimizer=torch.optim.AdamW([policy.weight],lr=p['learning_rate'],weight_decay=0.,foreach=False)
        ties=0
        for iteration in range(p['max_updates']):
            if time.time()>=p['training_deadline_unix']:break
            generated=generate(train,'native_high_hold',[.3,.4],p['seed']+iteration*4,'train')
            specs=[(r,'native_high_hold',[.3,.4],s['events'])for r,g in zip(train,generated)for s in g['samples']]
            cached=[];tails=[];short_tails=[]
            for offset in range(0,len(specs),p['replay_batch']):
                part=specs[offset:offset+p['replay_batch']];inp,sq=batch(part);h,_=policy.features(inp,sq)
                tails.extend(tail_nll(policy.prior(h),sq))
                cached.append((h.cpu(),{k:v.cpu()for k,v in sq.items()}))
                short=[(r,label,band,[dict(e,end_us=e['onset_us']+1000)if e['type']=='hold'else e for e in events])for r,label,band,events in part]
                inp,sq=batch(short);h,_=policy.features(inp,sq);short_tails.extend(tail_nll(policy.prior(h),sq));check()
                del inp,sq,h
            advantages=[];axes=[];feasible=[]
            for i,(row,g) in enumerate(zip(train,generated)):
                group=[hold_request_axes(tuples(s['events']),tuples(references[row['item']['chart_id']]),[.3,.4])for s in g['samples']]
                ok=[paired_tail_feasible(tails[i*4+j],short_tails[i*4+j],sum(e['type']=='hold'for e in s['events']),.25)for j,s in enumerate(g['samples'])]
                advantages.extend(pareto_preferences(group,feasible=ok));axes.extend(group);feasible.extend(ok)
            if not any(advantages):
                ties+=1;append('updates.jsonl',dict(iteration=iteration,updates=result['updates'],skipped='no_preference',axes=axes,feasible=feasible))
                if ties>=4:raise RuntimeError('No preference for four full panels')
                continue
            ties=0;decisions=[int((s['valid']&(s['allowed'].sum(-1)>1)).sum())for h,s in cached];total=max(1,sum(decisions))
            optimizer.zero_grad(set_to_none=True);pg_total=0.;kl_total=0.;offset=0
            for (h,sq),count in zip(cached,decisions):
                h=h.cuda();sq={k:v.cuda()for k,v in sq.items()};size=h.shape[0]
                pg,kl=policy_terms(policy.logits(h),policy.prior(h),sq,torch.tensor(advantages[offset:offset+size],device='cuda'))
                loss=pg*size/len(specs)+p['kl_weight']*kl*count/total;loss.backward()
                pg_total+=float(pg.detach())*size/len(specs);kl_total+=float(kl.detach())*count/total;offset+=size;check()
            if kl_total>p['max_kl']:raise RuntimeError('Pre-update KL exceeded')
            anchor_total=0.
            for offset in range(0,len(train),p['replay_batch']):
                part=train[offset:offset+p['replay_batch']]
                inp,sq=batch([(r,'native_unspecified',None,references[r['item']['chart_id']])for r in part]);h,_=policy.features(inp,sq)
                logits=policy.logits(h)
                anchor=torch.stack([ce(logits[i:i+1],{k:v[i:i+1]for k,v in sq.items()})for i in range(len(part))]).mean()
                (p['anchor_weight']*anchor*len(part)/len(train)).backward()
                anchor_total+=float(anchor.detach())*len(part)/len(train);check()
            norm=torch.nn.utils.clip_grad_norm_([policy.weight],1.,error_if_nonfinite=True)
            optimizer.step();result['updates']+=1
            after=0.
            with torch.no_grad():
                for (h,sq),count in zip(cached,decisions):
                    h=h.cuda();sq={k:v.cuda()for k,v in sq.items()}
                    _,kl=policy_terms(policy.logits(h),policy.prior(h),sq,torch.zeros(h.shape[0],device='cuda'))
                    after+=float(kl)*count/total
            record=dict(iteration=iteration,updates=result['updates'],optimizer_steps=int(optimizer.state[policy.weight]['step']),
                applied_training_requests=result['updates']*len(train),rollouts=len(specs),pg=pg_total,kl=kl_total,after_kl=after,
                anchor_ce=anchor_total,gradient_norm=float(norm),axes=axes,feasible=feasible,advantages=advantages,
                tail_nll=tails,short_tail_nll=short_tails,at=time.time())
            append('updates.jsonl',record);print(json.dumps({k:v for k,v in record.items()if k not in ('axes','feasible','advantages','tail_nll','short_tail_nll')}),flush=True)
            if after>p['max_kl']:raise RuntimeError('Post-update KL exceeded')
            if result['updates']%25==0:
                policy.save(out/f"output-{result['updates']}.pt",updates=result['updates'],metadata={'plan_sha256':result['plan_sha256'],'quality_eligible':False})
                if p.get('interim_validation'):
                    interim={}
                    for label,band in CONDITIONS:
                        for r in generate(val,label,band,0,'validation-update-'+str(result['updates'])):
                            interim[r['chart_id'],label]=r
                    scored=compare(baseline,interim,{r['item']['chart_id']:references[r['item']['chart_id']]for r in val},p['evaluation'])
                    scored['scope']='Fixed independent validation windows; repeated selection use, not final TEST'
                    scored['checkpoint_sha256']=file_sha(out/f"output-{result['updates']}.pt")
                    save(f"validation-{result['updates']}.json",scored)
                    for mode in {r['mode']for r in scored['rows']}:
                        selected=[r for r in scored['rows']if r['mode']==mode]
                        harms=[float(np.mean([r['arms'][label]['onset_delta']for r in selected]))
                               for label in ('native_unspecified','native_low_hold','native_high_hold')]
                        harms.append(float(np.mean([r['arms']['native_unspecified']['typed_delta']for r in selected])))
                        if min(harms)<-p['interim_max_retention_drop']:
                            raise RuntimeError('Interim independent retention stop: '+mode)
            del cached;check()
        if result['updates']==0:raise RuntimeError('No actual RL updates within training deadline')
        policy.save(out/'candidate-output.pt',updates=result['updates'],metadata={'plan_sha256':result['plan_sha256'],'quality_eligible':False})
        result['candidate_sha256']=file_sha(out/'candidate-output.pt')
        candidate={}
        for label,band in CONDITIONS:
            for record in generate(val,label,band,0,'validation-candidate'):candidate[record['chart_id'],label]=record
        scores=compare(baseline,candidate,{r['item']['chart_id']:references[r['item']['chart_id']]for r in val},p['evaluation'])
        scores['scope']='8 independent validation songs, native-prefix windows; whole-song quality untested'
        save('validation-scores.json',scores)
        # Tail guard is measured on every validation high-hold output, not inferred from training.
        validation_tails=[]
        for row in val:
            events=[s['events']for s in candidate[row['item']['chart_id'],'native_high_hold']['samples']]
            inp,sq=batch([(row,'native_high_hold',[.3,.4],e)for e in events]);h,_=policy.features(inp,sq);a=tail_nll(policy.prior(h),sq)
            inp,sq=batch([(row,'native_high_hold',[.3,.4],[dict(e,end_us=e['onset_us']+1000)if e['type']=='hold'else e for e in ev])for ev in events]);h,_=policy.features(inp,sq);b=tail_nll(policy.prior(h),sq)
            validation_tails.append(dict(chart_id=row['item']['chart_id'],original=a,short=b,feasible=[paired_tail_feasible(x,y,sum(e['type']=='hold'for e in ev),.25)for x,y,ev in zip(a,b,events)]));check()
        save('validation-tails.json',validation_tails)
        original=torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        changed=[k for k,v in model.state_dict().items()if k!='output.weight'and not torch.equal(v.cpu(),original['model'][k])]
        trained=policy.weight.detach().clone();policy.restore();policy.load(out/'candidate-output.pt')
        if changed or not torch.equal(trained,policy.weight):raise RuntimeError('Trunk/reload identity failed')
        check(final=True)
        stable_mean=sum(v['util_mean']*v['samples']for v in intervals)/max(1,sum(v['samples']for v in intervals))
        last_eight=float(np.mean([v['util_mean']for v in intervals[-8:]]))if len(intervals)>=8 else None
        resource_pass=last_eight is not None and min(stable_mean,last_eight)>=p['minimum_gpu_util']
        result.update(completed=True,optimizer_steps=int(optimizer.state[policy.weight]['step']),frozen_changed_keys=changed,
            independent_screen_passed=scores['passed']and all(all(r['feasible'])for r in validation_tails),
            gpu_stable_mean=stable_mean,gpu_last_eight_mean=last_eight,resource_passed=resource_pass)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if policy is not None and result['updates'] and not result['completed']:
            try:
                result.update(save_interrupted_output(policy,out/'interrupted-output.pt',updates=result['updates'],plan_sha256=result['plan_sha256']))
            except Exception as exc:
                result['interrupted_save_error']=f'{type(exc).__name__}: {exc}'
        result['gpu_intervals']=len(intervals)
        if intervals:
            result['gpu_stable_mean']=sum(v['util_mean']*v['samples']for v in intervals)/sum(v['samples']for v in intervals)
            result['gpu_last_eight_mean']=float(np.mean([v['util_mean']for v in intervals[-8:]]))if len(intervals)>=8 else None
        if monitor is not None:monitor.close()
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    run(parser.parse_args().plan)

"""Memory recovery probe: sequential rollouts/replays, one update per frozen group."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.conditional_rl import hold_request_axes, pareto_preferences, tail_prior_feasible
from beatfcn.train.consensus_rl import policy_terms, FIELD_TEMPERATURE
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from beatfcn.train.process_memory import physical_footprint
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_conditional_output_rl import tuples, onset_medoid
from scripts.run_output_policy_health import device_rows, ce


def run(plan):
    p=json.loads(Path(plan).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,updates=0,plan_sha256=file_sha(plan),scope='memory recovery mechanism probe, not original12-update success');resources=[]
    def save(n,v): (out/n).write_text(json.dumps(v,indent=2)+'\n')
    def append(n,v):
        with (out/n).open('a')as f:f.write(json.dumps(v)+'\n');f.flush()
    def check(stage):
        gc.collect();torch.mps.empty_cache();torch.mps.synchronize();mem=physical_footprint();resources.append(dict(stage=stage,at=time.time(),**mem));save('resources.json',resources)
        if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']:raise MemoryError('Physical footprint cap')
        if time.time()>p['deadline_unix']:raise TimeoutError('Original absolute deadline reached')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted':raise RuntimeError('Storage not admitted')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest:raise ValueError('Changed identity: '+path)
        torch.set_num_threads(4);model=load_checkpoint_model(p['checkpoint'],'mps');policy=OutputOnlyPolicy(model,p['checkpoint_sha256']);provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        raw={(r['chart_id'],r['condition']):r for r in map(json.loads,Path(p['raw_outputs']).read_text().splitlines())}
        limits={r['chart_id']:r['threshold']for r in json.loads(Path(p['calibration']).read_text())}
        optimizer=torch.optim.AdamW([policy.weight],lr=p['learning_rate'],weight_decay=0.,foreach=False);check('loaded')
        for group,item in enumerate(p['rows']):
            cid=item['chart_id'];inputs,req,_=provider.build(item,p['band']);samples=[];cached=[]
            for i in range(p['group_size']):
                seed=p['seed']+group*p['group_size']+i
                inp,empty=device_rows([dict(inputs=inputs,sequence=provider.sequence([],req))]);del empty
                check('before_generation')
                with torch.no_grad():
                    sample=generate_batch(model,inp,[req],p['max_tokens'],deadline=time.monotonic()+min(60.,max(1.,p['deadline_unix']-time.time()-10)),temperature=1.,field_temperature=FIELD_TEMPERATURE,generator=[torch.Generator(device='mps').manual_seed(seed)],cuda_graph=False)[0]
                append('rollouts.jsonl',dict(group=group,chart_id=cid,seed=seed,sample=sample));samples.append(sample);del inp
                check('after_generation')
                if not sample['complete']:raise RuntimeError('Incomplete generation')
                inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(sample['events'],req))]);h,logits=policy.features(inp,seq);del inp,logits
                prior=policy.prior(h);tail=tail_nll(prior,seq)[0]
                cached.append((h.cpu(),{k:v.cpu()for k,v in seq.items()},tail));del h,seq,prior;check('after_single_replay')
            ref_samples=raw[cid,'native_unspecified']['samples'];reference=ref_samples[onset_medoid(ref_samples)]['events']
            axes=[hold_request_axes(tuples(s['events']),tuples(reference),p['band'])for s in samples]
            tails=[c[2]for c in cached];feasible=[tail_prior_feasible(v,sum(e['type']=='hold'for e in s['events']),limits[cid])for v,s in zip(tails,samples)]
            advantages=pareto_preferences(axes,feasible=feasible)
            if not any(advantages):raise RuntimeError('No informative preference in fixed recovery group')
            decisions=[int((sq['valid']&(sq['allowed'].sum(-1)>1)).sum())for _,sq,_ in cached];total=max(sum(decisions),1)
            optimizer.zero_grad(set_to_none=True);pg_sum=0.;kl_sum=0.
            for index,(h,sq,_)in enumerate(cached):
                h=h.to('mps');sq={k:v.to('mps')for k,v in sq.items()};prior=policy.prior(h)
                pg,kl=policy_terms(policy.logits(h),prior,sq,torch.tensor([advantages[index]],device='mps'))
                weight=decisions[index]/total
                loss=pg/len(cached)+p['kl_weight']*kl*weight;loss.backward()
                pg_sum+=float(pg.detach())/len(cached);kl_sum+=float(kl.detach())*weight
                del h,sq,prior,pg,kl,loss;check('after_micro_backward')
            if kl_sum>p['max_kl']:raise RuntimeError('Pre-update KL exceeded')
            anchor_inputs,_,_=provider.build(item);inp,sq=device_rows([dict(inputs=anchor_inputs,sequence=provider.sequence(reference,req))]);h,logits=policy.features(inp,sq);del inp,logits
            anchor=ce(policy.logits(h),sq);(p['anchor_weight']*anchor).backward();anchor_value=float(anchor.detach());del h,sq,anchor
            norm=torch.nn.utils.clip_grad_norm_([policy.weight],p['gradient_clip'],error_if_nonfinite=True);optimizer.step();result['updates']+=1
            after=0.
            with torch.no_grad():
                for index,(h,sq,_)in enumerate(cached):
                    h=h.to('mps');sq={k:v.to('mps')for k,v in sq.items()};pg,kl=policy_terms(policy.logits(h),policy.prior(h),sq,torch.zeros(1,device='mps'))
                    after+=float(kl)*decisions[index]/total;del h,sq,pg,kl
            rec=dict(group=group,chart_id=cid,updates=result['updates'],axes=axes,tail_nll=tails,feasible=feasible,advantages=advantages,pg=pg_sum,kl=kl_sum,after_kl=after,anchor_ce=anchor_value,gradient_norm=float(norm))
            append('updates.jsonl',rec);print(json.dumps(rec),flush=True);del cached;check('updated')
            if after>p['max_kl']:raise RuntimeError('Post-update KL exceeded')
        original=torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        changed=[k for k,v in model.state_dict().items()if k!='output.weight'and not torch.equal(v.cpu(),original['model'][k])];del original
        if changed:raise RuntimeError('Frozen trunk changed')
        policy.save(out/'candidate-output.pt',updates=result['updates'],metadata={'plan_sha256':file_sha(plan),'scope':'recovery_probe_not_promoted'})
        trained=policy.weight.detach().clone();policy.restore();policy.load(out/'candidate-output.pt')
        if not torch.equal(trained,policy.weight):raise RuntimeError('Reload failed')
        result.update(completed=True,optimizer_steps=int(optimizer.state[policy.weight]['step']),candidate_sha256=file_sha(out/'candidate-output.pt'),frozen_changed_keys=changed)
        policy.restore();policy.close();check('saved')
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}');raise
    finally:
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

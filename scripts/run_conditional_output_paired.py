"""Bounded conditional RL with paired tail guard and sequential group accumulation."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.conditional_rl import hold_request_axes, pareto_preferences, paired_tail_feasible
from beatfcn.train.consensus_rl import policy_terms, FIELD_TEMPERATURE
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy, save_interrupted_output
from beatfcn.train.process_memory import physical_footprint
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_conditional_output_rl import tuples, onset_medoid
from scripts.run_output_policy_health import device_rows, ce


def run(plan):
    p=json.loads(Path(plan).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,updates=0,plan_sha256=file_sha(plan),scope='conditional output-layer candidate; independent quality not established');resources=[];policy=None
    def save(n,v): (out/n).write_text(json.dumps(v,indent=2)+'\n')
    def append(n,v):
        with (out/n).open('a')as f:f.write(json.dumps(v)+'\n');f.flush()
    def check(stage):
        gc.collect();torch.mps.empty_cache();torch.mps.synchronize();mem=physical_footprint();resources.append(dict(stage=stage,at=time.time(),**mem));save('resources.json',resources)
        if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']:raise MemoryError('Physical footprint cap')
        if time.time()>p['deadline_unix']:raise TimeoutError('Absolute training deadline reached')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted':raise RuntimeError('Storage not admitted')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest:raise ValueError('Changed identity: '+path)
        torch.set_num_threads(4);model=load_checkpoint_model(p['checkpoint'],'mps');policy=OutputOnlyPolicy(model,p['checkpoint_sha256']);provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        raw={(r['chart_id'],r['condition']):r for r in map(json.loads,Path(p['raw_outputs']).read_text().splitlines())}
        optimizer=torch.optim.AdamW([policy.weight],lr=p['learning_rate'],weight_decay=0.,foreach=False);check('loaded')
        consecutive_ties=0
        for group in range(p['max_groups']):
            item=p['rows'][group%len(p['rows'])]
            cid=item['chart_id'];inputs,req,_=provider.build(item,p['band']);samples=[];cached=[]
            for i in range(p['group_size']):
                seed=p['seed']+group*p['group_size']+i
                inp,empty=device_rows([dict(inputs=inputs,sequence=provider.sequence([],req))]);del empty
                check('before_generation')
                with torch.no_grad():
                    sample=generate_batch(model,inp,[req],p['max_tokens'],deadline=time.monotonic()+min(p['sample_timeout_s'],max(1.,p['deadline_unix']-time.time()-15)),temperature=1.,field_temperature=FIELD_TEMPERATURE,generator=[torch.Generator(device='mps').manual_seed(seed)],cuda_graph=False)[0]
                append('rollouts.jsonl',dict(group=group,chart_id=cid,seed=seed,sample=sample));samples.append(sample);del inp
                check('after_generation')
                if not sample['complete']:raise RuntimeError('Incomplete generation')
                inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(sample['events'],req))]);h,logits=policy.features(inp,seq);del inp,logits
                prior=policy.prior(h);tail=tail_nll(prior,seq)[0]
                frozen_h=h.cpu();frozen_seq={k:v.cpu()for k,v in seq.items()};del h,seq,prior;check('after_single_replay')
                short=[dict(e,end_us=e['onset_us']+1000)if e['type']=='hold'else e for e in sample['events']]
                inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(short,req))]);h,logits=policy.features(inp,seq)
                short_tail=tail_nll(policy.prior(h),seq)[0];del inp,seq,h,logits
                cached.append((frozen_h,frozen_seq,tail,short_tail));check('after_counterfactual')
            ref_samples=raw[cid,'native_unspecified']['samples'];reference=ref_samples[onset_medoid(ref_samples)]['events']
            axes=[hold_request_axes(tuples(s['events']),tuples(reference),p['band'])for s in samples]
            tails=[c[2]for c in cached];short_tails=[c[3]for c in cached];feasible=[paired_tail_feasible(v,u,sum(e['type']=='hold'for e in s['events']),p['tail_minimum_margin'])for v,u,s in zip(tails,short_tails,samples)]
            advantages=pareto_preferences(axes,feasible=feasible)
            if not any(advantages):
                consecutive_ties+=1;append('updates.jsonl',dict(group=group,chart_id=cid,updates=result['updates'],axes=axes,tail_nll=tails,short_tail_nll=short_tails,feasible=feasible,skipped='no_pairwise_preference'))
                del cached
                if consecutive_ties>=p['max_consecutive_ties']:raise RuntimeError('No learning preference in consecutive groups')
                continue
            consecutive_ties=0
            decisions=[int((sq['valid']&(sq['allowed'].sum(-1)>1)).sum())for _,sq,_,_ in cached];total=max(sum(decisions),1)
            optimizer.zero_grad(set_to_none=True);pg_sum=0.;kl_sum=0.
            for index,(h,sq,_,_)in enumerate(cached):
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
                for index,(h,sq,_,_)in enumerate(cached):
                    h=h.to('mps');sq={k:v.to('mps')for k,v in sq.items()};pg,kl=policy_terms(policy.logits(h),policy.prior(h),sq,torch.zeros(1,device='mps'))
                    after+=float(kl)*decisions[index]/total;del h,sq,pg,kl
            rec=dict(group=group,chart_id=cid,updates=result['updates'],axes=axes,tail_nll=tails,short_tail_nll=short_tails,feasible=feasible,advantages=advantages,pg=pg_sum,kl=kl_sum,after_kl=after,anchor_ce=anchor_value,gradient_norm=float(norm))
            append('updates.jsonl',rec);print(json.dumps(rec),flush=True);del cached;check('updated')
            if after>p['max_kl']:raise RuntimeError('Post-update KL exceeded')
            if result['updates']==p['max_updates']:break
        if result['updates']!=p['max_updates']:raise RuntimeError('Group budget exhausted without planned updates')
        original=torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        changed=[k for k,v in model.state_dict().items()if k!='output.weight'and not torch.equal(v.cpu(),original['model'][k])];del original
        if changed:raise RuntimeError('Frozen trunk changed')
        policy.save(out/'candidate-output.pt',updates=result['updates'],metadata={'plan_sha256':file_sha(plan),'scope':'paired_tail_candidate_not_promoted'})
        trained=policy.weight.detach().clone();policy.restore();policy.load(out/'candidate-output.pt')
        if not torch.equal(trained,policy.weight):raise RuntimeError('Reload failed')
        check('saved')
        result.update(completed=True,optimizer_steps=int(optimizer.state[policy.weight]['step']),candidate_sha256=file_sha(out/'candidate-output.pt'),frozen_changed_keys=changed)
        policy.restore();policy.close()
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}');raise
    finally:
        # Preserve one diagnostic delta on a graceful abort; never mark it eligible.
        # The outer deadline can still kill the process, so absence remains explicit.
        if policy is not None and result['updates'] and not result['completed']:
            try:
                interrupted=out/'interrupted-output.pt'
                result.update(save_interrupted_output(policy,interrupted,updates=result['updates'],plan_sha256=file_sha(plan)))
            except Exception as save_error:
                result['interrupted_save_error']=f'{type(save_error).__name__}: {save_error}'
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

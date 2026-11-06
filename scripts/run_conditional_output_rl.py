"""Bounded on-policy output-layer mechanism experiment; no product promotion."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.conditional_rl import hold_request_axes, pareto_preferences, tail_prior_feasible
from beatfcn.train.consensus_rl import event_f1, policy_terms, FIELD_TEMPERATURE
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from beatfcn.train.process_memory import physical_footprint
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_output_policy_health import device_rows, ce


def tuples(events):
    return [(e['onset_us'],e['end_us'],e['type'],e['lane']) for e in events]


def onset_medoid(samples):
    groups=[[(e['onset_us'],e['onset_us'],'tap',-1) for e in s['events']] for s in samples]
    scores=[sum(event_f1(g,h) for h in groups) for g in groups]
    return int(np.argmax(scores))


def numerical_parity(policy, hidden, seq, lr):
    """Compare the actual masked policy objective, gradient and Adam update."""
    results={}
    for device in ('cpu','mps'):
        w=policy.base.detach().to(device).clone().requires_grad_(True)
        h=hidden.detach().to(device);s={k:v.to(device) for k,v in seq.items()}
        advantages=torch.tensor([1.,-1.],device=device)
        optimizer=torch.optim.AdamW([w],lr=lr,weight_decay=0.,foreach=False)
        logits=F.linear(h,w);pg,kl=policy_terms(logits,logits.detach().clone(),s,advantages)
        loss=pg+.2*kl+.5*ce(logits,s);loss.backward()
        grad=w.grad.detach().cpu().clone()
        torch.nn.utils.clip_grad_norm_([w],1.,error_if_nonfinite=True);optimizer.step()
        results[device]=dict(loss=float(loss.detach()),grad=grad,weight=w.detach().cpu().clone(),
                            moment=optimizer.state[w]['exp_avg'].detach().cpu().clone())
        del w,h,s,optimizer,logits,loss,pg,kl
    a,b=results['cpu'],results['mps']
    return dict(loss_abs_error=abs(a['loss']-b['loss']),
        gradient_max_error=float((a['grad']-b['grad']).abs().max()),
        adam_weight_max_error=float((a['weight']-b['weight']).abs().max()),
        adam_moment_max_error=float((a['moment']-b['moment']).abs().max()))


def run(plan):
    p=json.loads(Path(plan).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,updates=0,cloud_cost_yuan=0,plan_sha256=file_sha(plan));resources=[]
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
    def append(name,value):
        with (out/name).open('a') as f: f.write(json.dumps(value)+'\n');f.flush()
    def check(stage):
        torch.mps.synchronize();mem=physical_footprint();resources.append(dict(stage=stage,at=time.time(),**mem));save('resources.json',resources)
        if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']: raise MemoryError('Physical footprint cap')
        if time.time()>p['deadline_unix']: raise TimeoutError('Absolute RL deadline')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted': raise RuntimeError('Storage admission failed')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest: raise ValueError('Identity changed: '+path)
        calibration=json.loads(Path(p['calibration']).read_text())
        if len(calibration)!=len(p['rows']) or not all(r['passed'] for r in calibration): raise ValueError('Calibration not passed')
        limits={r['chart_id']:r['threshold'] for r in calibration}
        torch.set_num_threads(4);model=load_checkpoint_model(p['checkpoint'],'mps')
        policy=OutputOnlyPolicy(model,p['checkpoint_sha256']);provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        raw={(r['chart_id'],r['condition']):r for r in map(json.loads,Path(p['raw_outputs']).read_text().splitlines())}
        anchors={};references={};parity_rows=[]
        for item in p['rows']:
            cid=item['chart_id'];samples=raw[cid,'native_unspecified']['samples']
            if not all(s['complete'] for s in samples): raise ValueError('Incomplete anchor')
            medoid=onset_medoid(samples);events=samples[medoid]['events'];references[cid]=events
            inputs,req,_=provider.build(item);inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(events,req))])
            h,_=policy.features(inp,seq);anchors[cid]=(h.detach().cpu(),{k:v.cpu() for k,v in seq.items()})
            high,rq,_=provider.build(item,p['band'])
            parity_rows.append(dict(inputs=high,sequence=provider.sequence(raw[cid,'native_high_hold']['samples'][0]['events'],rq)))
            del inp,seq,h
        inp,seq=device_rows([parity_rows[0],parity_rows[2]]);hidden,_=policy.features(inp,seq)
        parity=numerical_parity(policy,hidden,seq,p['learning_rate']);save('policy-gradient-parity.json',parity)
        if any(v>p['parity_tolerance'] for v in parity.values()): raise ValueError('Policy/gradient/Adam parity failed')
        del inp,seq,hidden,parity_rows;gc.collect();torch.mps.empty_cache();check('preflight')
        save('references.json',references)
        optimizer=torch.optim.AdamW([policy.weight],lr=p['learning_rate'],weight_decay=0.,foreach=False)
        skipped=0
        for attempt in range(p['max_groups']):
            check('before_group_'+str(attempt));item=p['rows'][attempt%len(p['rows'])];cid=item['chart_id']
            inputs,req,provenance=provider.build(item,p['band'])
            empty=provider.sequence([],req);inp,_=device_rows([dict(inputs=inputs,sequence=empty)]*p['group_size'])
            seeds=[p['seed']+attempt*p['group_size']+i for i in range(p['group_size'])]
            with torch.no_grad():
                samples=generate_batch(model,inp,[req]*p['group_size'],p['max_tokens'],
                    deadline=time.monotonic()+min(60.,max(1.,p['deadline_unix']-time.time()-10)),temperature=1.,
                    field_temperature=FIELD_TEMPERATURE,generator=[torch.Generator(device='mps').manual_seed(s) for s in seeds],cuda_graph=False)
            append('rollouts.jsonl',dict(group=attempt,chart_id=cid,seeds=seeds,samples=samples,provenance=provenance))
            if not all(s['complete'] for s in samples): raise RuntimeError('Incomplete on-policy group')
            inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(s['events'],req)) for s in samples])
            hidden,_=policy.features(inp,seq);reference_logits=policy.prior(hidden)
            tails=tail_nll(reference_logits,seq)
            feasible=[tail_prior_feasible(v,sum(e['type']=='hold' for e in s['events']),limits[cid]) for v,s in zip(tails,samples)]
            axes=[hold_request_axes(tuples(s['events']),tuples(references[cid]),p['band']) for s in samples]
            # Pairwise group preferences sum to zero. Do not call a dependent group baseline an unbiased RLOO estimator.
            advantages=pareto_preferences(axes,feasible=feasible)
            record=dict(group=attempt,chart_id=cid,axes=axes,tail_nll=tails,feasible=feasible,advantages=advantages,updates=result['updates'])
            if not any(advantages):
                skipped+=1;record['skipped']='no_pairwise_preference';append('updates.jsonl',record)
                del inp,seq,hidden,reference_logits;gc.collect();torch.mps.empty_cache()
                if skipped>=p['max_consecutive_ties']: raise RuntimeError('Repeated groups have no learning preference')
                continue
            skipped=0;optimizer.zero_grad(set_to_none=True)
            pg,kl=policy_terms(policy.logits(hidden),reference_logits,seq,torch.tensor(advantages,device='mps'))
            ah,ase=anchors[cid];ah=ah.to('mps');ase={k:v.to('mps') for k,v in ase.items()}
            anchor_loss=ce(policy.logits(ah),ase)
            if float(kl.detach())>p['max_kl']: raise RuntimeError('Pre-update KL trust region exceeded')
            loss=pg+p['kl_weight']*kl+p['anchor_weight']*anchor_loss;loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([policy.weight],p['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();result['updates']+=1
            with torch.no_grad(): _,after_kl=policy_terms(policy.logits(hidden),reference_logits,seq,torch.zeros(len(samples),device='mps'))
            record.update(updates=result['updates'],pg=float(pg.detach()),kl=float(kl.detach()),after_kl=float(after_kl),anchor_ce=float(anchor_loss.detach()),gradient_norm=float(norm))
            append('updates.jsonl',record);print(json.dumps(record),flush=True)
            if float(after_kl)>p['max_kl']: raise RuntimeError('Post-update KL trust region exceeded; candidate rejected')
            del inp,seq,hidden,reference_logits,ah,ase,loss,pg,kl,after_kl,anchor_loss;gc.collect();torch.mps.empty_cache();check('after_update')
            if result['updates']==p['max_updates']: break
        if result['updates']!=p['max_updates']: raise RuntimeError('Insufficient informative groups within fixed group cap')
        saved=torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        changed=[k for k,v in model.state_dict().items() if k!='output.weight' and not torch.equal(v.cpu(),saved['model'][k])];del saved
        if changed: raise RuntimeError('Frozen trunk changed: '+str(changed))
        policy.save(out/'candidate-output.pt',updates=result['updates'],metadata={'plan_sha256':file_sha(plan),'scope':'mechanism_candidate_not_promoted'})
        trained=policy.weight.detach().clone();policy.restore();policy.load(out/'candidate-output.pt')
        if not torch.equal(trained,policy.weight): raise RuntimeError('Candidate reload mismatch')
        result.update(completed=True,optimizer_steps=int(optimizer.state[policy.weight]['step']),frozen_changed_keys=changed,
                      candidate_sha256=file_sha(out/'candidate-output.pt'),trainable_parameters=policy.weight.numel())
        policy.restore();policy.close();check('saved')
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}');raise
    finally:
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

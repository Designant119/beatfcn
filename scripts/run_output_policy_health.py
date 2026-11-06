"""Bounded real-model health of a small output-matrix update on local MPS."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.check_experiment_storage import admission, GIB
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from beatfcn.model.full_event_generate import generate_batch, _EagerStep
from beatfcn.train.consensus_rl import event_f1
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.local_panel import collate
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from beatfcn.train.process_memory import physical_footprint


def device_rows(rows):
    inputs, seq = collate(rows)
    return ({k:v.to('mps') for k,v in inputs.items()}, {k:v.to('mps') for k,v in seq.items()})


def ce(logits, seq):
    losses=F.cross_entropy(logits.masked_fill(~seq['allowed'],-1e9).flatten(0,1),seq['targets'].flatten(),reduction='none').view_as(seq['valid'])
    valid=seq['valid'] & (seq['allowed'].sum(-1)>1)
    return losses[valid].mean()


def run(plan_path):
    spec=json.loads(Path(plan_path).read_text());out=Path(spec['out_dir'])
    out.mkdir(exist_ok=False)
    deadline=spec['deadline_unix'];resources=[];updates=0;summary={'passed':False}
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    def check(stage):
        torch.mps.synchronize();mem=physical_footprint();resources.append(dict(stage=stage,at=time.time(),**mem))
        if mem['physical_footprint_bytes']>spec['physical_footprint_cap_bytes']:
            raise MemoryError('Physical footprint cap')
        if time.time()>deadline:
            raise TimeoutError('Absolute health deadline')
        save('resources.json',resources)
    try:
        disk=admission(out,spec['planned_peak_bytes'],spec['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted':raise RuntimeError('Storage admission failed')
        if file_sha(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for path,digest in spec['code_sha256'].items():
            if file_sha(path)!=digest:raise ValueError('Health code changed')
        torch.set_num_threads(4);model=load_checkpoint_model(spec['checkpoint'],'mps')
        policy=OutputOnlyPolicy(model,spec['checkpoint_sha256']);check('load')
        provider=FrozenLocalInputs(spec['root'],spec['source_manifest'],model)
        raw={r['chart_id']:r for r in map(json.loads,Path(spec['raw_outputs']).read_text().splitlines()) if r['condition']=='native_high_hold'}
        rows=[];requests=[];targets=[]
        for item in spec['rows']:
            inputs,request,_=provider.build(item,[.3,.4]);events=raw[item['chart_id']]['samples'][0]['events']
            rows.append(dict(inputs=inputs,sequence=provider.sequence(events,request)));requests.append(request);targets.append(events)
        inputs,seq=device_rows(rows)
        hidden,base_logits=policy.features(inputs,seq)
        prior_error=float((policy.prior(hidden)-base_logits).abs().max())
        if prior_error>spec['logit_tolerance']:raise ValueError('Shared-hidden reference parity failed')
        # Same rows and masks, static sampling decoder against the teacher replay.
        with torch.no_grad():
            memory=model.encode(inputs);stepper=_EagerStep(model,memory,128)
            max_probability_error=0.;seen=set()
            for i in range(min(128,seq['tokens'].shape[1])):
                ints=np.stack([seq['tokens'][:,i].cpu().numpy(),seq['fields'][:,i].cpu().numpy()])
                times=seq['times'][:,i].cpu().numpy()
                step=stepper.step(ints,times)
                mask=seq['allowed'][:,i];valid=seq['valid'][:,i]
                error=(step.masked_fill(~mask,-1e9).softmax(-1)-base_logits[:,i].masked_fill(~mask,-1e9).softmax(-1)).abs()
                if valid.any():max_probability_error=max(max_probability_error,float(error[valid].max()))
                seen.update(seq['fields'][:,i][valid].cpu().tolist())
            del memory,stepper
        if max_probability_error>spec['probability_tolerance']:raise ValueError('Sampling/replay probability mismatch')
        save('parity.json',dict(reference_logits_max_error=prior_error,sampling_replay_probability_max_error=max_probability_error,fields=sorted(seen),steps=128))
        # Output-only optimization reuses EXACT frozen features, without a second backbone.
        optimizer=torch.optim.AdamW([policy.weight],lr=spec['learning_rate'],weight_decay=0.,foreach=False)
        initial=float(ce(base_logits,seq));losses=[]
        for index in range(spec['updates']):
            check('before_update_'+str(index+1));optimizer.zero_grad(set_to_none=True)
            loss=ce(policy.logits(hidden),seq);loss.backward()
            if policy.weight.grad is None or not torch.isfinite(policy.weight.grad).all():raise RuntimeError('Invalid output gradient')
            norm=torch.nn.utils.clip_grad_norm_([policy.weight],1.,error_if_nonfinite=True)
            optimizer.step();updates=index+1;losses.append(float(loss.detach()))
            if updates%10==0:print(json.dumps(dict(updates=updates,loss=losses[-1],grad_norm=float(norm))),flush=True)
        final=float(ce(policy.logits(hidden),seq));check('trained')
        policy.save(out/'health-output.pt',updates=updates,metadata={'plan_sha256':file_sha(plan_path),'purpose':'health_only_not_candidate'})
        # Reload the adapter, and verify its exact logits before free generation.
        trained=policy.logits(hidden).detach();policy.restore();policy.load(out/'health-output.pt')
        reload_error=float((trained-policy.logits(hidden)).abs().max())
        with torch.no_grad():
            result=generate_batch(model,inputs,requests,1024,deadline=time.monotonic()+max(1,deadline-time.time()-10),temperature=None,cuda_graph=False)
        save('free-health.json',result);check('free')
        def tuples(events):return [(e['onset_us'],e['end_us'],e['type'],e['lane']) for e in events]
        scores=[event_f1(tuples(pred['events']),tuples(target),tolerance_us=0) for pred,target in zip(result,targets)]
        # Verify the complete frozen trunk, not just requires_grad flags.
        saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        changed=[k for k,v in model.state_dict().items() if k!='output.weight' and not torch.equal(v.cpu(),saved['model'][k])]
        del saved
        policy.restore();restore_error=float((policy.logits(hidden)-base_logits).abs().max())
        state=optimizer.state[policy.weight]
        summary=dict(passed=final<=spec['max_final_ce'] and all(s>=spec['minimum_free_f1'] for s in scores)
                     and all(r['complete'] for r in result) and not changed and reload_error==0. and restore_error<=spec['logit_tolerance'],
                     updates=updates,optimizer_steps=int(state['step']),initial_ce=initial,final_ce=final,
                     free_exact_event_f1=scores,free_complete=[r['complete'] for r in result],
                     frozen_changed_keys=changed,adapter_reload_max_error=reload_error,restore_max_error=restore_error,
                     trainable_parameters=policy.weight.numel(),training_scope='health; original output matrix restored',
                     adapter_sha256=file_sha(out/'health-output.pt'),plan_sha256=file_sha(plan_path))
        policy.close();del optimizer,model,policy,hidden,inputs,seq,trained;gc.collect();torch.mps.empty_cache()
    except Exception as exc:
        summary.update(error=f'{type(exc).__name__}: {exc}',updates=updates)
        raise
    finally:
        summary['ended_unix']=time.time();save('result.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);a=p.parse_args();run(a.plan)

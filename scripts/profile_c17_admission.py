"""Single frozen training-case timing replay, no optimizer or new dev outcome."""
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from scripts import run_masked_native_events as core
from beatfcn.train import masked_native_events as law
from beatfcn.train.managed_mps import ManagedMPS
from scripts.run_local_portfolio_scale import sha,write,weight_digest


DEST=core.ROOT/'reports/local-integration-v1/c17-masked-native-admission-profile.json'


def main():
    if DEST.exists():raise FileExistsError('No timing repeats or selection')
    core.runtime();plan=core.read(core.PLAN);receipt=core.read(core.ART/'admission.json')
    path=core.ROOT/receipt['failure_checkpoint']
    if sha(path)!=receipt['failure_checkpoint_sha256']:raise ValueError('Frozen failure state drift')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['updates']!=2:raise ValueError('Expected globalCPU1+MPS1')
    model=core.fresh(plan);model.load_state_dict(saved['model']);model.eval();del saved
    before=weight_digest(model);panel,row=core.load_data();priors=core.read(core.ROOT/plan['priors_path'])
    spec=plan['fit_stream'][0][0];rr,x,t=core.row_inputs(panel,row,spec,'A')
    times=defaultdict(list);counts=defaultdict(int)
    def wrap(target,name,key):
        original=getattr(target,name)
        def measured(*args,**kwargs):
            torch.mps.synchronize();start=time.monotonic()
            result=original(*args,**kwargs)
            torch.mps.synchronize();times[key].append(time.monotonic()-start);counts[key]+=1
            return result
        setattr(target,name,measured)
    for name in ('encode','forward','head_logits','type_logits','tail_scores'):
        wrap(model,name,name)
    for layer in model.layers:
        wrap(layer.self_attention,'forward','self_attention')
        wrap(layer.cross_attention,'forward','cross_attention')
    wrap(law,'sample_logits','categorical_sample')
    guard=ManagedMPS(core.ROOT,core.ART,120);model.query_coverage={};wrap(guard,'clear','guard_clear')
    start=time.monotonic()
    try:
        pred=law.decode_masked(model,x,rr['request'],t['layout'],priors['duration'],seed=plan['seed'],public_id=(spec['index'],rr['request'].start_us,rr['request'].end_us),deadline=guard.start+guard.wall_s,clear_callback=guard.clear)
        torch.mps.synchronize();wall=time.monotonic()-start
        expected=receipt['actual_cost_probe']['prediction']
        if pred['events']!=expected['events'] or len(pred['trace'])!=len(expected['trace']):raise ValueError('Profile replay integer event identity failed')
        max_probability=max(abs(a['active_probability']-b['active_probability']) for a,b in zip(pred['trace'],expected['trace']))
        if max_probability>1e-4:raise ValueError('Profile probability identity')
        if weight_digest(model)!=before:raise ValueError('Profile changed weights')
        write(DEST,dict(status='frozen_profile_completed',source_checkpoint=str(path.relative_to(core.ROOT)),source_checkpoint_sha256=sha(path),
            global_applied_before=2,MPS_optimizer_applied_before=1,new_optimizer_updates=0,new_dev_outcomes=0,cloud_calls=0,
            train_index=spec['index'],integer_replay_equal=True,active_probability_max_difference=max_probability,weight_digest_unchanged=True,
            synchronized_profile_wall_s=wall,components={k:dict(calls=counts[k],sum_s=sum(v),mean_s=sum(v)/len(v),max_s=max(v)) for k,v in times.items()},
            scope='Single synchronized intrusive timing replay. forward includes attention sub-timers;do not sum nested fields. Not original async wall or universal speed claim.',
            static_context_KV='Four cross-attention modules recompute same-context projections at every block;cache benefit not measured or implemented. Head/end sampling/Python/guard are competing costs.'))
    finally:
        telemetry=guard.close();write(core.ART/'posthoc-profile-runtime.json',dict(telemetry=telemetry,wall_s=time.monotonic()-guard.start,new_updates=0))
    print(dict(status='profile_complete',wall_s=wall,new_updates=0,components={k:round(sum(v),4) for k,v in times.items()}))


if __name__=='__main__':main()

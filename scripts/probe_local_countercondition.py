"""A bounded real new-objective probe; weights never join the quality experiment."""
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.countercondition import countercondition_objective
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def run():
    path=ROOT/'bench/rhythm-v2/d69-objective-probe-v1.json';plan=json.loads(path.read_text())
    output=ROOT/plan['output']
    if output.exists():raise FileExistsError(output)
    for p,h in plan['inputs_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen input changed: '+p)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.6)
    saved=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=False)
    model=FullEventModel(FullEventConfig(**saved['config'])).to('mps').train()
    model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng']);torch.mps.set_rng_state(saved['mps_rng'])
    panel=LocalRequestPanel(ROOT,plan['tape'],plan['features'])
    metadata=json.loads((ROOT/plan['counterconditions']).read_text())
    cases=[]
    for selection in plan['cases']:
        step=metadata['steps'][selection['step']-1]
        item=next(n for n in step['negatives'] if n and n['index']==selection['index'])
        row=panel.records[item['index']]
        if row['split']!='train':raise ValueError('Probe reads heldout target')
        controls=np.asarray(row['controls'],np.float32).copy()
        i=FIELDS.index('kind')  # field ids of the grammar are distinct from control ids
        from beatfcn.train.event_conditions import CONTROL_FIELDS
        control_index=CONTROL_FIELDS.index(item['field'])
        controls[[i for i in range(len(controls)) if i!=control_index],:3]=0
        inp,seq=panel.batch([item['index']],controls=[controls])
        inp=pad_input_shapes(inp,plan['input_shapes'])
        cases.append((item,inp,seq,torch.tensor(item['opposing_controls'],dtype=torch.float32)[None]))
    report=dict(id=plan['id'],complete=False,passed=False,optimizer_updates=0,quality_updates=0,
        new_cloud_compute_calls=0,plan_sha256=sha(path),cases=[c[0] for c in cases],curve=[],
        tensor_stats=[{section:{k:tensor_health(v.numpy()) for k,v in values.items()} for section,values in [('inputs',c[1]),('sequence',c[2])]} for c in cases],
        scope='Two real training requests, new objective only; not heldout control or generation quality')
    device_cases=[(item,{k:v.to('mps') for k,v in inp.items()},{k:v.to('mps') for k,v in seq.items()},opp.to('mps')) for item,inp,seq,opp in cases]
    started=time.monotonic()
    try:
        # Real zero-weight loss AND gradients; no SGD update in this equivalence check.
        _,inp,seq,opp=device_cases[0]
        expected=event_loss(model(inp,seq),seq);expected.backward()
        base_grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        base_value=float(expected.detach());model.zero_grad(set_to_none=True)
        zero,_=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=0.,margin=plan['margin'])
        zero.backward()
        delta=max(float((p.grad.cpu()-base_grads[n]).abs().max()) for n,p in model.named_parameters() if p.grad is not None)
        loss_delta=abs(float(zero.detach())-base_value);model.zero_grad(set_to_none=True)
        total,_=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=plan['weight'],margin=plan['margin'])
        total.backward()
        condition_grad=model.control_projection.weight.grad.detach().cpu()
        contrast_grad=float((condition_grad-base_grads['control_projection.weight']).norm())
        report['equivalence']=dict(loss_difference=loss_delta,gradient_difference=delta,
            contrast_control_projection_gradient_norm=contrast_grad,
            passed=loss_delta<1e-6 and delta<1e-5 and np.isfinite(contrast_grad) and contrast_grad>0.)
        model.zero_grad(set_to_none=True)
        del expected,zero,total,base_grads,condition_grad
        if not report['equivalence']['passed']:raise ValueError('Real objective equivalence/gradient gate failed')

        def measure(step):
            model.eval();rows=[]
            with torch.no_grad():
                for item,inp,seq,opp in device_cases:
                    _,stats=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=plan['weight'],margin=plan['margin'])
                    logits=model(inp,seq).masked_fill(~seq['allowed'],-1e9)
                    correct=logits.argmax(-1)==seq['targets'];acc={}
                    for i,name in enumerate(FIELDS):
                        mask=seq['valid']&(seq['fields']==i)
                        if mask.any():acc[name]=float(correct[mask].float().mean())
                    rows.append(dict(chart_id=item['chart_id'],field=item['field'],ce=float(stats['event_ce']),
                        contrast=float(stats['contrast']),gap=float(stats['gap'][0]),field_accuracy=acc))
            report['curve'].append(dict(step=step,rows=rows));model.train()
            report.update(wall_s=time.monotonic()-started,driver_bytes=torch.mps.driver_allocated_memory())
            write_json(output,report)

        measure(0)
        for step in range(1,plan['steps']+1):
            if time.monotonic()-started>plan['max_wall_s']:raise TimeoutError('Bounded new-objective probe expired')
            optimizer.zero_grad(set_to_none=True)
            for _,inp,seq,opp in device_cases:
                loss,stats=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=plan['weight'],margin=plan['margin'])
                (loss/len(device_cases)).backward()
                del loss,stats
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            report.update(optimizer_updates=step,gradient_norm=float(grad))
            if step in (25,50,100):measure(step)
        final=report['curve'][-1]['rows']
        report.update(complete=True,passed=report['equivalence']['passed'] and all(
            r['ce']<=plan['maximum_ce'] and r['gap']>=plan['minimum_gap'] for r in final),
            wall_s=time.monotonic()-started,weights_discarded=True,
            loss_floor_note='softplus is reported separately; even gap=.1 leaves softplus(0)=.693. No near-zero total-loss claim.')
        write_json(output,report)
        print(json.dumps(dict(passed=report['passed'],equivalence=report['equivalence'],final=final,wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',complete=True,wall_s=time.monotonic()-started)
        write_json(output,report);raise


if __name__=='__main__':run()

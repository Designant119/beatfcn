"""Read-only moment/gradient audit of existing checkpoints; no step is applied."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.optimizer_audit import adamw_proposal
from scripts.audit_local_condition_gradients import state_sha
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(parameter, gradient, proposal):
    active = gradient != 0
    g, delta, weight = [t[active].double() for t in (gradient, proposal['delta'], parameter)]
    result = dict(elements=gradient.numel(), active_gradient_elements=int(active.sum()))
    if not len(g):
        return result
    gn, dn = float(g.norm()), float(delta.norm())
    result.update(gradient_norm=gn, proposed_update_norm=dn, parameter_norm=float(weight.norm()),
                  descent_cosine=float((-g)@delta)/(gn*dn) if gn*dn else None)
    for key in ('epsilon_fraction','history_second_moment_fraction'):
        values = proposal[key][active].double()
        result[key] = dict(median=float(values.median()), p90=float(torch.quantile(values,.9)), maximum=float(values.max()))
    return result


def run():
    path = ROOT/'bench/rhythm-v2/d76-optimizer-state-audit-v1.json'; plan = json.loads(path.read_text())
    output = ROOT/plan['output']
    if output.exists(): raise FileExistsError(output)
    for n,h in plan['inputs_sha256'].items():
        if sha(ROOT/n)!=h: raise ValueError('Frozen diagnostic source changed: '+n)
    torch.set_num_threads(2)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features']); batch=[]
    for pair in plan['pairs']:
        for index in pair['indices']:
            row=panel.records[index]
            if row['split']!='train': raise ValueError('Training requests required')
            control=np.asarray(row['controls'],np.float32).copy();field=CONTROL_FIELDS.index(pair['field'])
            control[[j for j in range(len(control)) if j!=field],:3]=0
            inp,seq=panel.batch([index],controls=[control]);batch.append((pad_input_shapes(inp,plan['input_shapes']),seq))
    report=dict(id=plan['id'],complete=False,passed=False,plan_sha256=sha(path),checkpoints=[],optimizer_updates=0,new_cloud_compute_calls=0)
    start=time.monotonic()
    try:
        for checkpoint in plan['checkpoints']:
            if time.monotonic()-start>plan['max_wall_s']:raise TimeoutError('Audit deadline')
            saved=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
            model=FullEventModel(FullEventConfig(**saved['config'])).eval();model.load_state_dict(saved['model'])
            before=state_sha(model); groups=saved['optimizer']['param_groups']
            if len(groups)!=1:raise ValueError('Expected frozen single-group optimizer')
            group=groups[0]; names=list(dict(model.named_parameters())); ids=group['params']
            if len(names)!=len(ids):raise ValueError('Parameter ordering differs')
            # Preserve exact checkpoint bytes on disk; reconstruct into new tensors only.
            state_before={n:{k:v.clone() for k,v in saved['optimizer']['state'][i].items()} for n,i in zip(names,ids)}
            ces=[]
            for inp,seq in batch:
                loss=event_loss(model(inp,seq),seq);ces.append(float(loss.detach()));(loss/4).backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            summary={}
            for (name,param),old_id in zip(model.named_parameters(),ids):
                state=saved['optimizer']['state'][old_id]
                if (name in ['control_projection.weight','control_ids.weight','output.weight','local_projection.weight','mel_projection.weight']
                        or name=='decoder.3.cross_attn.q.weight' or name=='decoder.3.cross_attn.v.weight'):
                    proposal=adamw_proposal(param,param.grad,state,group)
                    summary[name]=summarize(param,param.grad,proposal)
                    if name=='control_projection.weight':
                        for label,sl in [('endpoints',slice(0,2)),('flags',slice(2,4)),('scope',slice(4,6))]:
                            summary[name+'/'+label]=summarize(param[:,sl],param.grad[:,sl],{k:v[:,sl] for k,v in proposal.items()})
                for key,value in state.items():
                    if not torch.equal(value,state_before[name][key]):raise ValueError('Audit mutated source optimizer')
            if before!=state_sha(model):raise ValueError('Audit mutated model')
            report['checkpoints'].append(dict(checkpoint=checkpoint,ce=ces,raw_gradient_norm=norm,
                gradient_clip_coefficient=min(1.,1./(norm+1e-6)),
                optimizer={k:v for k,v in group.items() if k!='params'},blocks=summary,
                model_unchanged=True,optimizer_unchanged=True))
            del model,saved,state_before
        report.update(complete=True,passed=True,wall_s=time.monotonic()-start,
            scope='Current proposed update for four diagnostic training requests from existing step2/1000 moments; no applied step, not discarded D74/D75 terminal states or future quality.')
        write_json(output,report)
        print(json.dumps(dict(passed=True,wall_s=report['wall_s'],checkpoints=[{k:v for k,v in r.items() if k!='blocks'} for r in report['checkpoints']])))
    except Exception as exc:
        report.update(complete=True,passed=False,error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-start);write_json(output,report);raise


if __name__=='__main__':run()

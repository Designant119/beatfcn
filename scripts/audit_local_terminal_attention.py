"""Read-only instrumentation of the saved D77 failed terminal state."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_local_condition_path import sections
from scripts.audit_local_condition_gradients import state_sha
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    path = ROOT / 'bench/rhythm-v2/d78-terminal-attention-v1.json'
    plan = json.loads(path.read_text()); destination = ROOT / plan['output']
    if destination.exists():
        raise FileExistsError(destination)
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Changed frozen input: '+name)
    torch.set_num_threads(2)
    saved = torch.load(ROOT/plan['checkpoint'], map_location='cpu', weights_only=False)
    if not saved['diagnostic_only'] or saved['auto_resume'] or saved['eligible_for_promotion']:
        raise ValueError('Expected quarantined diagnostic state')
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model']); del saved
    before = state_sha(model)
    panel = LocalRequestPanel(ROOT,plan['tape'],plan['features'])
    result = dict(id=plan['id'], complete=False, passed=False, plan_sha256=sha(path), rows=[],
                  optimizer_updates=0, new_cloud_compute_calls=0, checkpoint=plan['checkpoint'])
    started=time.monotonic()
    try:
        with torch.no_grad():
            for pair in plan['pairs']:
                batch=[]
                for index in pair['indices']:
                    row=panel.records[index]
                    if row['split']!='train' or row['mode']!='phira':
                        raise ValueError('Expected frozen real Phira training request')
                    control=np.asarray(row['controls'],np.float32).copy()
                    field=CONTROL_FIELDS.index(pair['field'])
                    control[[i for i in range(len(control)) if i!=field],:3]=0
                    inp,seq=panel.batch([index],controls=[control])
                    batch.append((pad_input_shapes(inp,plan['input_shapes']),seq))
                if any(not torch.equal(v,batch[1][0][k]) for k,v in batch[0][0].items() if k!='controls'):
                    raise ValueError('Pair has another input discriminator')
                branch=next(i for i,(a,b) in enumerate(zip(batch[0][1]['targets'][0],batch[1][1]['targets'][0])) if a!=b)
                for name in ('tokens','fields','times','valid','allowed'):
                    if not torch.equal(batch[0][1][name][:,:branch+1],batch[1][1][name][:,:branch+1]):
                        raise ValueError('Pair prefix mismatch')
                target_a,target_b=[int(seq['targets'][0,branch]) for _,seq in batch]
                for case,(inp,seq) in enumerate(batch):
                    if time.monotonic()-started>plan['max_wall_s']:
                        raise TimeoutError('Read-only deadline')
                    prefix={k:v[:,:branch+1] for k,v in seq.items()}
                    args=[prefix[k] for k in ('tokens','fields','times','valid')]
                    memory=model.encode(inp); slices=sections(model,inp)
                    raw=model.decode(*args,memory)
                    layers=[]

                    def capture(module,args,output):
                        x,times,positions,(k,v),mask=args
                        q=module.q(x).view(x.shape[0],x.shape[1],module.c.heads,-1).transpose(1,2)
                        q=module._position(q,times,positions)
                        k=k.repeat_interleave(module.c.heads//module.c.kv_heads,1)
                        v=v.repeat_interleave(module.c.heads//module.c.kv_heads,1)
                        scores=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1])
                        probabilities=scores.masked_fill(~mask,float('-inf')).softmax(-1)
                        manual=module.out((probabilities@v).transpose(1,2).reshape_as(x))
                        error=float((manual-output).abs().max())
                        sum_error=float((probabilities.sum(-1)-1).abs().max())
                        if error>1e-5 or sum_error>1e-6:
                            raise ValueError('Instrumented softmax differs from native attention')
                        mass={n:probabilities[:,:,branch,part].sum(-1).flatten().tolist() for n,part in slices.items()}
                        field_index=slices['controls'].start+field
                        layers.append(dict(source_mass_per_head=mass,
                            specified_field_mass_per_head=probabilities[:,:,branch,field_index].flatten().tolist(),
                            manual_output_max_error=error, probability_sum_error=sum_error,
                            max_score_per_source={n:float(scores[:,:,branch,part].masked_fill(~memory['valid'][:,None,part],float('-inf')).max()) for n,part in slices.items()}))

                    handles=[layer.cross_attn.register_forward_hook(capture) for layer in model.decoder]
                    try:
                        observed=model.decode(*args,memory)
                    finally:
                        for handle in handles: handle.remove()
                    hook_error=float((observed-raw).abs().max())
                    if hook_error!=0:
                        raise ValueError('Read-only hooks altered logits')
                    opposite=model.replace_controls(memory,batch[case^1][0]['controls'])
                    swapped=model.decode(*args,opposite)
                    values={}
                    for name,part in slices.items():
                        norms=memory['values'][:,part][memory['valid'][:,part]].square().mean(-1).sqrt()
                        values[name]=dict(tokens=len(norms),rms_mean=float(norms.mean()),rms_min=float(norms.min()),rms_max=float(norms.max()))
                    result['rows'].append(dict(index=pair['indices'][case],field=pair['field'],branch=branch,
                        target_a=target_a,target_b=target_b,uniform_field_mass=1/int(memory['valid'].sum()),
                        original_margin_a_minus_b=float(raw[0,branch,target_a]-raw[0,branch,target_b]),
                        swapped_margin_a_minus_b=float(swapped[0,branch,target_a]-swapped[0,branch,target_b]),
                        memory_sources=values,layers=layers,hook_logit_difference=hook_error))
                    write_json(destination,result)
        unchanged=before==state_sha(model)
        result.update(complete=True,passed=unchanged,weights_unchanged=unchanged,wall_s=time.monotonic()-started,
                      scope='Only D77 terminal first ambiguous query on four real training requests; no new-song quality or causal proof from attention mass.')
        write_json(destination,result)
        if not unchanged: raise ValueError('Model weights changed')
        print(json.dumps(dict(passed=result['passed'],wall_s=result['wall_s'],rows=len(result['rows']))))
    except Exception as exc:
        result.update(complete=True,passed=False,error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(destination,result)
        raise


if __name__=='__main__':
    run()

"""Zero-update real-data equivalence of the broader candidate training path."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.train.condition_interaction import condition_interaction_loss
from beatfcn.train.local_condition_pairs import common_fork
from beatfcn.train.local_interaction_runtime import prepare_inputs, padded_common_prefix, backward_interaction
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_audio_audit import tensor_health
from scripts.probe_local_condition_interaction import load_frozen_components
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run():
    path=ROOT/'bench/rhythm-v2/d83-interaction-integration-audit-v2.json'
    plan=json.loads(path.read_text());output=ROOT/plan['output']
    if output.exists():raise FileExistsError(output)
    for source,h in plan['inputs_sha256'].items():
        if sha(ROOT/source)!=h:raise ValueError('Frozen input changed: '+source)
    if not torch.backends.mps.is_available():raise ValueError('Actual MPS required')
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.6)
    Config,Model,Standardizer=load_frozen_components(plan)
    normalizer=Standardizer.load(ROOT/plan['normalizer'])
    panel=LocalRequestPanel(ROOT,plan['tape'],plan['features'])
    pairs=json.loads((ROOT/plan['pair_tape']).read_text())['catalog']
    saved=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=False)
    model=Model(Config(**{**saved['config'],'audio_key_conditioning':True})).to('mps').train()
    loaded=model.load_state_dict(saved['model'],strict=False)
    if loaded.missing_keys!=['audio_key_conditioner.weight'] or loaded.unexpected_keys:raise ValueError('Initial bridge differs')
    del saved
    before=state_sha(model);started=time.monotonic()
    report=dict(id=plan['id'],complete=False,passed=False,optimizer_updates=0,new_cloud_spend_usd=0,plan_sha256=sha(path),rows=[])
    write_json(output,report)
    try:
        for pair_id in plan['pair_ids']:
            if time.monotonic()-started>plan['max_wall_s']:raise TimeoutError('Bounded audit deadline')
            pair=pairs[pair_id];a,b=pair['indices'];fork=pair['fork'];targets=pair['targets']
            if common_fork(panel.sequence(a),panel.sequence(b))!=(fork,targets):raise ValueError('Prepared branch changed')
            raw,sequence=panel.batch([a,b],controls=[np.asarray(c,np.float32) for c in pair['controls']])
            if any(not torch.equal(v[0],v[1]) for k,v in raw.items() if k!='controls'):raise ValueError('Non-condition pair input differs')
            inp=prepare_inputs(raw,normalizer,plan['input_shapes'])
            unknown=raw['controls'].clone();unknown[...,:3]=0
            if not torch.equal(normalizer({'controls':unknown})['controls'],unknown):raise ValueError('Unknown values changed')
            tensor_stats={k:tensor_health(v.numpy()) for k,v in inp.items()}
            device=[{k:v[i:i+1].to('mps') for k,v in inp.items()} for i in (0,1)]
            seqs=[{k:v[i:i+1].to('mps') for k,v in sequence.items()} for i in (0,1)]
            prefix=padded_common_prefix(seqs[0],fork,plan['auxiliary_prefix_length'])
            if any(not torch.equal(seqs[0][k][:,:fork+1],seqs[1][k][:,:fork+1]) for k in ('tokens','fields','times','valid','allowed')):raise ValueError('Actual collated prefix differs')
            row_logits=lambda i:model(device[i],prefix)[0,fork,targets]
            model.zero_grad(set_to_none=True)
            joint,gap=condition_interaction_loss(torch.stack([row_logits(i) for i in (0,1)]));joint.backward()
            reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            joint_value=float(joint.detach());del joint,gap
            model.zero_grad(set_to_none=True)
            streamed,_,errors=backward_interaction(row_logits,verify=True)
            actual={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
            if actual.keys()!=reference.keys():raise ValueError('Gradient parameter sets differ')
            error=max(float((actual[n]-reference[n]).abs().max()) for n in reference)
            norm=float(torch.cat([g.flatten() for g in reference.values()]).norm())
            delta_norm=float(torch.cat([(actual[n]-reference[n]).flatten() for n in reference]).norm())
            with torch.no_grad():
                prefix_error=max(float((row_logits(i)-model(device[i],seqs[i])[0,fork,targets]).abs().max()) for i in (0,1))
            row=dict(pair_id=pair_id,mode=pair['mode'],field=pair['field'],component=pair['component'],fork=fork,
                loss=joint_value,loss_error=abs(joint_value-float(streamed)),gradient_max_error=error,
                gradient_relative_norm_error=delta_norm/max(norm,1e-8),gradient_norm=norm,
                audio_key_gradient_norm=float(reference['audio_key_conditioner.weight'].norm()),
                prefix_full_error=prefix_error,recomputed_logit_errors=errors,
                mps_driver_bytes=torch.mps.driver_allocated_memory(),tensor_stats=tensor_stats)
            row['passed']=row['loss_error']<=1e-6 and error<=1e-5 and row['gradient_relative_norm_error']<=1e-4 and prefix_error<=1e-4 and np.isfinite(norm) and norm>0
            report['rows'].append(row);report['wall_s']=time.monotonic()-started;write_json(output,report)
            if not row['passed']:raise ValueError('Real integrated gradient/prefix gate failed')
            del inp,raw,sequence,seqs,device,prefix,reference,actual,streamed
        report.update(complete=True,passed=True,weights_unchanged=before==state_sha(model),wall_s=time.monotonic()-started)
        if not report['weights_unchanged']:raise ValueError('Zero-update audit changed model weights')
        write_json(output,report)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    except Exception as exc:
        report.update(complete=True,passed=False,error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(output,report);raise


if __name__=='__main__':run()

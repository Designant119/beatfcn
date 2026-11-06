"""D46: deterministic real-input corruption contract, without a model."""
import ast
from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import EMPTY
from beatfcn.model.group_canvas import new_group_canvas, masked_group_view
from scripts.audit_draft_exposure import canvas_payload, signature
from scripts.audit_edit_support import group_relation
from scripts.local_action_corruption import aligned_training_view, pooled_signature
from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import file_sha, verify_files

ROOT=Path(__file__).resolve().parents[1]
ARRAYS=('counts','heads','multiplicity','notes')


def statistics(value):
    a=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    x=a.astype(np.float64)
    if not np.isfinite(x).all(): raise ValueError('nonfinite input')
    return dict(shape=list(a.shape),dtype=str(a.dtype),nonfinite=0,
        minimum=float(x.min()) if x.size else None,maximum=float(x.max()) if x.size else None,
        mean=float(x.mean()) if x.size else None,std=float(x.std()) if x.size else None)


def inputs_hash(inputs):
    h=hashlib.sha256()
    for f in fields(inputs):
        value=getattr(inputs,f.name)
        if torch.is_tensor(value):
            h.update(f.name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def duration_pools(examples):
    pools=defaultdict(set)
    for _,labels,_ in examples:
        for group in canvas_payload(labels).values():
            for e in group:
                if e['type']=='hold':
                    if (e['end_us']-e['onset_us']) % 1000: raise ValueError('frozen duration pool not millisecond aligned')
                    pools[labels.mode].add((e['end_us']-e['onset_us'])//1000)
    return {k:sorted(v) for k,v in pools.items()}


def main():
    started=time.perf_counter()
    path=ROOT/'bench/rhythm-v2/d46-local-action-contract-v1.json'
    plan=json.loads(path.read_text());verify_files(plan)
    destination=ROOT/plan['result']
    if destination.exists():raise FileExistsError('preserve existing audit')
    manifest=json.loads((ROOT/plan['panel_manifest']).read_text());verify_files(manifest)
    panel=json.loads((ROOT/plan['panel_plan']).read_text())
    torch.set_num_threads(2)
    examples=[make_example(i,panel) for i in manifest['items']]
    pools=duration_pools(examples)
    input_health=[];target_hashes={}
    for inputs,labels,identity in examples:
        input_health.append(dict(identity=identity,input_sha256=inputs_hash(inputs),
            tensors={f.name:statistics(getattr(inputs,f.name)) for f in fields(inputs) if torch.is_tensor(getattr(inputs,f.name))},
            target_tensors={name:statistics(getattr(labels,name)) for name in ARRAYS}))
        target_hashes[identity['panel_id']]=hashlib.sha256(b''.join(getattr(labels,n).tobytes() for n in ARRAYS)).hexdigest()
    tree=ast.parse((ROOT/plan['corruption_source']).read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('blank_for','corruption')]
    if len(funcs)!=2:raise ValueError('corruption definition changed')
    scope=dict(np=np,new_group_canvas=new_group_canvas,masked_group_view=masked_group_view,EMPTY=EMPTY)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<frozen-data-corruption>','exec'),scope)
    prior=json.loads((ROOT/plan['previous_audit']).read_text())
    old_rng,new_rng=np.random.default_rng(plan['seed']),np.random.default_rng(plan['seed'])
    rows=[];totals=defaultdict(Counter);unique=defaultdict(set)
    for step in range(100):
        for inputs,labels,identity in examples:
            old,kind=scope['corruption'](labels,inputs,step,old_rng)
            view,audit=aligned_training_view(scope['corruption'],labels,inputs,step,new_rng,
                identity=identity,seed=plan['seed'],hold_durations_ms=pools[labels.mode])
            if old_rng.bit_generator.state!=new_rng.bit_generator.state:raise ValueError('original RNG trajectory changed')
            if not np.array_equal(old.multiplicity>=0,view.multiplicity>=0):raise ValueError('MASK visibility changed')
            if not np.array_equal(old.counts,view.counts) or not np.all(view.counts==-1):raise ValueError('hidden count leaked')
            old_payload,payload,truth=canvas_payload(old),canvas_payload(view),canvas_payload(labels)
            expected=prior['training_views'][len(rows)]
            if expected['identity']!=identity or expected['kind']!=kind or expected['stats']['visible_positive_groups']!=len(old_payload):
                raise ValueError('legacy replay no longer matches D44')
            changed={tuple(e['position']) for e in audit['edits']}
            for b,g in np.ndindex(labels.multiplicity.shape):
                pos=(b,g)
                if pos not in changed:
                    if (not np.array_equal(view.heads[pos],old.heads[pos]) or
                        not np.array_equal(view.notes[pos],old.notes[pos]) or view.multiplicity[pos]!=old.multiplicity[pos]):
                        raise ValueError('non-edited slot changed')
                else:
                    if labels.multiplicity[pos]<=0 or old.multiplicity[pos]<0 or signature(old_payload.get(pos,[]))==signature(truth[pos]):
                        raise ValueError('edited an unauthorized slot')
                    relation=group_relation(payload[pos],truth[pos],labels.mode)
                    if not relation['content_negative'] or not relation['locally_valid_payload']:
                        raise ValueError('new negative lacks the registered structure')
                    if pooled_signature(payload[pos])==pooled_signature(truth[pos]):raise ValueError('known pooling collision')
                    allowed={i for i,x in enumerate(inputs.allowed_types) if bool(x)}
                    if any(int(k) not in allowed for k in view.notes[pos][:view.multiplicity[pos],0]):raise ValueError('type whitelist broken')
            if kind!='synthetic_wrong_draft' and changed:raise ValueError('wrong intervention step')
            if hashlib.sha256(b''.join(getattr(labels,n).tobytes() for n in ARRAYS)).hexdigest()!=target_hashes[identity['panel_id']]:
                raise ValueError('supervision target mutated')
            counts=Counter(e['operation'] for e in audit['edits'])
            totals[labels.mode].update(counts)
            for edit in audit['edits']:
                for kind_name in {e['type'] for e in edit['target']}:
                    totals[labels.mode]['target_'+kind_name+'_edited']+=1
                unique[labels.mode].add((identity['panel_id'],tuple(edit['position'])))
            rows.append(dict(identity=identity,step=step+1,view=kind,edits=audit['edits'],
                changed_groups=len(changed),legacy_rng_equal=True,masks_equal=True,nonedited_slots_exact=True,
                tensors={name:statistics(getattr(view,name)) for name in ARRAYS}))
    for (inputs,_,_),before in zip(examples,input_health):
        if inputs_hash(inputs)!=before['input_sha256']:raise ValueError('audio, control, or request tensor mutated')
    if len(rows)!=800 or any(not unique[mode] for mode in ('mania_4k','phira')):raise ValueError('incomplete local edit support')
    result=dict(id='D46',complete=True,passed=True,plan_sha256=file_sha(path),input_sha256=plan['input_sha256'],
        scope='training corruption data contract only; no learned repair or generation capability claim',
        real_input_health=input_health,hold_duration_pool_ms=pools,views=rows,
        summary={m:dict(operation_and_type_exposures=dict(v),distinct_edited_source_slots=len(unique[m])) for m,v in totals.items()},
        all_800_masks_and_legacy_rng_equal=True,all_non_edited_payloads_bitwise_equal=True,
        original_empty_slot_noise_preserved=True,all_original_correct_groups_preserved=True,
        original_supervision_and_input_tensors_unchanged=True,known_additive_pooling_collisions=0,
        model_forwards=0,optimizer_updates=0,new_paid_compute_usd=0,model_promotion=False,
        wall_s=time.perf_counter()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','summary','hold_duration_pool_ms','wall_s','peak_rss_gib')},ensure_ascii=False))


if __name__=='__main__':main()

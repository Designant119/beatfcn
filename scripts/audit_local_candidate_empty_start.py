"""Zero-update final-checkpoint audit of the observed empty generation failure."""
import hashlib
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_codec import EventRequest, EventVocabulary, encode_events, teacher_sequence
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_interaction_runtime import prepare_inputs
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from scripts.audit_local_candidate_conditions import validate_checkpoint
from scripts.probe_local_condition_interaction import load_frozen_components
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def run():
    output=ROOT/'reports/d83_empty_start_audit_v2.json'
    if output.exists():raise FileExistsError(output)
    plan_path=ROOT/'bench/rhythm-v2/d83-independent-song-candidate-v1.json'
    plan=json.loads(plan_path.read_text()); state=json.loads((ROOT/plan['output']).read_text())
    for source,expected in plan['code_sha256'].items():
        if sha(ROOT/source)!=expected:raise ValueError('Frozen execution code changed')
    saved=torch.load(ROOT/state['checkpoint'],map_location='cpu',weights_only=False)
    validate_checkpoint(saved,plan,sha(plan_path),state,sha(ROOT/state['checkpoint']))
    Config,Model,Standardizer=load_frozen_components(plan)
    model=Model(Config(**saved['config'])).eval();model.load_state_dict(saved['model']);del saved
    normalizer=Standardizer.load(ROOT/plan['normalizer'])
    features=LocalFeatureStore(ROOT,plan['features'])
    protocol=json.loads((ROOT/'bench/rhythm-v2/d83-condition-protocol-v1.json').read_text())
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.6)
    cases=[]
    for i,song in enumerate(protocol['songs']):
        views=[('unknown',{})]
        if i==0:
            views += [(field['name']+'/'+side,{field['name']:field[side]})
                      for field in protocol['fields'] for side in ('low','high')]
        for label,conditions in views:
            a,b=song['scope_us'];request=EventRequest('phira',a,b,features.records[song['audio_sha256']]['duration_us'])
            inp,_=request_inputs(features,song['audio_sha256'],request,encode_controls('phira',conditions,(a,b),origin_us=a))
            inputs,seq=collate([dict(inputs=inp,sequence=teacher_sequence(encode_events([],request),request))])
            inputs=prepare_inputs(inputs,normalizer,plan['input_shapes'])
            cases.append(dict(song=song,view=label,inputs=inputs,sequence=seq))
    before=state_sha(model);started=time.monotonic();ids=[EventVocabulary().token('group',v) for v in (0,1)]
    report=dict(id='D83-final-empty-start-path-audit-v2',complete=False,optimizer_updates=0,new_cloud_compute_calls=0,
        checkpoint=state['checkpoint'],checkpoint_sha256=state['checkpoint_sha256'],rows=[],
        tolerance=1e-4,scope='Actual failed final state: first legal END/NEW decision only. No architecture change, optimizer or retraining.',
        source_sha256={p:sha(ROOT/p) for p in [str(Path(__file__).relative_to(ROOT)),str(plan_path.relative_to(ROOT)),
           'bench/rhythm-v2/d83-condition-protocol-v1.json',plan['output']]})
    write_json(output,report)
    for device in ('cpu','mps'):
        model=model.to(device)
        for i,case in enumerate(cases):
            inp={k:v.to(device) for k,v in case['inputs'].items()};seq={k:v.to(device) for k,v in case['sequence'].items()}
            full=model(inp,seq)[0,0,ids].cpu()
            memory=model.encode(inp);cache=model.start_cache(memory)
            cached=model.decode(*(seq[k][:,:1] for k in ('tokens','fields','times','valid')),memory,cache=cache)[0,0,ids].cpu()
            delta=float((full-cached).abs().max())
            if delta>1e-4 or full.argmax()!=cached.argmax():raise ValueError('Final cached/full legal choice mismatch')
            values=dict(legal_logits=full.tolist(),end_probability=float(full.softmax(0)[0]),
                        end_minus_new=float(full[0]-full[1]),argmax='end' if int(full.argmax())==0 else 'new',cache_max_abs=delta)
            if device=='cpu':report['rows'].append(dict(chart_id=case['song']['chart_id'],view=case['view'],cpu=values))
            else:
                row=report['rows'][i];row['mps']=values
                row['cpu_mps_max_abs']=float((full-torch.tensor(row['cpu']['legal_logits'])).abs().max())
                row['choice_agrees']=row['cpu']['argmax']==values['argmax']
                if row['cpu_mps_max_abs']>1e-4 or not row['choice_agrees']:raise ValueError('Final CPU/MPS legal choice mismatch')
    after=state_sha(model)
    report.update(complete=True,weights_unchanged=before==after,model_state_sha256=after,
                  wall_s=time.monotonic()-started,all_choose_end=all(r['cpu']['argmax']=='end' for r in report['rows']))
    if before!=after:raise ValueError('Read-only audit changed model weights')
    write_json(output,report)
    print(json.dumps({k:report[k] for k in ['complete','weights_unchanged','wall_s','all_choose_end']}))


if __name__=='__main__':run()

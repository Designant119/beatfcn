"""Oracle diagnostic at matched physical state; never a free-generation benchmark."""
import json
from pathlib import Path
import time

import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import BOS, EventRequest, EventState, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_sequence_support_trace import rank
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def histories(truth, generated, request):
    heads=sorted({e.onset_us for e in truth})
    if len(heads)<3:raise ValueError('Insufficient real groups for an internal anchor')
    anchor=heads[(len(heads)-1)//2]
    real=[e for e in truth if e.onset_us<=anchor]
    splice=[e for e in generated if e.onset_us<anchor]+[e for e in truth if e.onset_us==anchor]
    prefixes=[encode_events(es,request)[:-1]for es in (real,splice)]
    states=[]
    for prefix in prefixes:
        state=EventState(request)
        for token in prefix:state.apply(token)
        if state.phase!='group' or state.last_onset!=anchor or state.done:raise ValueError('Anchor is not a completed group')
        states.append(state)
    a,b=states
    if (a.phase,a.last_onset,a.onset,a.query(),a.allowed_tokens())!=(b.phase,b.last_onset,b.onset,b.query(),b.allowed_tokens()):
        raise ValueError('Physical state/support differs after splice')
    full=encode_events(truth,request)
    if full[:len(prefixes[0])]!=prefixes[0]:raise ValueError('Human prefix differs from full target')
    suffix=full[len(prefixes[0]):len(prefixes[0])+4]
    # These stage requests have no COPY; compare the same four oracle decisions.
    if [a.vocab.decision(t)[0]for t in suffix]!=['group','head_block','head_cell','head_ms']:
        raise ValueError('Expected a real next numeric head after the anchor')
    return dict(anchor_us=anchor,next_human_head_us=heads[heads.index(anchor)+1],prefixes=prefixes,suffix=suffix)


@torch.no_grad()
def run():
    output=ROOT/'reports/d87_candidate_prefix_splice_500_v1.json'
    if output.exists():raise FileExistsError(output)
    prep_path=ROOT/'bench/rhythm-v2/d87-prefix-splice-500-v1.json'
    prep=json.loads(prep_path.read_text())
    for p,digest in prep['inputs_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Frozen diagnostic input changed: '+p)
    plan=json.loads((ROOT/prep['training_plan']).read_text())
    source=json.loads((ROOT/prep['stage']).read_text())
    for p,digest in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
        if sha(ROOT/p)!=digest:raise ValueError('Frozen training input changed: '+p)
    if not source['complete'] or source.get('error'):raise ValueError('Stage audit incomplete')
    checkpoint=ROOT/source['checkpoint']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['updates']!=500 or saved['objective']!=plan['loss_mode'] or saved['plan_sha256']!=sha(ROOT/prep['training_plan']):
        raise ValueError('Wrong actual checkpoint')
    torch.set_num_threads(2)
    model=FullEventModel(FullEventConfig(**saved['config'])).eval();model.load_state_dict(saved['model']);del saved
    features=LocalFeatureStore(ROOT,plan['features'])
    report=dict(id='D87-candidate-prefix-splice-500-v1',complete=False,rows=[],rejected=[],optimizer_updates=0,
                new_cloud_compute_calls=0,model_decode_calls=0,plan_sha256=sha(prep_path),
                scope='Oracle history diagnostic only. Real anchor and continuation labels are intentionally supplied after generation.',
                limitations=['History contents and token lengths both change; this does not isolate either factor.',
                             'No new free output and no quality promotion. Only4 heldoutPhira activity windows.'])
    started=time.monotonic()
    try:
        for row in source['rows']:
            if row['role']!='active':continue
            if row['mode']!='phira':raise ValueError('Only this fixed Phira stage is supported')
            a,b=row['scope_us'];audio=row['audio_sha256']
            for p,digest in features.records[audio]['sha256'].items():
                if sha(ROOT/p)!=digest:raise ValueError('Feature changed')
            request=EventRequest(row['mode'],a,b,features.records[audio]['duration_us'])
            truth=[RhythmEvent(**e)for e in row['truth_events']]
            generated=[RhythmEvent(**e)for e in row['prediction']['events']]
            try:prepared=histories(truth,generated,request)
            except ValueError as exc:
                report['rejected'].append(dict(chart_id=row['chart_id'],reason=str(exc)));continue
            inp,_=request_inputs(features,audio,request,encode_controls('phira',{},(a,b),origin_us=a))
            inputs,_=collate([dict(inputs=inp,sequence=teacher_sequence(encode_events([],request),request))])
            memory=model.encode(pad_input_shapes(inputs,plan['input_shapes']))
            variants=[]
            for label,prefix in zip(('real_history','generated_then_real_anchor'),prepared['prefixes'],strict=True):
                state=EventState(request,model.vocab);cache=model.start_cache(memory);decisions=[]
                for j,token in enumerate((*prefix,*prepared['suffix'])):
                    if time.monotonic()-started>prep['max_wall_s']:raise TimeoutError('Oracle audit exceeded time bound')
                    field,t=state.query();previous=state.tokens[-1]if state.tokens else BOS
                    logits=model.decode(torch.tensor([[previous]]),torch.tensor([[field]]),torch.tensor([[t]],dtype=torch.float32),
                        torch.ones(1,1,dtype=torch.bool),memory,cache=cache)[0,0]
                    report['model_decode_calls']+=1
                    if not torch.isfinite(logits).all():raise ValueError('Nonfinite diagnostic logits')
                    support=state.allowed_tokens()
                    if token not in support:raise ValueError('Oracle continuation is not legal in matched state')
                    if j>=len(prefix):
                        values=logits[list(support)];target_pos=support.index(token)
                        decisions.append(dict(phase=state.phase,query_s=t,allowed=list(support),target=token,
                            target_nll=float(-torch.log_softmax(values,0)[target_pos]),
                            target_rank=1+int((values>values[target_pos]).sum()),prediction=rank(logits,support)))
                    state.apply(token)
                variants.append(dict(history=label,prefix_tokens=len(prefix),decisions=decisions))
            for x,y in zip(variants[0]['decisions'],variants[1]['decisions'],strict=True):
                if (x['phase'],x['query_s'],x['allowed'],x['target'])!=(y['phase'],y['query_s'],y['allowed'],y['target']):
                    raise ValueError('Oracle decisions are not physically matched')
            report['rows'].append(dict(chart_id=row['chart_id'],anchor_us=prepared['anchor_us'],
                next_human_head_us=prepared['next_human_head_us'],variants=variants))
            write_json(output,report)
        report.update(complete=True,wall_s=time.monotonic()-started,matched_windows=len(report['rows']),
                      checkpoint_unchanged=sha(checkpoint)==source['checkpoint_sha256'])
        if not report['checkpoint_unchanged']:raise ValueError('Checkpoint changed')
        write_json(output,report)
        print(json.dumps(dict(complete=True,matched_windows=report['matched_windows'],wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write_json(output,report);raise


if __name__=='__main__':run()

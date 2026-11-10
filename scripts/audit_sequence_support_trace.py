"""Replay saved generation tokens; shadow support never feeds back into the model."""
import argparse
from collections import defaultdict
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_codec import BOS, EventRequest, EventState, teacher_sequence, encode_events
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def static_head_support(state):
    if state.phase not in ('head_block', 'head_cell', 'head_ms'):
        return state.allowed_tokens()
    shadow = copy(state)
    # Keep request, selected coarse fields, COPY permissions and lane occupancy.
    shadow.last_onset = -1
    return shadow.allowed_tokens()


def rank(logits, support):
    values = logits[list(support)]
    order = torch.argsort(values, descending=True, stable=True)
    return dict(top_token=int(support[int(order[0])]),
                margin=float(values[order[0]]-values[order[1]]) if len(support)>1 else None,
                tied_top_count=int((values==values[order[0]]).sum()), size=len(support))


@torch.no_grad()
def run(arm, step):
    output = ROOT/f'reports/d87_{arm}_support_trace_{step}_v1.json'
    if output.exists():
        raise FileExistsError(output)
    path = ROOT/f'bench/rhythm-v2/d87-sequence-{arm}-v1.json'
    plan = json.loads(path.read_text())
    source_path = ROOT/f'reports/d87_{arm}_stage_{step}_v1.json'
    source = json.loads(source_path.read_text())
    if not source.get('complete') or source.get('error'):
        raise ValueError('Need completed real stage evidence')
    for p, digest in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        if sha(ROOT/p) != digest:
            raise ValueError('Frozen training source changed: '+p)
    checkpoint = ROOT/source['checkpoint']
    if sha(checkpoint) != source['checkpoint_sha256']:
        raise ValueError('Checkpoint identity changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['updates'] != step or saved['plan_sha256'] != sha(path) or saved['objective'] != plan['loss_mode']:
        raise ValueError('Wrong checkpoint or objective')
    torch.set_num_threads(2)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model']); del saved
    features = LocalFeatureStore(ROOT, plan['features'])
    for audio in {r['audio_sha256'] for r in source['rows']}:
        for p, digest in features.records[audio]['sha256'].items():
            if sha(ROOT/p) != digest:
                raise ValueError('Feature identity changed')
    report = dict(id=f'D87-{arm}-support-trace-{step}', complete=False, optimizer_updates=0,
                  new_cloud_compute_calls=0, rows=[], model_decode_calls=0,
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [path, source_path, checkpoint,
                     ROOT/'scripts/audit_sequence_support_trace.py', ROOT/'tests/test_sequence_support_trace.py']},
                  caveats=['Fixed saved tokens replay only; no shadow token is generated or trained.',
                           'Illegal logits have no supervision at this prefix; shadow rank is support sensitivity, not true desire.',
                           'This does not isolate prefix history, time representation or capacity as the training root cause.'])
    started = time.monotonic()
    try:
        for row in source['rows']:
            a,b = row['scope_us']
            request = EventRequest(row['mode'], a,b,features.records[row['audio_sha256']]['duration_us'])
            inp,_ = request_inputs(features,row['audio_sha256'],request,encode_controls(row['mode'],{},(a,b),origin_us=a))
            inputs,_ = collate([dict(inputs=inp,sequence=teacher_sequence(encode_events([],request),request))])
            inputs = pad_input_shapes(inputs,plan['input_shapes'])
            state = EventState(request,model.vocab)
            memory = model.encode(inputs); cache = model.start_cache(memory)
            details=[]
            for token in row['prediction']['tokens']:
                if time.monotonic()-started>180:
                    raise TimeoutError('Bounded fixed-trace audit exceeded180 seconds')
                field,t = state.query()
                previous = state.tokens[-1] if state.tokens else BOS
                logits = model.decode(torch.tensor([[previous]]),torch.tensor([[field]]),
                    torch.tensor([[t]],dtype=torch.float32),torch.ones(1,1,dtype=torch.bool),memory,cache=cache)[0,0]
                report['model_decode_calls']+=1
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite replay logits')
                support = state.allowed_tokens(); actual = rank(logits,support)
                if actual['top_token'] != token:
                    raise ValueError('Saved greedy trace did not replay exactly')
                entry = dict(index=len(state.tokens),phase=state.phase,query_s=t,token=token,
                             value=state.vocab.decision(token)[1],dynamic=actual)
                if state.phase.startswith('head_'):
                    static = static_head_support(state)
                    shadow_rank = rank(logits,static)
                    entry.update(static=shadow_rank,selected_legal_index=support.index(token),
                                 dynamic_min_token=support[0],static_top_removed=shadow_rank['top_token'] not in support)
                    if state.phase == 'head_ms':
                        lo,_ = state._heads_ms()
                        physical_lo = (max(request.start_us,state.last_onset+1)+999)//1000
                        onset_ms=state.head_block*2000+state.head_cell*40+entry['value']
                        entry.update(onset_ms=onset_ms,physical_lower_gap_ms=onset_ms-physical_lo,
                                     within_selected_cell_gap_ms=onset_ms-lo,
                                     ioi_us=onset_ms*1000-state.last_onset if state.last_onset>=0 else None)
                details.append(entry)
                state.apply(token)
            if [asdict(e) for e in sorted(state.events)] != row['prediction']['events'] or state.done != row['prediction']['complete']:
                raise ValueError('Fixed trace events differ from the original output')
            report['rows'].append(dict(record_index=row['record_index'],chart_id=row['chart_id'],role=row['role'],decisions=details))
            write_json(output,report)
        phases=defaultdict(list)
        for row in report['rows']:
            for d in row['decisions']:
                phases[d['phase']].append(d)
        report['summary']={phase:dict(n=len(items),forced=sum(d['dynamic']['size']==1 for d in items),
            tied=sum(d['dynamic']['tied_top_count']>1 for d in items),
            lower_choice=sum(d.get('selected_legal_index')==0 for d in items) if phase.startswith('head_') else None,
            static_top_removed=sum(d.get('static_top_removed',False) for d in items)) for phase,items in phases.items()}
        report.update(complete=True,wall_s=time.monotonic()-started,checkpoint_unchanged=sha(checkpoint)==source['checkpoint_sha256'])
        if not report['checkpoint_unchanged']:
            raise ValueError('Checkpoint changed during audit')
        write_json(output,report)
        print(json.dumps(dict(complete=True,wall_s=report['wall_s'],summary=report['summary'])))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(output,report)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('control','candidate'),required=True)
    parser.add_argument('--step',choices=(250,500,1000),type=int,required=True)
    args=parser.parse_args();run(args.arm,args.step)

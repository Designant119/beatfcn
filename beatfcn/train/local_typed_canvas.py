"""Lossless target transform and request-only inputs for typed canvas probes."""
from collections import Counter
from dataclasses import asdict
import math
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.local_typed_canvas import KINDS, CELL_US, FRAMES, SLOTS, COPIES, best_tail
from beatfcn.train.local_bundle import batch_rows


def request_layout(request):
    if request.mode != 'phira' or request.start_us % 1000:
        raise ValueError('Canvas experiment supports native-ms Phira windows')
    n = math.ceil((request.end_us-request.start_us)/CELL_US)
    if n > FRAMES:
        raise ValueError('Canvas frame resource overflow')
    base = request.start_us + np.arange(FRAMES)*CELL_US
    candidate = np.full((FRAMES, 40+COPIES), -1, np.int64)
    candidate[:, :40] = base[:, None] + np.arange(40)*1000
    for frame in range(n):
        copies = sorted(t for t in request.copy_times if base[frame] <= t < min(base[frame]+CELL_US, request.end_us))
        if len(copies) > COPIES:
            raise ValueError('Authorized-copy capacity overflow')
        candidate[frame, 40:40+len(copies)] = copies
    valid = np.arange(FRAMES) < n
    allowed = (candidate >= request.start_us) & (candidate < request.end_us) & valid[:, None]
    allowed[~valid, 0] = True  # finite dummy class; invalid frames never emit or train
    return dict(canvas_valid=valid, canvas_head_us=candidate, canvas_head_allowed=allowed,
                canvas_width_s=np.asarray((request.end_us-request.start_us)/1e6, np.float32))


def encode_canvas(events, request):
    layout = request_layout(request)
    count = np.zeros((FRAMES, 4), np.int64)
    shape = (FRAMES, 4, SLOTS)
    target = dict(count=count, offset=np.zeros(shape, np.int64), slot_valid=np.zeros(shape, bool),
                  tail_block=np.zeros((FRAMES, SLOTS), np.int64),
                  tail_cell=np.zeros((FRAMES, SLOTS), np.int64), tail_ms=np.zeros((FRAMES, SLOTS), np.int64))
    for e in sorted(events, key=lambda x:(x.onset_us, x.end_us, x.type, x.lane)):
        if e.type not in KINDS or not request.start_us <= e.onset_us < request.end_us or e.lane != -1:
            raise ValueError('Target outside typed canvas event contract')
        if (e.type == 'hold' and not e.onset_us < e.end_us <= request.duration_us) or (e.type != 'hold' and e.end_us != e.onset_us):
            raise ValueError('Invalid target duration')
        f = (e.onset_us-request.start_us)//CELL_US; k = KINDS.index(e.type); s = int(count[f, k])
        if s >= SLOTS:
            raise ValueError('Per-type event slot capacity overflow')
        options = np.flatnonzero(layout['canvas_head_allowed'][f] & (layout['canvas_head_us'][f] == e.onset_us))
        if not len(options):
            raise ValueError('Native onset lacks exact millisecond or authorized copy address')
        count[f, k] += 1; target['offset'][f, k, s] = options[0]; target['slot_valid'][f, k, s] = True
        if e.type == 'hold':
            if e.end_us % 1000:
                raise ValueError('Non-ms tail is outside the inherited codec contract')
            end = e.end_us//1000
            target['tail_block'][f, s], rest = divmod(end, 2000)
            target['tail_cell'][f, s], target['tail_ms'][f, s] = divmod(rest, 40)
    return target


def decode_target(target, request):
    heads = request_layout(request)['canvas_head_us']; events=[]
    for f,k in np.argwhere(target['count'] > 0):
        for s in range(int(target['count'][f,k])):
            head = int(heads[f, target['offset'][f,k,s]])
            end = ((int(target['tail_block'][f,s])*2000+int(target['tail_cell'][f,s])*40+int(target['tail_ms'][f,s]))*1000
                   if k == 1 else head)
            events.append(RhythmEvent(head,end,KINDS[k]))
    return sorted(events)


def canvas_batch(rows, device='cpu', *, mask_reference=False):
    # Existing collator handles only inputs/sequence. Targets are constructed
    # after input preparation and never passed to the model forward interface.
    inputs,_ = batch_rows(rows)
    layouts = [request_layout(r['request']) for r in rows]
    targets = [encode_canvas(r['targets'],r['request']) for r in rows]
    for key in layouts[0]:
        inputs[key]=torch.from_numpy(np.stack([v[key] for v in layouts]))
    if mask_reference:
        content = inputs['reference'][...,10] > 0
        inputs['reference_valid'] = inputs['reference_valid'] & ~content
        inputs['reference'] = inputs['reference'].masked_fill(content[...,None],0)
        inputs['reference_times'] = inputs['reference_times'].masked_fill(content,0)
    target = {k:torch.from_numpy(np.stack([v[k] for v in targets])) for k in targets[0]}
    return ({k:v.to(device) for k,v in inputs.items()}, {k:v.to(device) for k,v in target.items()})


@torch.no_grad()
def decode_canvas(model, inputs, requests, *, deadline=None):
    start=time.monotonic(); model.eval()
    output={k:v.float().cpu().numpy() for k,v in model(inputs).items()}
    heads=inputs['canvas_head_us'].cpu().numpy(); valid=inputs['canvas_valid'].cpu().numpy()
    counts=output['count'].argmax(-1); offsets=output['offset'].argmax(-1)
    records=[]
    for b,request in enumerate(requests):
        events=[]; failure=None; tokens=[]
        try:
            total=int((counts[b]*valid[b,:,None]).sum())
            if total > request.max_events:
                raise ValueError(f'Predicted event resource overflow: {total}')
            for f,k in np.argwhere((counts[b]>0) & valid[b,:,None]):
                if request.allowed_types is not None and KINDS[k] not in request.allowed_types:
                    raise ValueError('Predicted disallowed type')
                for s in range(int(counts[b,f,k])):
                    h=int(heads[b,f,offsets[b,f,k,s]])
                    if not request.start_us<=h<request.end_us:
                        raise ValueError('Predicted invalid native head')
                    end=best_tail(output['tail_block'][b,f,s],output['tail_cell'][b,f,s],output['tail_ms'][b,f,s],h,request.duration_us) if k==1 else h
                    events.append(RhythmEvent(h,end,KINDS[k]))
            events.sort()
            tokens=encode_events(events, request)
            if len(tokens)>request.max_tokens:
                raise ValueError('Native serialization token capacity overflow')
            if deadline is not None and time.monotonic()>deadline:
                raise TimeoutError('Canvas generation deadline')
        except (ValueError,TimeoutError) as exc:
            failure=f'{type(exc).__name__}: {exc}'
        records.append(dict(complete=failure is None,events=[asdict(e) for e in events],tokens=tokens,
                            resource_failure=failure,deadline_reached=failure is not None and 'deadline' in failure,
                            token_limit_reached=False,predicted_count=int((counts[b]*valid[b,:,None]).sum()),
                            wall_s=time.monotonic()-start))
    return records

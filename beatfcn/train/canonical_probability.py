"""Canonical native/COPY targets, separate from the inherited token codec."""
from dataclasses import replace

import numpy as np
import torch

from beatfcn.model.local_typed_canvas import CELL_US, FRAMES, KINDS, SLOTS
from .local_bundle import annotated_sequence, batch_rows
from .local_typed_canvas import request_layout


def explicit_row(row):
    visible = row.get('visible', ())
    request = replace(row['request'], known_carry=tuple(e for e in visible
        if e.type == 'hold' and e.onset_us < row['request'].start_us < e.end_us))
    # This non-token model only needs the collator's input half. An empty legal
    # sequence avoids routing native/COPY target tails through the old codec.
    return {**row, 'request':request, 'sequence':annotated_sequence([],request)}


def encode_canonical(events, request):
    layout=request_layout(request)
    shape=(FRAMES,4,SLOTS)
    target=dict(count=np.zeros((FRAMES,4),np.int64),offset=np.zeros(shape,np.int64),tail_us=np.zeros(shape,np.int64))
    for e in sorted(events,key=lambda e:(e.onset_us,e.end_us,e.type,e.lane)):
        if e.type not in KINDS or e.lane!=-1 or not request.start_us<=e.onset_us<request.end_us:
            raise ValueError('Target outside canonical event support')
        if request.allowed_types is not None and e.type not in request.allowed_types:
            raise ValueError('Target kind outside request support')
        if e.type=='hold':
            if not e.onset_us<e.end_us<=request.duration_us:raise ValueError('Illegal hold')
            if e.end_us%1000 and e.end_us not in request.copy_times:raise ValueError('Unauthorized tail COPY')
        elif e.end_us!=e.onset_us:raise ValueError('Non-hold duration')
        frame=(e.onset_us-request.start_us)//CELL_US;kind=KINDS.index(e.type)
        slot=int(target['count'][frame,kind])
        if slot>=SLOTS:raise ValueError('Canonical package capacity overflow')
        options=np.flatnonzero(layout['canvas_head_allowed'][frame] & (layout['canvas_head_us'][frame]==e.onset_us))
        if not len(options):raise ValueError('Unauthorized head address')
        target['count'][frame,kind]+=1
        target['offset'][frame,kind,slot]=options[0]
        target['tail_us'][frame,kind,slot]=e.end_us
    # Derive exact clock divisions on CPU before any MPS transfer. MPS2.6
    # integer division of large absolute microseconds may round via float32.
    target['tail_ms_native']=target['tail_us']//1000
    return target


def probability_batch(rows,device='cpu',*,mask_reference=False):
    clean=[explicit_row(row) for row in rows]
    inputs,_=batch_rows(clean)
    layouts=[request_layout(row['request']) for row in clean]
    for layout in layouts:
        for frame in range(FRAMES):
            seen=set()
            for index in np.flatnonzero(layout['canvas_head_allowed'][frame]):
                value=int(layout['canvas_head_us'][frame,index])
                if value in seen:layout['canvas_head_allowed'][frame,index]=False
                else:seen.add(value)
    for key in layouts[0]:inputs[key]=torch.from_numpy(np.stack([v[key] for v in layouts]))
    if mask_reference:
        content=inputs['reference'][...,10]>0
        inputs['reference_valid']=inputs['reference_valid'] & ~content
        inputs['reference']=inputs['reference'].masked_fill(content[...,None],0)
        inputs['reference_times']=inputs['reference_times'].masked_fill(content,0)
    targets=[encode_canonical(row['targets'],row['request']) for row in clean]
    target={k:torch.from_numpy(np.stack([t[k] for t in targets])) for k in targets[0]}
    return ({k:v.to(device) for k,v in inputs.items()},{k:v.to(device) for k,v in target.items()})

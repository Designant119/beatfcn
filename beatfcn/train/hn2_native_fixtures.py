"""Deterministic public tensors;zero model construction or feature downloads."""
import math
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from .hn2_native import prepare_public_support,validate_targets
from .typed_inventory import carry_reference_row


def native_fixtures():
    carry=RhythmEvent(0,1_000_000,'hold',-1)
    request=EventRequest('phira',1000,5000,4_123_456,
        copy_times=(1501,1502,2_040_001,4_123_456),known_carry=(carry,carry),max_events=512,max_tokens=8192)
    g=torch.Generator().manual_seed(34)
    x=dict(local=torch.randn(1,2,8,1,1024,generator=g)*.1,local_times=(torch.arange(8)[None]+.5)*.04,
        local_valid=torch.ones(1,8,dtype=torch.bool),mel=torch.randn(1,32,128,generator=g)*.1,
        mel_times=(torch.arange(32)[None]+.5)*.01,mel_valid=torch.ones(1,32,dtype=torch.bool),
        song=torch.randn(1,9,1024,generator=g)*.1,song_times=(torch.arange(9)[None]+.5)*.5-.001,
        song_valid=torch.ones(1,9,dtype=torch.bool),controls=torch.zeros(1,8,6),mode=torch.ones(1,dtype=torch.long))
    n=math.ceil(request.duration_us/10000);starts=torch.arange(n,dtype=torch.int64)*10000
    ends=(starts+10000).clamp(max=request.duration_us)
    x.update(whole_mel=torch.randn(1,n,128,generator=g)*.1,
        whole_mel_times=((starts+ends).double()/2/1e6-.001).float()[None],
        whole_mel_valid=torch.ones(1,n,dtype=torch.bool),
        whole_mel_intervals=(torch.stack((starts,ends),-1).double()/1e6-.001).float()[None])
    row=dict(request=request,visible=(),inputs={},index=-1,view='draft',source_role='human_input',known_regions=((2000,3000),))
    ref=carry_reference_row(row,'0'*64,regions=((10000,20000),))
    for key in ('reference','reference_times','reference_valid'):x[key]=torch.from_numpy(ref['inputs'][key])[None]
    x.update(prepare_public_support([request],[((2000,3000),)]))
    target=torch.zeros(1,512,4,dtype=torch.int64);valid=torch.zeros(1,512,dtype=torch.bool)
    rows=[[1000,1000,0,-1],[1501,1502,1,-1],[1501,4_123_456,1,-1],
          [1501,4_123_456,1,-1],[4000,4000,2,-1],[4000,4000,3,-1]]
    target[0,:len(rows)]=torch.tensor(rows);valid[0,:len(rows)]=True
    validate_targets(x,target,valid)
    protected={**x,**prepare_public_support([request],[((1000,5000),)])}
    return [dict(name='copy-short-duplicate-long-tail-carry-known-blank',inputs=x,targets=target,target_valid=valid,
                 fixture_metadata=dict(carry_additions=len(ref['carry_reference_additions']),tail_outside_local=True)),
            dict(name='fully-protected-empty',inputs=protected,targets=torch.zeros_like(target),target_valid=torch.zeros_like(valid),fixture_metadata={})]

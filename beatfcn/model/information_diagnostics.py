"""Target-informed decision probes; their probabilities are not product scores."""
import math
from copy import copy

import numpy as np
import torch

from . import features as F
from .batch import CORE_CELLS, MS_PER_CELL, collate
from .generation_batch import GenerationBatch
from .p1 import NEG, _ms_floor


def prefix_states(model, example, song, rows, *, device='cpu'):
    """Yield actual cached-generation states; consume each before advancing.

    Earlier true groups are supplied as a diagnostic prefix. Future groups never
    enter the generation batch. The returned batch is mutable on the next yield.
    """
    if not rows or min(rows) < 0 or max(rows) >= len(example.groups):
        raise ValueError('prefix rows must address existing target groups')
    work = copy(example)
    work.groups = []
    work.right = example.right[:1].copy()
    metadata = GenerationBatch(collate([work], [song], model.c.taps, model.c.top_tap), work, device)
    cache = {}
    with torch.no_grad():
        for row in range(max(rows) + 1):
            if row == 0:
                h, pointer, _ = model.encode(metadata.bt, cache)
                h = h[:, :1]
            else:
                h, pointer, _ = model.encode_step(metadata.bt, cache)
            if row in rows:
                yield row, metadata.bt, h, pointer
            if row < max(rows):
                metadata.append(example.groups[row], example.right[row + 1])


@torch.no_grad()
def decisions(model, bt, hrow, pointer, group, row):
    """Next time plus action heads at a fixed true onset and true first note.

    The fixed onset removes timing as a confound for type response; continuation
    and lane are conditional on the first true note/type. Never interpret these
    target-informed action distributions as freely generated model performance.
    """
    if model.c.selection or model.c.time_head != 'pointer' or bt['mode'].numel() != 1:
        raise ValueError('this diagnostic is for the single-core pointer baseline')
    dev = hrow.device
    previous = bt['grp_t'][:, row-1:row] if row else torch.full((1,1),-.001,device=dev,dtype=torch.float32)
    logits, mask, _ = model.time_logits(bt,hrow,pointer,previous,row_start=row)
    time_prob = logits.float().masked_fill(~mask,NEG).softmax(-1)[0,0]
    onset = torch.tensor([group['t_rel_us']/F.SEC],device=dev,dtype=torch.float32)
    right = torch.as_tensor(np.asarray(group['right_new']),device=dev)[None]
    busy = torch.as_tensor(np.asarray(group['busy_lanes']),device=dev)[None]
    history = model.history_features(bt,onset[:,None],start=row)
    state = model.gru_init(hrow[:,0],pointer,onset,right,busy,None if history is None else history[:,0])
    state = model.gru_cell(model.start[None],state)
    allowed = bt['allowed_types'] & model.mode_types[bt['mode']]
    type_prob = model.type_h(state).float().masked_fill(~allowed,NEG).softmax(-1)[0]
    note = group['notes'][0]
    integers = [torch.tensor([note[k]],device=dev) for k in ('type','lane','line','side')]
    x = torch.tensor([note['x']],device=dev,dtype=torch.float32)
    duration = torch.tensor([note.get('end_rel_us',0)/F.SEC],device=dev,dtype=torch.float32)
    if int(bt['mode'][0]) == 1 and not model.c.phira_location:
        integers[2],integers[3],x = integers[2].new_full((1,),-1),integers[3].new_full((1,),-1),torch.zeros_like(x)
    state_next = model.gru_cell(model._note_emb(*integers,x,duration),state)
    cont_prob = model.cont(state_next).float().softmax(-1)[0]
    lane_prob = None
    if int(bt['mode'][0]) == 0:
        typed = state + model.type_c(integers[0])
        lane_prob = model.lane_h(torch.cat([typed,busy.float()],-1)).float().masked_fill(busy,NEG).softmax(-1)[0]

    h0 = F.HALO_US//F.CELL_US
    qa = model.q_a(hrow).float()
    cells = pointer[:,h0:h0+CORE_CELLS].float()
    audio = torch.logsumexp(torch.einsum('byp,bcnp->bycn',qa,cells)/math.sqrt(qa.shape[-1]),-1)[0,0]
    cell_ms = torch.arange(CORE_CELLS,device=dev)*MS_PER_CELL+MS_PER_CELL//2
    bucket = (torch.log2((cell_ms[None,None]-_ms_floor(previous)[...,None]).clamp_min(0)/10+1)*6).long().clamp(max=63)
    relative = model.rel_bias(bucket).squeeze(-1)[0,0]
    state_bias = model.state_bias(bt['cell_state'][:,h0:h0+CORE_CELLS]).squeeze(-1)[0]
    kernel = model._kernel(bt['prev_onsets'][:,row:row+1],cell_ms.float()[None,None]/1000)[0,0]
    if not torch.allclose(audio+relative+state_bias+kernel,logits[0,0,1:1+CORE_CELLS].float(),atol=1e-4,rtol=1e-4):
        raise ValueError('pointer decomposition does not match real head')
    target = 1+group['t_rel_us']//F.CELL_US
    selected = int(time_prob.argmax())
    return dict(time_probability=time_prob.cpu().tolist(),type_probability=type_prob.cpu().tolist(),
        continuation_probability=cont_prob.cpu().tolist(),lane_probability=lane_prob.cpu().tolist() if lane_prob is not None else None,
        time_argmax=selected,true_cell_index=target,true_cell_probability=float(time_prob[target]),
        true_cell_rank=1+int(((logits[0,0]>logits[0,0,target]) & mask[0,0]).sum()),
        type_argmax=int(type_prob.argmax()),true_first_type=note['type'],true_type_probability=float(type_prob[note['type']]),
        hidden_norm=float(hrow.float().norm()),
        pointer_components={key:value.cpu().tolist() for key,value in
            [('audio',audio),('relative_time',relative),('cell_state',state_bias),('rhythm_kernel',kernel)]},
        target_informed=True)


def response(left,right):
    result={}
    for head in ('time','type','continuation','lane'):
        a,b=left[head+'_probability'],right[head+'_probability']
        if a is None and b is None: continue
        if a is None or b is None:
            raise ValueError('incompatible action heads')
        a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        if (a.ndim!=1 or not a.size or a.shape!=b.shape or not np.isfinite(a).all()
                or not np.isfinite(b).all() or min(a.min(),b.min())<0
                or not np.isclose(a.sum(),1,atol=1e-6) or not np.isclose(b.sum(),1,atol=1e-6)):
            raise ValueError('invalid probability comparison')
        result[head]=dict(total_variation=float(np.abs(a-b).sum()/2),
            maximum_absolute_difference=float(np.abs(a-b).max()),argmax_changed=bool(a.argmax()!=b.argmax()))
    return result

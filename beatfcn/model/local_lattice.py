"""10ms addressed non-AR lattice; exact-ms/COPY events, no type quotas.

This explicitly tests the older lattice+counterfactual recipe as a complete
alternative. Focal/log-prior objectives are not claimed to yield calibrated
natural sampling probabilities. Deterministic direct decoding is primary.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .local_typed_canvas import LocalTypedCanvas,TemporalBlock,KINDS
from .local_bundle import read_grid
from .event_canvas import RhythmEvent


FRAMES=800
SLOTS=12
COPIES=16


class LatticeCanvas(LocalTypedCanvas):
    def __init__(self,config,*,density_reference,type_prior):
        super().__init__(config,density_reference=density_reference)
        for name in ('frame_read','quantity','existence','kind_slot','slot_read','offset','tail_block','tail_cell','tail_ms','temporal'):
            delattr(self,name)
        d=config.d
        self.lattice_temporal=nn.ModuleList(TemporalBlock(d,r) for r in (1,1,2,2))
        self.presence=nn.Linear(d,1);self.multiplicity=nn.Linear(d,SLOTS)
        self.slot_ids=nn.Parameter(torch.randn(SLOTS,d)*.02)
        self.attributes=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
        self.kind=nn.Linear(d,4);self.residual=nn.Linear(d,1)
        self.head_copy=nn.Linear(d,1+COPIES)
        self.duration_fraction=nn.Linear(d,1)
        self.tail_copy=nn.Linear(d,513)
        nn.init.constant_(self.duration_fraction.bias,-7.)
        self.ratio_alpha=nn.Parameter(torch.full((3,),.54132485))
        self.register_buffer('type_prior',torch.tensor(type_prior,dtype=torch.float32))

    def forward(self,inputs):
        controls=inputs['controls'].clone()
        # Density and hold/flick/drag preferences each have one explicit path.
        controls[:,[0,2,6,7]]=0
        memory=self.encode({**inputs,'controls':controls})
        n=inputs['local'].shape[2]*self.c.band_slots;m=inputs['mel'].shape[1]
        query=torch.arange(FRAMES,device=controls.device).float()[None]*.01+.005
        x=read_grid(memory['values'][:,n:n+m],inputs['mel_times'],inputs['mel_valid'],query,.01)
        clock=torch.stack((query/8,(query/8).square(),torch.sin(query*math.pi),torch.cos(query*math.pi)),-1)
        x=x+self.clock_read(clock)+self.mode_ids(inputs['mode'])[:,None]
        visible=inputs['reference_valid'] & (inputs['reference'][...,10]>0)
        style=(inputs['reference']*visible[...,None]).sum(1)/visible.sum(1,keepdim=True).clamp_min(1)
        scale,bias=self.direct(controls.flatten(1)).chunk(2,-1)
        x=x*(1+.1*scale[:,None].tanh())+bias[:,None]+self.style(style)[:,None]
        for layer in self.address:x=layer(x,query,memory)
        for layer in self.lattice_temporal:x=layer(x,inputs['lattice_valid'])
        x=self.norm(x)
        density=inputs['controls'][:,0];rho=density[:,:2].mean(-1)*64
        tilt=torch.where(density[:,2]>.5,F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        presence=self.presence(x).squeeze(-1)+tilt[:,None]
        slots=x[:,:,None]+self.slot_ids[None,None];slots=slots+self.attributes(slots)
        kinds=self.kind(slots)
        for j,(field,kind) in enumerate(((2,1),(6,2),(7,3))):
            band=inputs['controls'][:,field];ratio=band[:,:2].mean(-1).clamp(.001,.999)
            baseline=self.type_prior[kind].clamp(.001,.999)
            bias=torch.where(band[:,2]>.5,F.softplus(self.ratio_alpha[j])*(torch.logit(ratio)-torch.logit(baseline)),0.)
            unit=F.one_hot(torch.tensor(kind,device=x.device),4).to(x.dtype)
            kinds=kinds+bias[:,None,None,None]*unit
        kinds=kinds.masked_fill(~inputs['lattice_kind_allowed'][:,None,None],-1e9)
        head_copy=self.head_copy(slots).masked_fill(~inputs['lattice_copy_allowed'][:,:,None],-1e9)
        return dict(presence=presence,multiplicity=self.multiplicity(x),kind=kinds,
            residual=self.residual(slots).squeeze(-1),head_copy=head_copy,
            duration_fraction=self.duration_fraction(slots).squeeze(-1),tail_copy=self.tail_copy(slots))


def expectation(output,inputs):
    number=(output['multiplicity'].softmax(-1)*torch.arange(1,SLOTS+1,device=output['presence'].device)).sum(-1)
    expected=output['presence'].sigmoid()*number*inputs['lattice_valid']
    # Slot participation is the exact positive-count survival function.
    survival=output['multiplicity'].softmax(-1).flip(-1).cumsum(-1).flip(-1)
    types=(output['kind'].softmax(-1)*survival[...,None]*output['presence'].sigmoid()[...,None,None]*inputs['lattice_valid'][...,None,None]).sum((1,2))
    return expected.sum(-1),types


def lattice_loss(model,inputs,target):
    out=model(inputs);valid=inputs['lattice_valid'];occupied=(target['count']>0)&valid
    label=occupied.float();bce=F.binary_cross_entropy_with_logits(out['presence'],label,reduction='none')
    correct=torch.where(occupied,out['presence'].sigmoid(),1-out['presence'].sigmoid())
    presence=(bce*(1-correct).square()*valid).sum()/valid.sum().clamp_min(1)
    positive=F.cross_entropy(out['multiplicity'][occupied],target['count'][occupied]-1) if bool(occupied.any()) else presence*0
    active=target['active'];zero=presence*0
    if bool(active.any()):
        # Train-time log-prior adjustment; inference uses the raw decision
        # logits. This is deliberate class-balanced action, not calibration.
        logits=out['kind']+model.type_prior.clamp_min(1e-8).log()
        losses=[]
        for kind in range(4):
            mask=active & (target['kind']==kind)
            if bool(mask.any()):losses.append(F.cross_entropy(logits[mask],target['kind'][mask]))
        kinds=sum(losses)/len(losses)
        head_copy=F.cross_entropy(out['head_copy'][active],target['head_copy'][active])
        native=active & (target['head_copy']==0)
        offset=out['residual'].sigmoid()*inputs['lattice_native_max'][...,None]
        residual=F.smooth_l1_loss(offset[native],target['offset_ms'][native]) if bool(native.any()) else zero
        holds=active & (target['kind']==1)
        if bool(holds.any()):
            head=target['head_us'][holds];first=target['head_ms_native'][holds]+1
            maximum=inputs['duration_ms'][0]-first
            duration=1+out['duration_fraction'][holds].sigmoid()*maximum.clamp_min(0)
            actual=target['tail_ms_native'][holds]-target['head_ms_native'][holds]
            native_tail=target['tail_copy'][holds]==0
            tail=F.smooth_l1_loss(torch.log1p(duration[native_tail]),torch.log1p(actual[native_tail].float())) if bool(native_tail.any()) else zero
            anchors=inputs['tail_copy_us'][0]
            permitted=(anchors[None]>head[:,None]) & (anchors[None]<=inputs['duration_us'][0]) & inputs['tail_copy_is_subms']
            # Native branch is legal exactly when a native endpoint exists.
            allowed=torch.cat(((first<=inputs['duration_ms'][0])[:,None],permitted),-1)
            tailcopy=F.cross_entropy(out['tail_copy'][holds].masked_fill(~allowed,-1e9),target['tail_copy'][holds])
        else:tail=tailcopy=zero
    else:kinds=head_copy=residual=tail=tailcopy=zero
    parts=dict(presence=presence,multiplicity=positive,kind=kinds,head_copy=head_copy,residual=residual,tail=tail,tail_copy=tailcopy)
    return sum(parts.values()),parts


def counterfactual_hinge(output,inputs,*,field_index,lower,upper):
    number,types=expectation(output,inputs)
    if field_index==0:
        value=number/(inputs['end_s']-inputs['origin_s'])
    else:
        kind={2:1,6:2,7:3}[field_index]
        value=types[:,kind]/number.clamp_min(1e-6)
    return (F.relu(lower-value)+F.relu(value-upper)).mean()


@torch.no_grad()
def generate_lattice(model,inputs,request):
    model.eval();out=model(inputs);events=[]
    counts=out['multiplicity'].argmax(-1)+1
    counts=torch.where((out['presence']>=0)&inputs['lattice_valid'],counts,0)[0]
    if int(counts.sum())>request.max_events:raise ValueError('Lattice event resource overflow')
    kinds=out['kind'][0].argmax(-1);copy=out['head_copy'][0].argmax(-1)
    offsets=torch.round(out['residual'][0].sigmoid()*inputs['lattice_native_max'][0,:,None]).long()
    for frame in (counts>0).nonzero().flatten().tolist():
        base=request.start_us+frame*10000
        for slot in range(int(counts[frame])):
            kind=int(kinds[frame,slot]);pointer=int(copy[frame,slot])
            head=int(inputs['lattice_copy_us'][0,frame,pointer-1]) if pointer else base+int(offsets[frame,slot])*1000
            if not request.start_us<=head<request.end_us:raise ValueError('Lattice illegal head')
            end=head
            if kind==1:
                first=head//1000+1;last=request.duration_us//1000
                anchors=inputs['tail_copy_us'][0]
                allowed=torch.cat((torch.tensor([first<=last],device=anchors.device),
                    (anchors>head)&(anchors<=request.duration_us)&inputs['tail_copy_is_subms'][0]))
                if not bool(allowed.any()):raise ValueError('Lattice hold lacks legal endpoint')
                chosen=int(out['tail_copy'][0,frame,slot].masked_fill(~allowed,-1e9).argmax())
                if chosen:end=int(anchors[chosen-1])
                else:end=(first+int(torch.round(out['duration_fraction'][0,frame,slot].sigmoid()*(last-first))))*1000
            events.append(RhythmEvent(head,end,KINDS[kind]))
    return sorted(events)

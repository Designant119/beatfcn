"""Rectified flow in an exactly reversible finite event code.

Count is one variable per cell/type, with16 ordered complete members. This is
continuous Gaussian-to-code velocity matching and16 Euler steps, not discrete
mask diffusion. Frozen native/COPY sets define the final nearest-code decision.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .event_canvas import RhythmEvent
from .local_mamba_canvas import MambaCanvas
from .local_typed_canvas import TemporalBlock,FRAMES,SLOTS,KINDS,best_tail
from .local_bundle import read_grid


FIELDS=5
CODE_DIM=1+SLOTS*FIELDS


def encode_event_code(target,request):
    count=target['count']
    if request.start_us%1000:raise ValueError('Native-ms cell origin required')
    base=request.start_us//1000+torch.arange(FRAMES,device=count.device)*40
    native_ms=target['tail_ms_native']-base[None,:,None,None]
    copies=list(request.copy_times)
    pointer=torch.zeros_like(native_ms)
    for i,t in enumerate(copies):
        if t%1000:pointer=torch.where(target['tail_us']==t,i+1,pointer)
    is_hold=torch.arange(4,device=count.device)==1
    member_valid=torch.arange(SLOTS,device=count.device)<count[...,None]
    pointer=torch.where(member_valid & is_hold[None,None,:,None],pointer,0)
    native_hold=member_valid & is_hold[None,None,:,None] & (pointer==0)
    delta=torch.where(native_hold,native_ms,0)
    head=target['offset'].float()/55
    # Count/head/pointer and three native tail digits form a finite reversible
    # code; no learned autoencoder is needed and no target gets discarded.
    attributes=torch.stack((head,delta.float().div(2000,rounding_mode='floor')/600,
        (delta%2000).float().div(40,rounding_mode='floor')/49,(delta%40).float()/39,pointer.float()/512),-1)
    attributes=torch.where(member_valid[...,None],attributes,torch.zeros_like(attributes))
    return torch.cat((count.float()[...,None]/SLOTS,attributes.flatten(-2)), -1)*2-1


def decode_event_code(code,inputs,request):
    values=(code.detach().cpu().double().numpy()[0]+1)/2
    valid=inputs['canvas_valid'][0].cpu().numpy();heads=inputs['canvas_head_us'][0].cpu().numpy()
    allowed=inputs['canvas_head_allowed'][0].cpu().numpy()
    counts=np.abs(values[...,:1]*SLOTS-np.arange(SLOTS+1)).argmin(-1)
    if request.allowed_types is not None:
        counts=np.where(np.asarray([k in request.allowed_types for k in KINDS])[None],counts,0)
    counts=np.where(valid[:,None],counts,0)
    if int(counts.sum())>request.max_events:raise ValueError('Event-code flow resource overflow')
    attrs=values[...,1:].reshape(FRAMES,4,SLOTS,FIELDS);events=[]
    for frame,kind in np.argwhere(counts>0):
        base=request.start_us+int(frame)*40000
        addresses=np.flatnonzero(allowed[frame])
        for slot in range(int(counts[frame,kind])):
            vector=attrs[frame,kind,slot]
            offset=int(addresses[np.abs(vector[0]*55-addresses).argmin()]);head=int(heads[frame,offset]);end=head
            if kind==1:
                # Nearest complete legal endpoint under the declared code
                # metric; not independent digit rounding or a repaired tail.
                native=None;native_score=-np.inf
                if head//1000+1<=request.duration_us//1000:
                    fields=[-(np.arange(n)/scale-value)**2 for n,scale,value in
                        ((601,600,vector[1]),(50,49,vector[2]),(40,39,vector[3]))]
                    delta=best_tail(*fields,head-base,request.duration_us-base)//1000
                    native=base+delta*1000
                    native_score=fields[0][delta//2000]+fields[1][delta%2000//40]+fields[2][delta%40]-vector[4]**2
                candidates=[] if native is None else [(float(native_score),native)]
                for i,t in enumerate(request.copy_times):
                    if t%1000 and head<t<=request.duration_us:
                        # Native-digit coordinates are zero for a COPY tail.
                        score=-sum(vector[j]**2 for j in (1,2,3))-(vector[4]-(i+1)/512)**2
                        candidates.append((float(score),t))
                if not candidates:raise ValueError('Flow hold has no legal native/COPY endpoint')
                end=max(candidates,key=lambda p:(p[0],-p[1]))[1]
            events.append(RhythmEvent(head,end,KINDS[kind]))
    return sorted(events)


class EventCodeFlow(MambaCanvas):
    def __init__(self,config,*,density_reference):
        super().__init__(config,density_reference=density_reference)
        for name in ('quantity','existence','kind_slot','slot_read','offset','tail_block','tail_cell','tail_ms'):
            delattr(self,name)
        self.code_input=nn.Linear(CODE_DIM,config.d)
        self.noise_input=nn.Sequential(nn.Linear(4,config.d),nn.SiLU(),nn.Linear(config.d,config.d))
        self.density_condition=nn.Linear(config.control_dim,config.d)
        self.flow_kind=nn.Embedding(4,config.d)
        self.flow_temporal=nn.ModuleList(TemporalBlock(config.d,r) for r in (1,2,4,8))
        self.flow_output=nn.Sequential(nn.Linear(config.d,2*config.d),nn.SiLU(),nn.Linear(2*config.d,CODE_DIM))

    def encode_audio(self,inputs):
        controls=inputs['controls'].clone();controls[:,0]=0
        memory=self.encode({**inputs,'controls':controls})
        frame_valid=inputs['canvas_valid'][0]
        query=(torch.arange(FRAMES,device=controls.device).float()[None]*.04)[:,frame_valid]
        four=query[...,None]+query.new_tensor([.005,.015,.025,.035])
        mel=read_grid(memory['audio'],memory['audio_times'],memory['audio_valid'],four,.01)
        x=self.frame_read(mel.flatten(-2))
        clock=torch.stack((query/8,(query/8).square(),torch.sin(query*torch.pi),torch.cos(query*torch.pi)),-1)
        x=x+self.clock_read(clock)+self.mode_ids(inputs['mode'])[:,None]
        visible=inputs['reference_valid'] & (inputs['reference'][...,10]>0)
        style=(inputs['reference']*visible[...,None]).sum(1)/visible.sum(1,keepdim=True).clamp_min(1)
        scale,bias=self.direct(controls.flatten(1)).chunk(2,-1)
        x=x*(1+.1*scale[:,None].tanh())+bias[:,None]+self.style(style)[:,None]
        context=memory['values'];n=x.shape[1]
        for layer in self.address:
            joint=layer(torch.cat((x,context),1));x,context=joint[:,:n],joint[:,n:]
        for layer in self.temporal:x=layer(x)
        compact=self.norm(x)
        # Request-layout validity is a contiguous prefix. Functional padding is
        # supported by MPS2.6, whereas aten::index_copy.out is not.
        if not torch.equal(frame_valid,torch.arange(FRAMES,device=frame_valid.device)<compact.shape[1]):
            raise ValueError('Event-code flow requires contiguous request frames')
        return F.pad(compact,(0,0,0,FRAMES-compact.shape[1]))

    def forward(self,inputs,state,times,*,audio=None):
        if audio is None:audio=self.encode_audio(inputs)
        t=times[:,None];clock=torch.cat((t,1-t,torch.sin(t*torch.pi),torch.cos(t*torch.pi)),-1)
        condition=self.noise_input(clock)+self.density_condition(inputs['controls'][:,0])
        x=self.code_input(state)+audio[:,:,None]+self.flow_kind.weight[None,None]+condition[:,None,None]
        context=x.mean(2)
        for layer in self.flow_temporal:context=layer(context,inputs['canvas_valid'])
        return self.flow_output(x+context[:,:,None])


def flow_loss(model,inputs,target,request,*,time_value=None,noise=None):
    code=encode_event_code(target,request)
    if time_value is None:time_value=torch.rand(len(code),device=code.device)
    if noise is None:noise=torch.randn_like(code)
    t=time_value[:,None,None,None]
    state=(1-t)*noise+t*code;velocity=code-noise
    predicted=model(inputs,state,time_value)
    valid=inputs['canvas_valid'][...,None,None]
    squared=(predicted-velocity).square()
    return (squared*valid).sum()/(valid.sum().clamp_min(1)*4*CODE_DIM)


@torch.no_grad()
def generate_flow(model,inputs,request,*,seed,steps=16,deadline=None):
    model.eval();audio=model.encode_audio(inputs)
    rng=np.random.default_rng(seed)
    state=torch.from_numpy(rng.standard_normal((1,FRAMES,4,CODE_DIM)).astype(np.float32)).to(audio.device)
    for step in range(steps):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Flow integration deadline')
        t=audio.new_tensor([step/steps]);state=state+model(inputs,state,t,audio=audio)/steps
    if not bool(torch.isfinite(state).all()):raise ValueError('Flow nonfinite terminal code')
    return decode_event_code(state,inputs,request),dict(ode_steps=steps,seed=seed,audio_encodes=1)

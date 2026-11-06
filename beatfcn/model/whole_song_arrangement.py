"""From-scratch full-song mel/flux + four-state canonical event emitter."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .arrangement_iohmm import forward_log_probability,prior_path
from .canonical_event_package import CanonicalPackageCanvas,package_support
from .cardinality_dp import cardinality_counts
from .full_event_model import EncoderBlock,RMSNorm
from .local_bundle import read_grid
from .local_event_package import _hurdle_count_log_probabilities
from .local_typed_canvas import CanvasBlock,TemporalBlock,FRAMES,SLOTS


class WholeSongArrangement(CanonicalPackageCanvas):
    def __init__(self,config,*,density_reference):
        super().__init__(config,density_reference=density_reference)
        # None of the frozen pretrained projections/encoders survive this path.
        for name in ('layer_mix','local_projection','band_embedding','band_queries','song_projection',
                     'source_ids','mel_source','encoders','address','mel_projection','control_projection','control_ids','empty_reference'):
            delattr(self,name)
        self.fine_projection=nn.Linear(config.mel_bins*4,config.d)
        self.fine_blocks=nn.ModuleList(TemporalBlock(config.d,r) for r in (1,2,4,8))
        self.song_input=nn.Linear(config.mel_bins*16+12,config.d)
        self.song_operators='CCGCCGCCGCCG'
        self.song_blocks=nn.ModuleList(EncoderBlock(config) if op=='G' else TemporalBlock(config.d,1)
            for op in self.song_operators)
        self.context_read=CanvasBlock(config)
        self.state_film=nn.Embedding(4,2*config.d)
        self.initial_read=nn.Linear(config.d,4)
        self.transition_read=nn.Linear(config.d,16)
        self.transition_bias=nn.Parameter((torch.eye(4)*.6+.1).log()+torch.randn(4,4)*.01)
        nn.init.zeros_(self.initial_read.weight);nn.init.zeros_(self.initial_read.bias)
        nn.init.zeros_(self.transition_read.weight);nn.init.zeros_(self.transition_read.bias)
        nn.init.normal_(self.state_film.weight,std=.02)

    def forward(self,inputs):
        if len(inputs['mode'])!=1:raise ValueError('Whole-song microbatch one')
        # Input dictionary deliberately contains no local/song pretrained keys.
        control=inputs['controls'].clone();density=control[:,0].clone();control[:,0]=0
        scale,bias=self.direct(control.flatten(1)).chunk(2,-1)
        style_mask=inputs['reference_valid'] & (inputs['reference'][...,10]>0)
        style=(inputs['reference']*style_mask[...,None]).sum(1)/style_mask.sum(1,keepdim=True).clamp_min(1)
        common=bias+self.style(style)
        song=self.song_input(inputs['whole_song'])+common[:,None]
        song_valid=inputs['whole_song_valid'];song_times=inputs['whole_song_times']
        if song.shape[1]>600:raise ValueError('Full-song local resource overflow')
        for op,layer in zip(self.song_operators,self.song_blocks):
            song=layer(song,song_times,song_valid) if op=='G' else layer(song,song_valid)
        reference=self.reference_projection(inputs['reference'])
        memory=dict(values=torch.cat((song,reference),1),
            times=torch.cat((song_times,inputs['reference_times']),1),
            valid=torch.cat((song_valid,inputs['reference_valid']),1),
            timed=torch.cat((song_valid,inputs['reference_valid']),1))
        fine=self.fine_projection(inputs['mel_flux'])
        for layer in self.fine_blocks:fine=layer(fine,inputs['mel_valid'])
        query=torch.arange(FRAMES,device=fine.device).float()[None]*.04
        four=query[...,None]+fine.new_tensor([.005,.015,.025,.035])
        gathered=read_grid(fine,inputs['mel_times'],inputs['mel_valid'],four,.01)
        x=self.frame_read(gathered.flatten(-2))
        clock=torch.stack((query/8,(query/8).square(),torch.sin(query*math.pi),torch.cos(query*math.pi)),-1)
        x=x+self.clock_read(clock)+self.mode_ids(inputs['mode'])[:,None]
        x=x*(1+.1*scale[:,None].tanh())+common[:,None]
        x=self.context_read(x,query+.02,memory)
        for layer in self.temporal:x=layer(x,inputs['canvas_valid'])
        film=self.state_film.weight
        a,b=film.chunk(2,-1)
        states=self.norm(x[:,None]*(1+.1*a[None,:,None].tanh())+b[None,:,None])
        gate=self.count_gate(states)
        positive=self.count_positive(states).view(1,4,FRAMES,4,SLOTS)
        logp=_hurdle_count_log_probabilities(gate,positive)
        rho=density[:,:2].mean(-1)*64
        tilt=torch.where(density[:,2]>.5,F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        logp=logp+tilt[:,None,None,None,None]*torch.arange(17,device=x.device)
        logp=logp-logp.logsumexp(-1,keepdim=True)
        initial=self.initial_read(song[:,0])[0].log_softmax(-1)
        transitions=(self.transition_read(song)[0].view(-1,4,4)+self.transition_bias).log_softmax(-1)
        return dict(x=states[0],count_log_probs=logp[0],initial=initial,transition=transitions)


def arrangement_loss(model,inputs,target,request):
    output=model(inputs);groups=package_support(inputs,request)[0];count=target['count'][0]
    if bool(((count>0)&~groups).any()):raise ValueError('Target outside arrangement support')
    nblocks=len(output['transition'])
    block_index=(request.start_us+torch.arange(FRAMES,device=count.device)*40_000)//2_000_000
    validframes=inputs['canvas_valid'][0]
    if bool((block_index[validframes]>=nblocks).any()):raise ValueError('Whole-song target clock outside context')
    emissions=[];count_diagnostics=[];onset_diagnostics=[];tail_diagnostics=[]
    for state in range(4):
        count_nll=-output['count_log_probs'][state].gather(-1,count[...,None]).squeeze(-1)
        per_frame=(count_nll*groups).sum(-1)
        onset_by={};tail_by={}
        onset_total=tail_total=per_frame.sum()*0
        for frame,kind in ((count>0)&groups).nonzero().tolist():
            n=int(count[frame,kind])
            _,onset,tail=model.ordered_members(output['x'][state],frame,kind,n,inputs,request,
                teacher={k:target[k][0,frame,kind,:n] for k in ('offset','tail_us')})
            onset_by[frame]=onset_by.get(frame,0)+onset;tail_by[frame]=tail_by.get(frame,0)+tail
            onset_total=onset_total+onset;tail_total=tail_total+tail
        attributes=torch.stack([onset_by.get(f,per_frame[f]*0)+tail_by.get(f,per_frame[f]*0) for f in range(FRAMES)])
        # No label outside H: emission log probability zero (E=1), never empty.
        emission=per_frame.new_zeros(nblocks).index_add(0,block_index[validframes],-(per_frame+attributes)[validframes])
        emissions.append(emission)
        count_diagnostics.append(per_frame.sum());onset_diagnostics.append(onset_total);tail_diagnostics.append(tail_total)
    emission=torch.stack(emissions,-1)
    denominator=groups.sum().clamp_min(1).float()
    marginal=forward_log_probability(output['initial'],output['transition'],emission)
    return -marginal/denominator,dict(count=torch.stack(count_diagnostics).mean()/denominator,
        onset=torch.stack(onset_diagnostics).mean()/denominator,tail=torch.stack(tail_diagnostics).mean()/denominator,
        valid_groups=denominator,raw_log_marginal=marginal)


@torch.no_grad()
def generate_arrangement(model,inputs,request,*,seed=20261054,deterministic=False,state_path=None,deadline=None):
    import time
    model.eval();output=model(inputs);rng=np.random.default_rng(seed)
    path=prior_path(output['initial'],output['transition'],rng=None if deterministic else rng) if state_path is None else np.asarray(state_path)
    if path.shape!=(len(output['transition']),):raise ValueError('Explicit prior path shape')
    groups=package_support(inputs,request)[0];addresses=groups.nonzero().tolist()
    block=(request.start_us+np.arange(FRAMES)*40_000)//2_000_000
    probabilities=[]
    for frame,kind in addresses:probabilities.append(output['count_log_probs'][path[block[frame]],frame,kind].exp().cpu().double().numpy())
    probabilities=np.asarray(probabilities)
    if deterministic:counts,info=cardinality_counts(probabilities,max_events=request.max_events);info.pop('polynomial')
    else:
        counts=np.asarray([rng.choice(17,p=p/p.sum()) for p in probabilities]);info={}
    if int(counts.sum())>request.max_events:raise ValueError('Arrangement event resource overflow')
    events=[]
    for (frame,kind),count in zip(addresses,counts):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Arrangement generation deadline')
        if count:
            members,_,_=model.ordered_members(output['x'][path[block[frame]]],frame,kind,int(count),inputs,request,
                rng=None if deterministic else rng,greedy=deterministic)
            events.extend(members)
    info.update(prior_state_path=path.tolist(),counts=counts.tolist(),generated=len(events),deterministic=deterministic)
    return sorted(events),info

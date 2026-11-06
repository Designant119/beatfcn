"""Complete local count/member law on physical cells with exact-us endpoints.

Canonical members are generated only within a cell; no sampled type history
crosses cell boundaries. Greedy conditional decisions are not joint MAP.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .local_bundle import read_grid
from .local_mamba_canvas import MambaCanvas
from .local_typed_canvas import CanvasBlock


class NativeCellChain(MambaCanvas):
    def __init__(self,config,*,density_reference):
        super().__init__(config,density_reference=density_reference)
        for name in ('quantity','existence','kind_slot','slot_read','offset','tail_block','tail_cell','tail_ms','density_alpha'):
            delattr(self,name)
        d=config.d
        self.context_read=CanvasBlock(config)
        self.cell_count=nn.Linear(d,66)  # 0..64, explicit OVERFLOW
        self.count_embedding=nn.Embedding(65,32)
        self.member_initial=nn.Linear(d+32,192)
        self.previous_member=nn.Linear(72,256)
        self.member_gru=nn.GRUCell(256,192)
        self.member_query=nn.Sequential(nn.Linear(192+d+32,d),nn.SiLU())
        self.address_feature=nn.Sequential(nn.Linear(d+128+33+4,d),nn.SiLU(),nn.LayerNorm(d))
        self.address_query=nn.Linear(d,128)
        self.address_key=nn.Linear(d,128)
        self.kind_read=nn.Sequential(nn.Linear(192+3*d,d),nn.SiLU(),nn.Linear(d,4))
        self.tail_query=nn.Sequential(nn.Linear(192+d+4,128),nn.SiLU())
        self.endpoint_key=nn.Sequential(nn.Linear(d+128+33+4,128),nn.SiLU())
        self.endpoint_level=nn.Embedding(3,128)
        self.endpoint_condition=nn.GRUCell(128,128)

    def time_features(self,times):
        angles=times[...,None]*(2*math.pi/self.time_periods)
        return torch.cat((times[...,None]/1200,angles.sin(),angles.cos()),-1)

    def forward(self,inputs):
        memory=self.encode(inputs);times=inputs['joint_frame_times']
        four=times[...,None]+times.new_tensor([.005,.015,.025,.035])
        mel=read_grid(memory['audio'],memory['audio_times'],memory['audio_valid'],four,.01)
        x=self.frame_read(mel.flatten(-2))
        clock=torch.stack((times/8,(times/8).square(),torch.sin(times*math.pi),torch.cos(times*math.pi)),-1)
        x=x+self.clock_read(clock)+self.mode_ids(inputs['mode'])[:,None]
        visible=inputs['reference_valid']&(inputs['reference'][...,10]>0)
        ref=inputs['reference'].masked_fill(~visible[...,None],0)
        style=ref.sum(1)/visible.sum(1,keepdim=True).clamp_min(1)
        scale,bias=self.direct(inputs['controls'].flatten(1)).chunk(2,-1)
        x=x*(1+.1*scale[:,None].tanh())+bias[:,None]+self.style(style)[:,None]
        context=memory['values'];frames=x.shape[1]
        for layer in self.address:
            joined=layer(torch.cat((x,context),1));x,context=joined[:,:frames],joined[:,frames:]
        for layer in self.temporal:x=layer(x)
        x=self.norm(x)
        fine=int(inputs['local_valid'][0].sum())*self.c.band_slots+int(inputs['mel_valid'][0].sum())
        songs=int(inputs['song_valid'][0].sum())
        song_values=memory['values'][:,fine:fine+songs]
        song_times=inputs['song_times'][:,inputs['song_valid'][0]]
        if not songs:raise ValueError('No ordered whole-song audio states')
        song_valid=torch.ones_like(song_times,dtype=torch.bool)
        direct_song=read_grid(song_values,song_times,song_valid,times+.02,.5)
        contextual={k:memory[k][:,fine:] for k in ('values','times','valid','timed')}
        x=self.context_read(x+direct_song,times+.02,contextual)
        # Preserve an explicit context read for the selected-head type interface.
        context_read=x  # mixed acoustic/reference context, not a pure reference ablation.
        valid=inputs['cell_has_address']
        count=self.cell_count(x).masked_fill(~valid[...,None],-1e9)
        count=count.clone();count[...,0]=torch.where(valid,count[...,0],torch.zeros_like(count[...,0]))
        return dict(count=count,x=x,song=direct_song,reference=context_read,
            song_values=song_values,song_times=song_times,song_valid=song_valid)

    def address_features(self,output,inputs,heads_us,cells,origin_us):
        device=output['x'].device
        heads=np.asarray(heads_us,np.int64)
        times=torch.tensor((heads-origin_us)/1e6,dtype=torch.float32,device=device)
        raw=read_grid(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],times.reshape(1,-1),.01)[0].reshape(*times.shape,128)
        low=inputs['mel_times'][0,0]-.005
        high=inputs['mel_times'][0].masked_fill(~inputs['mel_valid'][0],-1e9).max()+.005
        coverage=((times>=low)&(times<=high)).float()
        # Exact small integer phases distinguish authorized microsecond leaves
        # even where absolute float32 seconds cannot distinguish adjacent us.
        atom=np.stack((((heads-origin_us)%40000)/40000,(heads%1000)/1000,(heads%1000!=0).astype(float)),-1)
        atom=torch.tensor(atom,dtype=torch.float32,device=device)
        frames=output['x'][0,cells]
        while frames.ndim<raw.ndim:frames=frames.unsqueeze(-2)
        frames=frames.expand(*raw.shape[:-1],frames.shape[-1])
        return self.address_feature(torch.cat((frames,raw,self.time_features(times),coverage[...,None],atom),-1))

    def initial_members(self,output,cells,counts):
        x=output['x'][0,cells];embedding=self.count_embedding(counts)
        return self.member_initial(torch.cat((x,embedding),-1)).tanh(),embedding

    def member_step(self,output,cells,embedding,state,previous):
        state=self.member_gru(self.previous_member(previous),state)
        query=self.member_query(torch.cat((state,output['x'][0,cells],embedding),-1))
        return state,query

    def head_logits(self,query,features):
        return torch.einsum('md,mad->ma',self.address_query(query),self.address_key(features))/math.sqrt(128)

    def kind_logits(self,output,state,head_feature,cells):
        return self.kind_read(torch.cat((state,head_feature,output['song'][0,cells],output['reference'][0,cells]),-1))

    def endpoint_keys(self,output,inputs,absolute_us,origin_us):
        values=np.asarray(absolute_us,np.int64);device=output['x'].device
        times=torch.tensor((values-origin_us)/1e6,dtype=torch.float32,device=device)
        song=read_grid(output['song_values'],output['song_times'],output['song_valid'],times.reshape(1,-1),.5)[0]
        raw=read_grid(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],times.reshape(1,-1),.01)[0]
        lo=inputs['mel_times'][0,0]-.005;hi=inputs['mel_times'][0].masked_fill(~inputs['mel_valid'][0],-1e9).max()+.005
        coverage=((times>=lo)&(times<=hi)).float()
        atom=np.stack((((values-origin_us)%40000)/40000,(values%1000)/1000,(values%1000!=0).astype(float)),-1)
        extra=torch.tensor(atom,dtype=torch.float32,device=device)
        return self.endpoint_key(torch.cat((song,raw,self.time_features(times),coverage[:,None],extra),-1))

    def endpoint_scores(self,query,keys,level):
        return (keys*(query+self.endpoint_level.weight[level])).sum(-1)/math.sqrt(128)

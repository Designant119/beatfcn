"""Parameter-identical SSD canvas with a physical read of the song stream.

The original query-prefix concatenation places song states after thousands of
fine-audio rows. A frozen-weight audit found removing that stream changed no
events in32drafts and count probabilities by at most3.58e-7. This candidate
adds its already-encoded, time-addressed state directly to each canvas query.
There are no new weights and the original concatenation remains as context.
"""
import torch
from torch.nn import functional as F

from .local_bundle import read_grid
from .local_mamba_canvas import MambaCanvas
from .local_typed_canvas import FRAMES,SLOTS


class DirectSongCanvas(MambaCanvas):
    direct_song=True
    direct_song_strength=1.

    def forward(self,inputs):
        if not self.direct_song:return super().forward(inputs)
        controls=inputs['controls'].clone();density=controls[:,0].clone();controls[:,0]=0
        memory=self.encode({**inputs,'controls':controls});b=len(controls)
        frame_valid=inputs['canvas_valid'][0]
        if not bool(frame_valid.any()):raise ValueError('DirectSongCanvas requires valid frames')
        query=torch.arange(FRAMES,device=controls.device).float()[None]*.04
        query=query[:,frame_valid]
        four=query[...,None]+query.new_tensor([.005,.015,.025,.035])
        mel=read_grid(memory['audio'],memory['audio_times'],memory['audio_valid'],four,.01)
        x=self.frame_read(mel.flatten(-2))
        clock=torch.stack((query/8,(query/8).square(),torch.sin(query*torch.pi),torch.cos(query*torch.pi)),-1)
        x=x+self.clock_read(clock)+self.mode_ids(inputs['mode'])[:,None]
        visible=inputs['reference_valid'] & (inputs['reference'][...,10]>0)
        reference=inputs['reference'].masked_fill(~visible[...,None],0)
        style=reference.sum(1)/visible.sum(1,keepdim=True).clamp_min(1)
        scale,bias=self.direct(controls.flatten(1)).chunk(2,-1)
        x=x*(1+.1*scale[:,None].tanh())+bias[:,None]+self.style(style)[:,None]

        # These are exactly the separately encoded song rows in encode().
        # Neither raw labels nor an event history contributes to the address.
        fine=int(inputs['local_valid'][0].sum())*self.c.band_slots+int(inputs['mel_valid'][0].sum())
        count=int(inputs['song_valid'][0].sum())
        if not count:raise ValueError('Direct song read requires actual whole-song rows')
        values=memory['values'][:,fine:fine+count]
        times=inputs['song_times'][:,inputs['song_valid'][0]]
        if values.shape[1]!=times.shape[1]:raise ValueError('Song-stream extraction mismatch')
        global_read=read_grid(values,times,torch.ones_like(times,dtype=torch.bool),query,.5)
        x=x+self.direct_song_strength*global_read

        context=memory['values'];frame_count=x.shape[1]
        for layer in self.address:
            joint=layer(torch.cat((x,context),1));x,context=joint[:,:frame_count],joint[:,frame_count:]
        for layer in self.temporal:x=layer(x)
        compact=self.norm(x);x=compact.new_zeros(b,FRAMES,self.c.d);x[:,frame_valid]=compact
        base_count=self.quantity(x).view(b,FRAMES,4,SLOTS+1)
        rho=density[:,:2].mean(-1)*64
        tilt=torch.where(density[:,2]>.5,F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        count=base_count+tilt[:,None,None,None]*torch.arange(SLOTS+1,device=x.device)
        slot=x[:,:,None,None]+self.kind_slot[None,None];slot=slot+self.slot_read(slot)
        offset=self.offset(slot).masked_fill(~inputs['canvas_head_allowed'][:,:,None,None],-1e9)
        hold=slot[:,:,1]
        return dict(count=count,base_count=base_count,existence=self.existence(x),offset=offset,
            tail_block=self.tail_block(hold),tail_cell=self.tail_cell(hold),tail_ms=self.tail_ms(hold))

"""Complete SSD time field with canonical hurdle/GRU, separate from flow."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .canonical_event_package import CanonicalPackageCanvas
from .event_code_flow import EventCodeFlow
from .fullstack_mamba import FullStackMambaConfig,MambaBlock
from .local_event_package import _hurdle_count_log_probabilities
from .local_mamba_canvas import MambaCanvas
from .local_typed_canvas import FRAMES,SLOTS


class SSMEventPackage(CanonicalPackageCanvas):
    clock=MambaCanvas.clock
    encode=MambaCanvas.encode

    def __init__(self,config,*,density_reference):
        super().__init__(config,density_reference=density_reference)
        c=config
        mixer=FullStackMambaConfig(d=c.d,ffn=c.ffn,encoder_layers=c.encoder_layers,
            taps=c.taps,bands=c.bands,band_slots=c.band_slots,tap_dim=c.tap_dim,
            reference_dim=c.reference_dim,controls=c.controls,control_dim=c.control_dim,
            mel_bins=c.mel_bins,d_state=c.ssm_state,headdim=c.ssm_headdim,dropout=c.dropout)
        self.encoders=nn.ModuleList(nn.ModuleList(MambaBlock(mixer,bidirectional=True) for _ in range(c.encoder_layers)) for _ in range(3))
        self.address=nn.ModuleList(MambaBlock(mixer,bidirectional=True) for _ in range(2))
        self.temporal=nn.ModuleList(MambaBlock(mixer,bidirectional=True) for _ in range(2))
        self.time_projection=nn.Linear(33,c.d,bias=False)
        self.register_buffer('time_periods',torch.logspace(math.log10(.02),math.log10(2400),16))

    def forward(self,inputs):
        # Reuse only the audited SSD context equations, not its old slot heads.
        x=EventCodeFlow.encode_audio(self,inputs)
        gate=self.count_gate(x);positive=self.count_positive(x).view(len(x),FRAMES,4,SLOTS)
        logp=_hurdle_count_log_probabilities(gate,positive)
        density=inputs['controls'][:,0];rho=density[:,:2].mean(-1)*64
        tilt=torch.where(density[:,2]>.5,F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        logp=logp+tilt[:,None,None,None]*torch.arange(17,device=x.device);logp=logp-logp.logsumexp(-1,keepdim=True)
        return dict(x=x,count=logp.exp(),count_log_probs=logp)

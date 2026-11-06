"""C06 single probability-contract intervention on the existing reset canvas."""
import torch
from torch.nn import functional as F

from .local_mamba_canvas import MambaCanvas
from .local_typed_canvas import canvas_loss


class UnifiedCountCanvas(MambaCanvas):
    def forward(self,inputs):
        output=super().forward(inputs)
        existence=output['existence']
        positive=output['base_count'][...,1:].log_softmax(-1)
        base=torch.cat((F.logsigmoid(-existence)[...,None],
                        F.logsigmoid(existence)[...,None]+positive),-1)
        density=inputs['controls'][:,0]
        rho=density[:,:2].mean(-1)*64
        tilt=torch.where(density[:,2]>.5,
            F.softplus(self.density_alpha)*(rho.clamp_min(1e-4)/self.density_reference).log(),0.)
        logits=base+tilt[:,None,None,None]*torch.arange(17,device=base.device)
        return {**output,'base_count':base,'count':logits-logits.logsumexp(-1,keepdim=True)}


def unified_count_loss(output,target,inputs):
    # The old balanced existence score is logged but never contributes to the
    # new objective. Its parameters now receive the natural count NLL gradient.
    _,parts=canvas_loss(output,target,inputs)
    loss=parts['count']+parts['offset']+parts['tail']+.2*parts['total']+.2*parts['interval']
    return loss,parts

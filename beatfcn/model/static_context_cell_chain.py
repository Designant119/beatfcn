"""Parameter-identical candidate: only pad the post-SSD cross-attention shape.

No padding advances an SSD state. Padded attention keys are masked; padded
query rows are discarded after the pointwise cross-attention/FFN block.
"""
import torch
from torch.nn import functional as F

from beatfcn.model.local_typed_canvas import CanvasBlock
from beatfcn.model.native_cell_chain import NativeCellChain


class StaticContextBlock(CanvasBlock):
    query_capacity=200
    memory_capacity=2048

    def __init__(self,config):
        if config.dropout!=0:raise ValueError('Exact deterministic padding probe requires dropout0')
        super().__init__(config)

    def forward(self,x,times,memory):
        n=x.shape[1];m=memory['values'].shape[1]
        if n>self.query_capacity or m>self.memory_capacity:raise ValueError('Static attention capacity;no truncation')
        padded_x=F.pad(x,(0,0,0,self.query_capacity-n))
        padded_times=F.pad(times,(0,self.query_capacity-n))
        padded={}
        for key in ('values','times','valid','timed'):
            value=memory[key]
            padding=(0,0,0,self.memory_capacity-m) if value.ndim==3 else (0,self.memory_capacity-m)
            padded[key]=F.pad(value,padding,value=False if value.dtype==torch.bool else 0)
        return super().forward(padded_x,padded_times,padded)[:,:n]


class StaticContextCellChain(NativeCellChain):
    def __init__(self,config,*,density_reference):
        super().__init__(config,density_reference=density_reference)
        self.context_read=StaticContextBlock(config)

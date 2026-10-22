"""Bounded shared Pupu24/KDA4 scaffold, not the complete event decoder."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .pupu import FrozenLargeTeacher, load_teacher, make_rope, apply_rope
from .kda import KDABlock

@dataclass(frozen=True)
class Config:
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    kda_heads: int = 8
    insert_after: tuple = (6,12,18,24)
    ffn_hidden: int = 4096
    feature_dim: int = 8
    intent_tokens: int = 8
    local_max_frames: int = 1024  # S0 ONE native <=10.24s window; no implicit crop
    global_max_tokens: int = 600
    max_groups: int = 256
    max_prefix: int = 1536

@dataclass
class Context:
    """Already legal, caller-supplied features. Hidden labels NEVER belong here.

    local_mel normalized Pupu mel [B,1,T,128]; global_music is bounded
    audio-only raw feature summary [B,N,dim], NOT chart label summaries.
    reference encodes paired reference evidence (incl reference audio upstream).
    known holds observed/controls/scene features only. Provenance/semantic
    validation and real audio summary/reference builders are NOT implemented.
    """
    local_mel: torch.Tensor
    global_music: torch.Tensor
    reference: torch.Tensor
    known: torch.Tensor
    audio_valid: torch.Tensor | None = None
    global_valid: torch.Tensor | None = None
    reference_valid: torch.Tensor | None = None
    known_valid: torch.Tensor | None = None

    def fingerprint(self):
        h = hashlib.sha256()
        for name in self.__dataclass_fields__:
            x=getattr(self,name)
            h.update(name.encode())
            if x is not None:
                x=x.detach().cpu().contiguous()
                h.update(str((tuple(x.shape),x.dtype)).encode())
                h.update(x.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

@dataclass
class DecodeCache:
    context_hash: str
    revision: int
    kv: list
    kda: list
    events: int
    parameter_versions: tuple


def prefix_mask(lengths, valid):
    """True=allowed SDPA. A->A;R->R;I->R/I;W->prefix;E->prefix+causal E."""
    a,r,i,w,e=lengths; n=sum(lengths); p=a+r+i+w
    mask=torch.zeros(n,n,dtype=torch.bool,device=valid.device)
    mask[:a,:a]=True
    mask[a:a+r,a:a+r]=True
    mask[a+r:a+r+i,a:a+r+i]=True
    mask[a+r+i:p,:p]=True
    mask[p:,:p]=True
    mask[p:,p:]=torch.ones(e,e,dtype=torch.bool,device=valid.device).tril()
    # Invalid queries get harmless self attention, then outputs are zeroed.
    mask=mask[None,None] & valid[:,None,None,:]
    eye=torch.eye(n,dtype=torch.bool,device=valid.device)[None,None]
    return torch.where(valid[:,None,:,None],mask,eye)


def shared_attention(block, x, rope, mask, old_kv=None):
    """One module and one QKV/FFN parameter set for audio AND event rows."""
    z=block.norm1(x); b,n,d=z.shape; h=block.attn.heads
    q,k,v=block.attn.qkv(z).reshape(b,n,3,h,-1).permute(2,0,3,1,4).unbind(0)
    q,k=apply_rope(q,rope),apply_rope(k,rope)
    if old_kv is not None:
        k=torch.cat([old_kv[0],k],dim=2); v=torch.cat([old_kv[1],v],dim=2)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
    x=x+block.attn.proj(out.transpose(1,2).reshape(b,n,d))
    return x+block.mlp(block.norm2(x)), (k,v)


class SharedModel(nn.Module):
    def __init__(self,cfg=Config(),backbone=None):
        super().__init__(); self.cfg=cfg
        self.backbone=backbone if backbone is not None else FrozenLargeTeacher(cfg.dim,cfg.depth,cfg.heads)
        self.backbone.requires_grad_(False)
        self.kda_blocks=nn.ModuleList(KDABlock(cfg.dim,cfg.kda_heads,cfg.ffn_hidden) for _ in cfg.insert_after)
        self.group_embed=nn.Sequential(nn.Linear(cfg.feature_dim,cfg.dim),nn.SiLU(),nn.Linear(cfg.dim,cfg.dim))
        self.reference_embed=nn.Linear(cfg.feature_dim,cfg.dim)
        self.known_embed=nn.Linear(cfg.feature_dim,cfg.dim)
        self.intent=nn.Parameter(torch.randn(cfg.intent_tokens,cfg.dim)*.02)
        self.bos=nn.Parameter(torch.randn(1,1,cfg.dim)*.02)
        self.modality=nn.Embedding(5,cfg.dim)
        self.event_position=nn.Embedding(cfg.max_groups+1,cfg.dim)
        # Bounded outer-step scaffold only. Full GRU/hold/COPY_TIME/Phira deferred.
        self.outer_action=nn.Linear(cfg.dim,2)  # END_GAP / NEW_GROUP
        self.time_coarse=nn.Linear(cfg.dim,800)  # 8s @10ms
        self.time_fine=nn.Linear(cfg.dim,10)
        self.selection_query=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.selection_key=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.selection_readout=nn.Linear(cfg.dim,1)
        self.revision=0

    @classmethod
    def from_checkpoint(cls,path,*,dtype=torch.bfloat16,device='cpu'):
        # Allocate new modules directly in chosen dtype/device, no huge random base.
        base=load_teacher(path,dtype=dtype,device=device)
        with torch.device(device):
            old=torch.get_default_dtype(); torch.set_default_dtype(dtype)
            try: model=cls(Config(),base)
            finally: torch.set_default_dtype(old)
        return model

    def mark_updated(self):
        """Call after ANY optimizer step/load/edit. Inference caches never survive."""
        self.revision+=1

    def _prepare(self,c,feedback):
        cfg=self.cfg; b=feedback.shape[0]; device=feedback.device
        if c.local_mel.shape[-1]!=128 or c.local_mel.shape[2]<4 or c.local_mel.shape[2]%4:
            raise ValueError('S0 requires T>=4, multiple of4; explicit padding upstream')
        if c.local_mel.shape[2]>cfg.local_max_frames or c.global_music.shape[1]>cfg.global_max_tokens:
            raise ValueError('resource_exhausted: bounded audio context')
        if feedback.shape[1]>cfg.max_groups: raise ValueError('resource_exhausted: groups')
        local=self.backbone.patch_embed(c.local_mel)
        a=torch.cat([local,c.global_music],dim=1)
        r=self.reference_embed(c.reference); w=self.known_embed(c.known)
        intent=self.intent[None].expand(b,-1,-1)
        # Caller provides ONLY completed previous groups; output len=previous+1.
        e=torch.cat([self.bos.expand(b,-1,-1),self.group_embed(feedback)],dim=1)
        e=e+self.event_position(torch.arange(e.shape[1],device=device))[None]
        chunks=[a,r,intent,w,e]; lengths=[x.shape[1] for x in chunks]
        if sum(lengths[1:-1])+c.global_music.shape[1]>cfg.max_prefix: raise ValueError('resource_exhausted: prefix; do not truncate')
        x=torch.cat([z+self.modality.weight[j] for j,z in enumerate(chunks)],dim=1)
        def valid(v,n):
            if v is None:return torch.ones(b,n,dtype=torch.bool,device=device)
            if v.shape!=(b,n) or v.dtype!=torch.bool: raise ValueError('invalid mask shape/dtype')
            return v
        av=torch.cat([valid(c.audio_valid,local.shape[1]),valid(c.global_valid,c.global_music.shape[1])],dim=1)
        vm=torch.cat([av,valid(c.reference_valid,r.shape[1]),valid(None,intent.shape[1]),valid(c.known_valid,w.shape[1]),valid(None,e.shape[1])],dim=1)
        x=x*vm[...,None]
        hd=cfg.dim//cfg.heads
        local_rope=make_rope((c.local_mel.shape[2]//4,8),hd,device=device)
        # Identity RoPE for abstract summaries/events; explicit learned event index
        # and attribute-derived time embeddings, NOT fictitious frequency coords.
        neutral=torch.cat([torch.zeros(1,1,x.shape[1]-local.shape[1],hd,device=device),torch.ones(1,1,x.shape[1]-local.shape[1],hd,device=device)],dim=-1)
        rope=torch.cat([local_rope,neutral],dim=2)
        return x,lengths,vm,rope

    def forward(self,c:Context,previous_groups:torch.Tensor,*,return_cache=False,backend='reference'):
        if return_cache and backend!='reference':raise ValueError('S0 incremental cache validated only for reference backend')
        x,lengths,valid,rope=self._prepare(c,previous_groups)
        mask=prefix_mask(lengths,valid); kvs=[]; states=[]; j=0
        for l,block in enumerate(self.backbone.teacher.blocks,1):
            # NEVER no_grad: frozen originals still propagate input/KDA gradients.
            x,kv=shared_attention(block,x,rope,mask); x=x*valid[...,None]; kvs.append(kv)
            if l in self.cfg.insert_after:
                x,state=self.kda_blocks[j].routed(x,lengths,valid,backend=backend)
                states.append(state);j+=1
        h=self.backbone.teacher.norm(x)*valid[...,None]
        a,r,i,w,e=lengths; p=a+r+i+w
        intent=h[:,a+r:a+r+i]
        # Only target music + isolated intent; no target/event hidden inputs.
        q=self.selection_query(h[:,:a]);k=self.selection_key(intent)
        choice=(q@k.transpose(-1,-2))/(self.cfg.dim**.5)
        selected=choice.softmax(-1)@intent
        selection=self.selection_readout(selected).squeeze(-1)*valid[:,:a]
        local_n=c.local_mel.shape[2]//4*8
        local_selection=selection[:,:local_n].reshape(selection.shape[0],-1,8).mean(-1)
        out={'selection_cells':local_selection,'action':self.outer_action(h[:,p:]),'coarse':self.time_coarse(h[:,p:]),'fine':self.time_fine(h[:,p:]),'hidden':h,'intent':intent,'selection':selection,'lengths':lengths}
        if return_cache:
            if self.training or torch.is_grad_enabled():raise ValueError('cache is inference-only; use eval()+no_grad()')
            # Store legal mask: invalid prefix keys must remain invisible on steps.
            out['cache']=DecodeCache(c.fingerprint(),self.revision,[(k,v,valid.clone()) for k,v in kvs],states,e,tuple((id(p),p._version) for p in self.parameters()))
        return out

    def step(self,c:Context,completed_group:torch.Tensor,cache:DecodeCache):
        if self.training or torch.is_grad_enabled():raise ValueError('step is inference-only')
        if cache.context_hash!=c.fingerprint() or cache.revision!=self.revision or cache.parameter_versions!=tuple((id(p),p._version) for p in self.parameters()):
            raise ValueError('stale cache: context/weights changed; prefill again')
        if cache.events>self.cfg.max_groups:raise ValueError('resource_exhausted: groups')
        x=self.group_embed(completed_group[:,None])+self.modality.weight[4]+self.event_position.weight[cache.events]
        hd=self.cfg.dim//self.cfg.heads
        rope=torch.cat([x.new_zeros(1,1,1,hd),x.new_ones(1,1,1,hd)],dim=-1)
        newkv=[];states=[];j=0
        for l,block in enumerate(self.backbone.teacher.blocks,1):
            k,v,valid=cache.kv[l-1];valid=torch.cat([valid,torch.ones(x.shape[0],1,dtype=torch.bool,device=x.device)],dim=1)
            x,(k,v)=shared_attention(block,x,rope,valid[:,None,None,:],(k,v));newkv.append((k,v,valid))
            if l in self.cfg.insert_after:
                x,state=self.kda_blocks[j].stream(x,cache.kda[j]);states.append(state);j+=1
        h=self.backbone.teacher.norm(x)
        return {'action':self.outer_action(h),'coarse':self.time_coarse(h),'fine':self.time_fine(h),'cache':DecodeCache(cache.context_hash,self.revision,newkv,states,cache.events+1,cache.parameter_versions)}

    def inventory(self):
        modules={}
        for name,p in self.named_parameters():
            group='backbone' if name.startswith('backbone.') else ('.'.join(name.split('.')[:2]) if name.startswith('kda_blocks.') else name.split('.')[0])
            z=modules.setdefault(group,{'total':0,'trainable':0,'frozen':0})
            z['total']+=p.numel();z['trainable' if p.requires_grad else 'frozen']+=p.numel()
        total=sum(p.numel() for p in self.parameters())
        return {'identity':'shared-pupu24-kda4-s0-v1','modules':modules,'total':total,'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),'frozen':sum(p.numel() for p in self.parameters() if not p.requires_grad),'bf16_weight_bytes':2*total,'fp32_weight_bytes':4*total}

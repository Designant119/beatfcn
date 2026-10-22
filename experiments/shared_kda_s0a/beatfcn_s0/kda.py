"""Kimi Delta Attention CPU reference, explicit isolated state routing.

Independent PyTorch implementation of published recurrence/parameterization.
No claim that a causal scan implements an arbitrary block attention mask.
Official sources/epsilon choices documented in ARCHITECTURE.md. State is [B,H,K,V].
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def recurrence(q,k,v,log_alpha,beta,state=None):
    """Inputs [B,T,H,D], beta[B,T,H]; q/k ALREADY L2-normalized.

    Accumulate state FP32 for BF16/FP32 inputs, retain float64 for gradcheck.
    Gate decays KEY rows before delta prediction; beta scalar/head.
    """
    original=v.dtype
    work=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,k,v,g,beta=[z.to(work) for z in (q,k,v,log_alpha,beta)]
    b,t,h,d=q.shape
    s=q.new_zeros(b,h,d,v.shape[-1]) if state is None else state.to(work)
    outputs=[]
    for j in range(t):
        decayed=s*g[:,j].exp()[...,None]
        residual=v[:,j]-(k[:,j,...,None]*decayed).sum(-2)
        s=decayed+beta[:,j,:,None,None]*k[:,j,...,None]*residual[...,None,:]
        outputs.append(((q[:,j]*(d**-.5))[...,None]*s).sum(-2))
    out=torch.stack(outputs,1) if outputs else v.new_empty(b,0,h,v.shape[-1])
    return out.to(original),s


def official_fla(q,k,v,g,beta,state=None):
    """Optional actual official CUDA kernel, never silently substitute another op.

    Requires a pinned tested fla-core environment. Not installed or GPU-tested
    by this package. Already-transformed gates/norm/beta avoid fused ambiguity.
    """
    if not q.is_cuda:raise ValueError('FLA requires CUDA; use reference on CPU')
    from fla.ops.kda import chunk_kda
    return chunk_kda(q=q,k=k,v=v,g=g,beta=beta,initial_state=state,
                     output_final_state=True,use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False,use_beta_sigmoid_in_kernel=False,
                     state_v_first=False)


@dataclass
class StreamState:
    recurrent: torch.Tensor
    convolutions: tuple


class ShortConv(nn.Module):
    def __init__(self,dim,kernel=4):
        super().__init__();self.kernel=kernel
        self.weight=nn.Parameter(torch.empty(dim,kernel));self.bias=nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight,std=kernel**-.5)
    def forward(self,x,state=None):
        b,t,d=x.shape
        history=x.new_zeros(b,self.kernel-1,d) if state is None else state
        z=torch.cat([history,x],dim=1)
        # unfold shape B,T,D,K. Kernel convention matches torch.conv1d.
        out=(z.unfold(1,self.kernel,1)*self.weight[None,None]).sum(-1)+self.bias
        return F.silu(out),z[:,-(self.kernel-1):]


class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x):
        dtype=x.dtype; y=x.float() if dtype!=torch.float64 else x
        return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.eps)).to(dtype)*self.weight


class KDAMixer(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();assert dim%heads==0
        self.dim=dim;self.heads=heads;self.hd=dim//heads
        self.q_proj=nn.Linear(dim,dim,bias=False);self.k_proj=nn.Linear(dim,dim,bias=False);self.v_proj=nn.Linear(dim,dim,bias=False)
        self.q_conv=ShortConv(dim);self.k_conv=ShortConv(dim);self.v_conv=ShortConv(dim)
        self.alpha_down=nn.Linear(dim,self.hd,bias=False);self.alpha_up=nn.Linear(self.hd,dim,bias=False)
        self.gate_down=nn.Linear(dim,self.hd,bias=False);self.gate_up=nn.Linear(self.hd,dim,bias=False)
        self.beta_proj=nn.Linear(dim,heads,bias=False)
        self.A_log=nn.Parameter(torch.zeros(heads));self.dt_bias=nn.Parameter(torch.full((heads,self.hd),-3.))
        self.out_norm=RMSNorm(self.hd);self.out_proj=nn.Linear(dim,dim,bias=False)
    def forward(self,x,state=None,backend='reference'):
        b,t,d=x.shape
        if not t:raise ValueError('empty scan; skip explicitly at caller')
        old=(None,None,None) if state is None else state.convolutions
        q,cq=self.q_conv(self.q_proj(x),old[0]);k,ck=self.k_conv(self.k_proj(x),old[1]);v,cv=self.v_conv(self.v_proj(x),old[2])
        q=q.reshape(b,t,self.heads,self.hd);k=k.reshape_as(q);v=v.reshape_as(q)
        # Fixed epsilon, explicit pre-normalization shared by CPU and FLA.
        q=F.normalize(q.float(),dim=-1,eps=1e-6).to(x.dtype);k=F.normalize(k.float(),dim=-1,eps=1e-6).to(x.dtype)
        g=-self.A_log.float().exp()[None,None,:,None]*F.softplus(self.alpha_up(self.alpha_down(x)).float().reshape_as(q)+self.dt_bias.float())
        beta=self.beta_proj(x).float().sigmoid()
        fn=recurrence if backend=='reference' else official_fla if backend=='fla' else None
        if fn is None:raise ValueError('unknown KDA backend')
        out,s=fn(q,k,v,g,beta,None if state is None else state.recurrent)
        gate=self.gate_up(self.gate_down(x)).reshape_as(out).sigmoid()
        out=(self.out_norm(out)*gate).reshape(b,t,d)
        return self.out_proj(out),StreamState(s,(cq,ck,cv))


class KDABlock(nn.Module):
    def __init__(self,dim,heads,hidden):
        super().__init__();self.norm1=RMSNorm(dim);self.mixer=KDAMixer(dim,heads);self.norm2=RMSNorm(dim)
        self.ff_gate=nn.Linear(dim,hidden,bias=False);self.ff_value=nn.Linear(dim,hidden,bias=False);self.ff_out=nn.Linear(hidden,dim,bias=False)
    def finish(self,x,mixed):
        x=x+mixed;z=self.norm2(x)
        return x+self.ff_out(F.silu(self.ff_gate(z))*self.ff_value(z))
    def stream(self,x,state=None,backend='reference'):
        mixed,state=self.mixer(self.norm1(x),state,backend)
        return self.finish(x,mixed),state
    def routed(self,x,lengths,valid,backend='reference'):
        """Five separate scans using SAME mixer weights and separate histories.

        A scan: A only. R: R only. I: R then I. W: A,R,I then W.
        E: prefill A,R,I,W then BOS,completed-groups. All prefix rows are
        target-independent. R/I never prefill A/W/E. The attention layers
        provide bidirectional legal-prefix connectivity; these scans add
        explicit ordered recurrent context (NOT arbitrary-mask equivalence).

        Per-example compact valid positions ensure pad cannot advance either
        conv or recurrence. No flattened batch scan or cross-example carry.
        """
        a,r,i,w,e=lengths; p=a+r+i+w
        offsets=[0,a,a+r,a+r+i,p,p+e]
        routes=[list(range(0,a)),list(range(a,a+r)),list(range(a,a+r+i)),list(range(0,p)),list(range(0,p+e))]
        rows=[];event_states=[]
        for b in range(x.shape[0]):
            y=torch.zeros_like(x[b:b+1]); estate=None
            for role,indices in enumerate(routes):
                target_start,target_end=offsets[role],offsets[role+1]
                ids=torch.tensor(indices,dtype=torch.long,device=x.device)
                ids=ids[valid[b,ids]]
                if not ids.numel():continue
                z=x[b:b+1,ids]
                mixed,state=self.mixer(self.norm1(z),backend=backend)
                take=(ids>=target_start)&(ids<target_end)
                if take.any():
                    target=ids[take]
                    y[:,target]=self.finish(x[b:b+1,target],mixed[:,take])
                if role==4:estate=state
            rows.append(y);event_states.append(estate)
        # Event suffix always has BOS, so every example has valid event state.
        state=StreamState(torch.cat([s.recurrent for s in event_states],0),tuple(torch.cat([s.convolutions[j] for s in event_states],0) for j in range(3)))
        return torch.cat(rows,0),state

"""Music-conditioned20ms proposal attributes/rejection and native/COPY refinement.

Proposals come from a frozen free generator at inference. No target type/count
or true onset enters forward. The audio body can explicitly reuse the completed
local full-song model; such a warm start is not called fresh pretraining.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .local_bundle import read_grid
from .full_event_model import RMSNorm
from .whole_song_arrangement import WholeSongArrangement
from .event_canvas import RhythmEvent
from .local_typed_canvas import KINDS


REJECT=4


def monotone_proposal_matching(proposals,targets,tolerance_us=50_000):
    """Exact max-cardinality/min-distance1D matching, retaining multiplicity.

Sorted absolute-distance matching has a noncrossing optimum. Dynamic programming
therefore needs no external Hungarian dependency. Type is deliberately ignored.
"""
    import numpy as np
    n,m=len(proposals),len(targets)
    if any(proposals[i]>proposals[i+1] for i in range(n-1)) or any(targets[i]>targets[i+1] for i in range(m-1)):
        raise ValueError('Matching inputs must be sorted times')
    count=np.zeros((n+1,m+1),np.int32);distance=np.zeros((n+1,m+1),np.int64);back=np.zeros((n+1,m+1),np.uint8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            best=(int(count[i-1,j]),-int(distance[i-1,j]));choice=1
            other=(int(count[i,j-1]),-int(distance[i,j-1]))
            if other>best:best=other;choice=2
            gap=abs(proposals[i-1]-targets[j-1])
            if gap<=tolerance_us:
                matched=(int(count[i-1,j-1])+1,-int(distance[i-1,j-1])-gap)
                if matched>=best:best=matched;choice=3
            count[i,j],distance[i,j],back[i,j]=best[0],-best[1],choice
    pairs=[];i,j=n,m
    while i and j:
        choice=back[i,j]
        if choice==3:pairs.append((i-1,j-1));i-=1;j-=1
        elif choice==1:i-=1
        else:j-=1
    return pairs[::-1]


class CausalProposalBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.norm=RMSNorm(d);self.depth=nn.Conv1d(d,d,3,groups=d)
        self.mix=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,d))

    def forward(self,x):
        z=self.depth(F.pad(self.norm(x).transpose(1,2),(2,0))).transpose(1,2)
        return x+self.mix(F.silu(z))


class NoisyProposalRefiner(nn.Module):
    def __init__(self,config,*,density_reference):
        super().__init__();d=config.d
        self.body=WholeSongArrangement(config,density_reference=density_reference)
        for name,p in self.body.named_parameters():
            if name.startswith(('count_','density_alpha','kind_ids','onset_ids','tail_','member_','copy_','no_tail')):p.requires_grad_(False)
        self.fine=nn.Linear(config.mel_bins,d)
        self.proposal_clock=nn.Linear(4,d)
        self.request=nn.Sequential(nn.Linear(48,d),nn.SiLU(),nn.Linear(d,d))
        self.causal=nn.ModuleList(CausalProposalBlock(d) for _ in range(2))
        self.norm=RMSNorm(d);self.kind=nn.Linear(d,5)
        self.head=nn.Linear(d,613);self.tail_copy=nn.Linear(d,513);self.duration=nn.Linear(d,1)
        nn.init.constant_(self.duration.bias,-7.)

    def forward(self,inputs,proposal):
        out=self.body(inputs)
        prior=out['initial'].exp();priors=[prior]
        for block in range(1,len(out['transition'])):
            prior=prior@out['transition'][block].exp();priors.append(prior)
        weight=torch.stack(priors)[proposal['body_blocks'][0]]
        x=(out['x'].permute(1,0,2)*weight[:,:,None]).sum(1)[None]
        times=torch.arange(200,device=x.device).float()[None]*.04+.02
        grid=torch.arange(400,device=x.device).float()[None]*.02+.01
        coarse=read_grid(x,times,inputs['canvas_valid'],grid,.04)
        fine=read_grid(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],grid,.01)
        addressed=coarse+self.fine(fine)
        features=read_grid(addressed,grid,proposal['grid20_valid'],proposal['times_s'],.02)
        features=features+self.proposal_clock(proposal['clock'])+self.request(inputs['controls'].flatten(1))[:,None]
        for layer in self.causal:features=layer(features)
        features=self.norm(features)
        ones=torch.ones_like(proposal['head_has_endpoint'].any(-1))
        dynamic=torch.stack((ones,proposal['head_has_endpoint'].any(-1),ones,ones,ones),-1)
        return dict(kind=self.kind(features).masked_fill(~(proposal['kind_allowed'][:,None]&dynamic),-1e9),head=self.head(features).masked_fill(~proposal['head_allowed'],-1e9),
            duration=self.duration(features).squeeze(-1),tail_copy=self.tail_copy(features))


def refiner_loss(model,inputs,proposal,target):
    out=model(inputs,proposal);kind=target['kind'];parts=[]
    for k in range(5):
        keep=kind==k
        if bool(keep.any()):parts.append(F.cross_entropy(out['kind'][keep],kind[keep]))
    classification=sum(parts)/len(parts);positive=kind!=REJECT;zero=classification*0
    if bool(positive.any()):
        head_logits=out['head'].masked_fill((kind==1)[...,None]&~proposal['head_has_endpoint'],-1e9)
        head=-(target['head_distribution'][positive]*head_logits[positive].log_softmax(-1)).sum(-1).mean()
        entropy=-(target['head_distribution'][positive]*target['head_distribution'][positive].clamp_min(1e-30).log()).sum(-1).mean()
    else:head=entropy=zero
    holds=kind==1
    if bool(holds.any()):
        first=target['head_ms'][holds]+1;last=proposal['duration_ms'][0]
        predicted=1+(last-first).clamp_min(0)*out['duration'][holds].sigmoid()
        actual=target['end_ms'][holds]-target['head_ms'][holds]
        native=target['tail_copy'][holds]==0
        duration=F.smooth_l1_loss(torch.log1p(predicted[native]),torch.log1p(actual[native].float())) if bool(native.any()) else zero
        anchors=proposal['tail_times'][0];head_us=target['head_us'][holds]
        allowed=torch.cat(((first<=last)[:,None],(anchors[None]>head_us[:,None]) & proposal['tail_subms'][0,None]),-1)
        tail=F.cross_entropy(out['tail_copy'][holds].masked_fill(~allowed,-1e9),target['tail_copy'][holds])
    else:duration=tail=zero
    total=classification+.3*head+duration+tail
    return total,dict(classification=classification,head=head,head_target_entropy=entropy,duration=duration,tail_copy=tail)


@torch.no_grad()
def generate_refined(model,inputs,proposal,request):
    model.eval();out=model(inputs,proposal);classes=out['kind'][0].argmax(-1);events=[]
    for index in range(len(classes)):
        kind=int(classes[index])
        if kind==REJECT:continue
        head_logits=out['head'][0,index]
        if kind==1:head_logits=head_logits.masked_fill(~proposal['head_has_endpoint'][0,index],-1e9)
        head_id=int(head_logits.argmax())
        head=int(proposal['head_times'][0,index,head_id]);end=head
        if kind==1:
            first=head//1000+1;last=request.duration_us//1000
            anchors=proposal['tail_times'][0]
            allowed=torch.cat((torch.tensor([first<=last],device=classes.device),
                (anchors>head)&proposal['tail_subms'][0]))
            if not bool(allowed.any()):raise ValueError('Refined hold has no legal endpoint')
            selected=int(out['tail_copy'][0,index].masked_fill(~allowed,-1e9).argmax())
            end=int(anchors[selected-1]) if selected else (first+int(torch.round((last-first)*out['duration'][0,index].sigmoid())))*1000
        events.append(RhythmEvent(head,end,KINDS[kind]))
    if len(events)>request.max_events:raise ValueError('Refined event resource overflow')
    return sorted(events)

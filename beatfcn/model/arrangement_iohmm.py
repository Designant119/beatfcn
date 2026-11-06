"""Exact log-space sparse-emission IOHMM, with input-only prior decisions."""
import numpy as np
import torch


def forward_log_probability(initial, transition, emission):
    if initial.shape!=(emission.shape[1],) or transition.shape!=(len(emission),len(initial),len(initial)):
        raise ValueError('IOHMM path tensor shapes')
    alpha=initial+emission[0]
    for block in range(1,len(emission)):
        alpha=emission[block]+torch.logsumexp(alpha[:,None]+transition[block],0)
    return torch.logsumexp(alpha,0)


def prior_path(initial,transition,*,rng=None):
    """No hidden-target emission is accepted by this interface."""
    pi=initial.detach().cpu().double().numpy();tr=transition.detach().cpu().double().numpy()
    if rng is not None:
        p=np.exp(pi-pi.max());p/=p.sum();state=int(rng.choice(len(p),p=p));path=[state]
        for block in range(1,len(tr)):
            p=np.exp(tr[block,state]-tr[block,state].max());p/=p.sum()
            state=int(rng.choice(len(p),p=p));path.append(state)
        return np.asarray(path,np.int64)
    score=pi.copy();back=np.zeros((len(tr),len(pi)),np.int64)
    for block in range(1,len(tr)):
        pair=score[:,None]+tr[block]
        back[block]=pair.argmax(0);score=pair.max(0)
    path=np.zeros(len(tr),np.int64);path[-1]=int(score.argmax())
    for block in range(len(tr)-1,0,-1):path[block-1]=back[block,path[block]]
    return path

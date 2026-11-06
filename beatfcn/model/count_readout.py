"""Finite independent-count cardinality and conditional MAP, NumPy float64."""
import numpy as np


def normalized_counts(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or (p.sum(-1)<=0).any():
        raise ValueError('Invalid finite count distributions')
    return p/p.sum(-1,keepdims=True)


def cardinality_map(probabilities, *, event_limit):
    p=normalized_counts(probabilities)
    distribution=np.array([1.],dtype=np.float64)
    for row in p:
        distribution=np.convolve(distribution,row)
        mass=distribution.sum()
        if not np.isfinite(mass) or mass<=0:raise ValueError('Cardinality numerical failure')
        distribution/=mass
    total=int(distribution.argmax())
    if total>event_limit:raise ValueError('Cardinality mode exceeds event resource limit')
    logp=np.full_like(p,-np.inf);np.log(p,out=logp,where=p>0)
    score=np.full(total+1,-np.inf);score[0]=0
    back=np.zeros((len(p),total+1),np.uint8)
    for i,row in enumerate(logp):
        choices=min(len(row)-1,total)
        candidates=np.full((choices+1,total+1),-np.inf)
        for count in range(choices+1):
            candidates[count,count:]=score[:total+1-count]+row[count]
        back[i]=candidates.argmax(0)
        score=candidates[back[i],np.arange(total+1)]
    if not np.isfinite(score[total]):raise ValueError('Conditional MAP has no finite path')
    counts=np.empty(len(p),np.int64);remaining=total
    for i in range(len(p)-1,-1,-1):
        counts[i]=back[i,remaining];remaining-=counts[i]
    if remaining or counts.sum()!=total:raise AssertionError('Conditional MAP backtrack')
    return counts,dict(total_mode=total,expected=float((p*np.arange(p.shape[-1])).sum()),
        distribution_mass=float(distribution.sum()),mode_probability=float(distribution[total]),
        overflow_probability=float(distribution[event_limit+1:].sum()),
        complete_support_max=len(distribution)-1,conditional_log_probability=float(score[total]))


def expected_tail_rank(probabilities, *, event_limit):
    p=normalized_counts(probabilities)
    expected=float((p*np.arange(p.shape[-1])).sum())
    total=int(np.rint(expected))
    if total>event_limit:raise ValueError('Model expected count exceeds resource limit')
    tails=np.flip(np.cumsum(np.flip(p[:,1:],axis=1),axis=1),axis=1)
    # Stable tie order places member s before s+1, preserving prefix slots.
    order=np.argsort(-tails.reshape(-1),kind='stable')[:total]
    selected=np.zeros(tails.size,bool);selected[order]=True;selected=selected.reshape(tails.shape)
    if (selected[:,1:] & ~selected[:,:-1]).any():raise AssertionError('Non-prefix tail selection')
    return selected.sum(-1),dict(expected=expected,rounded_expected=total)

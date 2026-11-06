"""Temperature-one sampling on the frozen canvas probability supports."""
import numpy as np


def categorical(rng,logits):
    x=np.asarray(logits,np.float64)
    if not np.isfinite(x).any():raise ValueError('No finite categorical support')
    p=np.exp(x-np.max(x));p/=p.sum()
    return int(rng.choice(len(p),p=p))


def sample_tail(rng,block,cell,ms,onset_us,duration_us):
    lo,hi=onset_us//1000+1,duration_us//1000
    if lo>hi:raise ValueError('No legal native-ms hold endpoint')
    if hi//2000>=len(block):raise ValueError('Tail support resource overflow')
    pair=np.add.outer(np.asarray(cell,np.float64),np.asarray(ms,np.float64)).reshape(-1)
    logs=[];bounds=[]
    for b in range(lo//2000,hi//2000+1):
        first,last=max(0,lo-b*2000),min(1999,hi-b*2000)
        values=pair[first:last+1];maximum=values.max()
        z=maximum+np.log(np.exp(values-maximum).sum())
        logs.append(float(block[b])+z);bounds.append((b,first,last))
    b,first,last=bounds[categorical(rng,logs)]
    return (b*2000+first+categorical(rng,pair[first:last+1]))*1000

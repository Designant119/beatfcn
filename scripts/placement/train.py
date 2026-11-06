"""Small autoregressive placement model: rhythm + previous positions -> next x bin. No audio. Local CPU."""
import json,sys,time
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(3)
BINS=np.linspace(-675,675,28); NB=len(BINS)-1
def feats(a):
    """a: [n,4] start,end,kind,x -> features per note"""
    t,e,k=a[:,0],a[:,1],a[:,2].astype(int); n=len(a)
    prev=np.diff(t,prepend=t[0]-2.); nxt=np.diff(t,append=t[-1]+2.)
    f=np.zeros((n,11),np.float32)
    f[:,0]=np.log(prev+.01); f[:,1]=np.log(nxt+.01); f[np.arange(n),2+k]=1
    f[:,6]=np.log(np.maximum(e-t,0)+.01)*(k==1)
    f[:,7]=prev<.005; f[:,8]=nxt<.005
    ends=np.maximum.accumulate(e); f[:,9]=(np.r_[0,ends[:-1]]>t+.03)  # a hold is still down
    lo=np.searchsorted(t,t-1.); hi=np.searchsorted(t,t+1.); f[:,10]=np.log((hi-lo)/2+.5)
    return f
def tobin(x): return np.clip(np.digitize(np.clip(x,-674.9,674.9),BINS)-1,0,NB-1)
class M(nn.Module):
    def __init__(s,h=256):
        super().__init__(); s.prev=nn.Embedding(NB+1,48); s.inp=nn.Linear(11+48,h); s.g=nn.GRU(h,h,2,batch_first=True,dropout=.3); s.o=nn.Linear(h,NB)
    def forward(s,f,pb,st=None):
        o,st=s.g(torch.relu(s.inp(torch.cat([f,s.prev(pb)],-1))),st); return s.o(o),st
def main():
    D=np.load('placement_data.npz'); idx=json.load(open('placement_index.json'))
    tr=[x['chart'].replace(':','_') for x in idx if x['split']=='train']; va=[x['chart'].replace(':','_') for x in idx if x['split']=='val']
    data={k:(feats(D[k]),tobin(D[k][:,3])) for k in tr+va}
    rng=np.random.default_rng(0); net=M(); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=.01)
    print('train',len(tr),'val',len(va),'params',sum(p.numel() for p in net.parameters()),flush=True)
    def batch(keys,bs=48,L=128):
        F,Y,P=[],[],[]
        for k in rng.choice(keys,bs):
            f,y=data[k]; i=rng.integers(0,max(1,len(y)-L)); fj=f[i:i+L]; yj=y[i:i+L]
            if len(yj)<L: fj=np.pad(fj,((0,L-len(yj)),(0,0))); yj=np.r_[yj,np.full(L-len(yj),-1)]
            p=np.r_[NB,np.where(yj[:-1]<0,NB,yj[:-1])]; F.append(fj);Y.append(yj);P.append(p)
        return map(torch.as_tensor,(np.stack(F),np.stack(Y),np.stack(P)))
    def evaluate():
        net.eval(); tot=c=0
        with torch.no_grad():
            for k in va:
                f,y=data[k]; p=np.r_[NB,y[:-1]]; lg,_=net(torch.as_tensor(f)[None],torch.as_tensor(p)[None])
                tot+=nn.functional.cross_entropy(lg[0],torch.as_tensor(y),reduction='sum').item(); c+=len(y)
        net.train(); return tot/c
    # baselines: the marginal of x and a first-order bigram, from train
    ally=np.concatenate([data[k][1] for k in tr]); hist=np.bincount(ally,minlength=NB)+1.; hist/=hist.sum()
    bi=np.ones((NB,NB))
    for k in tr:
        y=data[k][1]; np.add.at(bi,(y[:-1],y[1:]),1)
    bi/=bi.sum(1,keepdims=True)
    ym=np.concatenate([data[k][1] for k in va]); base=-np.log(hist[ym]).mean()
    bg=np.mean([-np.log(bi[data[k][1][:-1],data[k][1][1:]]).mean() for k in va])
    print('val nats: uniform %.3f marginal %.3f bigram %.3f'%(np.log(NB),base,bg),flush=True)
    best=9e9;t0=time.time()
    for step in range(int(sys.argv[1]) if len(sys.argv)>1 else 1500):
        f,y,p=batch(tr); lg,_=net(f,p); loss=nn.functional.cross_entropy(lg.reshape(-1,NB),y.reshape(-1),ignore_index=-1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        if (step+1)%250==0:
            v=evaluate(); print(step+1,'train %.3f val %.3f %.0fs'%(loss.item(),v,time.time()-t0),flush=True)
            if v<best: best=v; torch.save(net.state_dict(),'placement.pt')
    print('best val',best)
if __name__=='__main__': main()

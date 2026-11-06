"""Does audio help placement? Same GRU; adds band energies around each note. Arms: none, true audio, audio of another song. Run on the instance."""
import glob,json,os,sys,time
import numpy as np,torch,torch.nn as nn
from train import M as _M,feats,tobin,NB
dev='cuda'; arm=sys.argv[1]; steps=int(sys.argv[2])
info=json.load(open('placement_meta.json'))
F='/root/data/feat_new/features/'
def mel_feats(a,audio,shift_s):
    m=np.load(F+audio+'/mel10_complete.npy',mmap_mode='r'); n=m.shape[0] if m.shape[0]>m.shape[1] else m.shape[1]
    M=np.asarray(m) if m.shape[0]>m.shape[1] else np.asarray(m).T
    b=M.shape[1]//8; B=np.stack([M[:,i*b:(i+1)*b].mean(1) for i in range(8)],1).astype(np.float32); B=(B-B.mean(0))/(B.std(0)+1e-5)
    idx=np.clip(((a[:,0]+shift_s)*100).astype(int),0,len(B)-1)
    on=B[np.clip(idx,0,len(B)-1)]; pre=B[np.clip(idx-5,0,len(B)-1)]; post=B[np.clip(idx+5,0,len(B)-1)]
    return np.concatenate([on,on-pre,post-on],1)
data={};split={}
items=[(c,v) for c,v in info.items() if os.path.exists(F+v['audio']+'/mel10_complete.npy') and os.path.exists(f'raw_notes/{c}.npy')]
audios=sorted({v['audio'] for _,v in items}); rot={a:audios[(i+7)%len(audios)] for i,a in enumerate(audios)}
for c,v in items:
    a=np.load(f'raw_notes/{c}.npy'); f=feats(a)
    if arm!='none':
        au=v['audio'] if arm=='true' else rot[v['audio']]
        try: f=np.concatenate([f,mel_feats(a,au,v['shift_s'])],1)
        except Exception: continue
    data[c]=(f.astype(np.float32),tobin(a[:,3])); split[c]=v['split']
tr=[c for c in data if split[c]=='train']; va=[c for c in data if split[c]=='val']; d=next(iter(data.values()))[0].shape[1]
print(arm,'train',len(tr),'val',len(va),'dim',d,flush=True)
class M(nn.Module):
    def __init__(s,h=256):
        super().__init__(); s.prev=nn.Embedding(NB+1,48); s.inp=nn.Linear(d+48,h); s.g=nn.GRU(h,h,2,batch_first=True,dropout=.3); s.o=nn.Linear(h,NB)
    def forward(s,f,pb): o,_=s.g(torch.relu(s.inp(torch.cat([f,s.prev(pb)],-1)))); return s.o(o)
rng=np.random.default_rng(0); net=M().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=.05)
def batch(bs=128,L=128):
    F_,Y,P=[],[],[]
    for k in rng.choice(tr,bs):
        f,y=data[k]; i=rng.integers(0,max(1,len(y)-L)); fj=f[i:i+L]; yj=y[i:i+L]
        if len(yj)<L: fj=np.pad(fj,((0,L-len(yj)),(0,0))); yj=np.r_[yj,np.full(L-len(yj),-1)]
        F_.append(fj);Y.append(yj);P.append(np.r_[NB,np.where(yj[:-1]<0,NB,yj[:-1])])
    return [torch.as_tensor(np.stack(x)).to(dev) for x in(F_,Y,P)]
def ev():
    net.eval(); t=c=0
    with torch.no_grad():
        for k in va:
            f,y=data[k]; lg=net(torch.as_tensor(f)[None].to(dev),torch.as_tensor(np.r_[NB,y[:-1]])[None].to(dev)); t+=nn.functional.cross_entropy(lg[0],torch.as_tensor(y).to(dev),reduction='sum').item(); c+=len(y)
    net.train(); return t/c
best=9e9;t0=time.time()
for step in range(steps):
    f,y,p=batch(); loss=nn.functional.cross_entropy(net(f,p).reshape(-1,NB),y.reshape(-1),ignore_index=-1)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
    if (step+1)%500==0:
        v=ev(); print(arm,step+1,'train %.3f val %.3f %.0fs'%(loss.item(),v,time.time()-t0),flush=True)
        if v<best: best=v; torch.save(net.state_dict(),f'placement_{arm}.pt')
print(arm,'best',best,flush=True)

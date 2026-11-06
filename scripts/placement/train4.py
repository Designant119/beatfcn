"""Same placement model on all charts fetched so far, with early stopping on held-out val charts."""
import glob,json,os,sys,time
import numpy as np,torch,torch.nn as nn
from train import M,feats,tobin,NB
torch.set_num_threads(3)
meta={r['chart_id'].split(':')[1]:r['split'] for r in json.load(open('/Users/yukoval/yukoval-projects/beatfcn/reports/research-20261104/data/meta.json')) if r['mode']=='phira'}
data={};split={}
for f in glob.glob('raw_notes/*.npy'):
    c=os.path.basename(f)[:-4]; s=meta.get(c)
    if s not in('train','val'): continue
    a=np.load(f); data[c]=(feats(a),tobin(a[:,3])); split[c]=s
tr=[c for c in data if split[c]=='train']; va=[c for c in data if split[c]=='val']
print('train',len(tr),'val',len(va),'notes',sum(len(data[c][1]) for c in data),flush=True)
rng=np.random.default_rng(0); net=M(); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=.05)
def batch(bs=48,L=128):
    F,Y,P=[],[],[]
    for k in rng.choice(tr,bs):
        f,y=data[k]; i=rng.integers(0,max(1,len(y)-L)); fj=f[i:i+L]; yj=y[i:i+L]
        if len(yj)<L: fj=np.pad(fj,((0,L-len(yj)),(0,0))); yj=np.r_[yj,np.full(L-len(yj),-1)]
        p=np.r_[NB,np.where(yj[:-1]<0,NB,yj[:-1])]; F.append(fj);Y.append(yj);P.append(p)
    return map(torch.as_tensor,(np.stack(F),np.stack(Y),np.stack(P)))
def ev():
    net.eval(); t=c=0
    with torch.no_grad():
        for k in va:
            f,y=data[k]; lg,_=net(torch.as_tensor(f)[None],torch.as_tensor(np.r_[NB,y[:-1]])[None]); t+=nn.functional.cross_entropy(lg[0],torch.as_tensor(y),reduction='sum').item(); c+=len(y)
    net.train(); return t/c
best=9e9;t0=time.time()
for step in range(int(sys.argv[1])):
    f,y,p=batch(); lg,_=net(f,p); loss=nn.functional.cross_entropy(lg.reshape(-1,NB),y.reshape(-1),ignore_index=-1)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
    if (step+1)%250==0:
        v=ev(); print(step+1,'train %.3f val %.3f %.0fs'%(loss.item(),v,time.time()-t0),flush=True)
        if v<best: best=v; torch.save(net.state_dict(),'placement4.pt')
print('best',best)

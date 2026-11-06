"""Sample a learned layout for a rhythm draft and compare jumps with humans and the rule layout."""
import json,sys,zipfile
import numpy as np, torch, yaml
sys.path[:0]=['/tmp/bf_deploy','/tmp/bf_deploy/scripts','.']
from pathlib import Path
from train import M,feats,BINS,NB
from beatfcn.events.phira import parse_phira
from beatfcn.events.phira_postprocess import clean_types, limit_fingers
from scripts.build_phira_preview import plain_background, rpe
KIND={'tap':0,'hold':1,'flick':2,'drag':3}
def layout(ev,net,T=.8,seed=0):
    ev=sorted(ev,key=lambda e:(e['onset_us'],e['type']!='hold'))
    a=np.array([(e['onset_us']/1e6,e['end_us']/1e6,KIND[e['type']],0.) for e in ev],np.float32); f=torch.as_tensor(feats(a))[None]
    g=torch.Generator().manual_seed(seed); st=None; prev=NB; xs=[]; near=[]
    c=(BINS[:-1]+BINS[1:])/2
    with torch.no_grad():
        for i in range(len(ev)):
            lg,st=net(f[:,i:i+1],torch.tensor([[prev]]),st); lg=lg[0,0]/T
            t=a[i,0]
            # two notes struck together cannot share a spot
            mask=[abs(c[j]-x) < 135 for j in range(NB) for x in [0]] and None
            blocked=[x for tt,x in near if t-tt<.03]
            if blocked:
                for x in blocked: lg[torch.as_tensor(np.abs(c-x)<180)]-=50
            b=torch.multinomial(lg.softmax(-1),1,generator=g).item(); prev=b
            x=float(c[b]+torch.empty(1).uniform_(-20,20,generator=g).item())
            xs.append(x); near=[(tt,xx) for tt,xx in near if t-tt<.03]+[(t,x)]
    return [dict(id=f'event:{i}',t_us=int(e['onset_us']),end_us=int(e['end_us']),type=e['type'],x=round(x,1)) for i,(e,x) in enumerate(zip(ev,xs))]
def stats(L):
    Ls=sorted(L,key=lambda p:p['t_us']); x=np.array([p['x'] for p in Ls]); t=np.array([p['t_us'] for p in Ls],float); dt=np.diff(t)/1e6; dx=np.abs(np.diff(x))
    return 'x sd %.0f jump median %.0f '%(x.std(),np.median(dx))+str({f'{a}-{b}':(int(np.median(dx[(dt>=a)&(dt<b)])),round(float(np.mean(dx[(dt>=a)&(dt<b)]>600)),3),round(float(np.mean(dx[(dt>=a)&(dt<b)]<30)),2)) for a,b in((.01,.1),(.1,.2),(.2,.35),(.35,.6),(.6,2)) if ((dt>=a)&(dt<b)).sum()})
if __name__=='__main__':
    net=M(); net.load_state_dict(torch.load('placement.pt')); net.eval()
    B=Path(sys.argv[1]); audio=B/'in/fang/audio.ogg'
    ev=json.load(open(B/'in/fang/new/draft-events.json'))['events']; r=json.load(open(B/'in/fang/new/draft-receipt.json')); c=int((r.get('clock') or {}).get('c_ms',0)*1000)
    ev=[dict(onset_us=int(e['onset_us'])-c,end_us=int(e['end_us'])-c,type=e['type']) for e in ev]; ev=[e for e in ev if e['onset_us']>=0]
    ev,_=limit_fingers(ev,2); ev,_=clean_types(ev)
    for T in(1.,.8,.6): print('T',T,stats(layout(ev,net,T=T)))
    D=np.load('placement_data.npz'); idx=json.load(open('placement_index.json'))
    J=[]
    for x in idx:
        if x['split']=='val':
            a=D[x['chart'].replace(':','_')]; J.append(dict(zip(('t','x'),(a[:,0],a[:,3]))))
    tt=np.concatenate([np.diff(j['t']) for j in J]); xx=np.concatenate([np.abs(np.diff(j['x'])) for j in J])
    print('human val jump median %.0f '%np.median(xx)+str({f'{a}-{b}':(int(np.median(xx[(tt>=a)&(tt<b)])),round(float(np.mean(xx[(tt>=a)&(tt<b)]>600)),3),round(float(np.mean(xx[(tt>=a)&(tt<b)]<30)),2)) for a,b in((.01,.1),(.1,.2),(.2,.35),(.35,.6),(.6,2))}))
    L=layout(ev,net,T=.8)
    doc=rpe(L,'BeatFCN draft'); doc['META'].update(name='方瑞嘉十给 [D]',composer='unknown',song='audio.ogg',charter='BeatFCN r5 rhythm + learned placement')
    raw=json.dumps(doc,ensure_ascii=False).encode(); p=parse_phira(raw); assert not p.quarantined and len(p.events)==len(ev)
    info=dict(name='方瑞嘉十给 [D]',composer='unknown',level='BeatFCN draft',difficulty=0.,charter='BeatFCN r5 + learned placement',illustrator='Plain background',chart='chart.json',music='audio.ogg',illustration='background.png',offset=0.,aspectRatio=16/9,backgroundDim=.5,lineLength=6.,intro='AI draft: same notes as C, positions by a small learned model.')
    out=B/'round2'; out.mkdir(exist_ok=True)
    with zipfile.ZipFile(out/'方瑞嘉十给-D.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('chart.json',raw); z.writestr('info.yml',yaml.safe_dump(info,allow_unicode=True)); z.writestr('background.png',plain_background()); z.write(audio,'audio.ogg')
    print('wrote D notes',len(L))

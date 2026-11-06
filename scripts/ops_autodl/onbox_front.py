import json,sys,subprocess
from concurrent.futures import ProcessPoolExecutor
sys.path[:0]=['/root/beatfcn','/root/beatfcn/scripts']
import run_workflow_benchmark as W
from beatfcn.train.workflow_metrics import repair_hold_release
G='/data/evolve/g5/P_r5_product/'; O='/data/evolve/front/'; import os; os.makedirs(O,exist_ok=True)
meta=json.load(open(G+'run-meta.json')); smp=json.load(open(G+'drafts-samples.json'))
RULES=[(1,None),(2,1),(3,2),(4,2),(5,2),(6,2),(6,3),(8,3),(8,4)]
def job(s):
    ds=[W.draft_tuples(s,smp[str(i)][s['song_id']]) for i in range(8)]; out={}
    for n,v in RULES:
        if n==1: c=ds[0]
        else:
            c,_=W.consensus_draft(ds[:n],s['mode'],s['duration_us'],min_jack_us=100000,pattern_source='anchor',min_votes=v,fragment_guard=True); c=repair_hold_release(c,s['mode'],min_release_us=40000)[0]
        out[f'{n}_{v}']=dict(events=[dict(onset_us=a,end_us=b,type=t,lane=l) for a,b,t,l in c])
    return s['song_id'],out
songs=[s for s in meta['panel']['songs'] if all(s['song_id'] in smp[str(i)] for i in range(8))]; print(len(songs),flush=True)
allr={}
with ProcessPoolExecutor(8) as ex:
    for sid,o in ex.map(job,songs):
        for k,v in o.items(): allr.setdefault(k,{})[sid]=v
T='/data/full-database-v1/native-v2-tape'
V=['--floor','/data/evolve/floor.json','--tape',T,'--timing','/data/full-database-v1/chart-timing/chart_timing-v2.json','--features-dir','/root/data/feat_new/features']
ps=[]
for k,v in allr.items():
    f=O+k+'.json'; json.dump(v,open(f,'w'))
    ps.append(subprocess.Popen(['/root/envs/bf/bin/python','scripts/benchmark_v3.py','score','--drafts',f,'--panel',G+'run-meta.json','--out',O+k+'_v3.json','--label',k]+V,cwd='/root/beatfcn',stdout=subprocess.DEVNULL,stderr=open(O+k+'.err','w'),env=dict(os.environ,OMP_NUM_THREADS='1')))
    if len(ps)>=5: [p.wait() for p in ps]; ps=[]
[p.wait() for p in ps]; print('front done')

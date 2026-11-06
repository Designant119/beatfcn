"""Post-terminal fixed-weight greedy mark diagnostic, never primary replacement."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain,KINDS
from beatfcn.model.local_typed_canvas import best_tail
from beatfcn.train.anchor_marks import anchor_inputs,teacher_example
from scripts.run_c07_anchor_marks import device,record
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c07-mark-decision-diagnostic-v1'


@torch.no_grad()
def main():
    started=time.monotonic();p=json.loads((ROOT/'bench/rhythm-v2/c07-mark-decision-diagnostic-v1.json').read_text())
    for f,h in p['sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen identity changed '+f)
    plan=json.loads((ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json').read_text())
    base=json.loads((ROOT/'bench/rhythm-v2/c06-reset-unified-count-law-v1.json').read_text());panel,row=load_data()
    model=AnchorMarkChain().to('mps');model.load_state_dict(torch.load(ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1/quality-1000.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    original=json.loads((ROOT/plan['baseline']/'quality-1000-draft.json').read_text());lookup={r['index']:r for r in original}
    summary={};oracle=[]
    for track in ('machine','oracle'):
        rows=[];traces=[]
        for index in plan['indices']['dev']:
            if time.monotonic()-started>90:raise TimeoutError('Frozen mark diagnostic cap')
            rr=row(index,'draft');target=None
            if track=='oracle':x,target,heads=teacher_example(rr)
            else:
                heads=sorted(v['onset_us'] for v in lookup[index]['prediction']['events']);x=anchor_inputs(rr,heads)
            features,hidden=model.encode(device(x));previous=torch.full((1,1),4,dtype=torch.long,device='mps');events=[];probabilities=[]
            for j,head in enumerate(heads):
                h,hidden=model.gru(torch.cat((features[:,j:j+1],model.types(previous)),-1),hidden)
                logits=model.kind(h[:,0]).cpu().numpy()[0].astype(np.float64)
                if rr['request'].duration_us//1000-head//1000<1:logits[1]=-np.inf
                if rr['request'].allowed_types is not None:
                    logits[[k for k,t in enumerate(KINDS) if t not in rr['request'].allowed_types]]=-np.inf
                probs=np.exp(logits-logits.max());probs/=probs.sum();kind=int(probs.argmax());probabilities.append(probs.tolist());end=head
                if kind==1:
                    fields=[layer(h[:,0]).cpu().numpy()[0] for layer in (model.tail_block,model.tail_cell,model.tail_ms)]
                    end=(head//1000)*1000+best_tail(*fields,0,(rr['request'].duration_us//1000-head//1000)*1000)
                events.append(dict(onset_us=int(head),end_us=int(end),type=KINDS[kind],lane=-1));previous=torch.tensor([[kind]],dtype=torch.long,device='mps')
            rows.append(record(events,rr,index,panel.records[index]['component'],'draft',base['shifts_us']))
            traces.append(dict(index=index,heads=heads,type_probability_under_generated_prefix=probabilities))
            if target is not None:
                labels=target['kind'][0].numpy();prob=np.asarray(probabilities);nll=-np.log(prob[np.arange(len(labels)),labels].clip(1e-300))
                oracle.append(dict(index=index,component=panel.records[index]['component'],events=len(labels),nll=float(nll.mean())))
        write(ART/f'{track}-events.json',rows);write(ART/f'{track}-trace.json',traces);summary[track]=metrics(rows)
        summary[track]['typed50']=float(np.mean([r['scores']['note_50ms']['f1'] for r in rows]))
    result=dict(id=p['id'],status='completed',scope=p['scope'],optimizer_updates=0,cloud_calls=0,
        summary=summary,oracle_free_prefix=oracle,wall_s=time.monotonic()-started,
        primary_unchanged=True,plan_sha256=sha(ROOT/'bench/rhythm-v2/c07-mark-decision-diagnostic-v1.json'))
    write(ROOT/'reports/local-evolution-v3/mark-decision-diagnostic.json',result)
    print(json.dumps({k:{f:v[f] for f in ('full_note_f1','typed50','onset10')} for k,v in summary.items()},ensure_ascii=False))


if __name__=='__main__':
    import os
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5);main()

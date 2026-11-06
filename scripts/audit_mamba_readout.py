"""Zero-update CPU comparison of frozen SSD1000 decisions; no new training."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.canvas_diffusion import _sample_tail
from beatfcn.model.count_readout import expected_tail_rank,cardinality_map
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_typed_canvas import KINDS,best_tail
from beatfcn.model.sample import match_f1
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/local-mamba-readout-v1'
PLAN=ROOT/'bench/rhythm-v2/local-mamba-readout-v1.json';REPORT=ROOT/'reports/local_mamba_readout_v1.json'


def decode(out,inputs,request,decision,seed):
    rng=np.random.default_rng(seed);valid=inputs['canvas_valid'][0].numpy()
    heads=inputs['canvas_head_us'][0].numpy();allowed=inputs['canvas_head_allowed'][0].numpy()
    ids=np.flatnonzero(np.repeat(valid[:,None],4,axis=1).reshape(-1))
    logits=out['count'][0].astype(np.float64);p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    active=p.reshape(-1,17)[ids];detail={}
    if decision=='expected_tail_rank':chosen,detail=expected_tail_rank(active,event_limit=request.max_events)
    elif decision=='cardinality':chosen,detail=cardinality_map(active,event_limit=request.max_events)
    elif decision=='mode':chosen=active.argmax(-1)
    elif decision=='ancestral':chosen=np.asarray([rng.choice(17,p=row/row.sum()) for row in active])
    else:raise ValueError('Unknown frozen decision')
    if int(chosen.sum())>request.max_events:raise ValueError('Readout event overflow')
    counts=np.zeros(800,np.int64);counts[ids]=chosen;counts=counts.reshape(200,4);events=[]
    for frame,kind in np.argwhere(counts>0):
        base=request.start_us+int(frame)*40000
        for slot in range(int(counts[frame,kind])):
            hlog=out['offset'][0,frame,kind,slot].astype(np.float64)
            if decision=='ancestral':
                weight=np.exp(hlog-hlog.max());weight*=allowed[frame];weight/=weight.sum()
                offset=int(rng.choice(len(weight),p=weight))
            else:offset=int(hlog.argmax())
            head=int(heads[frame,offset]);end=head
            if kind==1:
                fields=[out[k][0,frame,slot].astype(np.float64) for k in ('tail_block','tail_cell','tail_ms')]
                if decision=='ancestral':end=base+1000*_sample_tail(*fields,(head-base)//1000+1,(request.duration_us-base)//1000,rng)
                else:end=base+best_tail(*fields,head-base,request.duration_us-base)
            events.append(RhythmEvent(head,end,KINDS[kind]))
    events.sort();tokens=encode_events(events,request)
    if len(tokens)>request.max_tokens:raise ValueError('Readout native serialization overflow')
    return events,detail


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve same-weight audit')
    torch.set_num_threads(2)
    parent_path=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(parent_path.read_text())
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt'
    plan=dict(id='local-mamba-readout-v1',seed=20261056,checkpoint=str(weight.relative_to(ROOT)),checkpoint_sha256=sha(weight),
        parent_plan_sha256=sha(parent_path),indices=parent['indices'],wall_s=1200,device='cpu',optimizer_updates=0,
        decisions=['expected_tail_rank','mode','cardinality','ancestral0','ancestral1','ancestral2','ancestral3'],
        interpretation='Same frozen1000weights, fourT1draws equal-weighted, all decisions reported. Original expected-tail-rank primary remains unchanged; this cannot select a better checkpoint or claim backbone causality.',
        shifts_us=parent['shifts_us'],cloud_compute_calls=0,confirmation_scored=False,
        source_sha256={k:sha(ROOT/k) for k in ('scripts/audit_mamba_readout.py','beatfcn/model/local_mamba_canvas.py','beatfcn/model/count_readout.py','beatfcn/model/canvas_diffusion.py')})
    write(PLAN,plan);started=time.monotonic()
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval()
    saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(parent_path):raise ValueError('Frozen SSD checkpoint identity')
    model.load_state_dict(saved['model'])
    report=dict(id=plan['id'],status='evaluating',optimizer_updates=0,processed=0,failures=[],cloud_compute_calls=0,confirmation_scored=False);write(REPORT,report)
    records={task:{decision:[] for decision in plan['decisions']} for task in ('draft','empty')}
    try:
        for task,indices in [('draft',plan['indices']['dev']),('empty',plan['indices']['empty_dev'])]:
            for index in indices:
                if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Frozen same-weight audit wall')
                e=row(index,'draft');inp,_=relative_batch([e],'cpu')
                with torch.no_grad():out={k:v.numpy() for k,v in model(inp).items()}
                path=ART/'probabilities'/f'{index}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
                for name in plan['decisions']:
                    decision='ancestral' if name.startswith('ancestral') else name;sample=int(name[-1]) if decision=='ancestral' else 0
                    try:
                        events,detail=decode(out,inp,e['request'],decision,plan['seed']+17*index+sample)
                        prediction=dict(complete=True,events=[asdict(v) for v in events],count_decision=detail)
                    except (ValueError,RuntimeError) as exc:events=[];prediction=dict(complete=False,events=[],failure=str(exc))
                    start=e['request'].start_us;width=e['request'].end_us-start
                    shifts=[match_f1([start+(v.onset_us-start+d)%width for v in events],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                    records[task][name].append(dict(index=index,component=panel.records[index]['component'],request=asdict(e['request']),prediction=prediction,target=[asdict(v) for v in e['targets']],scores=score(events,e['targets']),shifted_onset10=shifts))
                    write(ART/f'{task}-{name}.json',records[task][name])
                report['processed']+=1;report['wall_s']=time.monotonic()-started;write(REPORT,report)
        report.update(status='completed',results={t:{k:metrics(v) for k,v in d.items()} for t,d in records.items()},wall_s=time.monotonic()-started)
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started);report['failures'].append(f'{type(exc).__name__}: {exc}');write(REPORT,report);raise


if __name__=='__main__':main()

"""Measure C06 cancellation criteria and count support before new training."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.count_readout import expected_tail_rank
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_local_evolution import fresh, load_data
from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/c06-candidate-preflight-v1.json'
REPORT=ROOT/'reports/local-evolution-v2/baseline-preflight.json'


def rank_stats(probability,positive):
    p=np.asarray(probability);y=np.asarray(positive,dtype=bool)
    if not y.any() or y.all():return dict(auroc=None,average_precision=None)
    a,b=p[y],p[~y]
    auroc=float(((a[:,None]>b[None]).astype(float)+.5*(a[:,None]==b[None])).mean())
    order=np.argsort(-p,kind='stable');yy=y[order]
    ap=float(((yy.cumsum()/(np.arange(len(yy))+1))*yy).sum()/yy.sum())
    return dict(auroc=auroc,average_precision=ap,prevalence=float(y.mean()),
        mean_positive_probability=float(a.mean()),mean_negative_probability=float(b.mean()))


@torch.no_grad()
def main():
    if REPORT.exists():raise FileExistsError('Preserve the preflight')
    torch.set_num_threads(2);p=json.loads(PLAN.read_text())
    parent=json.loads((ROOT/p['parent_plan']).read_text());start=time.monotonic()
    weight=ROOT/'artifacts/local-evolution-v1/ssd_reset/quality-1000.pt'
    expected=json.loads((ROOT/'reports/local-evolution-v1/ssd_reset.json').read_text())['final_weight_sha256']
    if sha(weight)!=expected:raise ValueError('Frozen reset checkpoint identity')
    p.update(script_sha256=sha(Path(__file__)),weight_sha256=expected);write(PLAN,p)
    model=fresh(parent,'ssd_reset','cpu').eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model'])
    panel,row=load_data();seen=set();indices=[]
    for i in parent['ssd']['indices']['dev']:
        c=panel.records[i]['component']
        if c not in seen:seen.add(c);indices.append(i)
    errors=[];ce=[];entropies=[];masked=0;slots=0;records=[];logits={};types={k:dict(probability=[],positive=[],count_nll=[],brier=[]) for k in range(4)}
    pred_total=true_total=intersect=l1=0;component_counts={}
    for index in indices+parent['ssd']['indices']['empty_dev']:
        e=row(index,'draft');inp,target=relative_batch([e],'cpu');out=model(inp)
        prob=out['count'][0].softmax(-1);valid=inp['canvas_valid'][0]
        chosen,_=expected_tail_rank(prob[valid].flatten(0,1).numpy(),event_limit=e['request'].max_events)
        pred=np.zeros((200,4),np.int64);pred[valid.numpy()]=chosen.reshape(-1,4)
        truth=target['count'][0].numpy();active=valid.numpy()
        logits[index]={k:out[k].clone() for k in ('count','base_count','existence','offset')}
        if index in indices:
            pm=pred[active,1:];tm=truth[active,1:]
            a,b,c=int(pm.sum()),int(tm.sum()),int(np.minimum(pm,tm).sum());d=int(abs(pm-tm).sum())
            pred_total+=a;true_total+=b;intersect+=c;l1+=d
            component_counts[panel.records[index]['component']]=dict(pred=a,true=b,intersection=c,f1=2*c/max(a+b,1),l1=d)
            mask=target['slot_valid'][0]&valid[:,None,None]
            predicted=out['offset'][0].argmax(-1);labels=target['offset'][0]
            head=inp['canvas_head_us'][0][:,None,None,:].expand(-1,4,16,-1)
            ph=head.gather(-1,predicted[...,None]).squeeze(-1);th=head.gather(-1,labels[...,None]).squeeze(-1)
            err=((ph-th).abs().double()/1000)[mask];errors.extend(err.tolist())
            allowed=inp['canvas_head_allowed'][0][:,None,None,:].expand_as(head)
            masked+=int((~allowed.gather(-1,labels[...,None]).squeeze(-1)&mask).sum());slots+=int(mask.sum())
            lp=out['offset'][0].log_softmax(-1)
            ce.extend((-lp.gather(-1,labels[...,None]).squeeze(-1))[mask].tolist())
            entropies.extend((-(lp.exp()*lp).sum(-1))[mask].tolist())
            onehot=F.one_hot(target['count'][0],17).float()
            nll=-out['count'][0].log_softmax(-1).gather(-1,target['count'][0,...,None]).squeeze(-1)
            for kind in range(4):
                t=types[kind];t['probability'].extend((1-prob[valid,kind,0]).tolist());t['positive'].extend((target['count'][0,valid,kind]>0).tolist())
                t['count_nll'].extend(nll[valid,kind].tolist());t['brier'].extend((prob[valid,kind]-onehot[valid,kind]).square().sum(-1).tolist())
        records.append(dict(index=index,predicted_counts=pred.tolist(),true_counts=truth.tolist(),
            expected_events=float((prob[valid]*torch.arange(17)).sum()),decoded_events=int(pred.sum())))
        if time.monotonic()-start>p['wall_s']:raise TimeoutError('Preflight wall')
    exposure=np.zeros((4,17),np.int64);cached={};window_ids=set();components=set()
    for batch in parent['ssd']['sample_stream'][:1000]:
        for spec in batch:
            key=(spec['index'],spec['view'],spec['field_index'])
            if key not in cached:
                e=row(*key);inp,target=relative_batch([e],'cpu');valid=inp['canvas_valid'][0].numpy();counts=target['count'][0].numpy()[valid]
                cached[key]=np.stack([np.bincount(counts[:,k],minlength=17) for k in range(4)])
            exposure+=cached[key];window_ids.add(spec['index']);components.add(panel.records[spec['index']]['component'])
        if time.monotonic()-start>p['wall_s']:raise TimeoutError('Exposure audit wall')
    median=float(np.median(errors));offset=dict(teacher_median_error_ms=median,teacher_mean_error_ms=float(np.mean(errors)),
        teacher_hit10=float(np.mean(np.asarray(errors)<=10)),mean_nll=float(np.mean(ce)),mean_entropy=float(np.mean(entropies)),
        target_slots=slots,masked_target_fraction=masked/max(1,slots),
        training_recommended=median>10 and masked/max(1,slots)<=.01,
        decision='cancel_taps_training_per_Grok_rule' if median<=10 else 'eligible_for_taps_implementation_probe')
    count=dict(non_tap=dict(pred=pred_total,true=true_total,intersection=intersect,f1=2*intersect/max(pred_total+true_total,1),l1=l1),
        components=component_counts,per_type={('tap','hold','flick','drag')[k]:dict(**rank_stats(v['probability'],v['positive']),
            natural_count_nll=float(np.mean(v['count_nll'])),natural_brier=float(np.mean(v['brier']))) for k,v in types.items()},
        exposure_histogram=exposure.tolist(),exposed_positive_cells=(exposure[:,1:].sum(-1)).tolist(),
        unique_exposure_windows=len(window_ids),unique_exposure_components=len(components),requests=2000,
        training_support_sufficient=bool((exposure[:,1:].sum(-1)>0).all()))
    outdir=ROOT/'artifacts/c06-candidate-preflight-v1';outdir.mkdir(parents=True,exist_ok=True)
    torch.save(logits,outdir/'reset-logits.pt');write(outdir/'reset-counts.json',records)
    result=dict(status='completed',optimizer_updates=0,cloud_compute_calls=0,plan_sha256=sha(PLAN),offset=offset,count=count,
        logits_sha256=sha(outdir/'reset-logits.pt'),wall_s=time.monotonic()-start,
        limits='Teacher uses real occupied slots/cells for diagnosis; not free output or hidden-target inference. Same16developmentcomponents reused. Rank discrimination does not uniquely identify a loss or representation cause.')
    write(REPORT,result);print(json.dumps(dict(offset=offset,count=count['non_tap'],positive_exposure=count['exposed_positive_cells'],wall_s=result['wall_s']),ensure_ascii=False))


if __name__=='__main__':main()

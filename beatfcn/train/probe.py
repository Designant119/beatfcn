"""Read-only diagnostics and explicit acceptance for fixed-batch memorisation."""
from contextlib import contextmanager
import math

import torch


def tensor_summary(value, chunk_size=262144):
    flat=value.detach().reshape(-1)
    count=nan=posinf=neginf=0
    total=squared=0.
    minimum,maximum=math.inf,-math.inf
    for chunk in flat.split(chunk_size):
        x=chunk.to(device='cpu',dtype=torch.float64)
        nan+=int(torch.isnan(x).sum());posinf+=int(torch.isposinf(x).sum());neginf+=int(torch.isneginf(x).sum())
        finite=x[torch.isfinite(x)]
        if not finite.numel():continue
        count+=finite.numel();total+=float(finite.sum());squared+=float(finite.square().sum())
        minimum=min(minimum,float(finite.min()));maximum=max(maximum,float(finite.max()))
    mean=total/count if count else None
    return dict(shape=list(value.shape),dtype=str(value.dtype),elements=flat.numel(),finite=count,
                nan=nan,posinf=posinf,neginf=neginf,min=minimum if count else None,
                max=maximum if count else None,mean=mean,
                std=math.sqrt(max(0.,squared/count-mean*mean)) if count else None)


def metrics_json(loss, metrics):
    return dict(loss=float(loss.detach()),illegal=int(metrics['_illegal']),
                heads={name:dict(loss=float(values[0]),count=int(values[1]),correct=int(values[2]),
                                  accuracy=float(values[2]/values[1]) if int(values[1]) else None)
                       for name,values in metrics.items() if name!='_illegal'})


@contextmanager
def capture_decisions():
    """Observe real masked-CE calls, retaining only valid rows; restore on failure."""
    from ..model import p1
    original=p1.masked_ce
    result={}
    def recorded(logits,target,valid,mask=None,name='',sigma=0.):
        output=original(logits,target,valid,mask,name,sigma)
        with torch.no_grad():
            lg=logits.detach().float().reshape(-1,logits.shape[-1])
            keep=valid.reshape(-1).bool()
            if mask is not None:lg=lg.masked_fill(~mask.reshape_as(lg),-1e9)
            lg=lg[keep];tg=target.reshape(-1)[keep].long()
            if lg.numel():
                if bool(((tg<0)|(tg>=lg.shape[-1])).any()):raise ValueError(f'invalid target in {name}')
                pred=lg.argmax(-1);prob=lg.softmax(-1)
                entry=dict(count=int(tg.numel()),accuracy=float((pred==tg).float().mean()),
                           mean_target_probability=float(prob.gather(1,tg[:,None]).mean()))
                if name=='type':
                    entry['confusion_truth_rows_pred_columns']=torch.bincount(
                        tg*lg.shape[-1]+pred,minlength=lg.shape[-1]**2).view(lg.shape[-1],lg.shape[-1]).cpu().tolist()
                    entry['truth_counts']=torch.bincount(tg,minlength=lg.shape[-1]).cpu().tolist()
                    entry['prediction_counts']=torch.bincount(pred,minlength=lg.shape[-1]).cpu().tolist()
                if name in ('offset','end_off'):
                    err=(pred-tg).float()
                    entry['signed_error_ms_quantiles']=torch.quantile(err,err.new_tensor([.1,.5,.9])).cpu().tolist()
                result[name]=entry
        return output
    p1.masked_ce=recorded
    try:yield result
    finally:p1.masked_ce=original


def assess_memorisation(history, gradients, gates, expected_steps):
    if not history or history[0]['step']!=0 or history[-1]['step']!=expected_steps:
        return dict(passed=False,reasons=['missing required initial/final measurements'])
    first,last=history[0],history[-1]
    reasons=[]
    if not all(math.isfinite(r['loss']) and r['illegal']==0 for r in history):reasons.append('nonfinite loss or illegal target')
    if [g['step'] for g in gradients]!=list(range(1,expected_steps+1)) or not all(g['finite'] for g in gradients):
        reasons.append('incomplete or nonfinite gradients')
    if last['loss']>gates['final_loss_at_most']:reasons.append('final supervised loss above threshold')
    reduction=1-last['loss']/max(first['loss'],1e-12)
    if reduction<gates['reduction_at_least']:reasons.append('insufficient loss reduction')
    active=[(k,v) for k,v in last['heads'].items() if v['count'] and k in
            ('time','offset','cont','type','lane','end_block','end_cell','end_off')]
    if not active:reasons.append('no supervised targets')
    for name,values in active:
        accuracy=values['accuracy']
        if accuracy is None or not math.isfinite(accuracy) or accuracy<gates['per_head_accuracy_at_least']:
            reasons.append(f'{name} accuracy below threshold')
    return dict(passed=not reasons,reasons=reasons,loss_reduction_fraction=reduction,
                scope='Fixed training batch only; not held-out quality or condition-control acceptance.')


def assess_generation(records, expected_cores, threshold):
    """Use the existing full-note matches, including both hold endpoints and mania lanes."""
    scores=[]
    for row in records:
        match=row['scores']['edit_utility']['full_note']
        denominator=match['pred']+match['true']
        f1=2*match['matched']/denominator if denominator else 1.
        scores.append(dict(chart_id=row['chart_id'],c0_us=row['c0_us'],full_note_f1=f1,
                           complete=row['complete'],legal=row['legal'],
                           passed=bool(row['complete'] and row['legal'] and f1>=threshold)))
    identities={(r['chart_id'],r['c0_us']) for r in scores}
    return dict(passed=len(records)==len(identities)==expected_cores and all(r['passed'] for r in scores),
                full_note_f1_at_least=threshold,cores=scores)

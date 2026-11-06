"""Pre-registered, zero-update CPU readout and acoustic-intervention audit."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from beatfcn.model.count_readout import cardinality_map,expected_tail_rank
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.local_typed_canvas import best_tail
from beatfcn.model.sample import match_f1
from beatfcn.train.local_bundle import make_row
from beatfcn.train.local_typed_canvas import canvas_batch
from scripts.run_local_typed_canvas import fresh,panel_for,score
from scripts.run_local_bundle import ROOT,sha,write,log


PLAN=ROOT/'bench/rhythm-v2/canvas-readout-support-v1.json'
REPORT=ROOT/'reports/canvas_readout_support_v1.json'
ART=ROOT/'artifacts/canvas-readout-support-v1'
KINDS=('tap','hold','flick','drag')
METHODS=('marginal_argmax','cardinality_conditional_map','aux_existence','expected_tail_rank')
AUDIO_KEYS=('local','local_times','local_valid','mel','mel_times','mel_valid','song','song_times','song_valid')


def softmax(x):
    x=np.asarray(x,np.float64);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve previous diagnostic')
    prior_path=ROOT/'bench/rhythm-v2/local-typed-canvas-v1.json';prior=json.loads(prior_path.read_text())
    weight=ROOT/'artifacts/local-typed-canvas-v1/quality-0900.pt'
    plan=dict(id='canvas-readout-support-v1',optimizer_updates=0,device='cpu',threads=2,
        weight=str(weight.relative_to(ROOT)),weight_sha256=sha(weight),
        prior_plan=str(prior_path.relative_to(ROOT)),prior_plan_sha256=sha(prior_path),
        indices=prior['indices']['val'],donors=prior['indices']['val'][2:]+prior['indices']['val'][:2],
        methods=list(METHODS),primary='cardinality_conditional_map',wall_limit_s=600,
        shifts_us=[137000,281000,419000,733000,1103000,1741000,2593000],
        decoder_contract='All rules frozen before inference. Original marginal MAP, modal model cardinality then conditional MAP, auxiliary balanced-existence score>0.5 plus conditional positive modal count, rounded model expectation with prefix tail ranks. No hidden or requested target count, no NMS/minimum interval/quota/threshold search.',
        acoustic_intervention='Cyclic donor component: replace all local/mel/song values with donor relative clocks/masks; original request, controls, reference, integer address/endpoint permissions and scoring targets fixed. This is deliberate mismatched-audio input, not a valid product request.',
        support_gate=dict(minimum_nonempty=12,density_ratio=[.5,1.75],onset50=.15,
                          onset10_null_gain=.05,minimum_song_directions=5,acoustic_nll_direction='real_count_NLL < donor_count_NLL'),
        interpretation='Exploratory support of readout-only repair; no training, release or whole-family conclusion. Secondary readouts all reported, no selecting the best seed or threshold.')
    paths=set(prior['source_sha256'])|{'scripts/probe_canvas_readout_support.py','beatfcn/model/count_readout.py',
                                     'bench/rhythm-v2/local-typed-canvas-v1.json','reports/local_typed_canvas_v1.json'}
    plan['source_sha256']={p:sha(ROOT/p) for p in paths};write(PLAN,plan)
    ART.mkdir(exist_ok=True);write(ART/'budget-before.json',json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text()))
    write(REPORT,dict(id=plan['id'],status='prepared',complete=False,optimizer_updates=0,cloud_compute_calls=0))
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:
        f.write('\ncanvas-readout-support-v1 预注册：复用冻结900权重、0更新CPU600秒，单次共享前向比较四种固定读出，主方法为完整模型基数众数+条件MAP。7个事前非整拍移位及全部声学流同步置换，只测是否已有定位支持，不调阈值/数量/种子、不把期望量当能力。旧画布失败后的council仍在收集，本次执行Argon/Sol共同的支持检查，新训练待齐备后裁决。\n')
    log(dict(prepared=True,weight_sha256=plan['weight_sha256'],cases=32,optimizer_updates=0))


def readouts(output,valid,limit):
    prob=softmax(output['count'][0]);ids=np.flatnonzero(np.repeat(valid[:,None],4,1).reshape(-1))
    p=prob.reshape(-1,17)[ids];result={}
    for method in METHODS:
        try:
            detail={}
            if method=='marginal_argmax':counts=p.argmax(-1)
            elif method=='cardinality_conditional_map':counts,detail=cardinality_map(p,event_limit=limit)
            elif method=='expected_tail_rank':counts,detail=expected_tail_rank(p,event_limit=limit)
            else:
                exists=output['existence'][0].reshape(-1)[ids]>0
                counts=np.where(exists,p[:,1:].argmax(-1)+1,0)
            full=np.zeros(valid.size*4,np.int64);full[ids]=counts
            result[method]=(full.reshape(valid.size,4),detail,None)
        except ValueError as exc:result[method]=(None,{},str(exc))
    return result,prob


def decode(output,inputs,request,counts,failure=None):
    events=[];tokens=[]
    if failure is None:
        try:
            if counts.sum()>request.max_events:raise ValueError('Predicted event capacity overflow')
            heads=inputs['canvas_head_us'][0].numpy();offset=output['offset'][0].argmax(-1)
            for frame,kind in np.argwhere(counts>0):
                for slot in range(int(counts[frame,kind])):
                    h=int(heads[frame,offset[frame,kind,slot]])
                    if not request.start_us<=h<request.end_us:raise ValueError('Invalid head')
                    end=best_tail(output['tail_block'][0,frame,slot],output['tail_cell'][0,frame,slot],
                        output['tail_ms'][0,frame,slot],h,request.duration_us) if kind==1 else h
                    events.append(RhythmEvent(h,end,KINDS[kind]))
            events.sort();tokens=encode_events(events,request)
            if len(tokens)>request.max_tokens:raise ValueError('Native serialization token overflow')
        except ValueError as exc:failure=str(exc)
    return events,dict(complete=failure is None,failure=failure,events=[asdict(e) for e in events],tokens=tokens)


def distribution_scores(output,prob,target,valid):
    truth=target['count'][0].numpy();where=valid[:,None]
    true_p=np.take_along_axis(prob,truth[...,None],-1)[...,0]
    nll=float((-np.log(true_p.clip(1e-300))*where).sum()/(valid.sum()*4))
    positive=truth>0;pos=positive & where;neg=~positive & where
    return dict(count_nll=nll,
        positive_count_nll=float(-np.log(true_p[pos].clip(1e-300)).mean()) if pos.any() else None,
        empty_count_nll=float(-np.log(true_p[neg].clip(1e-300)).mean()) if neg.any() else None,
        expected_events=float(((prob*np.arange(17)).sum(-1)*where).sum()),
        maximum_nonzero_probability=float(((1-prob[...,0])*where).max()))


def summarize(rows,plan):
    data={};shifts=plan['shifts_us']
    for method in METHODS:
        by_input={}
        for acoustic in ('real','permuted'):
            active=[r for r in rows if r['method']==method and r['acoustic']==acoustic]
            by_song=defaultdict(list);types=Counter()
            for r in active:by_song[r['chart_id']].append(r);types.update(r['predicted_types'])
            by_input[acoustic]=dict(complete=sum(r['prediction']['complete'] for r in active),
                nonempty=sum(bool(r['prediction']['events']) for r in active),
                predicted=sum(len(r['prediction']['events']) for r in active),
                target=sum(r['target_count'] for r in active),predicted_types=dict(types),
                onset10=float(np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in active])),
                onset50=float(np.mean([r['scores']['onset_f1'] for r in active])),
                full_note_f1=float(np.mean([r['scores']['full_note_f1'] for r in active])),
                shifted_onset10=float(np.mean([np.mean([n['onset10'] for n in r['shift_null']]) for r in active])),
                per_song={s:dict(onset10=float(np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in rs])),
                    shifted_onset10=float(np.mean([np.mean([n['onset10'] for n in r['shift_null']]) for r in rs]))) for s,rs in by_song.items()})
        a=by_input['real'];g=plan['support_gate'];a['density_ratio']=a['predicted']/max(a['target'],1)
        directions=sum(x['onset10']>x['shifted_onset10'] for x in a['per_song'].values())
        gate=(a['complete']==16 and a['nonempty']>=g['minimum_nonempty'] and
              g['density_ratio'][0]<=a['density_ratio']<=g['density_ratio'][1] and
              a['onset50']>=g['onset50'] and a['onset10']-a['shifted_onset10']>=g['onset10_null_gain'] and
              directions>=g['minimum_song_directions'])
        data[method]=dict(inputs=by_input,alignment_nonempty_guard=bool(gate),positive_song_directions=directions)
    return data


def run():
    plan=json.loads(PLAN.read_text());report=json.loads(REPORT.read_text())
    if report['status']!='prepared':raise ValueError('Support audit already attempted')
    for p,s in plan['source_sha256'].items():
        if sha(ROOT/p)!=s:raise ValueError('Source changed: '+p)
    if sha(ROOT/plan['weight'])!=plan['weight_sha256']:raise ValueError('Weight changed')
    torch.set_num_threads(plan['threads']);prior=json.loads((ROOT/plan['prior_plan']).read_text())
    saved=torch.load(ROOT/plan['weight'],map_location='cpu',weights_only=False)
    if saved['updates']!=900 or saved['stage']!='quality' or saved['plan_sha256']!=plan['prior_plan_sha256']:
        raise ValueError('Frozen checkpoint metadata mismatch')
    model=fresh(prior);model.load_state_dict(saved['model']);model.eval();panel=panel_for(prior)
    started=time.monotonic();deadline=started+plan['wall_limit_s'];rows=[];distributions=[];cpu_equal=[]
    report['status']='running';write(REPORT,report)
    old=json.loads((ROOT/'artifacts/local-typed-canvas-v1/draft-final.json').read_text())
    try:
        for position,(index,donor) in enumerate(zip(plan['indices'],plan['donors'])):
            if panel.records[index]['component']==panel.records[donor]['component']:raise ValueError('Acoustic donor from same component')
            row=make_row(panel,index);donor_row=make_row(panel,donor)
            x,y=canvas_batch([row]);dx,_=canvas_batch([donor_row]);valid=x['canvas_valid'][0].numpy()
            for acoustic in ('real','permuted'):
                if time.monotonic()>=deadline:raise TimeoutError('Frozen support deadline')
                inp=x if acoustic=='real' else {**x,**{k:dx[k] for k in AUDIO_KEYS}}
                with torch.no_grad():output={k:v.cpu().numpy() for k,v in model(inp).items()}
                selections,prob=readouts(output,valid,row['request'].max_events)
                distributions.append(dict(index=index,donor_index=donor,acoustic=acoustic,**distribution_scores(output,prob,y,valid)))
                np.savez_compressed(ART/f'probabilities-{index}-{acoustic}.npz',count_probability=prob,
                                    auxiliary_existence_logits=output['existence'][0],offset_argmax=output['offset'][0].argmax(-1))
                for method,(counts,detail,failure) in selections.items():
                    if time.monotonic()>=deadline:raise TimeoutError('Frozen support deadline')
                    pred,prediction=decode(output,x,row['request'],counts,failure)
                    s=score(pred,row['targets']);head=[e.onset_us for e in pred];truth=[e.onset_us for e in row['targets']]
                    width=row['request'].end_us-row['request'].start_us;start=row['request'].start_us
                    null=[]
                    for shift in plan['shifts_us']:
                        shifted=[start+(t-start+shift)%width for t in head]
                        null.append(dict(shift_us=shift,onset10=match_f1(shifted,truth,10000)[3],onset50=match_f1(shifted,truth,50000)[3]))
                    rows.append(dict(index=index,chart_id=panel.records[index]['chart_id'],donor_index=donor,
                        acoustic=acoustic,method=method,prediction=prediction,scores=s,shift_null=null,
                        count_decision=detail,target_count=len(truth),predicted_types=dict(Counter(e.type for e in pred))))
                    if acoustic=='real' and method=='marginal_argmax':
                        cpu_equal.append(prediction['events']==old[position]['prediction']['events'])
                write(ART/'predictions.json',rows);write(ART/'distribution-scores.json',distributions)
            log(dict(index=index,position=position+1,expected=distributions[-2]['expected_events'],wall_s=time.monotonic()-started))
        summary=summarize(rows,plan)
        real=np.mean([r['count_nll'] for r in distributions if r['acoustic']=='real'])
        donor=np.mean([r['count_nll'] for r in distributions if r['acoustic']=='permuted'])
        nll_direction=real<donor
        for value in summary.values():value['readout_only_support']=bool(value['alignment_nonempty_guard'] and nll_direction)
        report.update(status='completed',complete=True,optimizer_updates=0,wall_s=time.monotonic()-started,
            readouts=summary,mean_count_nll=dict(real=float(real),permuted=float(donor)),
            CPU_reproduces_all16_saved_MPS_event_multisets=all(cpu_equal),
            primary_support=summary[plan['primary']]['readout_only_support'],
            acoustic_nll_direction=bool(nll_direction),cloud_compute_calls=0)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}')
        write(REPORT,report);raise
    write(REPORT,report)
    budget_path=ROOT/'reports/rhythm_v2_budget.json';budget=json.loads(budget_path.read_text())
    budget['local_cpu_diagnostics'][plan['id']]={'status':report['status'],'optimizer_updates':0,
        'wall_s':report['wall_s'],'new_cloud_spend_usd':0,'report':str(REPORT.relative_to(ROOT)),
        'weight_sha256':plan['weight_sha256'],'primary_readout_support':report['primary_support']}
    write(budget_path,budget)
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\ncanvas-readout-support-v1 完成：0更新CPU{report["wall_s"]:.3f}s，16窗×2声学条件×4固定读出，主读出支持={report["primary_support"]}。全部输出保留，未调整阈值/种子/模型权重，云0；既有失败council中用于裁决下一合同。\n')
    log(dict(status=report['status'],primary_support=report['primary_support'],readouts=summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    (prepare if args.stage=='prepare' else run)()

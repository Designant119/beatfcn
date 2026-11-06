"""Frozen count-tail-rank replication on previously unused song components."""
import argparse
from collections import defaultdict,Counter
from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
import torch

from beatfcn.model.count_readout import expected_tail_rank
from beatfcn.model.count_sampling import categorical,sample_tail
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.sample import match_f1
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_request_panel import event_request,request_inputs
from beatfcn.train.local_bundle import annotated_sequence,make_row
from beatfcn.train.local_typed_canvas import canvas_batch,encode_canvas,decode_target
from scripts.run_local_typed_canvas import fresh,panel_for,score
from scripts.run_local_bundle import ROOT,write,sha,log
from scripts.probe_canvas_readout_support import decode,softmax,AUDIO_KEYS,distribution_scores


PLAN=ROOT/'bench/rhythm-v2/canvas-count-pilot-v1.json'
REPORT=ROOT/'reports/canvas_count_pilot_v1.json'
ART=ROOT/'artifacts/canvas-count-pilot-v1'
KINDS=('tap','hold','flick','drag')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve existing pilot')
    prior_path=ROOT/'bench/rhythm-v2/local-typed-canvas-v1.json';prior=json.loads(prior_path.read_text())
    panel=panel_for(prior);used=prior['indices']['train']+prior['indices']['val']
    excluded_c={panel.records[i]['component'] for i in used};excluded_a={panel.records[i]['audio_sha256'] for i in used}
    by=defaultdict(list)
    for i,r in enumerate(panel.records):
        if (r['split']=='val' and r['mode']=='phira' and r['component'] not in excluded_c and r['audio_sha256'] not in excluded_a
            and r['request']['end_us']-r['request']['start_us']==8_000_000):by[r['component']].append(i)
    buckets=defaultdict(list)
    for component,ids in by.items():
        charts=defaultdict(list)
        for i in ids:
            if 20<=panel.records[i]['tokens']<=768:charts[panel.records[i]['chart_id']].append(i)
        eligible=sorted(c for c,v in charts.items() if len(v)>=2)
        if not eligible:continue
        chart=eligible[0];indices=sorted(charts[chart],key=lambda i:digest(f"20261028:{chart}:{panel.records[i]['request']['start_us']}"))[:2]
        buckets[panel.records[indices[0]]['stratum']].append((digest(f'20261028:{component}'),component,indices))
    selected=[]
    for stratum in sorted(buckets,key=lambda s:digest('20261028:stratum:'+s)):
        selected.append(min(buckets[stratum]));
        if len(selected)==8:break
    if len(selected)!=8:raise ValueError('Need eight independent unused components')
    indices=[i for _,_,ids in selected for i in ids]
    empties=[]
    for _,component,_ in selected:
        candidates=[i for i in by[component] if panel.records[i]['event_count']==0]
        if candidates:empties.append(min(candidates,key=lambda i:digest(f'20261028:empty:{i}')))
    source=json.loads((ROOT/'bench/rhythm-v2/canvas-readout-support-v1.json').read_text())
    plan=dict(id='canvas-count-pilot-v1',optimizer_updates=0,device='cpu',threads=2,wall_limit_s=600,
        prior_plan=str(prior_path.relative_to(ROOT)),prior_plan_sha256=sha(prior_path),
        weight=source['weight'],weight_sha256=source['weight_sha256'],indices=indices,
        donors=indices[2:]+indices[:2],empty_indices=empties,old_indices=prior['indices']['val'],
        components=[c for _,c,_ in selected],strata=[panel.records[ids[0]]['stratum'] for _,_,ids in selected],
        selection='Unused in current recipe train/selection panels, disjoint components and audio SHA; hash round-robin strata, chart-id and window hash. Eligible source <=768 tokens as earlier scope. Not claimed never evaluated elsewhere in project.',
        primary='Locked900 weights; round own expected count then descending P(C>=slot) prefix slots; original MAP offset/absolute-tail heads unchanged',
        support_gate=source['support_gate'],shifts_us=source['shifts_us'],
        secondary_sampling=dict(temperature=1,samples=4,seed_rule='20261027 + 17*record_index + sample_id',
            official_sample=0,null='Joint time-frame logit permutation; preset seed20261029+index, same permutation for four samples'),
        gate_purpose='Bounded local resource-screen only; type,hold,reference,conditions,empty false positives recorded as risks, not release conjunction',
        cloud_compute_limit=0)
    audit=[];paths=set(prior['source_sha256'])|{'scripts/probe_canvas_readout_support.py','scripts/run_canvas_count_pilot.py',
        'beatfcn/model/count_readout.py','beatfcn/model/count_sampling.py','bench/rhythm-v2/local-typed-canvas-v1.json'}
    for index in indices+empties:
        req=event_request(panel.records[index]['request']);events=panel.targets(index)
        target=encode_canvas(events,req)
        if decode_target(target,req)!=sorted(events):raise ValueError('Fresh cohort lossless audit')
        c=panel.charts[panel.records[index]['chart_index']];path=c['target_path']
        if sha(ROOT/path)!=c['target_sha256']:raise ValueError('Fresh target changed')
        paths.add(path);audit.append(dict(index=index,chart_id=c['chart_id'],component=c['component'],
            audio_sha256=c['audio_sha256'],lossless=True,max_cell_type_count=int(target['count'].max())))
    plan['source_sha256']={p:sha(ROOT/p) for p in paths};plan['selection_audit']=audit
    write(PLAN,plan);ART.mkdir(exist_ok=True);write(ART/'budget-before.json',json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text()))
    write(REPORT,dict(id=plan['id'],status='prepared',complete=False,optimizer_updates=0,cloud_compute_calls=0,
                     new_components=8,new_positive_windows=16,new_empty_windows=len(empties)))
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\ncanvas-count-pilot-v1预注册：采用实际支持检查中事前次要expected-tail-rank，冻结900权重，8个未参与本次选择的验证组件16窗+可用空窗，组件/音频与训练及旧8歌不交叉。沿用支持门不降阈值；同步声学置换和7错位，另40旧条件/参考任务及4固定种子T1祖先采样/帧置换。CPU600秒、0更新/0云。目标是是否值得有限本地加资源，不是产品出场；未宣称全项目新盲测。\n')
    log(dict(prepared=True,components=plan['components'],indices=indices,empty_indices=empties,optimizer_updates=0))


def draft_input(panel,index):
    r=panel.records[index];request=event_request(r['request'])
    controls=encode_controls('phira',{},(request.start_us,request.end_us),origin_us=request.start_us)
    inputs,_=request_inputs(panel.features,r['audio_sha256'],request,controls)
    row=dict(inputs=inputs,sequence=annotated_sequence([],request),origin_s=request.start_us/1e6,targets=[],request=request)
    x,_=canvas_batch([row]);return x,request


def rank_decode(output,x,request):
    valid=x['canvas_valid'][0].numpy();ids=np.flatnonzero(np.repeat(valid[:,None],4,1).reshape(-1))
    prob=softmax(output['count'][0]);counts,detail=expected_tail_rank(prob.reshape(-1,17)[ids],event_limit=request.max_events)
    full=np.zeros(valid.size*4,np.int64);full[ids]=counts
    pred,record=decode(output,x,request,full.reshape(valid.size,4))
    return pred,record,prob,detail


def sample_decode(output,x,request,index,sample_id,permutation=None):
    rng=np.random.default_rng(20261027+17*index+sample_id)
    valid=x['canvas_valid'][0].numpy();heads=x['canvas_head_us'][0].numpy();allowed=x['canvas_head_allowed'][0].numpy()
    events=[];failure=None;tokens=[]
    try:
        for f in np.flatnonzero(valid):
            q=int(permutation[f]) if permutation is not None else f
            for kind in range(4):
                count=categorical(rng,output['count'][0,q,kind])
                if len(events)+count>request.max_events:raise ValueError('Sample event capacity')
                for slot in range(count):
                    logits=output['offset'][0,q,kind,slot].copy();logits[~allowed[f]]=-np.inf
                    h=int(heads[f,categorical(rng,logits)])
                    end=sample_tail(rng,output['tail_block'][0,q,slot],output['tail_cell'][0,q,slot],
                        output['tail_ms'][0,q,slot],h,request.duration_us) if kind==1 else h
                    events.append(RhythmEvent(h,end,KINDS[kind]))
        events.sort();tokens=encode_events(events,request)
        if len(tokens)>request.max_tokens:raise ValueError('Sample serialization capacity')
    except ValueError as exc:failure=str(exc)
    return events,dict(complete=failure is None,failure=failure,events=[asdict(e) for e in events],tokens=tokens)


def record(pred,prediction,index,request,targets,plan,**extra):
    truth=[e.onset_us for e in targets];heads=[e.onset_us for e in pred];width=request.end_us-request.start_us
    null=[]
    for shift in plan['shifts_us']:
        moved=[request.start_us+(h-request.start_us+shift)%width for h in heads]
        null.append(dict(shift_us=shift,onset10=match_f1(moved,truth,10000)[3]))
    return dict(index=index,request=asdict(request),prediction=prediction,target=[asdict(e) for e in targets],
                scores=score(pred,targets),shift_null=null,**extra)


def aggregate(rows,panel):
    per=defaultdict(list)
    for r in rows:per[panel.records[r['index']]['component']].append(r)
    truth=sum(len(r['target']) for r in rows);pred=sum(len(r['prediction']['events']) for r in rows)
    return dict(attempted=len(rows),complete=sum(r['prediction']['complete'] for r in rows),
        nonempty=sum(bool(r['prediction']['events']) for r in rows),predicted=pred,target=truth,
        density_ratio=pred/max(truth,1),onset10=float(np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in rows])),
        onset50=float(np.mean([r['scores']['onset_f1'] for r in rows])),
        full_note_f1=float(np.mean([r['scores']['full_note_f1'] for r in rows])),
        shifted_onset10=float(np.mean([np.mean([v['onset10'] for v in r['shift_null']]) for r in rows])),
        per_component={c:dict(onset10=float(np.mean([r['scores']['onset_10ms']['f1'] or 0 for r in rs])),
            shifted_onset10=float(np.mean([np.mean([v['onset10'] for v in r['shift_null']]) for r in rs]))) for c,rs in per.items()})


def run():
    p=json.loads(PLAN.read_text());r=json.loads(REPORT.read_text())
    if r['status']!='prepared':raise ValueError('Pilot already attempted')
    for path,expected in p['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Frozen source changed '+path)
    if sha(ROOT/p['weight'])!=p['weight_sha256']:raise ValueError('Weight changed')
    torch.set_num_threads(2);prior=json.loads((ROOT/p['prior_plan']).read_text());panel=panel_for(prior)
    saved=torch.load(ROOT/p['weight'],map_location='cpu',weights_only=False)
    if saved['plan_sha256']!=p['prior_plan_sha256'] or saved['updates']!=900:raise ValueError('Checkpoint metadata')
    model=fresh(prior);model.load_state_dict(saved['model']);model.eval()
    started=time.monotonic();deadline=started+p['wall_limit_s'];r['status']='running';write(REPORT,r)
    fresh_rows=[];distributions=[];sample_rows=[];empty_rows=[];old_rows={}
    try:
        for index,donor in zip(p['indices'],p['donors']):
            x,request=draft_input(panel,index);dx,_=draft_input(panel,donor)
            for acoustic in ('real','permuted'):
                if time.monotonic()>=deadline:raise TimeoutError('Pilot wall limit')
                inp=x if acoustic=='real' else {**x,**{k:dx[k] for k in AUDIO_KEYS}}
                with torch.no_grad():out={k:v.cpu().numpy() for k,v in model(inp).items()}
                pred,prediction,prob,detail=rank_decode(out,x,request)
                # Scoring targets are loaded after the primary native output.
                targets=panel.targets(index);target={k:torch.from_numpy(v)[None] for k,v in encode_canvas(targets,request).items()}
                fresh_rows.append(record(pred,prediction,index,request,targets,p,acoustic=acoustic,count_decision=detail))
                distributions.append(dict(index=index,acoustic=acoustic,**distribution_scores(out,prob,target,x['canvas_valid'][0].numpy())))
                if acoustic=='real':
                    perm=np.arange(len(x['canvas_valid'][0]));valid=np.flatnonzero(x['canvas_valid'][0].numpy())
                    rng=np.random.default_rng(20261029+index);perm[valid]=rng.permutation(valid)
                    for null in (False,True):
                        for sid in range(4):
                            pp,pr=sample_decode(out,x,request,index,sid,permutation=perm if null else None)
                            sample_rows.append(record(pp,pr,index,request,targets,p,sample_id=sid,frame_permuted=null))
            write(ART/'fresh-predictions.json',fresh_rows);write(ART/'samples.json',sample_rows)
            write(ART/'distribution-scores.json',distributions);log(dict(stage='fresh',index=index,wall_s=time.monotonic()-started))
        for index in p['empty_indices']:
            if time.monotonic()>=deadline:raise TimeoutError('Pilot wall limit')
            x,request=draft_input(panel,index)
            with torch.no_grad():out={k:v.cpu().numpy() for k,v in model(x).items()}
            pred,pr,_,detail=rank_decode(out,x,request);empty_rows.append(record(pred,pr,index,request,panel.targets(index),p,count_decision=detail))
        write(ART/'empty-predictions.json',empty_rows)
        cases={'draft':(p['old_indices'],'draft',None,False),
            'completion':(p['old_indices'][::2],'middle',None,False),
            'density_low':(p['old_indices'][::4],'draft',{'density_notes_per_second':[2.,3.]},False),
            'density_high':(p['old_indices'][::4],'draft',{'density_notes_per_second':[6.,8.]},False),
            'reference_masked':(p['old_indices'][::2],'middle',None,True)}
        for task,(ids,view,bands,masked) in cases.items():
            outputs=[]
            for index in ids:
                if time.monotonic()>=deadline:raise TimeoutError('Pilot wall limit')
                row=make_row(panel,index,view,manual_bands=bands);x,_=canvas_batch([row],mask_reference=masked)
                with torch.no_grad():out={k:v.cpu().numpy() for k,v in model(x).items()}
                pred,pr,_,detail=rank_decode(out,x,row['request'])
                outputs.append(record(pred,pr,index,row['request'],row['targets'],p,mask_reference=masked,count_decision=detail))
            old_rows[task]=outputs;write(ART/f'{task}-old-cohort.json',outputs)
        main=aggregate([v for v in fresh_rows if v['acoustic']=='real'],panel)
        swapped=aggregate([v for v in fresh_rows if v['acoustic']=='permuted'],panel)
        real_nll=float(np.mean([v['count_nll'] for v in distributions if v['acoustic']=='real']))
        donor_nll=float(np.mean([v['count_nll'] for v in distributions if v['acoustic']=='permuted']))
        directions=sum(v['onset10']>v['shifted_onset10'] for v in main['per_component'].values());g=p['support_gate']
        gates=dict(legal_complete=main['complete']==16,nonempty=main['nonempty']>=g['minimum_nonempty'],
            density=g['density_ratio'][0]<=main['density_ratio']<=g['density_ratio'][1],onset50=main['onset50']>=g['onset50'],
            onset10_null=main['onset10']-main['shifted_onset10']>=g['onset10_null_gain'],
            song_directions=directions>=g['minimum_song_directions'],acoustic_count_nll=real_nll<donor_nll)
        samples={str(s):{'real':aggregate([v for v in sample_rows if not v['frame_permuted'] and v['sample_id']==s],panel),
                         'frame_permuted':aggregate([v for v in sample_rows if v['frame_permuted'] and v['sample_id']==s],panel)} for s in range(4)}
        r.update(status='completed',complete=True,optimizer_updates=0,wall_s=time.monotonic()-started,
            promising_for_bounded_local_scale_pilot=all(gates.values()),primary_gates=gates,primary=main,
            acoustic_permuted=swapped,mean_count_nll=dict(real=real_nll,permuted=donor_nll),samples=samples,
            old_cohort={t:aggregate(v,panel) for t,v in old_rows.items()},
            empty_cases=[dict(index=v['index'],predicted=len(v['prediction']['events'])) for v in empty_rows],
            product_release_ready=False,cloud_compute_calls=0)
    except BaseException as exc:
        r.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}')
        write(REPORT,r);raise
    write(REPORT,r)
    if sha(ROOT/p['weight'])!=p['weight_sha256']:raise ValueError('Post-pilot weight changed')
    bp=ROOT/'reports/rhythm_v2_budget.json';budget=json.loads(bp.read_text());budget['local_cpu_diagnostics'][p['id']]={
        'status':r['status'],'optimizer_updates':0,'wall_s':r['wall_s'],'new_cloud_spend_usd':0,
        'report':str(REPORT.relative_to(ROOT)),'promising_for_bounded_local_scale_pilot':r['promising_for_bounded_local_scale_pilot']}
    write(bp,budget)
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\ncanvas-count-pilot-v1终态：0更新CPU{r["wall_s"]:.3f}s，新8组件16窗完整F1={main["full_note_f1"]:.6f}、onset10={main["onset10"]:.6f}、onset50={main["onset50"]:.6f}，固定主读出门{gates}。扩算筛选={r["promising_for_bounded_local_scale_pilot"]}，不是产品就绪；4种子全部采样/帧置换和旧40任务完整保存，原权重未改、云0。\n')
    log(dict(status=r['status'],promising=r['promising_for_bounded_local_scale_pilot'],primary=main,gates=gates,wall_s=r['wall_s']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)()

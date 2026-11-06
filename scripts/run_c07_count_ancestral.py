"""One frozen T=1 count-law audit; deterministic members, no training."""
from dataclasses import asdict
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1
from beatfcn.train.relative_canvas import generate_relative, relative_batch
from scripts.diagnose_evolution_terminals import FrozenOutput
from scripts.run_evolution_unified_count import fresh
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/c07-count-ancestral-audit-v1.json'
ART = ROOT / 'artifacts/c07-count-ancestral-audit-v1'
REPORT = ROOT / 'reports/local-evolution-v2/count-ancestral.json'
VIEWS = ('draft', 'empty', 'completion', 'reference_masked', 'density_low', 'density_high')
KINDS = ('tap', 'hold', 'flick', 'drag')


def read(path):
    return json.loads((ROOT / path).read_text())


def specs(plan):
    dev = plan['indices']['dev']
    return dict(draft=[(i, 'draft', None, False) for i in dev],
        empty=[(i, 'draft', None, False) for i in plan['indices']['empty_dev']],
        completion=[(i, 'middle', None, False) for i in dev[::2]],
        reference_masked=[(i, 'middle', None, True) for i in dev[::2]],
        density_low=[(i, 'draft', (2., 3.), False) for i in dev[::8]],
        density_high=[(i, 'draft', (6., 8.), False) for i in dev[::8]])


def inverse_cdf(p, uniforms):
    counts = (uniforms[..., None] >= np.cumsum(p, axis=-1)).sum(-1)
    if (counts >= p.shape[-1]).any():
        raise ValueError('Numerically invalid categorical inverse CDF')
    return counts


def normalized(logits):
    x = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite count logits')
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def prepare():
    if PLAN.exists():
        raise FileExistsError('Do not overwrite a registered audit')
    base = read('bench/rhythm-v2/c06-reset-unified-count-law-v1.json')
    files = set(base['sources']) | {'scripts/run_c07_count_ancestral.py',
        'scripts/run_local_evolution.py', 'scripts/diagnose_evolution_terminals.py',
        'scripts/run_local_typed_canvas.py', 'beatfcn/model/count_readout.py',
        'beatfcn/model/sample.py', 'beatfcn/model/full_event_codec.py', 'tests/test_c07_count_ancestral.py'}
    plan = dict(id='c07-count-ancestral-audit-v1', experiment_class='local_decision_probe',
        predecessors=['c06-reset-unified-count-law-v1'], seed=20261032,
        indices=base['indices'], model=base['model'], density_reference=base['density_reference'],
        baseline_plan='bench/rhythm-v2/c06-reset-unified-count-law-v1.json',
        baseline_plan_sha256=sha(ROOT / 'bench/rhythm-v2/c06-reset-unified-count-law-v1.json'),
        weights=dict(old=base['baseline_weight'], new='artifacts/c06-reset-unified-count-law-v1/quality-1000.pt'),
        weight_sha256=dict(old=sha(ROOT/base['baseline_weight']), new=read('reports/local-evolution-v2/unified-count.json')['final_weight_sha256']),
        seeds=[2026102901,2026102902,2026102903,2026102904],
        views=list(VIEWS), temperature=1., probability_law='Independent normalized final categorical count per valid cell/type; PCG64 SeedSequence(seed,index,view_enum,0), common uniform across weights and null.',
        null='Roll all four type count-probability rows by10 valid cells; members unchanged. Distribution histograms preserved; realized sampled N need not match.',
        single_intervention='Count selection only; all member offset/tail, input features and native event support fixed.',
        shared_failure_modes=['tail_rank_type_competition'],
        mechanism_changes=dict(tail_rank_type_competition=dict(change='Sample final normalized count law once per fixed draw instead of global expected tail rank',prediction='Correct typed-cell mass is recoverable and beats within-window probability shift',discriminating_control='Four fixed common-uniform draws, time-shifted probability null, original deterministic replay and empty requests')),
        wall_s=dict(export=300, evaluation=600, process=900), memory=dict(driver_gib=8,recommended_fraction=.7),
        gate=dict(non_tap_f1_gain=.03, non_tap_vs_null=.02, components_vs_null=9,
                  full_noninferiority=.01,onset_noninferiority=.01,empty_mean_events_max=20,all_draws_complete=True),
        cloud_compute_calls=0, optimizer_updates=0, shifts_us=base['shifts_us'],
        sources={f:sha(ROOT/f) for f in sorted(files)}, execution_ready=False,
        failure_policy='No retry,threshold,temperature,seed or training change. Failed draw scores zero and stays in denominator; terminal requires mechanism review.')
    write(PLAN,plan)
    for f in files:
        dst=ART/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    shutil.copy2(PLAN,ART/'source/initial-plan.json')


def resource(plan, started):
    torch.mps.synchronize()
    driver, recommended=torch.mps.driver_allocated_memory(),torch.mps.recommended_max_memory()
    if driver>min(plan['memory']['driver_gib']*2**30,plan['memory']['recommended_fraction']*recommended):
        raise MemoryError('Frozen MPS memory cap')
    if time.monotonic()-started>plan['wall_s']['process']:
        raise TimeoutError('Frozen whole-audit time cap')
    return dict(driver_bytes=driver,recommended_bytes=recommended,allocated_bytes=torch.mps.current_allocated_memory())


def count_score(counts, target, request):
    truth=np.zeros_like(counts)
    for e in target:
        truth[(e.onset_us-request.start_us)//40000,KINDS.index(e.type)]+=1
    a,b=counts[:,1:],truth[:,1:]
    return dict(pred=int(a.sum()),true=int(b.sum()),intersection=int(np.minimum(a,b).sum()),l1=int(abs(a-b).sum()))


def run():
    if REPORT.exists():
        raise FileExistsError('No automatic rerun')
    plan=read(str(PLAN.relative_to(ROOT)));started=time.monotonic();report=dict(id=plan['id'],status='exporting',optimizer_updates=0,cloud_compute_calls=0,neural_forward_calls=0,failures=[],exports=[])
    for f,h in plan['sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Source identity changed: '+f)
    for name,f in plan['weights'].items():
        if sha(ROOT/f)!=plan['weight_sha256'][name]:raise ValueError('Weight identity changed: '+name)
    # Exact boundaries and zeros are not approximated by an epsilon floor.
    toy=np.array([[.25,.75],[0.,1.],[1.,0.]])
    if inverse_cdf(toy,np.array([.25,0.,.999])).tolist()!=[1,1,0]:raise AssertionError('Inverse CDF boundary')
    panel,row=load_data();task_specs=specs(plan);export_start=time.monotonic();source_index={}
    try:
        with torch.no_grad():
            for name in ('old','new'):
                model=fresh(plan,'mps',old=name=='old');model.load_state_dict(torch.load(ROOT/plan['weights'][name],map_location='cpu',weights_only=False)['model']);model.eval()
                folder=ART/'export'/name;folder.mkdir(parents=True,exist_ok=True)
                for task,items in task_specs.items():
                    original_folder='artifacts/local-evolution-v1/ssd_reset' if name=='old' else 'artifacts/c06-reset-unified-count-law-v1/unified'
                    originals=read(f'{original_folder}/quality-1000-{task}.json')
                    source_index[(name,task)]={v['index']:v for v in originals}
                    for index,view,bands,masked in items:
                        if time.monotonic()-export_start>plan['wall_s']['export']:raise TimeoutError('Export time cap')
                        memory=resource(plan,started);rr=row(index,view,0,bands);x,_=relative_batch([rr],'mps',mask_reference=masked)
                        out={k:v.detach().cpu() for k,v in model(x).items()};report['neural_forward_calls']+=1
                        layout={k:x[k].cpu() for k in ('canvas_valid','canvas_head_us','canvas_head_allowed')}
                        replay=generate_relative(FrozenOutput(out),layout,rr['request'])
                        if replay['events']!=source_index[(name,task)][index]['prediction']['events']:raise AssertionError(f'Deterministic replay {name}/{task}/{index}')
                        digest={k:dict(shape=list(v.shape),sha256=__import__('hashlib').sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in x.items()}
                        path=folder/f'{task}-{index}.pt';torch.save(dict(out=out,layout=layout,request=asdict(rr['request']),input_summary=digest),path)
                        report['exports'].append(dict(model=name,task=task,index=index,path=str(path.relative_to(ROOT)),sha256=sha(path)))
                        write(REPORT,report)
                del model;gc.collect();torch.mps.empty_cache()
        report['export_wall_s']=time.monotonic()-export_start
        plan.update(execution_ready=True,pretraining_probe_receipts=[dict(status='completed',scope='frozen_model_export',evidence=str(REPORT.relative_to(ROOT)))])
        write(PLAN,plan);shutil.copy2(PLAN,ART/'source/frozen-plan.json');report['plan_sha256']=sha(PLAN)
        moment=time.monotonic();deadline=min(started+900,moment+600);report['status']='sampling';all_summary={};count_rows={}
        for name in ('old','new'):
            all_summary[name]={};count_rows[name]={}
            for condition in ('real','null'):
                summaries=[];counts_by_draw=[]
                for draw,seed in enumerate(plan['seeds']):
                    by_task={};draw_count=[]
                    for task,items in task_specs.items():
                        records=[]
                        for index,view,bands,masked in items:
                            if time.monotonic()>deadline:raise TimeoutError('Sampling/score cap')
                            rr=row(index,view,0,bands);f=ART/'export'/name/f'{task}-{index}.pt';v=torch.load(f,map_location='cpu',weights_only=False)
                            out,layout=v['out'],v['layout'];valid=layout['canvas_valid'][0].numpy();p=normalized(out['count'][0].numpy())
                            if condition=='null':p[valid]=np.roll(p[valid],10,axis=0)
                            u=np.random.default_rng(np.random.SeedSequence([seed,index,VIEWS.index(task),0])).random(p[valid].shape[:-1])
                            counts=np.zeros(p.shape[:-1],np.int64);counts[valid]=inverse_cdf(p[valid],u)
                            forced=torch.full_like(out['count'],-1e9);forced.scatter_(-1,torch.from_numpy(counts)[None,...,None],0.)
                            pred=generate_relative(FrozenOutput({**out,'count':forced}),layout,rr['request'],deadline=deadline)
                            values=[RhythmEvent(**e) for e in pred['events']];scores=score(values,rr['targets'])
                            if not pred['complete']:
                                scores['full_note_f1']=scores['onset_f1']=0.;scores['onset_10ms']['f1']=0.
                                report['failures'].append(dict(model=name,condition=condition,draw=draw,task=task,index=index,error=pred['failure']))
                            start=rr['request'].start_us;width=rr['request'].end_us-start
                            shifts=[match_f1([start+(e.onset_us-start+d)%width for e in values],[e.onset_us for e in rr['targets']],10000)[3] for d in plan['shifts_us']]
                            records.append(dict(index=index,component=panel.records[index]['component'],view=view,request=asdict(rr['request']),prediction=pred,target=[asdict(e) for e in rr['targets']],scores=scores,shifted_onset10=shifts))
                            if task=='draft' and index in plan['indices']['dev'][::2]:draw_count.append(dict(index=index,**count_score(counts,rr['targets'],rr['request'])))
                        write(ART/name/condition/f'draw{draw}-{task}.json',records);by_task[task]=metrics(records)
                    summaries.append(by_task);counts_by_draw.append(draw_count)
                    print(json.dumps(dict(model=name,condition=condition,draw=draw,draft=by_task['draft']['onset10']),ensure_ascii=False),flush=True)
                all_summary[name][condition]=summaries;count_rows[name][condition]=counts_by_draw
        gates={}
        for name in ('old','new'):
            baseline=read('reports/local-evolution-v1/ssd_reset.json' if name=='old' else 'reports/local-evolution-v2/unified-count.json')['evaluations']['draft']
            def count_f1(rows):return 2*sum(r['intersection'] for r in rows)/max(sum(r['pred']+r['true'] for r in rows),1)
            real=float(np.mean([count_f1(v) for v in count_rows[name]['real']]));null=float(np.mean([count_f1(v) for v in count_rows[name]['null']]))
            wins=sum(np.mean([count_f1([v[i]]) for v in count_rows[name]['real']])>np.mean([count_f1([v[i]]) for v in count_rows[name]['null']]) for i in range(16))
            q=all_summary[name]['real'];empty=float(np.mean([sum(v['empty']['predicted_types'].values()) for v in q]))
            gates[name]=dict(non_tap_f1=real,null_f1=null,components_positive=wins,empty_mean=empty,
                passed=bool(real-({'old':18/327,'new':20/305}[name])>=.03 and real-null>=.02 and wins>=9 and
                    np.mean([v['draft']['full_note_f1'] for v in q])>=baseline['full_note_f1']-.01 and
                    np.mean([v['draft']['onset10'] for v in q])>=baseline['onset10']-.01 and empty<=20 and
                    not any(v['model']==name for v in report['failures'])))
        report.update(status='terminal_pending_mechanism_review',summary=all_summary,count_diagnostics=count_rows,mechanism_gate=gates,
            evaluation_wall_s=time.monotonic()-moment,process_wall_s=time.monotonic()-started,
            boundary='74requests x2weights x2conditions x4draws; failed requests preserved. Probability null preserves histograms, not realized sample counts.')
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped_pending_mechanism_review',error=f'{type(exc).__name__}: {exc}',process_wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    def stop(*_):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    {'prepare':prepare,'run':run}[args.action]()

"""Complete frozen B/Z/T source and TT/SELF probe; zero optimizer updates."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import signal
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.c14_probe_support import ProbeGuard,load_frozen_model,prediction_identity,probability_difference
from beatfcn.train.c14_type_probe import type_arm,type_source_coverage,tt_diagnostic,self_diagnostic
from beatfcn.train.native_trial_evaluation import case_inputs,score_outcome
from beatfcn.train.native_quality_metrics import summarize
from scripts import run_native_cell_recovery as engine
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import ROOT,write,sha,weight_digest
from scripts.run_native_cell_chain import closure
from scripts.check_experiment_storage import admission,GIB


ID='c14-type-song-prefix-v1';ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
REPORT=ROOT/f'reports/local-integration-v1/{ID}.json';ARMS=('B','Z','T');DIAGNOSTIC_TASKS=('draft','completion','reference_masked')


def conditional_aggregate(rows):
    summaries=[r['summary'] for r in rows];n=sum(r['type_decisions'] for r in summaries);events=sum(r['events'] for r in summaries)
    def total(key):return sum(r[key] for r in summaries) if all(r[key] is not None for r in summaries) else None
    numerator=total('type_nll_sum')
    holds=sum(r['hold_decisions'] for r in summaries);tail=total('endpoint_nll_sum');tail_prior=total('duration_prior_nll_sum')
    return dict(rows=len(rows),events=events,type_decisions=n,forced=sum(r['forced'] for r in summaries),
        support_failures=sum(r['support_failures'] for r in summaries),head_support_failures=sum(r['head_support_failures'] for r in summaries),
        type_support_failures=sum(r['type_support_failures'] for r in summaries),endpoint_support_failures=sum(r['endpoint_support_failures'] for r in summaries),
        numerical_underflow=sum(r['numerical_underflow'] for r in summaries),
        type_nll=numerator/n if numerator is not None and n else None,
        type_all_target_nll=numerator/events if numerator is not None and events else None,
        frequency_prior_nll=total('global_prior_nll_sum')/n if total('global_prior_nll_sum') is not None and n else None,
        history_prior_nll=total('history_prior_nll_sum')/n if total('history_prior_nll_sum') is not None and n else None,
        hold_decisions=holds,endpoint_nll=tail/holds if tail is not None and holds else None,duration_prior_nll=tail_prior/holds if tail_prior is not None and holds else None,
        scope='Any unsupported target retains infinite-loss semantics;None is never averaged out')


def mechanism_decision(report,groups,conditional):
    b=conditional['B']['TT']['draft'];z=conditional['Z']['TT']['draft'];bs=conditional['B']['SELF']['draft'];zs=conditional['Z']['SELF']['draft']
    baseline=report['free_summary']['B']['draft'];candidate=report['free_summary']['Z']['draft'];improved={}
    for kind in ('hold','flick','drag'):
        bc={};zc={}
        for arm,dest in [('B',bc),('Z',zc)]:
            for r in groups[arm]['draft']:dest[str(r['component'])]=dest.get(str(r['component']),0)+r['scores']['per_type'][kind]['tp']
        improved[kind]=dict(tp_gain=candidate['types'][kind]['tp']-baseline['types'][kind]['tp'],
            improved_components=sum(zc[c]>bc[c] for c in bc),matched_components=candidate['types'][kind]['matched_components'])
    component_improvements=0
    for component in {str(r['component']) for r in report['condition_rows']['B']['TT']['draft']}:
        def component_value(arm):
            values=[r for r in report['condition_rows'][arm]['TT']['draft'] if str(r['component'])==component]
            return conditional_aggregate(values)['type_nll']
        a=component_value('B');v=component_value('Z')
        component_improvements+=a is not None and v is not None and v<a
    finite=b['type_nll'] is not None and z['type_nll'] is not None
    checks=dict(protocol=report.get('all_free_complete',False) and report.get('all_conditionals_complete',False),
        tt_gain=finite and b['type_nll']-z['type_nll']>=.05,components=component_improvements>=10,
        prior_gain=finite and z['frequency_prior_nll']-z['type_nll']>=.01 and z['history_prior_nll']-z['type_nll']>=.01,
        non_tap=any(v['tp_gain']>=3 and v['improved_components']>=2 for v in improved.values()),
        hold_retention=candidate['types']['hold']['tp']>=baseline['types']['hold']['tp'],
        self_support=zs['support_failures']<=bs['support_failures'])
    return dict(checks={k:bool(v) for k,v in checks.items()},supported=all(checks.values()),component_improvements=component_improvements,
        per_type=improved,scope='Primarydraft32 mechanism gate only;no research/scaling qualification;T is never chosen as a winner')


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve type probe registration')
    parent_path=ROOT/'bench/rhythm-v2/c13-native-quality-v1.json';parent=json.loads(parent_path.read_text())
    q=json.loads((ROOT/'artifacts/c13-native-quality-v1/quality.json').read_text());prior=ROOT/'artifacts/c14-training-prior-audit-v1a/prior.json'
    if admission(ROOT,512*2**20,15*GIB)['status']!='admitted':raise MemoryError('Disk admission')
    p=dict(id=ID,parent_plan=str(parent_path.relative_to(ROOT)),parent_plan_sha256=sha(parent_path),checkpoint=q['checkpoint'],checkpoint_sha256=q['checkpoint_sha256'],
        decision_sha256=sha(ROOT/'reports/council-quality-c14/parent-final-decision.md'),wall_s=1800,disk_cap_bytes=512*2**20,
        optimizer_updates=0,cloud_calls=0,arms=list(ARMS),candidate='Z',control_only='T',prior_path=str(prior.relative_to(ROOT)),prior_sha256=sha(prior),
        expected_metadata=parent['expected_free_metadata'],expected_metadata_sha256=parent['expected_free_metadata_sha256'],
        raw_base_sha256={name:sha(ROOT/f'artifacts/c13-native-quality-v1/{name}.json') for name in json.loads((ROOT/parent['expected_free_metadata']).read_text())},
        diagnostic_tasks=list(DIAGNOSTIC_TASKS),primary_mechanism_task='draft',
        self_score_policy='Fixed original targets and cell/rank;no nearest matching or truth history insertion;invalid support means infinite loss. Also report all-target denominator to avoid changed forced-mask denominator.',
        mechanism_gate=dict(tt_nll_gain=.05,positive_components=10,prior_gain=.01,non_tap_tp_gain=3,improved_tp_components=2,hold_tp_nondecreasing=True,self_support_nondecreasing=False),
        predecessors=['c13-native-quality-v1'],shared_failure_modes=['type_conditional_learning','exposure_interface'],
        mechanism_changes=dict(type_conditional_learning=dict(change='Delete only type direct-song input;retain all other paths',prediction='All-target type and actual free TP can improve',discriminating_control='OriginalBASE/full170+Tshift andfixedpriors'),
            exposure_interface=dict(change='Score true current head under own rollout at same cell/rank',prediction='Discriminate pre-existing conditional weakness from generated-history support error',discriminating_control='TT/SELF alltargets;support failures never dropped')),
        pretraining_probe_receipts=[dict(status='completed',evidence='reports/council-quality-c14/type-probe-tests.json',scope='real_small_model_TT_SELF_and_source_isolation')],execution_ready=True)
    p['source_sha256']={f:sha(ROOT/f) for f in set(closure('scripts/run_c14_type_song_prefix.py'))|{'tests/test_c14_type_probe.py'}}
    write(PLAN,p)
    for f in p['source_sha256']:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    write(ART/'source/frozen-plan.json',p)


@torch.no_grad()
def run():
    if REPORT.exists():raise FileExistsError('No automatic type probe retry')
    p=json.loads(PLAN.read_text())
    for name,h in p['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Type probe source drift')
    if sha(ROOT/p['prior_path'])!=p['prior_sha256'] or sha(ROOT/p['expected_metadata'])!=p['expected_metadata_sha256']:raise ValueError('Prior/metadata drift')
    metadata=json.loads((ROOT/p['expected_metadata']).read_text());groups={arm:deepcopy(metadata) for arm in ARMS}
    for arm in ARMS:
        for task,rows in groups[arm].items():write(ART/f'{arm}-{task}.json',rows)
    report=dict(id=ID,status='running',phase='initialization',optimizer_updates=0,cloud_calls=0,cloud_spend_usd=0,base_certificates=[],
        condition_rows={a:{prefix:{task:[] for task in DIAGNOSTIC_TASKS} for prefix in ('TT','SELF')} for a in ARMS},
        expected_free_outcomes=510,expected_conditional_requests=384,research_qualified=False,scaling_authorized=False,confirmation_scored=False)
    expected_conditionals=[dict(arm=arm,prefix=prefix,task=task,index=r['index'],component=r['component'],target_events=len(r['target'])) for arm in ARMS for prefix in ('TT','SELF') for task in DIAGNOSTIC_TASKS for r in metadata[task]]
    write(ART/'expected-conditional-requests.json',expected_conditionals)
    guard=ProbeGuard(ROOT,ART,p['wall_s'],p['disk_cap_bytes']);deadline=guard.start+p['wall_s'];model=None
    def stop(*_):raise TimeoutError('Type probe wall/interrupt')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['wall_s'])
    raw={}
    def persist():
        for arm in ARMS:
            for task,rows in groups[arm].items():write(ART/f'{arm}-{task}.json',rows)
        write(ART/'report.json',report)
    try:
        parent,model,before=load_frozen_model(p);report['weight_digest_before']=before;panel,row=load_data();prior=json.loads((ROOT/p['prior_path']).read_text())
        for task,h in p['raw_base_sha256'].items():
            path=ROOT/f'artifacts/c13-native-quality-v1/{task}.json'
            if sha(path)!=h:raise ValueError('Raw baseline drift')
            raw[task]={r['index']:r for r in json.loads(path.read_text())}
        for arm in ARMS:
            for task,records in groups[arm].items():
                report['phase']=f'free_{arm}_{task}'
                for rec in records:
                    guard.check(report['phase']);rec['status']='attempted';index=rec['index'];old=raw[task][index]
                    bands=tuple(rec['bands']) if rec['bands'] is not None else None
                    rr,x,t,control=case_inputs(engine,parent,panel,row,task,index,rec['view'],bands,rec['masked'])
                    if engine.input_digest(x)!=old['input_sha256']:raise ValueError('Complete model input identity')
                    out=model(x);cp=out['count'][0].softmax(-1).cpu().numpy();count_diff=probability_difference(cp,old['count_probability'])
                    with type_arm(model,out,x,arm):pred=engine.prediction(model,rr,x,t,guard,deadline,out)
                    if pred['count_decision']!=old['prediction']['count_decision']:raise ValueError('Count changed under type-only intervention')
                    if arm=='B':report['base_certificates'].append(dict(task=task,index=index,count_probability_diff=count_diff,**prediction_identity(old['prediction'],pred)))
                    events=[RhythmEvent(**e) for e in pred['events']];scores,partial=score_outcome(engine,events,rr['targets'],pred['complete'])
                    null=engine.output_null(events,rr['targets'],rr['request'],parent['shifts_us'])
                    if not pred['complete']:
                        for n in null:
                            n.update(onset10=0.,full50=0.)
                            for v in n['types'].values():v.update(tp=0,f1=0.)
                    rec.update(status='completed' if pred['complete'] else 'decode_failed',prediction=pred,scores=scores,raw_partial_scores=partial,
                        output_null=null,output_null_onset10=[n['onset10'] for n in null],control=control,
                        count_identity_diff=count_diff,type_song_coverage=type_source_coverage(out,x,arm),
                        merged_visible_plus_generated=[asdict(e) for e in sorted([*rr['visible'],*events])])
                    write(ART/f'{arm}-{task}.json',records);del x,t,out,rr,events,cp;guard.clear()
                print('free',arm,task,len(records),flush=True)
        report['all_free_complete']=all(r['prediction']['complete'] for tasks in groups.values() for rows in tasks.values() for r in rows)
        report['free_summary']={arm:{task:summarize(records) for task,records in tasks.items()} for arm,tasks in groups.items()};persist()
        for arm in ARMS:
            for task in DIAGNOSTIC_TASKS:
                for rec in groups[arm][task]:
                    report['phase']=f'conditional_{arm}_{task}';guard.check(report['phase']);index=rec['index']
                    bands=tuple(rec['bands']) if rec['bands'] is not None else None
                    rr,x,t,_=case_inputs(engine,parent,panel,row,task,index,rec['view'],bands,rec['masked']);out=model(x)
                    for prefix,function in [('TT',tt_diagnostic),('SELF',self_diagnostic)]:
                        report['current_conditional']=dict(arm=arm,prefix=prefix,task=task,index=index)
                        kwargs=dict(deadline=deadline)
                        if prefix=='SELF':kwargs['checkpoint']=guard.clear
                        with type_arm(model,out,x,arm):value=function(model,out,x,rr['request'],t,prior,**kwargs)
                        if prefix=='TT' and arm=='B' and abs(value['exact_factor_nll']['kind']-raw[task][index]['teacher_scores']['exact_factor_nll']['kind'])>1e-4:raise ValueError('BASE TT exact factor identity')
                        value.update(index=index,component=rec['component'],task=task,arm=arm,prefix=prefix)
                        write(ART/f'diagnostic-{arm}-{prefix}-{task}-{index}.json',value)
                        report['condition_rows'][arm][prefix][task].append(dict(index=index,component=rec['component'],status='completed',summary=value['summary']))
                        del value;guard.clear()
                    del x,t,out,rr;guard.clear();write(ART/'report.json',report)
                print('conditionals',arm,task,flush=True)
        conditional={a:{prefix:{task:conditional_aggregate(rows) for task,rows in tasks.items()} for prefix,tasks in kinds.items()} for a,kinds in report['condition_rows'].items()}
        report.update(conditional_summary=conditional,all_conditionals_complete=True,weight_digest_after=weight_digest(model))
        if report['weight_digest_after']!=before:raise ValueError('Frozen model modified')
        report.update(status='completed_pending_mechanism_review',mechanism=mechanism_decision(report,groups,conditional),
            research_scope='Count/empty failure remains;mechanism gate is not research/product/scaling qualification')
    except BaseException as exc:
        report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
        for tasks in groups.values():
            for records in tasks.values():
                for r in records:
                    if r['status']=='attempted':r.update(status='execution_failed',failure=report['failure'])
    finally:
        completed={(arm,prefix,task,r['index']) for arm,kinds in report['condition_rows'].items() for prefix,tasks in kinds.items() for task,rows in tasks.items() for r in rows}
        report['conditional_coverage']=[dict(**e,status='completed' if (e['arm'],e['prefix'],e['task'],e['index']) in completed else 'not_completed',loss_semantics='recorded' if (e['arm'],e['prefix'],e['task'],e['index']) in completed else 'infinite_or_unmeasured') for e in expected_conditionals]
        signal.alarm(0);report['telemetry']=guard.close();report['wall_s']=time.monotonic()-guard.start;persist();write(REPORT,report)
    print(json.dumps({k:report.get(k) for k in ('status','failure','wall_s','mechanism')}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args();globals()[args.action]()

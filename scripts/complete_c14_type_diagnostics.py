"""Complete C14 diagnostics only, reusing all510 immutable free outcomes."""
import argparse
from copy import deepcopy
import json
import signal
import time

import torch

from beatfcn.train.c14_probe_support import ProbeGuard,load_frozen_model
from beatfcn.train.c14_type_probe import type_arm,tt_diagnostic,self_diagnostic
from beatfcn.train.native_trial_evaluation import case_inputs
from scripts import run_native_cell_recovery as engine
from scripts.run_c14_type_song_prefix import ARMS,DIAGNOSTIC_TASKS,conditional_aggregate,mechanism_decision
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import ROOT,write,sha,weight_digest
from scripts.run_native_cell_chain import closure
from scripts.check_experiment_storage import admission,GIB


ID='c14-type-diagnostic-completion-v1';ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
OLD=ROOT/'artifacts/c14-type-song-prefix-v1';REPORT=ROOT/f'reports/local-integration-v1/{ID}.json'


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve completion registration')
    old_plan=json.loads((ROOT/'bench/rhythm-v2/c14-type-song-prefix-v1.json').read_text());old=json.loads((OLD/'report.json').read_text())
    proof=json.loads((ROOT/'reports/council-quality-c14/type-serialization-repair.json').read_text())
    reuse=json.loads((ROOT/'reports/council-quality-c14/type-free-reuse-certificate.json').read_text())
    if proof['status']!='verified' or reuse['status']!='verified' or old['failure']['type']!='TypeError':raise ValueError('Required actual defect/reuse receipts')
    changed=[f for f,h in old_plan['source_sha256'].items() if sha(ROOT/f)!=h]
    if set(changed)!={'beatfcn/train/c14_type_probe.py','tests/test_c14_type_probe.py'}:raise ValueError('Only diagnosed representation/test repair allowed')
    before=(OLD/'source/beatfcn/train/c14_type_probe.py').read_text();after=(ROOT/'beatfcn/train/c14_type_probe.py').read_text()
    if before.replace('forced=sum(kind_allowed)==1 and supported','forced=bool(sum(kind_allowed)==1 and supported)')!=after:raise ValueError('Unexpected inference/math change')
    old_bytes=sum(f.stat().st_size for f in OLD.rglob('*') if f.is_file());disk_cap=min(256*2**20,512*2**20-old_bytes-16*2**20)
    if disk_cap<96*2**20 or admission(ROOT,disk_cap,15*GIB)['status']!='admitted':raise MemoryError('Combined original+completion disk admission')
    p=dict(id=ID,parent_plan=old_plan['parent_plan'],parent_plan_sha256=old_plan['parent_plan_sha256'],checkpoint=old_plan['checkpoint'],checkpoint_sha256=old_plan['checkpoint_sha256'],
        original_probe_plan='bench/rhythm-v2/c14-type-song-prefix-v1.json',original_probe_plan_sha256=sha(ROOT/'bench/rhythm-v2/c14-type-song-prefix-v1.json'),
        old_report_sha256=sha(OLD/'report.json'),prior_path=old_plan['prior_path'],prior_sha256=old_plan['prior_sha256'],
        inherited_free_outcomes=510,new_free_outcomes=0,inherited_conditional_requests=1,new_conditional_requests=383,
        inherited_failed_conditional_attempts=1,
        inherited_wall_s=old['wall_s'],wall_s=1200,cumulative_wall_cap_s=1800,disk_cap_bytes=disk_cap,old_artifact_bytes=old_bytes,
        optimizer_updates=0,cloud_calls=0,allowlist_repair=changed,model_math_unchanged=True,
        reused_free_files={f['path']:f['sha256'] for f in reuse['files']},
        inherited_diagnostic='artifacts/c14-type-song-prefix-v1/diagnostic-B-TT-draft-458.json',
        inherited_diagnostic_sha256=sha(OLD/'diagnostic-B-TT-draft-458.json'))
    if p['inherited_wall_s']+p['wall_s']>p['cumulative_wall_cap_s']:raise ValueError('No reset of original wall budget')
    p['source_sha256']={f:sha(ROOT/f) for f in set(closure('scripts/complete_c14_type_diagnostics.py'))|{'tests/test_c14_type_probe.py'}}
    write(PLAN,p)
    for f in p['source_sha256']:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    write(ART/'source/frozen-plan.json',p)


@torch.no_grad()
def run():
    if REPORT.exists():raise FileExistsError('No completion retry')
    p=json.loads(PLAN.read_text())
    for f,h in p['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Completion source drift')
    if sha(OLD/'report.json')!=p['old_report_sha256'] or sha(ROOT/p['prior_path'])!=p['prior_sha256']:raise ValueError('Old result/prior drift')
    if sha(ROOT/p['inherited_diagnostic'])!=p['inherited_diagnostic_sha256']:raise ValueError('InheritedTTdrift')
    groups={arm:{} for arm in ARMS}
    for name,h in p['reused_free_files'].items():
        path=ROOT/name
        if sha(path)!=h:raise ValueError('Inherited free outcome drift')
        arm,task=path.stem.split('-',1);groups[arm][task]=json.loads(path.read_text())
    old=json.loads((OLD/'report.json').read_text());expected=json.loads((OLD/'expected-conditional-requests.json').read_text())
    report=dict(id=ID,status='running',phase='initialization',optimizer_updates=0,cloud_calls=0,cloud_spend_usd=0,
        inherited_free_outcomes=510,new_free_outcomes=0,all_free_complete=True,free_summary=old['free_summary'],
        condition_rows=deepcopy(old['condition_rows']),inherited_conditional_requests=1,new_conditional_requests_completed=0,
        inherited_failed_conditional_attempts=p['inherited_failed_conditional_attempts'],
        original_failure=old['failure'],original_failed_report_sha256=p['old_report_sha256'],research_qualified=False,scaling_authorized=False,confirmation_scored=False)
    completed={(a,k,t,r['index']) for a,kinds in report['condition_rows'].items() for k,tasks in kinds.items() for t,rows in tasks.items() for r in rows}
    if len(completed)!=1:raise ValueError('Unexpected inherited diagnostics')
    guard=ProbeGuard(ROOT,ART,p['wall_s'],p['disk_cap_bytes']);deadline=guard.start+p['wall_s'];model=None
    def stop(*_):raise TimeoutError('Completion wall/interrupt')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['wall_s'])
    try:
        parent,model,before=load_frozen_model(p);report['weight_digest_before']=before
        if before!=old['weight_digest_before']:raise ValueError('Exact frozen predictor required')
        panel,row=load_data();prior=json.loads((ROOT/p['prior_path']).read_text());raw={task:{r['index']:r for r in json.loads((ROOT/f'artifacts/c13-native-quality-v1/{task}.json').read_text())} for task in DIAGNOSTIC_TASKS}
        write(ART/'report.json',report)
        for arm in ARMS:
            for task in DIAGNOSTIC_TASKS:
                for rec in groups[arm][task]:
                    report['phase']=f'{arm}_{task}_{rec["index"]}';guard.check(report['phase']);index=rec['index']
                    rr,x,t,_=case_inputs(engine,parent,panel,row,task,index,rec['view'],tuple(rec['bands']) if rec['bands'] is not None else None,rec['masked'])
                    if engine.input_digest(x)!=raw[task][index]['input_sha256']:raise ValueError('Original request/input identity')
                    out=model(x)
                    for prefix,function in [('TT',tt_diagnostic),('SELF',self_diagnostic)]:
                        key=(arm,prefix,task,index)
                        if key in completed:continue
                        report['current_conditional']=dict(arm=arm,prefix=prefix,task=task,index=index)
                        kwargs=dict(deadline=deadline)
                        if prefix=='SELF':kwargs['checkpoint']=guard.clear
                        with type_arm(model,out,x,arm):value=function(model,out,x,rr['request'],t,prior,**kwargs)
                        if prefix=='TT' and arm=='B' and abs(value['exact_factor_nll']['kind']-raw[task][index]['teacher_scores']['exact_factor_nll']['kind'])>1e-4:raise ValueError('BASE TT factor identity')
                        value.update(index=index,component=rec['component'],task=task,arm=arm,prefix=prefix)
                        write(ART/f'diagnostic-{arm}-{prefix}-{task}-{index}.json',value)
                        report['condition_rows'][arm][prefix][task].append(dict(index=index,component=rec['component'],status='completed',summary=value['summary']))
                        completed.add(key);report['new_conditional_requests_completed']+=1;del value;guard.clear()
                    del x,t,out,rr;guard.clear();write(ART/'report.json',report)
                print('completed',arm,task,report['new_conditional_requests_completed'],flush=True)
        if len(completed)!=384 or report['new_conditional_requests_completed']!=383:raise ValueError('Full original diagnostic denominator')
        conditional={a:{k:{task:conditional_aggregate(rows) for task,rows in tasks.items()} for k,tasks in kinds.items()} for a,kinds in report['condition_rows'].items()}
        report.update(conditional_summary=conditional,all_conditionals_complete=True,weight_digest_after=weight_digest(model))
        if report['weight_digest_after']!=before:raise ValueError('Frozen state changed')
        report.update(status='completed_pending_mechanism_review',mechanism=mechanism_decision(report,groups,conditional))
    except BaseException as exc:report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
    finally:
        report['conditional_coverage']=[dict(**e,status='completed' if (e['arm'],e['prefix'],e['task'],e['index']) in completed else 'not_completed',loss_semantics='recorded' if (e['arm'],e['prefix'],e['task'],e['index']) in completed else 'infinite_or_unmeasured') for e in expected]
        signal.alarm(0);report['telemetry']=guard.close();report['wall_s']=time.monotonic()-guard.start;report['cumulative_managed_wall_s']=p['inherited_wall_s']+report['wall_s'];write(ART/'report.json',report);write(REPORT,report)
    print(json.dumps({k:report.get(k) for k in ('status','failure','new_conditional_requests_completed','wall_s','cumulative_managed_wall_s','mechanism')}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args();globals()[args.action]()

"""Frozen count input ablation; no member decode, F1 or optimizer."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from beatfcn.train.c14_probe_support import ProbeGuard,load_frozen_model,probability_difference
from beatfcn.train.native_cell_chain import cell_row,cell_batch
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import ROOT,write,sha,weight_digest
from scripts.run_native_cell_chain import closure
from scripts.run_native_cell_recovery import input_digest
from scripts.check_experiment_storage import admission,GIB


ID='c14-count-inputs-v1';ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
REPORT=ROOT/f'reports/local-integration-v1/{ID}.json';ARMS=('mel_zero','song_zero','controls_unspecified')


def changed_input(inputs,arm):
    if arm not in ARMS:raise ValueError('Unknown intervention')
    changed=dict(inputs);key={'mel_zero':'mel','song_zero':'song','controls_unspecified':'controls'}[arm]
    changed[key]=torch.zeros_like(inputs[key]);return changed


def count_summary(probability,target):
    p=np.asarray(probability,float);target=np.asarray(target);mode=p.argmax(-1);expected=(p[:,:65]*np.arange(65)[None]).sum(-1)
    if p.shape!=(len(target),66) or not np.isfinite(p).all() or np.max(np.abs(p.sum(-1)-1))>1e-4:raise ValueError('Count probability support')
    return dict(mode_events=int(mode.sum()),positive_expected=float(expected[target>0].sum()),zero_expected=float(expected[target==0].sum()),
        expected_events=float(expected.sum()),maximum_presence=float((1-p[:,0]).max()),overflow_mass=float(p[:,65].sum()),overflow_mode_cells=int((mode==65).sum()))


def aggregate(rows,arm,task):
    values=[r['arms'][arm]['summary'] for r in rows if r['task']==task and arm in r['arms']]
    return dict(completed=len(values),**{k:sum(v[k] for v in values) for k in ('mode_events','positive_expected','zero_expected','expected_events','overflow_mass','overflow_mode_cells')},
        maximum_presence=max((v['maximum_presence'] for v in values),default=None))


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve probe registration')
    parent=json.loads((ROOT/'bench/rhythm-v2/c13-native-quality-v1.json').read_text())
    q=json.loads((ROOT/'artifacts/c13-native-quality-v1/quality.json').read_text());decision=ROOT/'reports/council-quality-c14/parent-final-decision.md'
    if admission(ROOT,128*2**20,15*GIB)['status']!='admitted':raise MemoryError('Probe disk admission')
    p=dict(id=ID,parent_plan='bench/rhythm-v2/c13-native-quality-v1.json',parent_plan_sha256=sha(ROOT/'bench/rhythm-v2/c13-native-quality-v1.json'),
        checkpoint=q['checkpoint'],checkpoint_sha256=q['checkpoint_sha256'],decision_sha256=sha(decision),wall_s=600,disk_cap_bytes=128*2**20,
        optimizer_updates=0,cloud_calls=0,arms=list(ARMS),base_policy='Saved BASE;single fresh count forward only certifies identity,never replaces or selects score',
        stream_contract='Only exact named raw input key;local MERT and all clocks/valid/reference/mode untouched',
        indices=parent['indices']['dev']+parent['indices']['empty_dev'],
        raw_base_sha256={name:sha(ROOT/f'artifacts/c13-native-quality-v1/{name}.json') for name in ('draft','empty')},
        trigger=dict(mode_decrease=200,zero_expected_decrease=200,other_mode_decrease_max=80,empty_expected_max=22.108,empty_mode_max=13,control_mode_change_max=30),
        predecessors=['c13-native-quality-v1'],shared_failure_modes=['count_allocation'],
        mechanism_changes=dict(count_allocation=dict(change='Separate named raw source deletions,not count output repair',prediction='Identify route dependence before proposing representation change',discriminating_control='Saved baseline count identity;separate zero and positive cells plus empty')),
        pretraining_probe_receipts=[dict(status='completed',evidence='reports/council-quality-c14/count-input-tests.json',scope='exact_intervention_and_summary_regression')],execution_ready=True)
    p['source_sha256']={f:sha(ROOT/f) for f in set(closure('scripts/run_c14_count_inputs.py'))|{'tests/test_c14_count_inputs.py'}}
    write(PLAN,p)
    for f in p['source_sha256']:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    write(ART/'source/frozen-plan.json',p)


@torch.no_grad()
def run():
    if REPORT.exists():raise FileExistsError('No automatic probe retry')
    p=json.loads(PLAN.read_text())
    for name,h in p['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Probe source drift')
    rows=[]
    for task,h in p['raw_base_sha256'].items():
        path=ROOT/f'artifacts/c13-native-quality-v1/{task}.json'
        if sha(path)!=h:raise ValueError('Saved baseline drift')
        rows.extend(dict(task=task,index=r['index'],baseline=r,arms={},status='not_attempted') for r in json.loads(path.read_text()))
    report=dict(id=ID,status='running',rows=rows,optimizer_updates=0,model_forward_attempts=0,model_forward_completed=0,cloud_calls=0,cloud_spend_usd=0)
    guard=ProbeGuard(ROOT,ART,p['wall_s'],p['disk_cap_bytes']);model=None
    def stop(*_):raise TimeoutError('Probe wall/interrupt')
    signal.signal(signal.SIGALRM,stop);signal.alarm(p['wall_s'])
    try:
        parent,model,before=load_frozen_model(p);report['weight_digest_before']=before;panel,row=load_data()
        for rec in rows:
            guard.check('count_inputs');rec['status']='attempted';old=rec['baseline'];rr=cell_row(panel,row(rec['index'],'draft'));x,t=cell_batch(rr,'mps')
            if input_digest(x)!=old['input_sha256'] or t['counts'].tolist()!=old['count_target']:raise ValueError('Exact source/request/target identity')
            report['model_forward_attempts']+=1;out=model(x);report['model_forward_completed']+=1
            base=out['count'][0].softmax(-1).cpu().numpy();rec['base_probability_max_diff']=probability_difference(base,old['count_probability'])
            rec['arms']['base']=dict(summary=count_summary(old['count_probability'],t['counts']))
            del out;guard.clear()
            for arm in ARMS:
                altered=changed_input(x,arm);report['model_forward_attempts']+=1;out=model(altered);report['model_forward_completed']+=1
                probabilities=out['count'][0].softmax(-1).cpu().numpy()
                rec['arms'][arm]=dict(probability=probabilities.tolist(),summary=count_summary(probabilities,t['counts']))
                del altered,out,probabilities;guard.clear()
            rec['status']='completed';rec.pop('baseline');del x,t,rr;base=None;guard.clear();write(ART/'report.json',report)
        report['summary']={task:{arm:aggregate(rows,arm,task) for arm in ('base',*ARMS)} for task in ('draft','empty')}
        d=report['summary']['draft'];e=report['summary']['empty'];g=p['trigger'];qualifying=[]
        for arm,other in [('mel_zero','song_zero'),('song_zero','mel_zero')]:
            if (d['base']['mode_events']-d[arm]['mode_events']>=g['mode_decrease'] and d['base']['zero_expected']-d[arm]['zero_expected']>=g['zero_expected_decrease']
                and d['base']['mode_events']-d[other]['mode_events']<=g['other_mode_decrease_max'] and e[arm]['expected_events']<=g['empty_expected_max']
                and e[arm]['mode_events']<=g['empty_mode_max'] and abs(d['base']['mode_events']-d['controls_unspecified']['mode_events'])<=g['control_mode_change_max']):qualifying.append(arm)
        report.update(status='completed_pending_mechanism_review',qualifying_routes=qualifying,grok_trigger=len(qualifying)==1,
            trigger_scope='Dependence trigger only;does not establish1ms-resolution benefit or authorize new training',weight_digest_after=weight_digest(model))
        if report['weight_digest_after']!=before:raise ValueError('Frozen model changed')
    except BaseException as exc:
        report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
    finally:
        signal.alarm(0);report['telemetry']=guard.close();report['wall_s']=time.monotonic()-guard.start;write(ART/'report.json',report);write(REPORT,report)
    print(json.dumps({k:report.get(k) for k in ('status','failure','grok_trigger','qualifying_routes','model_forward_completed','wall_s')}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args();globals()[args.action]()

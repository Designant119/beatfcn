"""Fresh quality only after a complete read-only health receipt and C13 decision.

An isolated instance of the frozen runner supplies unchanged training/eval
functions. Its output namespace and diagnostic callable are injected explicitly;
the original C12 module and artifacts are never rebound or overwritten.
"""
import argparse
import importlib.util
import inspect
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import random
import numpy as np
import torch

from beatfcn.train.field_integrity import replay_fields
from beatfcn.train.native_trial_evaluation import evaluate as evaluate_trial,metadata_outcomes,ORDER
from scripts.check_experiment_storage import admission,GIB
from scripts.run_local_portfolio_scale import write,sha
from scripts.run_native_cell_chain import closure


ROOT=Path(__file__).resolve().parents[1];ID='c13-native-quality-v1'
ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
REPORT=ROOT/'reports/local-integration-v1/native-quality-repaired.json'
HEALTH=ROOT/'artifacts/c13-frozen-health-completion-v1/report.json'
SUPPLEMENT=ROOT/'reports/local-integration-v1/c13-health-probability-supplement.json'
DECISION=ROOT/'reports/council-replay-c13/parent-final-decision.md'


def load_runner():
    spec=importlib.util.spec_from_file_location('beatfcn_c13_isolated_quality_runner',ROOT/'scripts/run_native_cell_recovery.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.ID=ID;runner.ART=ART;runner.PLAN=PLAN;runner.REPORT=REPORT
    runner.replay_fields=replay_fields
    if runner.ART.resolve()==(ROOT/'artifacts/c12-native-cell-recovery-v1').resolve():raise ValueError('Old artifact namespace forbidden')
    if runner.train.__globals__['ART']!=ART or inspect.unwrap(runner.evaluate).__globals__['ART']!=ART:raise ValueError('Output binding failed')
    runner.evaluate=functools.partial(evaluate_trial,runner)
    return runner


def validate_health(plan=None):
    r=json.loads(HEALTH.read_text())
    if r['status']!='health_passed' or not all(r['checks'].values()) or r['diagnostic_optimizer_updates']!=0:
        raise ValueError('Complete actual health receipt required')
    if r['weight_digest_before']!=r['weight_digest_after'] or not r['rng_unchanged']:raise ValueError('Frozen health mutated')
    supplement=json.loads(SUPPLEMENT.read_text())
    if supplement['status']!='passed' or supplement['health_receipt_sha256']!=sha(HEALTH):raise ValueError('Full probability/optimizer supplement required')
    if plan and sha(HEALTH)!=plan['health_receipt_sha256']:raise ValueError('Health receipt drift')
    if plan and sha(SUPPLEMENT)!=plan['health_supplement_sha256']:raise ValueError('Supplement drift')
    return r


def prepare():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve registered quality')
    if not DECISION.exists():raise ValueError('Collect full C13 and write concrete parent decision first')
    validate_health();parent=json.loads((ROOT/'bench/rhythm-v2/c12-native-cell-recovery-v1.json').read_text())
    p=dict(parent);p.update(id=ID,health_receipt=str(HEALTH.relative_to(ROOT)),health_receipt_sha256=sha(HEALTH),
        parent_decision=str(DECISION.relative_to(ROOT)),parent_decision_sha256=sha(DECISION),
        inherited_health_updates=100,new_health_updates=0,health_weights_used_for_quality=False,
        health_supplement=str(SUPPLEMENT.relative_to(ROOT)),health_supplement_sha256=sha(SUPPLEMENT),
        process_wall_s=10500,health_rerun_authorized=False,
        diagnostic_repair='Original active-rank replay plus full probability/coverage assertions;free170 persisted before teacher/P2;fixed failure denominators;training/loss/model/ordinarydecoder unchanged',
        evaluation_contract=dict(free_order=list(ORDER),expected_free_outcomes=170,teacher_outcomes=170,p2_outcomes=8,
            failure_score=0,raw_partial_events_preserved=True,all_controls_must_complete=True,
            minimum_common_core_fraction=.95,minimum_covered_components=10,minimum_positive_covered_components=10,
            confirmation_sealed=True),
        stages=['preflight','quality','evaluation'],predecessors=['c13-frozen-health-completion-v1','c12-native-cell-recovery-v1'],
        shared_failure_modes=['diagnostic_batch_shape','total_process_memory'],
        mechanism_changes=dict(
            diagnostic_batch_shape=dict(change='All field arms retain original active-rank grouping',prediction='BASE matches unmodified decode withouttolerancerelaxation',discriminating_control='Complete5window12armreceipt andoriginalcounterexample'),
            total_process_memory=dict(change='Verifiedruntime/lifetime/physicalguard;formal64prefixsameoptimizer',prediction='Sustainedformalvariedupdatesbounded',discriminating_control='Allrequesttelemetryandprefix64median/growth')),
        pretraining_probe_receipts=[dict(status='completed',evidence=str(HEALTH.relative_to(ROOT)),scope='fullactualhealthanddiagnosticcompletion'),
            dict(status='completed',evidence=str(SUPPLEMENT.relative_to(ROOT)),scope='fullsupportandprobabilityplusoptimizerhyperparameteraudit'),
            dict(status='completed',evidence='reports/local-integration-v1/c13-quality-binding-regression.json',scope='newisolatedrunnerbindingregression')],
        execution_ready=True)
    if p['initial_state_path']!='artifacts/c12-mps-runtime-probe-v1/common-initial-state.pt':raise ValueError('Fresh common initialization only')
    baseline=ROOT/'reports/local-integration-v1/strong-baseline-admission.json'
    p.update(comparison_baseline=json.loads(baseline.read_text())['metrics'],comparison_baseline_path=str(baseline.relative_to(ROOT)),comparison_baseline_sha256=sha(baseline))
    disk=admission(ROOT,GIB,15*GIB)
    if disk['status']!='admitted':raise MemoryError('Pre-registration disk admission')
    p['disk_admission']=disk
    runner=load_runner();panel,row=runner.load_data();metadata=metadata_outcomes(runner,p,panel,row)
    path=ART/'expected-free-metadata.json';write(path,metadata)
    p.update(expected_free_metadata=str(path.relative_to(ROOT)),expected_free_metadata_sha256=sha(path))
    files=set(closure('scripts/run_repaired_native_quality.py'))|set(closure('scripts/run_native_cell_recovery.py'))|{'tests/test_repaired_quality_context.py','tests/test_native_trial_evaluation.py','scripts/audit_health_completion.py'}
    p['source_sha256']={name:sha(ROOT/name) for name in sorted(files)};write(PLAN,p)
    for name in p['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def worker(stage):
    plan=json.loads(PLAN.read_text());validate_health(plan)
    if plan['health_weights_used_for_quality'] or plan['new_health_updates']!=0:raise ValueError('Quality warm start or health retraining forbidden')
    torch.manual_seed(plan['seed']);random.seed(plan['seed']);np.random.seed(plan['seed'])
    runner=load_runner();runner.worker(stage)


def run():
    if REPORT.exists():raise FileExistsError('No automatic quality rerun')
    plan=json.loads(PLAN.read_text());validate_health(plan)
    for path,h in {**plan['source_sha256'],**plan['runtime_records']}.items():
        if sha(ROOT/path)!=h:raise ValueError('Registered source/runtime drift '+path)
    runner=load_runner();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)
    started=time.monotonic();report=dict(id=ID,status='registered',stages={},cloud_compute_calls=0,cloud_spend_usd=0,
        health_receipt=plan['health_receipt'],health_receipt_sha256=plan['health_receipt_sha256'],
        inherited_health_updates=100,new_health_updates=0,health_weights_used_for_quality=False)
    try:
        for stage in plan['stages']:
            cap=plan['preflight_wall_s'] if stage=='preflight' else plan[stage+'_wall_s']
            if time.monotonic()-started+cap>plan['process_wall_s']:raise TimeoutError('Insufficient stage budget')
            if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Disk admission')
            report.update(status=stage+'_running',active_stage=stage);runner.publish(report)
            with (ART/f'{stage}.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_repaired_native_quality','worker',stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap+30,check=True)
            result=json.loads((ART/f'{stage}.json').read_text())
            report['stages'][stage]=dict(status=result['status'],updates=result['updates'],wall_s=result['wall_s'])
            expected={'preflight':'preflight_completed','quality':'quality_completed','evaluation':'terminal_pending_mechanism_review'}[stage]
            if result['status']!=expected:report.update(status=result['status'],failed_stage=stage);break
        else:report['status']='terminal_pending_mechanism_review'
    except BaseException as exc:report.update(status='supervisor_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:report['wall_s']=time.monotonic()-started;runner.publish(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);parser.add_argument('stage',nargs='?');args=parser.parse_args()
    if args.action=='worker':worker(args.stage)
    else:globals()[args.action]()

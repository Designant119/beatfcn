"""One bounded Modal invocation, three real 0.25B implementation probes."""
from pathlib import Path
import hashlib,json,time
import modal
from beatfcn.train.campaign_budget import reserve,update_stage
from beatfcn.train.paid_call import await_bounded_call

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/d64-backbone-health-v1.json'
STAGE='d64-backbone-health-v1'
app=modal.App('beatfcn-d64-backbone-health')
volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
MAMBA='https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
CONV='https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0',index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('numpy==2.2.6','einops==0.8.1','packaging==25.0','ninja==1.13.0','transformers==4.53.2')
    .pip_install(MAMBA,CONV,extra_options='--no-deps')
    .env({'PYTHONPATH':'/work','OMP_NUM_THREADS':'8'})
    .add_local_dir(ROOT/'beatfcn','/work/beatfcn')
    .add_local_file(ROOT/'scripts/d64_backbone_probe.py','/work/scripts/d64_backbone_probe.py')
    .add_local_file(ROOT/'scripts/probe_full_event_model.py','/work/scripts/probe_full_event_model.py')
    .add_local_file(PLAN,'/work/plan.json')
    .add_local_dir(ROOT/'inputs/diagnostic_panels/d64-full-event-mel','/work/fixture'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(image=image,gpu='L40S',cpu=(8,8),memory=(32768,32768),timeout=1800,
              startup_timeout=600,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':volume})
def probe(expected):
    for path,sha in expected.items():
        if digest(Path('/work')/path)!=sha:raise ValueError('Code/input identity changed: '+path)
    from scripts.d64_backbone_probe import run
    plan=json.loads(Path('/work/plan.json').read_text())
    output=Path('/data/evolution/capability-evolution-20261024')/STAGE
    if output.exists():raise FileExistsError('Inspect existing run; no overwrite: '+str(output))
    output.mkdir(parents=True)
    def publish(report):
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        return run(plan,'/work/fixture',output,publish)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    plan=json.loads(PLAN.read_text())
    expected={f'beatfcn/{p.relative_to(ROOT/"beatfcn")}':digest(p) for p in (ROOT/'beatfcn').rglob('*.py')}
    for name in ['d64_backbone_probe.py','probe_full_event_model.py']:
        expected['scripts/'+name]=digest(ROOT/'scripts'/name)
    expected['plan.json']=digest(PLAN)
    for name in ['manifest.json','inputs.npz']:
        expected['fixture/'+name]=digest(ROOT/plan['data']/name)
    identity=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    ledger=ROOT/'reports/rhythm_v2_budget.json'
    reserve(ledger,STAGE,plan['resources'],json.loads((ROOT/'reports/rhythm_v2_rates.json').read_text()),
            json.loads((ROOT/'reports/rhythm_v2_connection_status.json').read_text()),app_id=app.app_id,source_sha256=identity)
    try:
        call=probe.spawn(expected)
        update_stage(ledger,STAGE,status='running',function_call_id=call.object_id,plan_sha256=digest(PLAN))
        print(json.dumps(dict(app=app.app_id,call=call.object_id,stage=STAGE)),flush=True)
        result=await_bounded_call(call,timeout_s=plan['resources']['timeout_s']+plan['resources'].get('startup_allowance_s',120),app_id=app.app_id,modal_command=[str(ROOT/'.venv/bin/modal')])
        (ROOT/'reports/d64_backbone_health_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        update_stage(ledger,STAGE,status='complete' if result['complete'] else 'incomplete',
            probe_passed=result.get('passed',False),ended_at=time.time(),function_wall_s=result['wall_s'],
            result_path=f'/data/evolution/capability-evolution-20261024/{STAGE}/result.json')
        print(json.dumps(dict(complete=result['complete'],passed=result.get('passed'),candidates=[
            {k:c.get(k) for k in ['name','passed','parameters','train_wall_s','peak_allocated_gib']} for c in result['candidates']])),flush=True)
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}',observed_at=time.time())
        raise

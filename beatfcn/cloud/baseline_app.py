"""Explicitly budgeted corrected-baseline launch. Does nothing on import.

modal run beatfcn/cloud/baseline_app.py::baseline --budget-usd 5 --no-plan-only
Prices checked 2026-10-23 at modal.com/pricing. Uploads a separate versioned manifest.
"""
from pathlib import Path
import hashlib
import json
import time
from beatfcn.cloud.train_app import app, data, train, prep_train, prep_bench_batches, eval_whole_songs, bench_corrected

CPU_SECOND=0.0000131
GIB_SECOND=0.00000222
L40S_SECOND=0.000542
H100_SECOND=0.001097


def reservations():
    # Reserve upper resource limits, full timeout and 120 seconds startup/teardown allowance per stage.
    return [('prep',None,8,32,600),('bench_prep',None,8,32,600),
            ('bench_l40s','L40S',4,48,300),('bench_h100','H100',4,48,300),
            ('train','H100',12,48,1200),('whole_eval','L40S',4,32,1800)]


def budget_plan():
    prices={None:0,'L40S':L40S_SECOND,'H100':H100_SECOND}
    return [{'stage':name,'gpu':gpu,'cpu':cpu,'memory_gib':mem,'timeout_s':timeout,
             'reserved_usd':round((timeout+120)*(cpu*CPU_SECOND+mem*GIB_SECOND+prices[gpu]),4)}
            for name,gpu,cpu,mem,timeout in reservations()]


@app.local_entrypoint()
def baseline(budget_usd: float = 0., root: str = 'inputs/training_audit_20261023',
         config: str = 'scripts/baseline_offset_v2.json', plan_only: bool = True):
    plan=budget_plan()
    total=sum(s['reserved_usd'] for s in plan)
    print(json.dumps({'stages':plan,'reserved_usd':round(total,3),'approved_usd':budget_usd,'plan_only':plan_only},indent=2))
    if plan_only:
        return
    if budget_usd<total:
        raise ValueError(f'budget {budget_usd} is below reserved {total:.3f}; no compute started')
    from beatfcn.train.audit import fingerprint
    from beatfcn.ingest.offset import DECISION_VERSION
    root=Path(root)
    manifest=root/'manifest/charts.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    if any(r.get('offset_decision',{}).get('decision_version')!=DECISION_VERSION for r in rows if r['status']=='parsed'):
        raise ValueError('manifest has stale offset decisions')
    digest=fingerprint(rows)
    name=f'charts.offset-v2-{digest[:12]}.jsonl'
    remote=f'manifest/{name}'
    existing=[e for e in data.listdir('manifest') if e.path == remote]
    if existing:
        actual=hashlib.sha256(b''.join(data.read_file(remote))).hexdigest()
        if actual!=hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError('existing snapshot has different bytes')
    else:
        with data.batch_upload() as upload:
            upload.put_file(str(manifest),remote)
    cfg=json.loads(Path(config).read_text())
    cfg['manifest_name']=name
    # Explicit zero retries and one container: no automatic expansion of the reviewed run budget.
    def bounded(fn,stage):
        return fn.with_options(cpu=(stage['cpu'],stage['cpu']),
            memory=(stage['memory_gib']*1024,stage['memory_gib']*1024),timeout=stage['timeout_s'],
            max_containers=1,retries=0,scaledown_window=2,**({'gpu':stage['gpu']} if stage['gpu'] else {}))
    results={'manifest_sha256':digest,'manifest_name':name,'budget':plan,'started_at':time.time(),'stages':{}}
    out=Path('reports/baseline_offset_v2_execution.json')
    def save():
        out.write_text(json.dumps(results,indent=2))
    def call(fn, stage, *args, **kwargs):
        dispatched = time.perf_counter()
        try:
            return bounded(fn, stage).remote(*args, **kwargs)
        finally:
            # Includes scheduling/network wait, some of which is NOT billed. Do not label this an invoice.
            results.setdefault('dispatch_wall_s', {})[stage['stage']] = time.perf_counter() - dispatched
            save()
    try:
        stage=plan[0]
        prep=call(prep_train,stage,cfg['taps_key'],tuple(cfg['taps']),pca=cfg['pca'],
            val_cores=cfg['val_cores'],audit=False,manifest_name=name)
        results['stages']['prep']=prep
        save()
        results['stages']['bench_prep']=call(prep_bench_batches,plan[1],n_batches=4,workers=8,
            out='bench/meta_offset_v2_dynamic.pt',manifest_name=name,static=False,micro_batch=8)
        save()
        measurements, rates = {}, {}
        for stage in plan[2:4]:
            result=call(bench_corrected,stage,steps=100,tag=stage['gpu'].lower())
            results['stages'][stage['stage']]=result
            measurements[stage['gpu']] = result
            rates[stage['gpu']] = ((L40S_SECOND if stage['gpu']=='L40S' else H100_SECOND)
                                   + stage['cpu']*CPU_SECOND + stage['memory_gib']*GIB_SECOND)
            save()
        from beatfcn.train.comparison import compare_compute_benchmarks
        comparison = compare_compute_benchmarks(measurements, rates)
        selected = comparison['provisional_gpu']
        results['compute_comparison'] = comparison
        results['acceptance'] = {'status': 'unestablished', 'reason':
            'Initial baseline only. Require paired whole-song quality and total elapsed resource cost before promotion.'}
        save()
        cfg['prep']=prep['key']
        cfg['taps']=tuple(cfg['taps'])
        results['baseline_gpu'] = selected
        run=call(train,{**plan[4],'gpu':selected},**cfg)
        results['stages']['train']=run
        save()
        if run.get('aborted'):
            raise RuntimeError(f"training guard stopped: {run['aborted']}")
        evaluation=call(eval_whole_songs,plan[5],cfg['run_id'],n_songs=4)
        results['stages']['whole_eval']=evaluation
        save()
    except Exception as exc:
        results['failure']=f'{type(exc).__name__}:{exc}'
        raise
    finally:
        results['finished_at']=time.time()
        save()

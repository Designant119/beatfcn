"""One explicitly approved $2 pilot; stages are recorded before dispatch, never retried implicitly."""
from pathlib import Path
import hashlib
import json
import time

from beatfcn.cloud.train_app import app, data, prep_bench_batches
from beatfcn.cloud.baseline_app import CPU_SECOND, GIB_SECOND, L40S_SECOND, H100_SECOND
from beatfcn.cloud.pareto_pilot import run_pilot, graph_diagnostic

LEDGER = Path('reports/budget2_execution_2026-10-23.json')
META = 'bench/offset_v2_budget2_20261023.pt'
STAGES = {
    'prepare': dict(gpu=None, cpu=6, memory=24, timeout=240),
    'l40s': dict(gpu='L40S', cpu=4, memory=48, timeout=360),
    'h100': dict(gpu='H100', cpu=4, memory=48, timeout=600),
    'h100_compact': dict(gpu='H100', cpu=4, memory=48, timeout=180),
    'graph_diag': dict(gpu='H100', cpu=4, memory=48, timeout=120),
}


def stage_rate(stage):
    gpu_rate = {None: 0, 'L40S': L40S_SECOND, 'H100': H100_SECOND}[stage['gpu']]
    return gpu_rate + stage['cpu'] * CPU_SECOND + stage['memory'] * GIB_SECOND


@app.local_entrypoint()
def pilot(phase: str = 'prepare', budget_usd: float = 2.):
    if phase not in STAGES or budget_usd > 2 or budget_usd <= 0:
        raise ValueError('approved budget is at most $2; choose a registered pilot stage')
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'approved_usd': budget_usd, 'stages': {}}
    if phase in ledger['stages']:
        raise ValueError('stage already dispatched; inspect the ledger instead of rerunning it')
    stage = STAGES[phase]
    reservation = stage_rate(stage) * (stage['timeout'] + 120)
    spent = sum(s['reserved_usd'] for s in ledger['stages'].values())
    if spent + reservation > min(budget_usd, ledger['approved_usd']):
        raise ValueError('insufficient remaining approved budget; no compute started')
    if phase == 'prepare':
        from beatfcn.train.audit import fingerprint
        manifest = Path('inputs/training_audit_20261023/manifest/charts.jsonl')
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if any(r.get('offset_decision', {}).get('decision_version') != 2 for r in rows if r['status'] == 'parsed'):
            raise ValueError('stale offset decisions')
        digest = fingerprint(rows)
        name = f'charts.offset-v2-{digest[:12]}.jsonl'
        remote = f'manifest/{name}'
        if any(e.path == remote for e in data.listdir('manifest')):
            if hashlib.sha256(b''.join(data.read_file(remote))).digest() != hashlib.sha256(manifest.read_bytes()).digest():
                raise ValueError('snapshot bytes differ')
        else:
            with data.batch_upload() as upload:
                upload.put_file(str(manifest), remote)
        ledger.update(manifest_name=name, manifest_sha256=digest)
        fn = prep_bench_batches
        kwargs = dict(n_batches=4, workers=4, out=META, manifest_name=name, static=False, micro_batch=8)
    else:
        if ledger['stages'].get('prepare', {}).get('status') != 'complete':
            raise ValueError('preparation has not completed')
        fn = run_pilot
        kwargs = dict(batches=META, manifest_name=ledger['manifest_name'], graph_compare=phase == 'h100',
                      steps=100, run_tag=phase, wall_budget_s=stage['timeout']-20,
                      compact_inner=phase=='h100_compact')
        if phase == 'graph_diag':
            fn, kwargs = graph_diagnostic, dict(batches=META)
    record = {'reserved_usd': reservation, 'resources': stage, 'status': 'dispatched', 'started_at': time.time()}
    ledger['stages'][phase] = record
    def save():
        LEDGER.write_text(json.dumps(ledger, indent=2))
    save()
    call = None
    try:
        fn = fn.with_options(cpu=(stage['cpu'], stage['cpu']), memory=(stage['memory']*1024, stage['memory']*1024),
                             timeout=stage['timeout'], retries=0, max_containers=1, scaledown_window=2,
                             **({'gpu': stage['gpu']} if stage['gpu'] else {}))
        call = fn.spawn(**kwargs)
        record['call_id'] = call.object_id
        save()
        record['result'] = call.get(timeout=stage['timeout']+110)
        record['status'] = 'complete'
    except BaseException as exc:
        if call is not None:
            call.cancel(terminate_containers=True)
        record.update(status='failed', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        record['dispatch_wall_s'] = time.time() - record['started_at']
        # Scheduling/network time can be unbilled; this conservative estimate is NOT an invoice.
        record['dispatch_time_cost_estimate_usd'] = stage_rate(stage) * (record['dispatch_wall_s'] + 2)
        save()
    print(json.dumps(record, indent=2))

"""Reserve and run the formal full-database multi-stream eager MERT encode."""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from beatfcn.train.data_preparation_budget import (
    CAP, FULL_CORPUS_EVIDENCE, GRANT, gate, record_usage, reserve)
from scripts.run_full_database_mert_batch_v3 import (
    PREREQUISITES, SOURCES as BATCH_SOURCES, normalized_rates, poll_call,
    reserve_upper, sha, stopped_evidence, write)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
JOBS = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
PLAN = ROOT / 'bench/rhythm-v2/mert-eager-formal-v1.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-eager-formal-run-v1.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/mert-eager-formal-connection-v1.json'
RUN_ID = 'full-db-mert-eager-formal-v1'
RESOURCES = dict(timeout_s=2050, startup_allowance_s=80, stop_allowance_s=80,
    cpu=12, memory_gib=16, gpu='A10G', retries=0, max_containers=1, other_upper_usd='0.02')
SOURCES = tuple(dict.fromkeys((*BATCH_SOURCES,
    'beatfcn/cloud/full_database_mert_eager_app.py',
    'beatfcn/train/data_preparation_budget.py',
    'beatfcn/train/cuda_pipeline.py',
    'beatfcn/ingest/pipeline.py',
    'scripts/run_full_database_mert_eager.py')))


def contract_digest(contract):
    payload = {key: value for key, value in contract.items() if key != 'contract_sha256'}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def build_contract():
    from beatfcn.cloud.full_database_mert_eager_app import EXPECTED_PIPELINE
    jobs = [json.loads(line) for line in JOBS.read_text().splitlines() if line.strip()]
    if len({job['audio_sha256'] for job in jobs}) != len(jobs):
        raise ValueError('Full-corpus jobs must be unique')
    prerequisite_sha = {name: sha(ROOT / path) for name, path in PREREQUISITES.items()}
    contract = dict(schema='beatfcn.full-database-mert-pilot.v1',
        id='full-database-mert-eager-formal-v1',
        grant_id=GRANT, run_id=RUN_ID, optimizer_updates=0, jobs=jobs,
        full_corpus_encode=True, diagnostic_only=False,
        full_corpus_evidence=dict(path=FULL_CORPUS_EVIDENCE,
            sha256=sha(ROOT / FULL_CORPUS_EVIDENCE)),
        output_root='full-database-v1/mert-graph-diagnostic-v2',
        checkpoint='full-database-v1/mert2-checkpoint',
        source_sha256={name: sha(ROOT / name) for name in SOURCES},
        jobs_sha256=sha(JOBS),
        jobs_receipt_sha256=sha(ROOT / 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json'),
        prerequisites=PREREQUISITES, prerequisite_sha256=prerequisite_sha,
        pipeline=EXPECTED_PIPELINE,
        owner_gpu_evidence=dict(path='reports/full-database-cloud-v1/owner-gpu-utilization-85.md',
            sha256=sha(ROOT / 'reports/full-database-cloud-v1/owner-gpu-utilization-85.md')),
        gpu_utilization=dict(method='legacy_GpuUtil_take_interval_means_with_last8_resource_guard',
            required_stable_mean_percent=85, sampler_period_ms=500, take_interval_s=30,
            rolling_intervals=8, check_after_s=480, setup_excluded=True,
            final_partial_interval_included=True, missing_samples_fail_closed=True,
            pipeline_decode_write_and_gaps_included=True, failure_blocks_full_corpus=True),
        resources=RESOURCES, automatic_retry=False,
        raw_manifest_sha256='676032329c43f829e0aa4439cdab6ac6fddd9c8361a5014e350c3e2e382e4652',
        raw_content_sha256='f8cad7e07c7815e0486d4e6be65dc2bfe9433eed30ba65e0d0014faba34ebfbf',
        acceptance=dict(stable_mean_percent=85, check_after_s=480,
            both_total_and_last_eight_means_required=True,
            full_corpus_completion_required=True, recovery_smoke_required=True,
            formal_85_percent_acceptance_claim=True),
        note='Authorized full-corpus MERT encode on A10G x2 eager streams; per-song '
             'reference forward proven bitwise identical under concurrency '
             '(mert-eager-streams-probe-v1 and -a10g-v1); batched MERT permanently '
             'closed; resume identity-checks and quarantines prior receipts.')
    contract['contract_sha256'] = contract_digest(contract)
    errors = gate(contract, ROOT)
    if errors:
        raise ValueError('; '.join(errors))
    write(PLAN, contract)
    return contract


def fresh_connection(contract):
    from scripts.refresh_rhythm_state import refresh
    value = refresh()
    if not value.get('fresh_check_succeeded'):
        raise ValueError('Fresh Modal evidence failed')
    evidence = {name: sha(ROOT / result['path']) for name, result in value['results'].items()
                if result.get('ok')}
    if set(evidence) != {'apps', 'rates', 'billing'}:
        raise ValueError('Incomplete Modal evidence')
    rates = normalized_rates()
    upper = reserve_upper(contract, rates)
    result = dict(value, evidence_sha256=evidence, full_lifecycle_bound_verified=upper <= CAP,
        formal_upper_usd=str(upper), normalized_rates=rates)
    write(CONNECTION, result)
    return result


def launch(contract):
    connection = fresh_connection(contract)
    started = time.time()
    reservation = reserve(LEDGER, RUN_ID, contract, ROOT, RESOURCES,
        connection['normalized_rates'], connection)
    soft_deadline = started + RESOURCES['timeout_s'] + RESOURCES['startup_allowance_s']
    receipt = dict(run_id=RUN_ID, status='reserved', started_at_unix=started,
        deadline_unix=soft_deadline, reservation=reservation, cloud_calls=[],
        stop_verified=False)
    write(RECEIPT, receipt)
    import modal
    from beatfcn.cloud import full_database_mert_eager_app
    state = dict(app_id=None, call=None)
    lock = threading.RLock()
    stop = threading.Event()
    closed = threading.Event()

    def cancel_owned():
        with lock:
            call = state['call']
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as exc:
                with lock:
                    receipt['cancel_error'] = type(exc).__name__
                    write(RECEIPT, receipt)

    def meter():
        while not closed.is_set():
            try:
                with lock:
                    app_id = state['app_id']
                    should_stop = False
                    if app_id:
                        usage = record_usage(LEDGER, RUN_ID,
                            elapsed_s_since_dispatch=time.time() - started, app_id=app_id)
                        receipt['usage'] = usage['job']
                        should_stop = usage['grant']['stop_required']
                    if time.time() >= soft_deadline:
                        should_stop = True
                    if should_stop:
                        stop.set()
                    write(RECEIPT, receipt)
                if should_stop:
                    cancel_owned()
                    if app_id:
                        subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id, '--yes'],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
            except BaseException as exc:
                with lock:
                    receipt['meter_error'] = type(exc).__name__
                    write(RECEIPT, receipt)
                stop.set();cancel_owned()
            closed.wait(2)
    thread = threading.Thread(target=meter, daemon=True);thread.start()
    try:
        with modal.enable_output():
            with full_database_mert_eager_app.app.run(detach=True, name=RUN_ID):
                with lock:
                    state['app_id'] = full_database_mert_eager_app.app.app_id
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                function = full_database_mert_eager_app.encode_mert_eager.with_options(
                    gpu='A10G', cpu=(RESOURCES['cpu'], RESOURCES['cpu']),
                    memory=(RESOURCES['memory_gib'] * 1024, RESOURCES['memory_gib'] * 1024),
                    timeout=RESOURCES['timeout_s'], retries=0, max_containers=1,
                    scaledown_window=120)
                call = function.spawn(contract['run_id'],
                    json.dumps(contract, ensure_ascii=False, separators=(',', ':')),
                    soft_deadline)
                with lock:
                    state['call'] = call
                    receipt['cloud_calls'].append(dict(call_id=call.object_id))
                    write(RECEIPT, receipt)
                result = poll_call(call, modal, stop, soft_deadline)
                with lock:
                    state['call'] = None
                    receipt['result_summary'] = dict(
                        planned=result.get('planned'),
                        attempted=result.get('attempted'),
                        existing_reused=result.get('existing_reused'),
                        quarantined=len(result.get('quarantined', [])),
                        ineligible=len(result.get('ineligible', [])),
                        failures=len(result.get('failures', [])),
                        stopped=result.get('stopped'),
                        throughput=result.get('throughput'),
                        gpu_utilization=result.get('gpu_utilization'),
                        recovery_smoke=result.get('recovery_smoke'))
                    write(RECEIPT, receipt)
                if (result.get('stopped') or result.get('failures')
                        or len(result.get('success_ids', []))
                            + len(result.get('ineligible', [])) != len(contract['jobs'])):
                    raise ValueError('Formal eager run did not complete the full corpus')
                utilization = result['gpu_utilization']
                if (not utilization.get('applicable') or not utilization.get('passed')
                        or utilization.get('stable_observed_s', 0) < 480
                        or utilization.get('stable_period_mean_utilization_percent', 0) < 85
                        or utilization.get('last_eight_interval_mean_utilization_percent', 0) < 85):
                    raise ValueError('Formal 480s/85% stable-period gate was not passed: '
                        + json.dumps(utilization))
                smoke = result.get('recovery_smoke', {})
                if smoke.get('verified') != smoke.get('sampled'):
                    raise ValueError('Recovery smoke verification failed')
                with lock:
                    receipt['status'] = 'formal_acceptance_passed'
                    write(RECEIPT, receipt)
    except BaseException as exc:
        with lock:
            receipt['status'] = 'failed'
            receipt['failure'] = dict(type=type(exc).__name__, reason=str(exc))
            write(RECEIPT, receipt)
        cancel_owned();raise
    finally:
        closed.set();thread.join(timeout=5);cancel_owned()
        with lock:
            app_id = state['app_id']
        evidence = None
        if app_id:
            subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id, '--yes'],
                cwd=ROOT, capture_output=True, text=True, timeout=20)
            evidence = stopped_evidence(app_id, time.time() + 30)
        with lock:
            receipt['stop_verified'] = evidence is not None
            receipt['stop_evidence'] = evidence
            if app_id and evidence:
                receipt['usage'] = record_usage(LEDGER, RUN_ID,
                    elapsed_s_since_dispatch=time.time() - started, app_id=app_id,
                    closed=True, stop_evidence=evidence)['job']
            else:
                receipt['ledger_close_pending'] = True
            if receipt['status'] == 'formal_acceptance_passed' and not evidence:
                receipt['status'] = 'cleanup_failed'
            write(RECEIPT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--launch', action='store_true')
    arguments = parser.parse_args()
    contract = build_contract()
    if not arguments.launch:
        print(json.dumps(dict(status='local_checks_passed_no_cloud_calls', run_id=RUN_ID)))
        return
    result = launch(contract)
    print(json.dumps(dict(status=result['status'], stop_verified=result['stop_verified'])))
    if result['status'] != 'formal_acceptance_passed' or not result['stop_verified']:
        raise RuntimeError('Formal eager run did not reach an accepted terminal state')


if __name__ == '__main__':
    main()

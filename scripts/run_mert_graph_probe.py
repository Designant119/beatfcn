"""Reserve and run the bounded MERT CUDA-graph diagnostic probe."""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from beatfcn.train.data_preparation_budget import CAP, GRANT, gate, record_usage, reserve
from scripts.run_full_database_mert_batch_v3 import (
    PREREQUISITES, SOURCES as BATCH_SOURCES, normalized_rates, poll_call,
    reserve_upper, sha, stopped_evidence, write)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
SELECTION = ROOT / 'reports/full-database-cloud-v1/mert-pilot-selection-v1.json'
PLAN = ROOT / 'bench/rhythm-v2/mert-cuda-graph-probe-v2.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-cuda-graph-probe-run-v1.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/mert-cuda-graph-probe-connection-v1.json'
RUN_ID = 'full-db-mert-graph-probe-v2'
RESOURCES = dict(timeout_s=150, startup_allowance_s=40, stop_allowance_s=40,
    cpu=4, memory_gib=4, gpu='L40S', retries=0, max_containers=1, other_upper_usd='0.01')
SOURCES = tuple(dict.fromkeys((*BATCH_SOURCES,
    'beatfcn/cloud/mert_graph_probe_app.py', 'scripts/run_mert_graph_probe.py')))


def selected_job():
    selection = json.loads(SELECTION.read_text())
    return min(selection['jobs'], key=lambda job: abs(job['audio_sec'] - 30))


def contract_digest(contract):
    payload = {key: value for key, value in contract.items() if key != 'contract_sha256'}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def build_contract():
    job = selected_job()
    prerequisite_sha = {name: sha(ROOT / path) for name, path in PREREQUISITES.items()}
    contract = dict(schema='beatfcn.full-database-mert-pilot.v1', id='mert-cuda-graph-probe-v2',
        grant_id=GRANT, run_id=RUN_ID, optimizer_updates=0, jobs=[job],
        output_root='full-database-v1/mert-cuda-graph-probe-v2',
        checkpoint='full-database-v1/mert2-checkpoint',
        source_sha256={name: sha(ROOT / name) for name in SOURCES},
        selection_sha256=sha(SELECTION),
        jobs_receipt_sha256=sha(ROOT / 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json'),
        prerequisites=PREREQUISITES, prerequisite_sha256=prerequisite_sha,
        owner_gpu_evidence=dict(path='reports/full-database-cloud-v1/owner-gpu-utilization-85.md',
            sha256=sha(ROOT / 'reports/full-database-cloud-v1/owner-gpu-utilization-85.md')),
        gpu_utilization=dict(method='legacy_GpuUtil_take_interval_means_with_last8_resource_guard',
            required_stable_mean_percent=85, sampler_period_ms=500, take_interval_s=30,
            rolling_intervals=8, check_after_s=480, setup_excluded=True,
            final_partial_interval_included=True, missing_samples_fail_closed=True,
            pipeline_decode_write_and_gaps_included=True, failure_blocks_full_corpus=True),
        resources=RESOURCES, full_corpus_encode=False, automatic_retry=False,
        diagnostic_only=True, probe='mert_cuda_graph_v1', phase_seconds=30,
        raw_manifest_sha256='676032329c43f829e0aa4439cdab6ac6fddd9c8361a5014e350c3e2e382e4652',
        raw_content_sha256='f8cad7e07c7815e0486d4e6be65dc2bfe9433eed30ba65e0d0014faba34ebfbf',
        acceptance=dict(graph_single_max_abs=0, graph_single_float16_exact_fraction=1,
            exact_clock=True, eager_and_graph_phase_metrics=True,
            formal_85_percent_acceptance_claim=False),
        note='One-song CUDA-graph diagnostic only; no formal efficiency pass, training or full-corpus authorization.')
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
        probe_upper_usd=str(upper), normalized_rates=rates)
    write(CONNECTION, result)
    return result


def launch(contract):
    connection = fresh_connection(contract)
    started = time.time()
    reservation = reserve(LEDGER, RUN_ID, contract, ROOT, RESOURCES,
        connection['normalized_rates'], connection)
    deadline = started + RESOURCES['timeout_s'] + RESOURCES['startup_allowance_s']
    receipt = dict(run_id=RUN_ID, status='reserved', started_at_unix=started,
        reservation=reservation, cloud_calls=[], stop_verified=False)
    write(RECEIPT, receipt)
    import modal
    from beatfcn.cloud import mert_graph_probe_app
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
                    if time.time() >= deadline:
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
            with mert_graph_probe_app.app.run(detach=True, name=RUN_ID):
                with lock:
                    state['app_id'] = mert_graph_probe_app.app.app_id
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                function = mert_graph_probe_app.probe_graph.with_options(
                    gpu='L40S', cpu=(4, 4), memory=(4096, 4096), timeout=150,
                    retries=0, max_containers=1, scaledown_window=2)
                call = function.spawn(RUN_ID, json.dumps(contract,
                    ensure_ascii=False, separators=(',', ':')))
                with lock:
                    state['call'] = call
                    receipt['cloud_calls'].append(dict(call_id=call.object_id))
                    write(RECEIPT, receipt)
                result = poll_call(call, modal, stop, deadline)
                with lock:
                    state['call'] = None
                    receipt['result'] = result
                    receipt['parity'] = result['parity']
                    receipt['eager_phase'] = result['eager']
                    receipt['graph_phase'] = result['graph']
                    write(RECEIPT, receipt)
                if not result.get('diagnostic_only') or result.get('stable_period_started'):
                    raise ValueError('Graph probe entered an unauthorized stable period')
                if not result['parity']['passed']:
                    raise ValueError('Graph/single parity was not accepted')
                for phase in ('eager', 'graph'):
                    value = result[phase]
                    if value.get('replays_per_second', 0) <= 0 or value.get('gpu_samples', 0) <= 0:
                        raise ValueError('Incomplete graph-probe phase metrics: ' + phase)
                with lock:
                    receipt['status'] = 'compute_returned'
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
            evidence = stopped_evidence(app_id, time.time() + 20)
        with lock:
            receipt['stop_verified'] = evidence is not None
            receipt['stop_evidence'] = evidence
            if app_id and evidence:
                receipt['usage'] = record_usage(LEDGER, RUN_ID,
                    elapsed_s_since_dispatch=time.time() - started, app_id=app_id,
                    closed=True, stop_evidence=evidence)['job']
            else:
                receipt['ledger_close_pending'] = True
            if receipt['status'] == 'compute_returned':
                receipt['status'] = 'completed' if evidence else 'cleanup_failed'
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
    if result['status'] != 'completed' or not result['stop_verified']:
        raise RuntimeError('Graph probe did not reach a verified stopped terminal state')


if __name__ == '__main__':
    main()

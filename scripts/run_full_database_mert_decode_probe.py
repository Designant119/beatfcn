"""Reserve and run the balanced CPU decode concurrency probe."""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from decimal import Decimal
from pathlib import Path

from beatfcn.train.data_preparation_budget import CAP, GRANT, gate, record_usage, reserve
from scripts.run_full_database_mert_batch_v3 import (
    PREREQUISITES, SOURCES as BATCH_SOURCES, normalized_rates, poll_call, sha,
    stopped_evidence, write)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
SELECTION = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-selection-v1.json'
PLAN = ROOT / 'bench/rhythm-v2/mert-decode-probe-v1.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-run-v1.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/mert-decode-probe-connection-v1.json'
RUN_ID = 'full-db-mert-decode-probe-v1'
RESOURCES = dict(timeout_s=600, startup_allowance_s=120, stop_allowance_s=120,
    cpu=8, memory_gib=8, retries=0, max_containers=1, other_upper_usd='0.02')
SOURCES = tuple(dict.fromkeys((*BATCH_SOURCES,
    'beatfcn/cloud/mert_decode_probe_app.py',
    'scripts/prepare_full_database_mert_decode_probe.py',
    'scripts/run_full_database_mert_decode_probe.py')))


def contract_digest(contract):
    payload = {key: value for key, value in contract.items() if key != 'contract_sha256'}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def upper_bound(resources, rates):
    duration = Decimal(str(resources['timeout_s'])) + Decimal(str(resources['startup_allowance_s'])) \
        + Decimal(str(resources['stop_allowance_s']))
    rate = Decimal(str(resources['cpu'])) * Decimal(rates['cpu_usd_s']) \
        + Decimal(str(resources['memory_gib'])) * Decimal(rates['memory_gib_usd_s'])
    return rate * duration + Decimal(str(resources['other_upper_usd']))


def build_contract():
    selection = json.loads(SELECTION.read_text())
    prerequisite_sha = {name: sha(ROOT / path) for name, path in PREREQUISITES.items()}
    contract = dict(schema='beatfcn.full-database-mert-pilot.v1', id='mert-decode-probe-v1',
        grant_id=GRANT, run_id=RUN_ID, optimizer_updates=0, jobs=selection['jobs'],
        output_root='full-database-v1/mert-decode-probe-v1',
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
        diagnostic_only=True, probe='mert_decode_concurrency_v1',
        worker_counts=[16, 32, 48, 48, 32, 16],
        raw_manifest_sha256='676032329c43f829e0aa4439cdab6ac6fddd9c8361a5014e350c3e2e382e4652',
        raw_content_sha256='f8cad7e07c7815e0486d4e6be65dc2bfe9433eed30ba65e0d0014faba34ebfbf',
        acceptance=dict(decoded_clock_equals_authoritative_mel_receipt=True,
            all_worker_phases_reported=True, formal_85_percent_acceptance_claim=False),
        note='CPU-only decode throughput/concurrency diagnostic; no GPU calls, feature writes or training.')
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
    upper = upper_bound(RESOURCES, rates)
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
    from beatfcn.cloud import mert_decode_probe_app
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
            with mert_decode_probe_app.app.run(detach=True, name=RUN_ID):
                with lock:
                    state['app_id'] = mert_decode_probe_app.app.app_id
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                function = mert_decode_probe_app.probe_decode_concurrency.with_options(
                    cpu=(8, 8), memory=(8192, 8192), timeout=600, retries=0,
                    max_containers=1, scaledown_window=2)
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
                    receipt['phases'] = result['phases']
                    write(RECEIPT, receipt)
                if result.get('probe') != 'mert_decode_concurrency_v1' \
                        or result.get('gpu_calls') != 0:
                    raise ValueError('Decode probe identity changed')
                if len(result.get('phases', [])) != 6 \
                        or any(phase['jobs'] != len(contract['jobs']) for phase in result['phases']):
                    raise ValueError('Decode probe phases are incomplete')
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
        raise RuntimeError('Decode probe did not reach a verified stopped terminal state')


if __name__ == '__main__':
    main()

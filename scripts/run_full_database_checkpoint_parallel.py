"""Download the pinned checkpoint in parallel with the independent mel verifier."""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from beatfcn.train.cloud_parallel_checkpoint_budget import reserve_parallel_checkpoint
from beatfcn.train.cloud_preparation_budget import gate, record_usage
from scripts.run_full_database_cloud_preparation import (
    LEDGER, SOURCES as PREPARATION_SOURCES, normalized_rates, poll_call,
    reservation_upper, sha, stopped_evidence, write)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/full-database-checkpoint-parallel-v1.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/checkpoint-parallel-connection-v1.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/checkpoint-parallel-run-v1.json'
RUN_ID = 'full-database-checkpoint-parallel-v1'
RESOURCES = dict(timeout_s=1800, startup_allowance_s=120, stop_allowance_s=120,
    cpu=8, memory_gib=4, retries=0, max_containers=1, other_upper_usd='0.02')
SOURCES = tuple(dict.fromkeys((*PREPARATION_SOURCES,
    'beatfcn/train/cloud_parallel_checkpoint_budget.py',
    'scripts/run_full_database_checkpoint_parallel.py')))


def build_contract():
    if PLAN.exists():
        contract = json.loads(PLAN.read_text())
    else:
        contract = dict(schema='beatfcn.full-database-cloud-preparation.v1',
            grant_id='owner-full-database-cloud-preparation-20261031', run_id=RUN_ID,
            optimizer_updates=0, manifest='manifest/charts.offset-v2-3d99ff6bd5e7.jsonl',
            output_root='full-database-v1', phases=['targets', 'raw', 'mel', 'checkpoint'],
            execution_phases=['checkpoint'], detach=True,
            resources={'checkpoint': RESOURCES}, automatic_retry=False, local_corpus_traffic=False,
            source_sha256={name: sha(ROOT / name) for name in SOURCES},
            owner_evidence='reports/full-database-cloud-v1/owner-cloud-preparation-cap-increase-3-20261031.md',
            cpu_workers=8, execution='modal_container_only',
            parallel_with='mel', reason='checkpoint download is independent of mel array verification')
        contract['contract_sha256'] = hashlib.sha256(json.dumps(contract,
            ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write(PLAN, contract)
    if contract['source_sha256'] != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError('Parallel checkpoint source changed')
    errors = gate(contract, ROOT)
    if errors:
        raise ValueError('; '.join(errors))
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
    upper = reservation_upper(RESOURCES, rates)
    result = dict(value, evidence_sha256=evidence, full_lifecycle_bound_verified=upper <= 8,
        lifecycle_upper_usd=str(upper), normalized_rates=rates)
    write(CONNECTION, result)
    return result


def launch(contract):
    connection = fresh_connection(contract)
    started = time.time()
    reservation = reserve_parallel_checkpoint(LEDGER, contract, ROOT, RESOURCES,
        connection['normalized_rates'], connection)
    receipt = dict(run_id=RUN_ID, status='reserved', started_at_unix=started,
        reservation=reservation, cloud_calls=[], stop_verified=False)
    write(RECEIPT, receipt)
    import modal
    from beatfcn.cloud import full_database_preparation_app
    state = dict(app_id=None, call=None)
    state_lock = threading.RLock()
    stop = threading.Event()
    closed = threading.Event()
    deadline = started + RESOURCES['timeout_s'] + RESOURCES['startup_allowance_s']

    def cancel_owned():
        with state_lock:
            call = state['call']
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as exc:
                with state_lock:
                    receipt['cancel_error'] = type(exc).__name__
                    write(RECEIPT, receipt)

    def meter():
        while not closed.is_set():
            try:
                with state_lock:
                    app_id = state['app_id']
                    should_stop = False
                    if app_id:
                        usage = record_usage(LEDGER, 'checkpoint',
                            elapsed_s_since_dispatch=time.time() - started, app_id=app_id)
                        receipt['current_usage'] = usage['job']
                        should_stop = usage['grant']['stop_required']
                    if time.time() >= deadline:
                        should_stop = True
                    if should_stop:
                        stop.set()
                    write(RECEIPT, receipt)
                if should_stop:
                    cancel_owned()
                    if app_id:
                        subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
            except BaseException as exc:
                with state_lock:
                    receipt['meter_error'] = type(exc).__name__
                    write(RECEIPT, receipt)
                stop.set()
                cancel_owned()
            closed.wait(2)
    thread = threading.Thread(target=meter, daemon=True)
    thread.start()
    try:
        with modal.enable_output():
            with full_database_preparation_app.app.run(detach=True, name=RUN_ID):
                with state_lock:
                    state['app_id'] = full_database_preparation_app.app.app_id
                    if not state['app_id']:
                        raise RuntimeError('Actual app identity missing')
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                function = full_database_preparation_app.download_checkpoint.with_options(
                    timeout=RESOURCES['timeout_s'], cpu=(8, 8), memory=(4096, 4096),
                    retries=0, max_containers=1, scaledown_window=2)
                call = function.spawn(RUN_ID, json.dumps(contract,
                    ensure_ascii=False, separators=(',', ':')))
                with state_lock:
                    state['call'] = call
                    receipt['cloud_calls'].append(dict(call_id=call.object_id))
                    write(RECEIPT, receipt)
                result = poll_call(call, modal, stop, deadline)
                with state_lock:
                    state['call'] = None
                if not result.get('complete'):
                    raise ValueError('Checkpoint phase incomplete: ' + str(result))
                with state_lock:
                    receipt['result'] = result
                    receipt['status'] = 'compute_returned'
                    write(RECEIPT, receipt)
    except BaseException as exc:
        with state_lock:
            receipt['status'] = 'failed'
            receipt['failure'] = dict(type=type(exc).__name__, reason=str(exc))
            write(RECEIPT, receipt)
        cancel_owned()
        raise
    finally:
        closed.set()
        thread.join(timeout=21)
        cancel_owned()
        with state_lock:
            app_id = state['app_id']
        evidence = None
        if app_id:
            try:
                subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                    cwd=ROOT, capture_output=True, text=True, timeout=20)
                evidence = stopped_evidence(app_id)
            except BaseException as exc:
                with state_lock:
                    receipt['stop_verification_error'] = type(exc).__name__
        with state_lock:
            receipt['stop_verified'] = evidence is not None
            receipt['stop_evidence'] = evidence
            if app_id and evidence:
                receipt['current_usage'] = record_usage(LEDGER, 'checkpoint',
                    elapsed_s_since_dispatch=time.time() - started, app_id=app_id,
                    closed=True, stop_evidence=evidence)['job']
            else:
                receipt['ledger_close_pending'] = True
            if receipt['status'] == 'compute_returned':
                receipt['status'] = 'completed' if evidence else 'cleanup_failed'
            write(RECEIPT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    arguments = parser.parse_args()
    contract = build_contract()
    if not arguments.launch:
        print(json.dumps(dict(status='local_checks_passed_no_cloud_calls', run_id=RUN_ID)))
        return
    result = launch(contract)
    print(json.dumps(dict(status=result['status'], stop_verified=result['stop_verified'])))
    if result['status'] != 'completed' or not result['stop_verified']:
        raise RuntimeError('Checkpoint did not reach a verified stopped terminal state')


if __name__ == '__main__':
    main()

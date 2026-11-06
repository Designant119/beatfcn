"""Reserve and launch in-container CPU full-database preparation phases."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from beatfcn.train.cloud_preparation_budget import CAP, gate, record_usage, reserve


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
PLAN = ROOT / 'bench/rhythm-v2/full-database-cloud-preparation-v9.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/cloud-preparation-connection-v9.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/cloud-preparation-run-v9.json'
SOURCES = (
    'beatfcn/train/full_database.py',
    'beatfcn/events/canonical.py',
    'beatfcn/model/event_canvas.py',
    'beatfcn/model/batch.py',
    'beatfcn/model/features.py',
    'beatfcn/ingest/phira_policy.py',
    'beatfcn/ingest/coverage.py',
    'beatfcn/audio/local_mel_cache.py',
    'beatfcn/audio/preprocess.py',
    'beatfcn/train/data.py',
    'beatfcn/train/event_density_drag_filter.py',
    'beatfcn/train/campaign_budget.py',
    'beatfcn/audio/mert2.py',
    'beatfcn/cloud/full_database_preparation_app.py',
    'beatfcn/train/cloud_preparation_budget.py',
    'scripts/run_full_database_cloud_preparation.py',
)
RESOURCES = dict(startup_allowance_s=120, stop_allowance_s=120,
    cpu=8, memory_gib=4, retries=0, max_containers=1, other_upper_usd='0.02')
PHASE_RESOURCES = {
    'targets': dict(RESOURCES, timeout_s=1800),
    'raw': dict(RESOURCES, timeout_s=5400),
    'mel': dict(RESOURCES, timeout_s=5400),
    'checkpoint': dict(RESOURCES, timeout_s=1800),
}
PHASE_ORDER = ('targets', 'raw', 'mel', 'checkpoint')
EXECUTION_PHASES = ('raw', 'mel', 'checkpoint')
WRITE_LOCK = threading.RLock()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)


def build_contract():
    if PLAN.exists():
        contract = json.loads(PLAN.read_text())
    else:
        contract = dict(schema='beatfcn.full-database-cloud-preparation.v1',
            grant_id='owner-full-database-cloud-preparation-20261031',
            run_id='full-database-cloud-preparation-v9', optimizer_updates=0,
            manifest='manifest/charts.offset-v2-3d99ff6bd5e7.jsonl',
            output_root='full-database-v1', phases=list(PHASE_ORDER),
            resources=PHASE_RESOURCES, automatic_retry=False, local_corpus_traffic=False,
            execution_phases=list(EXECUTION_PHASES), detach=True,
            completed_phases=dict(targets=dict(run_id='full-database-cloud-preparation-v7',
                complete=True, success_count=23211, permanent_failure_count=0)),
            recovery=dict(previous_run='full-database-cloud-preparation-v8',
                previous_raw_job='raw-v8', new_raw_job='raw-v9',
                previous_app_id='ap-SV22kXF45sydcbIon0NYA1',
                previous_call_id='fc-01M67KK59C0Q223W98CFZEY5CN',
                evidence='10897 verified partial records; phira:10256 missing expected audio member and already kept=false/quarantined',
                reason='resume only the unverified pre-quarantined audio; do not rehash 10897 verified records'),
            source_sha256={name: sha(ROOT / name) for name in SOURCES},
            owner_evidence='reports/full-database-cloud-v1/owner-cloud-preparation-cap-increase-3-20261031.md',
            cpu_workers=8, execution='modal_container_only')
        contract['contract_sha256'] = hashlib.sha256(json.dumps(contract,
            ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write(PLAN, contract)
    if contract['source_sha256'] != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError('Preparation source changed')
    errors = gate(contract, ROOT)
    if errors:
        raise ValueError('; '.join(errors))
    return contract


def normalized_rates():
    value = json.loads((ROOT / 'reports/rhythm_v2_rates.json').read_text())
    return dict(gpu_usd_s='0',
        cpu_usd_s=str(Decimal(value['cpu_hour_cost']) / Decimal(3600)),
        memory_gib_usd_s=str(Decimal(value['mem_gib_hour_cost']) / Decimal(3600)))


def reservation_upper(resources, rates):
    duration = (Decimal(str(resources['timeout_s']))
        + Decimal(str(resources.get('startup_allowance_s', 120)))
        + Decimal(str(resources['stop_allowance_s'])))
    rate = (Decimal(str(resources['cpu'])) * Decimal(rates['cpu_usd_s'])
        + Decimal(str(resources['memory_gib'])) * Decimal(rates['memory_gib_usd_s']))
    return rate * duration + Decimal(str(resources.get('other_upper_usd', 0)))


def fresh_connection(resources):
    from scripts.refresh_rhythm_state import refresh
    value = refresh()
    if not value.get('fresh_check_succeeded'):
        raise ValueError('Fresh Modal evidence failed')
    evidence = {name: sha(ROOT / result['path']) for name, result in value['results'].items()
                if result.get('ok')}
    if set(evidence) != {'apps', 'rates', 'billing'}:
        raise ValueError('Incomplete Modal evidence')
    rates = normalized_rates()
    upper = reservation_upper(resources, rates)
    result = dict(value, evidence_sha256=evidence,
        full_lifecycle_bound_verified=upper <= CAP,
        lifecycle_upper_usd=str(upper), normalized_rates=rates)
    write(CONNECTION, result)
    return result


def stopped_evidence(app_id, attempts=5):
    result = None
    for attempt in range(attempts):
        result = subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'list', '--json'],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            row = next((value for value in json.loads(result.stdout)
                        if value.get('app_id') == app_id), None)
            if row is not None and row.get('state') == 'stopped' and str(row.get('tasks')) == '0':
                return dict(app_id=app_id, state='stopped', tasks=0,
                    stopped_at=row.get('stopped_at'), checked_at_unix=time.time(),
                    stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())
        if attempt + 1 < attempts:
            time.sleep(1)
    return None


def poll_call(call, modal_module, stop, deadline):
    """Poll without masking Modal's polling timeout as a launcher failure."""
    while True:
        if stop.is_set():
            raise RuntimeError('Cloud-preparation budget or watchdog stop')
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Cloud-preparation phase lifecycle cutoff')
        try:
            return call.get(timeout=min(10, max(.1, remaining)))
        except modal_module.exception.FunctionTimeoutError:
            raise
        except (modal_module.exception.TimeoutError, TimeoutError):
            # Modal 1.6.2 poll_function raises the built-in; other SDK paths use Modal's type.
            continue


def launch(contract):
    first_phase = EXECUTION_PHASES[0]
    first_resources = contract['resources'][first_phase]
    connection = fresh_connection(first_resources)
    rates = connection['normalized_rates']
    started = time.time()
    first_job = 'raw-v9'
    first_reservation = reserve(LEDGER, first_job, contract, ROOT,
        first_resources, rates, connection, phase=first_phase, recovery_of='raw-v8')
    phase_lifecycle = first_resources['timeout_s'] + first_resources['startup_allowance_s']
    receipt = dict(run_id=contract['run_id'], status='reserved', started_at_unix=started,
        phases=[dict(phase=first_phase, job_id=first_job, status='reserved',
                     reservation=first_reservation)],
        cloud_calls=[], stop_verified=False)
    state = dict(app_id=None, call=None, current_phase=first_job,
        phase_started=started, phase_deadline=started + phase_lifecycle, stop_reason=None)
    write(RECEIPT, receipt)
    # Reservation above precedes the first Modal-capable import.
    import modal
    from beatfcn.cloud import full_database_preparation_app
    stop = threading.Event()
    closed = threading.Event()
    state_lock = threading.RLock()
    overall_deadline = started + sum(contract['resources'][phase]['timeout_s']
        + contract['resources'][phase]['startup_allowance_s']
        + contract['resources'][phase]['stop_allowance_s'] for phase in EXECUTION_PHASES) + 600

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
        last_stop = -float('inf')
        while not closed.is_set():
            try:
                should_stop = False
                with state_lock:
                    phase = state['current_phase']
                    app_id = state['app_id']
                    if phase and app_id:
                        usage = record_usage(LEDGER, phase,
                            elapsed_s_since_dispatch=time.time() - state['phase_started'],
                            app_id=app_id)
                        receipt['current_usage'] = usage['job']
                        should_stop = usage['grant']['stop_required']
                        if should_stop:
                            state['stop_reason'] = 'budget_stop'
                    if time.time() > overall_deadline:
                        should_stop = True
                        state['stop_reason'] = state['stop_reason'] or 'hard_deadline'
                        receipt['watchdog_hard_stop_at_unix'] = time.time()
                    if should_stop:
                        stop.set()
                    write(RECEIPT, receipt)
                if should_stop:
                    cancel_owned()
                    if app_id and time.time() - last_stop >= 5:
                        last_stop = time.time()
                        subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
            except BaseException as exc:
                with state_lock:
                    receipt['meter_error'] = type(exc).__name__
                    state['stop_reason'] = 'meter_failure:' + type(exc).__name__
                    write(RECEIPT, receipt)
                stop.set()
                cancel_owned()
            closed.wait(2)
    thread = threading.Thread(target=meter, daemon=True)
    thread.start()
    try:
        with modal.enable_output():
            with full_database_preparation_app.app.run(detach=True, name=contract['run_id']):
                with state_lock:
                    state['app_id'] = full_database_preparation_app.app.app_id
                    if not state['app_id']:
                        raise RuntimeError('Actual app identity missing')
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                for phase in EXECUTION_PHASES:
                    resources = contract['resources'][phase]
                    if phase == first_phase:
                        job_id = first_job
                    else:
                        # Evidence is refreshed for every phase, not reused beyond 600s.
                        connection = fresh_connection(resources)
                        job_id = phase
                        phase_started = time.time()
                        phase_lifecycle = resources['timeout_s'] + resources['startup_allowance_s']
                        reservation = reserve(LEDGER, job_id, contract, ROOT,
                            resources, connection['normalized_rates'], connection, phase=phase)
                        with state_lock:
                            state['current_phase'] = job_id
                            state['phase_started'] = phase_started
                            state['phase_deadline'] = phase_started + phase_lifecycle
                            receipt['phases'].append(dict(phase=phase, job_id=job_id,
                                status='reserved', reservation=reservation))
                            write(RECEIPT, receipt)
                    function = getattr(full_database_preparation_app, {
                        'targets': 'prepare_targets', 'raw': 'verify_raw',
                        'mel': 'verify_mel', 'checkpoint': 'download_checkpoint'}[phase])
                    function = function.with_options(timeout=resources['timeout_s'],
                        cpu=(resources['cpu'], resources['cpu']),
                        memory=(resources['memory_gib'] * 1024, resources['memory_gib'] * 1024),
                        retries=0, max_containers=1, scaledown_window=2)
                    call = function.spawn(contract['run_id'],
                        json.dumps(contract, ensure_ascii=False, separators=(',', ':')))
                    with state_lock:
                        state['call'] = call
                        receipt['cloud_calls'].append(dict(phase=phase, call_id=call.object_id))
                        deadline = state['phase_deadline']
                        write(RECEIPT, receipt)
                    result = poll_call(call, modal, stop, deadline)
                    with state_lock:
                        state['call'] = None
                    if not result.get('complete'):
                        raise ValueError(f'Phase {phase} incomplete: {result}')
                    with state_lock:
                        # The same lock excludes the meter while this job transitions to closed.
                        receipt['phases'].append(dict(phase=phase, job_id=job_id,
                            status='completed', result=result))
                        record_usage(LEDGER, job_id,
                            elapsed_s_since_dispatch=time.time() - state['phase_started'],
                            app_id=state['app_id'], closed=True,
                            stop_evidence=dict(phase=phase, function_returned=True,
                                               app_id=state['app_id']))
                        state['current_phase'] = None
                        write(RECEIPT, receipt)
                with state_lock:
                    receipt['status'] = 'completed'
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
            active_job = state['current_phase']
            phase_started = state['phase_started']
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
            if receipt['status'] == 'completed' and evidence is None:
                receipt['status'] = 'cleanup_failed'
            if active_job and evidence:
                record_usage(LEDGER, active_job,
                    elapsed_s_since_dispatch=time.time() - phase_started,
                    app_id=app_id, closed=True, stop_evidence=evidence)
            elif active_job:
                receipt['ledger_close_pending'] = active_job
            write(RECEIPT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    arguments = parser.parse_args()
    contract = build_contract()
    if not arguments.launch:
        print(json.dumps(dict(status='local_checks_passed_no_cloud_calls',
            run_id=contract['run_id'], cloud_calls=0), ensure_ascii=False))
        return
    result = launch(contract)
    print(json.dumps(dict(status=result['status'], stop_verified=result['stop_verified']),
        ensure_ascii=False))
    if result['status'] != 'completed' or not result['stop_verified']:
        raise RuntimeError('Cloud preparation did not reach a verified stopped terminal state')


if __name__ == '__main__':
    main()

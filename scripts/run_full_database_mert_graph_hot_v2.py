"""Check, reserve, then explicitly launch the Owner-capped MERT pilot.

Without --launch this performs local/read-only checks only. The first Modal
import occurs only after the independent USD 5 pilot reservation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time

from beatfcn.train.data_preparation_budget import CAP, GRANT, gate, record_usage, reserve


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
SELECTION = ROOT / 'reports/full-database-cloud-v1/mert-graph-hot-selection-v2.json'
PLAN = ROOT / 'bench/rhythm-v2/mert-graph-hot-v2.json'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-graph-hot-run-v2.json'
CONNECTION = ROOT / 'reports/full-database-cloud-v1/mert-graph-hot-connection-v2.json'
RATE_PATH = ROOT / 'reports/rhythm_v2_rates.json'
SOURCES = (
    'beatfcn/audio/mert2.py',
    'beatfcn/audio/mert2_batch.py',
    'beatfcn/audio/cuda_mel.py',
    'beatfcn/audio/local_mel_cache.py',
    'beatfcn/audio/preprocess.py',
    'beatfcn/train/full_database.py',
    'beatfcn/train/cuda_pipeline.py',
    'beatfcn/events/canonical.py',
    'beatfcn/events/validate.py',
    'beatfcn/events/group_split.py',
    'beatfcn/events/controls.py',
    'beatfcn/model/event_canvas.py',
    'beatfcn/model/batch.py',
    'beatfcn/model/features.py',
    'beatfcn/ingest/pipeline.py',
    'beatfcn/ingest/phira_policy.py',
    'beatfcn/ingest/coverage.py',
    'beatfcn/ingest/offset.py',
    'beatfcn/ingest/timing.py',
    'beatfcn/train/data.py',
    'beatfcn/train/event_density_drag_filter.py',
    'beatfcn/train/drag_density_filter.py',
    'beatfcn/cloud/full_database_app.py',
    'beatfcn/cloud/full_database_concurrent_app.py',
    'beatfcn/cloud/full_database_batch_app.py',
    'beatfcn/cloud/full_database_mert_graph_app.py',
    'beatfcn/cloud/gpu_util.py',
    'beatfcn/train/data_preparation_budget.py',
    'beatfcn/train/campaign_budget.py',
    'scripts/refresh_rhythm_state.py',
    'scripts/prepare_full_database_mert_graph_hot_v2.py',
    'scripts/run_full_database_mert_graph_hot_v2.py',
)
RESOURCES = dict(timeout_s=240, startup_allowance_s=50, stop_allowance_s=50,
    cpu=8, memory_gib=16, gpu='L40S', retries=0, max_containers=1,
    other_upper_usd='0.05')
PREREQUISITES = dict(
    target_preparation='reports/full-database-cloud-v1/target-preparation-v1.json',
    raw_content='reports/full-database-cloud-v1/raw-content-receipt-v1.json',
    checkpoint_upload='reports/full-database-cloud-v1/mert-checkpoint-upload-v1.json')
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


def normalized_rates():
    value = json.loads(RATE_PATH.read_text())
    return dict(gpu_usd_s=str(Decimal(value['gpu_hour_cost_l40s']) / Decimal(3600)),
        cpu_usd_s=str(Decimal(value['cpu_hour_cost']) / Decimal(3600)),
        memory_gib_usd_s=str(Decimal(value['mem_gib_hour_cost']) / Decimal(3600)),
        source='reports/rhythm_v2_rates.json')


def reserve_upper(contract, rates):
    resources = contract['resources']
    duration = (Decimal(str(resources['timeout_s']))
        + Decimal(str(resources['startup_allowance_s']))
        + Decimal(str(resources['stop_allowance_s'])))
    rate = (Decimal(rates['gpu_usd_s'])
        + Decimal(str(resources['cpu'])) * Decimal(rates['cpu_usd_s'])
        + Decimal(str(resources['memory_gib'])) * Decimal(rates['memory_gib_usd_s']))
    return rate * duration + Decimal(str(resources.get('other_upper_usd', 0)))


def contract_digest(contract):
    payload = {key: value for key, value in contract.items() if key != 'contract_sha256'}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def build_contract():
    selection = json.loads(SELECTION.read_text())
    admitted = PLAN.exists()
    if admitted:
        contract = json.loads(PLAN.read_text())
    else:
        prerequisite_sha = {name: sha(ROOT / path) for name, path in PREREQUISITES.items()
                            if (ROOT / path).is_file()}
        contract = dict(
            schema='beatfcn.full-database-mert-pilot.v1', id='full-database-mert-graph-hot-v2',
            grant_id=GRANT, run_id='full-db-mert-graph-hot-v2', optimizer_updates=0,
            jobs=selection['jobs'], output_root='full-database-v1/mert-graph-diagnostic-v2',
            checkpoint='full-database-v1/mert2-checkpoint',
            source_sha256={name: sha(ROOT / name) for name in SOURCES},
            selection_sha256=sha(SELECTION),
            jobs_receipt_sha256=sha(ROOT / 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json'),
            prerequisites=PREREQUISITES, prerequisite_sha256=prerequisite_sha,
            owner_gpu_evidence=dict(
                path='reports/full-database-cloud-v1/owner-gpu-utilization-85.md',
                sha256=sha(ROOT / 'reports/full-database-cloud-v1/owner-gpu-utilization-85.md')),
            gpu_utilization=dict(
                method='legacy_GpuUtil_take_interval_means_with_last8_resource_guard',
                required_stable_mean_percent=85, sampler_period_ms=500,
                take_interval_s=30, rolling_intervals=8, check_after_s=480,
                setup_excluded=True, final_partial_interval_included=True,
                missing_samples_fail_closed=True,
                pipeline_decode_write_and_gaps_included=True,
                failure_blocks_full_corpus=True),
            resources=dict(RESOURCES), full_corpus_encode=False, automatic_retry=False,
            pipeline=dict(graph='mert_exact_shape_cuda_graph_v1', prefetch_workers=24,
                write_workers=8, in_flight_groups=24, commit_every=16,
                direct_verified_raw=True, cuda_mel=True, single_pass_write_hash=True),
            diagnostic_only=True,
            raw_manifest_sha256='676032329c43f829e0aa4439cdab6ac6fddd9c8361a5014e350c3e2e382e4652',
            raw_content_sha256='f8cad7e07c7815e0486d4e6be65dc2bfe9433eed30ba65e0d0014faba34ebfbf',
            previous_attempt=dict(run_id='full-db-mert-graph-probe-v2',
                graph_probe_gpu_mean_percent=99.1, eager_gpu_mean_percent=42.6,
                graph_single_parity_exact=True,
                reason='measure end-to-end decode/mel/capture/replay/write occupancy before the formal 480s gate'),
            acceptance=dict(
                all_selected_jobs_complete=True, no_permanent_failures=True,
                output_shape_matches_mel_clock=True, all_output_files_finite=True,
                each_feature_receipt_hash_verified=True, volume_reopen_recovery=True,
                graph_single_parity_exact=True, end_to_end_stage_profile_required=True,
                formal_85_percent_acceptance_claim=False),
            note='Bounded end-to-end exact-shape MERT graph diagnostic; no formal efficiency pass, training or full-corpus authorization.')
        contract['contract_sha256'] = contract_digest(contract)
    expected = {name: sha(ROOT / name) for name in SOURCES}
    if contract.get('source_sha256') != expected:
        raise ValueError('Frozen pilot source identity changed')
    if contract.get('selection_sha256') != sha(SELECTION) or contract.get('jobs') != selection['jobs']:
        raise ValueError('Frozen pilot selection or selected jobs changed')
    if contract.get('jobs_receipt_sha256') != sha(ROOT / 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json'):
        raise ValueError('Frozen MERT jobs receipt changed')
    if contract.get('resources') != RESOURCES or contract['resources'].get('gpu') != 'L40S':
        raise ValueError('Frozen pilot resource envelope or L40S rate identity changed')
    if contract.get('prerequisites') != PREREQUISITES:
        raise ValueError('Frozen prerequisite paths changed')
    if contract.get('contract_sha256') != contract_digest(contract):
        raise ValueError('Frozen pilot contract digest changed')
    errors = gate(contract, ROOT)
    if errors:
        raise ValueError('; '.join(errors))
    # An unadmitted draft is never persisted; only a complete gated contract becomes immutable.
    if not admitted:
        write(PLAN, contract)
    return contract


def fresh_connection(contract):
    from scripts.refresh_rhythm_state import refresh
    value = refresh()
    if not value.get('fresh_check_succeeded'):
        raise ValueError('Fresh read-only Modal connection check failed')
    evidence = {name: sha(ROOT / result['path']) for name, result in value['results'].items()
                if result.get('ok')}
    if set(evidence) != {'apps', 'rates', 'billing'}:
        raise ValueError('Fresh Modal evidence incomplete')
    rates = normalized_rates()
    upper = reserve_upper(contract, rates)
    result = dict(value, evidence_sha256=evidence,
        full_lifecycle_bound_verified=upper <= CAP,
        pilot_upper_usd=str(upper), normalized_rates=rates)
    write(CONNECTION, result)
    return result


def stopped_evidence(app_id, deadline, attempts=3):
    for attempt in range(attempts):
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        result = subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'list', '--json'],
            cwd=ROOT, capture_output=True, text=True, timeout=min(10, remaining))
        if result.returncode == 0:
            values = json.loads(result.stdout)
            row = next((item for item in values if item.get('app_id') == app_id), None)
            if row is not None and row.get('state') == 'stopped' and str(row.get('tasks')) == '0':
                return dict(app_id=app_id, state='stopped', tasks=0,
                    stopped_at=row.get('stopped_at'), checked_at_unix=time.time(),
                    stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())
        if attempt + 1 < attempts:
            time.sleep(min(1, max(0, deadline - time.time())))
    return None


def poll_call(call, modal_module, stop, deadline):
    while True:
        if stop.is_set():
            raise RuntimeError('MERT pilot budget or watchdog stop')
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('MERT pilot lifecycle cutoff')
        try:
            return call.get(timeout=min(10, max(.1, remaining)))
        except modal_module.exception.FunctionTimeoutError:
            raise
        except (modal_module.exception.TimeoutError, TimeoutError):
            continue


def recompute_gpu_acceptance(value):
    """Independently verify raw stable-period evidence; never trust a boolean alone."""
    means = value.get('raw_interval_means', [])
    counts = value.get('raw_sample_counts', [])
    durations = value.get('raw_interval_durations_s', [])
    observed = value.get('stable_observed_s', 0)
    cadence_valid = (len(durations) == len(means)
        and all(math.isfinite(float(item)) and 0 < float(item) <= 65 for item in durations)
        and all(20 <= float(item) <= 65 for item in durations[:-1])
        and abs(sum(durations) - observed) <= 2)
    valid = (value.get('applicable') is True and len(means) == len(counts)
        and len(means) >= 8 and value.get('error_interval_count') == 0
        and observed >= 480 and cadence_valid
        and all(math.isfinite(float(item)) and 0 <= float(item) <= 100 for item in means)
        and all(type(count) is int and count > 0 for count in counts)
        and value.get('rolling_stop_reason') is None)
    stable = sum(means) / len(means) if means else None
    rolling = sum(means[-8:]) / 8 if len(means) >= 8 else None
    passed = bool(valid and stable >= 85 and rolling >= 85)
    return dict(passed=passed, stable_period_mean_utilization_percent=stable,
        last_eight_interval_mean_utilization_percent=rolling,
        interval_count=len(means), raw_sample_count=sum(counts),
        interval_coverage_s=sum(durations), interval_coverage_valid=cadence_valid,
        raw_evidence_valid=valid)


def launch(contract):
    connection = fresh_connection(contract)
    rates = connection['normalized_rates']
    started = time.time()
    reservation = reserve(LEDGER, contract['run_id'], contract, ROOT,
        contract['resources'], rates, connection)
    deadline = started + float(contract['resources']['startup_allowance_s']
        + contract['resources']['timeout_s'] + contract['resources']['stop_allowance_s'])
    soft_deadline = deadline - contract['resources']['stop_allowance_s']
    receipt = dict(run_id=contract['run_id'], status='reserved', reservation=reservation,
        started_at_unix=started, deadline_unix=deadline, cloud_calls=[],
        result=None, failure=None, stop_verified=False)
    write(RECEIPT, receipt)
    # The first cloud-capable import is intentionally after reservation.
    import modal
    from beatfcn.cloud import full_database_mert_graph_app as full_database_app
    state = dict(app_id=None, call=None, stop_reason=None)
    closed = threading.Event()
    stop = threading.Event()
    state_lock = threading.RLock()

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
                with state_lock:
                    app_id = state['app_id']
                usage = None
                if app_id:
                    # Never hold the state/receipt lock across fcntl or network-dependent ledger I/O.
                    usage = record_usage(LEDGER, contract['run_id'],
                        elapsed_s_since_dispatch=time.time() - started, app_id=app_id)
                with state_lock:
                    if closed.is_set():
                        break
                    should_stop = False
                    if usage:
                        receipt['usage'] = usage['job']
                        should_stop = usage['grant']['stop_required']
                        if should_stop:
                            state['stop_reason'] = 'budget_stop'
                    if time.time() >= soft_deadline:
                        should_stop = True
                        state['stop_reason'] = state['stop_reason'] or 'soft_lifecycle_cutoff'
                    if time.time() >= deadline:
                        should_stop = True
                        state['stop_reason'] = 'hard_deadline_exceeded'
                        receipt['watchdog_hard_stop_at_unix'] = time.time()
                    if should_stop:
                        stop.set()
                    write(RECEIPT, receipt)
                if should_stop:
                    cancel_owned()
                    if app_id and time.time() - last_stop >= 5:
                        last_stop = time.time()
                        subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                            cwd=ROOT, capture_output=True, text=True,
                            timeout=min(20, max(.1, deadline - time.time())))
            except BaseException as exc:
                with state_lock:
                    if closed.is_set():
                        break
                    state['stop_reason'] = 'meter_failure:' + type(exc).__name__
                    receipt['meter_error'] = type(exc).__name__
                    write(RECEIPT, receipt)
                stop.set()
                cancel_owned()
            closed.wait(2)
    thread = threading.Thread(target=meter, daemon=True)
    thread.start()
    try:
        with modal.enable_output():
            with full_database_app.app.run(detach=False, name=contract['run_id']):
                with state_lock:
                    state['app_id'] = full_database_app.app.app_id
                    if not state['app_id']:
                        raise RuntimeError('Actual app identity missing')
                    receipt['app_id'] = state['app_id']
                    write(RECEIPT, receipt)
                resources = contract['resources']
                function = full_database_app.encode_mert_graph.with_options(
                    gpu=resources['gpu'], cpu=(resources['cpu'], resources['cpu']),
                    memory=(resources['memory_gib'] * 1024, resources['memory_gib'] * 1024),
                    timeout=resources['timeout_s'], retries=0, max_containers=1,
                    scaledown_window=120)
                call = function.spawn(contract['run_id'],
                    json.dumps(contract, ensure_ascii=False, separators=(',', ':')),
                    soft_deadline)
                with state_lock:
                    state['call'] = call
                    receipt['cloud_calls'].append(dict(call_id=call.object_id))
                    write(RECEIPT, receipt)
                result = poll_call(call, modal, stop, soft_deadline)
                with state_lock:
                    state['call'] = None
                    receipt['result'] = result
                if (result.get('stopped') or result.get('failures')
                        or len(result.get('success_ids', [])) != len(contract['jobs'])):
                    raise ValueError('Graph diagnostic did not complete every selected job')
                selected = {job['audio_sha256'] for job in contract['jobs']}
                if set(result.get('recovery_verified', [])) != selected:
                    raise ValueError('Remote Volume recovery did not verify every selected output')
                if result.get('diagnostic_only') is not True:
                    raise ValueError('End-to-end graph run lost its diagnostic identity')
                utilization = result.get('gpu_utilization', {})
                recomputed = recompute_gpu_acceptance(utilization)
                if (utilization.get('error_interval_count') != 0
                        or not utilization.get('raw_interval_means')
                        or not all(count > 0 for count in utilization.get('raw_sample_counts', []))):
                    raise ValueError('End-to-end graph GPU evidence is missing or invalid')
                throughput = result.get('throughput', {})
                if not all(throughput.get(key) is not None and throughput[key] > 0
                           for key in ('new_audio_seconds', 'stable_pipeline_seconds',
                                       'audio_seconds_per_second', 'new_jobs_per_second')):
                    raise ValueError('Measured graph throughput is absent or invalid')
                profile = result.get('profile', {})
                parity = profile.get('graph_single_parity', {})
                if not parity.get('passed') or parity.get('max_abs') != 0 \
                        or parity.get('float16_exact_fraction') != 1:
                    raise ValueError('End-to-end graph/single numerical parity was not accepted')
                if any(profile.get('stage_totals', {}).get(key) is None
                        for key in ('decode_s', 'mel_s', 'capture_s', 'replay_s',
                                    'copy_s', 'write_s')):
                    raise ValueError('End-to-end graph stage profile is absent')
                with state_lock:
                    receipt['gpu_utilization'] = utilization
                    receipt['gpu_recomputed'] = recomputed
                    receipt['throughput'] = throughput
                    receipt['profile'] = profile
                    write(RECEIPT, receipt)
                remote = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
                def validate_remote_receipt(audio):
                    path = f"{contract['output_root']}/features/{audio}/receipt.json"
                    saved = json.loads(b''.join(remote.read_file(path)))
                    if not saved.get('complete') or saved.get('audio_sha256') != audio:
                        raise ValueError('Remote feature receipt missing or incomplete: ' + audio)
                    return audio
                # Only small JSON receipts are fetched; feature arrays stay in Modal.
                with ThreadPoolExecutor(max_workers=8) as pool:
                    receipt_ids = list(pool.map(validate_remote_receipt, result['success_ids']))
                if len(receipt_ids) != len(contract['jobs']) or stop.is_set():
                    raise RuntimeError('Remote receipt validation did not finish within the lifecycle')
                with state_lock:
                    receipt['remote_receipts_verified'] = len(result['success_ids'])
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
        thread.join(timeout=min(5, max(0, deadline - time.time())))
        cancel_owned()
        with state_lock:
            app_id = state['app_id']
            meter_alive = thread.is_alive()
        evidence = None
        if app_id and time.time() < deadline:
            try:
                subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=min(10, max(.1, deadline - time.time())))
            except BaseException as exc:
                with state_lock:
                    receipt['stop_verification_error'] = type(exc).__name__
            evidence = stopped_evidence(app_id, deadline)
        ledger_closed = False
        closeout = {}
        if app_id and evidence and not meter_alive and time.time() < deadline:
            def close_ledger():
                try:
                    closeout['usage'] = record_usage(LEDGER, contract['run_id'],
                        elapsed_s_since_dispatch=time.time() - started, app_id=app_id,
                        closed=True, stop_evidence=evidence)['job']
                except BaseException as exc:
                    closeout['error'] = type(exc).__name__
            closer = threading.Thread(target=close_ledger, daemon=True)
            closer.start()
            closer.join(timeout=min(5, max(0, deadline - time.time())))
            ledger_closed = not closer.is_alive() and 'usage' in closeout and time.time() <= deadline
        with state_lock:
            receipt['stop_verified'] = evidence is not None
            receipt['stop_evidence'] = evidence
            if ledger_closed:
                receipt['usage'] = closeout['usage']
            else:
                receipt['ledger_close_pending'] = True
                if closeout.get('error'):
                    receipt['usage_closeout_error'] = closeout['error']
            if receipt['status'] == 'compute_returned':
                receipt['status'] = ('completed' if evidence is not None and ledger_closed
                    else 'cleanup_failed')
            receipt['elapsed_s_since_dispatch'] = time.time() - started
            receipt['billing_scope'] = 'conservative live upper;not a settled invoice'
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
        raise RuntimeError('MERT pilot did not reach a verified stopped terminal state')


if __name__ == '__main__':
    main()

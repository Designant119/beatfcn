"""Independent Owner-capped full-database data-preparation grant."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from .campaign_budget import locked_ledger


GRANT = 'owner-full-database-mert-pilot-20261030'
CAP = Decimal('19.70')
PRIOR_CAP = Decimal('5.00')
SECOND_PRIOR_CAP = Decimal('10.00')
THIRD_PRIOR_CAP = Decimal('13.00')
FOURTH_PRIOR_CAP = Decimal('15.00')
FIFTH_PRIOR_CAP = Decimal('16.50')
SIXTH_PRIOR_CAP = Decimal('17.50')
SEVENTH_PRIOR_CAP = Decimal('18.10')
EIGHTH_PRIOR_CAP = Decimal('18.70')
KNOWN_GPUS = {'L40S': 'gpu_usd_s', 'A10G': 'gpu_usd_s_a10g'}
FULL_CORPUS_EVIDENCE = 'reports/full-database-cloud-v1/owner-mert-pilot-cap-increase-2-20261031.md'


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('Cost must be finite and nonnegative')
    return result


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def event(ledger, kind, **fields):
    ledger.setdefault('data_preparation_events', []).append(dict(kind=kind,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), **fields))


def totals(grant):
    jobs = grant.setdefault('jobs', [])
    used = sum((max(money(job['reserved_upper_usd']), money(job['live_upper_usd']),
                     money(job['reported_actual_usd'])) for job in jobs), Decimal(0))
    grant.update(committed_upper_usd=str(used), remaining_usd=str(max(CAP - used, Decimal(0))),
        reported_actual_usd=str(sum((money(job['reported_actual_usd']) for job in jobs), Decimal(0))),
        stop_required=used > CAP or any(job.get('stop_required') and job['status'] != 'closed' for job in jobs))


def authorize(path, *, owner_evidence_path, root):
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner approval evidence missing')
    with locked_ledger(path) as ledger:
        grants = ledger.setdefault('data_preparation_authorizations', {})
        grant = grants.get(GRANT)
        if grant is None:
            grant = dict(id=GRANT, purpose='bounded full-database MERT throughput pilot only',
                owner_authorized=False, jobs=[], paid_dispatched=False)
            grants[GRANT] = grant
        if grant.get('owner_authorized'):
            raise ValueError('Existing approval is immutable; amendments require a new grant')
        grant.update(owner_authorized=True, new_usd_cap=str(CAP),
            owner_evidence=owner_evidence_path, owner_evidence_sha256=digest(evidence),
            funding='new explicit Owner data-preparation cap;outside historical80USD/probe6USD/hn2-4.06USD')
        totals(grant)
        event(ledger, 'data_preparation_authorized', grant_id=GRANT, cap_usd=str(CAP),
            evidence=owner_evidence_path)


def authorize_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD5 MERT/CUDA amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != PRIOR_CAP:
            raise ValueError('MERT/CUDA increase requires the original USD5 cap')
        amendment = dict(id='owner-mert-pilot-increase-5-20261031',
            previous_cap_usd=str(PRIOR_CAP), increase_usd='5.00', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            previous_evidence=dict(path=grant['owner_evidence'],
                sha256=grant['owner_evidence_sha256']),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';additional explicit USD5 MERT/CUDA amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='5.00', new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_second_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD3 MERT/CUDA amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner second MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != SECOND_PRIOR_CAP:
            raise ValueError('Second MERT/CUDA increase requires the USD10 cap')
        amendment = dict(id='owner-mert-pilot-second-increase-3-20261031',
            previous_cap_usd=str(SECOND_PRIOR_CAP), increase_usd='3.00',
            new_usd_cap=str(CAP), evidence=owner_evidence_path,
            evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';second explicit USD3 MERT/CUDA amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='3.00', previous_cap_usd=str(SECOND_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_third_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD2 full-database amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner third MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != THIRD_PRIOR_CAP:
            raise ValueError('Third MERT/CUDA increase requires the USD13 cap')
        amendment = dict(id='owner-mert-pilot-third-increase-2-20261031',
            previous_cap_usd=str(THIRD_PRIOR_CAP), increase_usd='2.00', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';third explicit USD2 full-database amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='2.00', previous_cap_usd=str(THIRD_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_fourth_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD1.5 post-OOM retry amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner fourth MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != FOURTH_PRIOR_CAP:
            raise ValueError('Fourth MERT/CUDA increase requires the USD15 cap')
        amendment = dict(id='owner-mert-pilot-fourth-increase-1.5-20261031',
            previous_cap_usd=str(FOURTH_PRIOR_CAP), increase_usd='1.50', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';fourth explicit USD1.5 post-OOM retry amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='1.50', previous_cap_usd=str(FOURTH_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_fifth_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD1.0 final-attempt amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner fifth MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != FIFTH_PRIOR_CAP:
            raise ValueError('Fifth MERT/CUDA increase requires the USD16.5 cap')
        amendment = dict(id='owner-mert-pilot-fifth-increase-1.0-20261031',
            previous_cap_usd=str(FIFTH_PRIOR_CAP), increase_usd='1.00', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';fifth explicit USD1.0 final-attempt amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='1.00', previous_cap_usd=str(FIFTH_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_sixth_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD0.6 instrumented-final-run amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner sixth MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != SIXTH_PRIOR_CAP:
            raise ValueError('Sixth MERT/CUDA increase requires the USD17.5 cap')
        amendment = dict(id='owner-mert-pilot-sixth-increase-0.6-20261031',
            previous_cap_usd=str(SIXTH_PRIOR_CAP), increase_usd='0.60', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';sixth explicit USD0.6 instrumented-final amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='0.60', previous_cap_usd=str(SIXTH_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_seventh_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD0.6 final-completion amendment without resetting jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner seventh MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != SEVENTH_PRIOR_CAP:
            raise ValueError('Seventh MERT/CUDA increase requires the USD18.1 cap')
        amendment = dict(id='owner-mert-pilot-seventh-increase-0.6-20261031',
            previous_cap_usd=str(SEVENTH_PRIOR_CAP), increase_usd='0.60', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';seventh explicit USD0.6 final-completion amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='0.60', previous_cap_usd=str(SEVENTH_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def authorize_eighth_increase(path, *, owner_evidence_path, root):
    # Append the explicit +USD1.0 final re-dispatch amendment without resetting jobs.
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner eighth MERT/CUDA increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != EIGHTH_PRIOR_CAP:
            raise ValueError('Eighth MERT/CUDA increase requires the USD18.7 cap')
        amendment = dict(id='owner-mert-pilot-eighth-increase-1.0-20261031',
            previous_cap_usd=str(EIGHTH_PRIOR_CAP), increase_usd='1.00', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';eighth explicit USD1.0 final re-dispatch amendment')
        totals(grant)
        event(ledger, 'data_preparation_cap_increased', grant_id=GRANT,
            increase_usd='1.00', previous_cap_usd=str(EIGHTH_PRIOR_CAP),
            new_cap_usd=str(CAP), evidence=owner_evidence_path)


def gate(contract, root):
    root = Path(root)
    errors = []
    required = {
        'target_preparation': 'reports/full-database-cloud-v1/target-preparation-v1.json',
        'raw_content': 'reports/full-database-cloud-v1/raw-content-receipt-v1.json',
        'checkpoint_upload': 'reports/full-database-cloud-v1/mert-checkpoint-upload-v1.json',
        'mert_jobs': 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json',
        'local_source_probe': 'reports/full-database-cloud-v1/mert-local-source-probe-v1.json',
    }
    if contract.get('schema') != 'beatfcn.full-database-mert-pilot.v1':
        errors.append('Wrong pilot contract schema')
    if contract.get('grant_id') != GRANT or contract.get('optimizer_updates', 0) != 0:
        errors.append('Pilot grant/no-update identity missing')
    for name, relative in required.items():
        path = root / relative
        if not path.is_file():
            errors.append(name + '_receipt missing')
            continue
        value = json.loads(path.read_text())
        passed = value.get('complete') is True or value.get('status') in {'uploaded', 'already_present'}
        if not passed:
            errors.append(name + '_receipt not passed')
        if name in {'target_preparation', 'raw_content', 'checkpoint_upload'}:
            expected = contract.get('prerequisite_sha256', {}).get(name)
            if not expected or digest(path) != expected:
                errors.append(name + '_receipt identity differs from frozen pilot contract')
    for name, expected in contract.get('source_sha256', {}).items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            errors.append('Frozen source drift: ' + name)
    if not contract.get('source_sha256'):
        errors.append('Source closure absent')
    gpu = contract.get('gpu_utilization', {})
    if (gpu.get('method') != 'legacy_GpuUtil_take_interval_means_with_last8_resource_guard'
            or gpu.get('required_stable_mean_percent') != 85
            or gpu.get('sampler_period_ms') != 500
            or gpu.get('take_interval_s') != 30
            or gpu.get('rolling_intervals') != 8
            or gpu.get('check_after_s') != 480
            or gpu.get('setup_excluded') is not True
            or gpu.get('final_partial_interval_included') is not True
            or gpu.get('missing_samples_fail_closed') is not True
            or gpu.get('pipeline_decode_write_and_gaps_included') is not True
            or gpu.get('failure_blocks_full_corpus') is not True):
        errors.append('Owner legacy stable-period >=85% GPU-utilization gate absent or changed')
    evidence = contract.get('owner_gpu_evidence', {})
    evidence_path = root / evidence.get('path', '')
    if (not evidence.get('sha256') or not evidence_path.is_file()
            or digest(evidence_path) != evidence.get('sha256')):
        errors.append('Owner GPU-utilization evidence missing or changed')
    jobs = contract.get('jobs', [])
    if not jobs or len({job['audio_sha256'] for job in jobs}) != len(jobs):
        errors.append('Pilot jobs absent or duplicated')
    if contract.get('automatic_retry'):
        errors.append('Pilot cannot execute or retry full corpus')
    if contract.get('full_corpus_encode'):
        full = contract.get('full_corpus_evidence', {})
        path = root / full.get('path', '')
        if (full.get('path') != FULL_CORPUS_EVIDENCE or not full.get('sha256')
                or not path.is_file() or digest(path) != full.get('sha256')):
            errors.append('Full-corpus authorization evidence missing or changed')
    return errors


def reserve(path, job_id, contract, root, resources, rates, connection):
    errors = gate(contract, root)
    if errors:
        raise ValueError('; '.join(errors))
    if not connection.get('fresh_check_succeeded') or not 0 <= time.time() - connection.get('checked_at_unix', 0) <= 600:
        raise ValueError('Fresh Modal account/rate evidence required')
    if not connection.get('evidence_sha256') or not connection.get('full_lifecycle_bound_verified'):
        raise ValueError('Full-lifecycle evidence absent')
    if resources.get('retries', 0) != 0 or resources.get('max_containers', 0) != 1:
        raise ValueError('Single container and zero retries required')
    for key in ('timeout_s', 'cpu', 'memory_gib', 'stop_allowance_s'):
        if money(resources.get(key, 0)) <= 0:
            raise ValueError('Positive resource bound required: ' + key)
    for key in ('gpu_usd_s', 'cpu_usd_s', 'memory_gib_usd_s'):
        if money(rates.get(key, 0)) <= 0:
            raise ValueError('Current positive rate required: ' + key)
    if resources.get('gpu') and resources['gpu'] not in KNOWN_GPUS:
        raise ValueError('Unknown GPU rate class: ' + str(resources.get('gpu')))
    duration = money(resources['timeout_s']) + money(resources.get('startup_allowance_s', 300)) + money(resources['stop_allowance_s'])
    gpu_rate_key = KNOWN_GPUS[resources['gpu']] if resources.get('gpu') else None
    rate = (money(rates[gpu_rate_key]) if gpu_rate_key else Decimal(0)) \
        + money(resources['cpu']) * money(rates['cpu_usd_s']) \
        + money(resources['memory_gib']) * money(rates['memory_gib_usd_s'])
    upper = (rate * duration + money(resources.get('other_upper_usd', 0))).quantize(Decimal('.000001'))
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        if not grant.get('owner_authorized') or money(grant['new_usd_cap']) != CAP:
            raise ValueError('Owner approval missing or changed')
        if digest(Path(root) / grant['owner_evidence']) != grant['owner_evidence_sha256']:
            raise ValueError('Owner approval evidence drift')
        totals(grant)
        if grant['stop_required'] or upper > money(grant['remaining_usd']):
            raise ValueError('Data-preparation cap exhausted')
        if any(job['id'] == job_id for job in grant['jobs']):
            raise ValueError('Duplicate attempt;no silent retry')
        if any(job['status'] != 'closed' for job in grant['jobs']):
            raise ValueError('One active data-preparation job at a time')
        job = dict(id=job_id, status='reserved', reserved_upper_usd=str(upper),
            reported_actual_usd='0', live_upper_usd='0', rate_upper_usd_s=str(rate),
            other_upper_usd=str(money(resources.get('other_upper_usd', 0))), resources=resources,
            contract_sha256=contract.get('contract_sha256'), evidence_sha256=connection['evidence_sha256'],
            reserved_at_unix=time.time())
        grant['jobs'].append(job)
        totals(grant)
        event(ledger, 'data_preparation_reserved', grant_id=GRANT, job_id=job_id, upper_usd=str(upper))
        return job


def record_usage(path, job_id, *, elapsed_s_since_dispatch, app_id,
                 reported_actual_usd=None, closed=False, stop_evidence=None):
    with locked_ledger(path) as ledger:
        grant = ledger['data_preparation_authorizations'][GRANT]
        job = next(item for item in grant['jobs'] if item['id'] == job_id)
        if job['status'] == 'closed':
            raise ValueError('Closed data-preparation job cannot resume')
        if not app_id or job.get('app_id', app_id) != app_id:
            raise ValueError('Actual app identity required')
        elapsed = money(elapsed_s_since_dispatch)
        if elapsed < money(job.get('elapsed_s_since_dispatch', 0)):
            raise ValueError('Elapsed meter cannot go backwards')
        actual = money(job['reported_actual_usd']) if reported_actual_usd is None else money(reported_actual_usd)
        if actual < money(job['reported_actual_usd']):
            raise ValueError('Reported charges cannot decrease')
        if closed and not stop_evidence:
            raise ValueError('App-stop evidence required')
        live = max(money(job['live_upper_usd']), actual,
                   elapsed * money(job['rate_upper_usd_s']) + money(job['other_upper_usd']))
        stop = live + money(job['resources']['stop_allowance_s']) * money(job['rate_upper_usd_s']) >= money(job['reserved_upper_usd'])
        job.update(app_id=app_id, elapsed_s_since_dispatch=str(elapsed),
            reported_actual_usd=str(actual), live_upper_usd=str(live),
            stop_required=bool(stop), status='closed' if closed else 'stopping' if stop else 'running')
        if closed:
            job['stop_evidence'] = stop_evidence
        grant['paid_dispatched'] = True
        totals(grant)
        event(ledger, 'data_preparation_usage', job_id=job_id, elapsed_s=str(elapsed),
            live_upper_usd=str(live), status=job['status'])
        return dict(job=job, grant=grant)

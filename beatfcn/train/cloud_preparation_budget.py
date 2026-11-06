"""Independent Owner-capped Modal-container full-database preparation grant."""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from .campaign_budget import locked_ledger


GRANT = 'owner-full-database-cloud-preparation-20261031'
CAP = Decimal('8.00')
PRIOR_CAP = Decimal('5.00')
PHASES = {'targets', 'raw', 'mel', 'checkpoint'}


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('Cost must be finite and nonnegative')
    return result


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def event(ledger, kind, **fields):
    ledger.setdefault('cloud_preparation_events', []).append(dict(kind=kind,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), **fields))


def totals(grant):
    jobs = grant.setdefault('jobs', [])
    committed = sum((max(money(job['reserved_upper_usd']), money(job['live_upper_usd']),
                          money(job['reported_actual_usd'])) for job in jobs), Decimal(0))
    reported = sum((money(job['reported_actual_usd']) for job in jobs), Decimal(0))
    live = sum((max(money(job['live_upper_usd']), money(job['reported_actual_usd'])) for job in jobs), Decimal(0))
    grant.update(committed_upper_usd=str(committed), reported_actual_usd=str(reported),
        live_upper_usd=str(live), remaining_usd=str(max(Decimal(0), CAP - committed)),
        budget_exhausted=committed >= CAP,
        stop_required=live >= CAP or any(job.get('stop_required') and job['status'] != 'closed' for job in jobs))


def authorize(path, *, owner_evidence_path, root):
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner cloud-preparation approval missing')
    with locked_ledger(path) as ledger:
        grants = ledger.setdefault('cloud_preparation_authorizations', {})
        if GRANT in grants:
            raise ValueError('Cloud-preparation grant already exists; preserve its identity')
        grant = dict(id=GRANT, purpose='bounded in-container target/raw/mel/checkpoint preparation only',
            owner_authorized=True, new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=digest(evidence), jobs=[], paid_dispatched=False,
            funding='new explicit Owner data-preparation cap;separate from MERT pilot and historical pools')
        grants[GRANT] = grant
        totals(grant)
        event(ledger, 'cloud_preparation_authorized', grant_id=GRANT, cap_usd=str(CAP),
            evidence=owner_evidence_path)


def authorize_increase(path, *, owner_evidence_path, root):
    """Append the explicit +USD3 amendment; never reset or rewrite prior jobs."""
    evidence = Path(root) / owner_evidence_path
    if not evidence.is_file():
        raise ValueError('Owner cloud-preparation increase approval missing')
    with locked_ledger(path) as ledger:
        grant = ledger['cloud_preparation_authorizations'][GRANT]
        if money(grant['new_usd_cap']) != PRIOR_CAP:
            raise ValueError('Cloud-preparation increase requires the original USD5 cap')
        previous_evidence = dict(path=grant['owner_evidence'],
            sha256=grant['owner_evidence_sha256'])
        amendment = dict(id='owner-cloud-preparation-increase-3-20261031',
            previous_cap_usd=str(PRIOR_CAP), increase_usd='3.00', new_usd_cap=str(CAP),
            evidence=owner_evidence_path, evidence_sha256=digest(evidence),
            previous_evidence=previous_evidence,
            recorded_at_utc=datetime.now(timezone.utc).isoformat())
        grant.setdefault('cap_amendments', []).append(amendment)
        grant.update(new_usd_cap=str(CAP), owner_evidence=owner_evidence_path,
            owner_evidence_sha256=amendment['evidence_sha256'],
            funding=grant['funding'] + ';additional explicit USD3 Owner amendment')
        totals(grant)
        event(ledger, 'cloud_preparation_cap_increased', grant_id=GRANT,
            increase_usd='3.00', new_cap_usd=str(CAP), evidence=owner_evidence_path)


def gate(contract, root):
    root = Path(root)
    errors = []
    required = {
        'readiness': 'reports/full-database-cloud-v1/readiness-audit-v1.json',
        'manifest_download': 'inputs/full-database-cloud-v1/manifest/download-receipt.json',
        'local_source_probe': 'reports/full-database-cloud-v1/mert-local-source-probe-v1.json',
    }
    if contract.get('schema') != 'beatfcn.full-database-cloud-preparation.v1':
        errors.append('Wrong cloud-preparation contract schema')
    if contract.get('grant_id') != GRANT or contract.get('optimizer_updates', 0) != 0:
        errors.append('Preparation grant/no-update identity missing')
    if set(contract.get('phases', [])) != PHASES:
        errors.append('All four preparation phases required exactly once')
    for name, relative in required.items():
        path = root / relative
        if not path.is_file():
            errors.append(name + ' missing')
            continue
        value = json.loads(path.read_text())
        if value.get('complete') is not True:
            errors.append(name + ' not complete')
    for name, expected in contract.get('source_sha256', {}).items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            errors.append('Frozen source drift: ' + name)
    if not contract.get('source_sha256'):
        errors.append('Source closure absent')
    if contract.get('automatic_retry') or contract.get('local_corpus_traffic'):
        errors.append('No retry or local-corpus traffic permitted')
    return errors


def reserve(path, job_id, contract, root, resources, rates, connection, *,
            phase=None, recovery_of=None):
    errors = gate(contract, root)
    if errors:
        raise ValueError('; '.join(errors))
    phase = phase or job_id
    if phase not in PHASES:
        raise ValueError('Unknown preparation phase')
    if job_id != phase and not recovery_of:
        raise ValueError('A new attempt ID requires explicit manual recovery lineage')
    if not connection.get('fresh_check_succeeded') or not 0 <= time.time() - connection.get('checked_at_unix', 0) <= 600:
        raise ValueError('Fresh Modal evidence required')
    if not connection.get('evidence_sha256') or not connection.get('full_lifecycle_bound_verified'):
        raise ValueError('Full lifecycle evidence required')
    if resources.get('retries', 0) != 0 or resources.get('max_containers', 0) != 1:
        raise ValueError('Single container/zero retries required')
    for key in ('timeout_s', 'cpu', 'memory_gib', 'stop_allowance_s'):
        if money(resources.get(key, 0)) <= 0:
            raise ValueError('Positive resource bound required: ' + key)
    for key in ('gpu_usd_s', 'cpu_usd_s', 'memory_gib_usd_s'):
        if key != 'gpu_usd_s' and money(rates.get(key, 0)) <= 0:
            raise ValueError('Current positive rate required: ' + key)
        if key == 'gpu_usd_s' and resources.get('gpu') is not None and money(rates.get(key, 0)) <= 0:
            raise ValueError('Current positive rate required: ' + key)
    duration = money(resources['timeout_s']) + money(resources.get('startup_allowance_s', 120)) + money(resources['stop_allowance_s'])
    gpu_rate = Decimal(0) if resources.get('gpu') is None else money(rates['gpu_usd_s'])
    rate = gpu_rate + money(resources['cpu']) * money(rates['cpu_usd_s']) + money(resources['memory_gib']) * money(rates['memory_gib_usd_s'])
    upper = (rate * duration + money(resources.get('other_upper_usd', 0))).quantize(Decimal('.000001'))
    with locked_ledger(path) as ledger:
        grant = ledger['cloud_preparation_authorizations'][GRANT]
        if not grant.get('owner_authorized') or money(grant['new_usd_cap']) != CAP:
            raise ValueError('Owner cloud-preparation approval changed')
        if digest(Path(root) / grant['owner_evidence']) != grant['owner_evidence_sha256']:
            raise ValueError('Owner evidence drift')
        totals(grant)
        if grant['stop_required'] or upper > money(grant['remaining_usd']):
            raise ValueError('Cloud-preparation cap exhausted')
        if any(job['id'] == job_id for job in grant['jobs']):
            raise ValueError('Duplicate preparation attempt; no silent retry')
        if any(job['status'] != 'closed' for job in grant['jobs']):
            raise ValueError('One active preparation phase at a time')
        if recovery_of:
            prior = next((job for job in grant['jobs'] if job['id'] == recovery_of), None)
            if (prior is None or prior.get('phase', prior['id']) != phase
                    or prior['status'] != 'closed'
                    or prior.get('stop_evidence', {}).get('state') != 'stopped'
                    or str(prior.get('stop_evidence', {}).get('tasks')) != '0'):
                raise ValueError('Manual recovery requires the prior phase to be closed with stopped/tasks=0 evidence')
        job = dict(id=job_id, phase=phase, recovery_of=recovery_of,
            status='reserved', reserved_upper_usd=str(upper),
            reported_actual_usd='0', live_upper_usd='0', rate_upper_usd_s=str(rate),
            other_upper_usd=str(money(resources.get('other_upper_usd', 0))), resources=resources,
            contract_sha256=contract.get('contract_sha256'), evidence_sha256=connection['evidence_sha256'],
            reserved_at_unix=time.time())
        grant['jobs'].append(job)
        totals(grant)
        event(ledger, 'cloud_preparation_reserved', grant_id=GRANT, job_id=job_id, upper_usd=str(upper))
        return job


def record_usage(path, job_id, *, elapsed_s_since_dispatch, app_id,
                 reported_actual_usd=None, closed=False, stop_evidence=None):
    with locked_ledger(path) as ledger:
        grant = ledger['cloud_preparation_authorizations'][GRANT]
        job = next(item for item in grant['jobs'] if item['id'] == job_id)
        if job['status'] == 'closed':
            raise ValueError('Closed preparation phase cannot resume')
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
        event(ledger, 'cloud_preparation_usage', job_id=job_id, elapsed_s=str(elapsed),
            live_upper_usd=str(live), status=job['status'])
        return dict(job=job, grant=grant)

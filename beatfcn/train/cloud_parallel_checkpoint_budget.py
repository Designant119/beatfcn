"""Explicit parallel checkpoint reservation within the existing preparation cap."""
from decimal import Decimal
from pathlib import Path
import time

from .campaign_budget import locked_ledger
from .cloud_preparation_budget import CAP, GRANT, digest, event, gate, money, totals


def reserve_parallel_checkpoint(path, contract, root, resources, rates, connection):
    """Allow only the independent checkpoint to overlap the active mel verifier."""
    errors = gate(contract, root)
    if errors:
        raise ValueError('; '.join(errors))
    if contract.get('execution_phases') != ['checkpoint'] or not contract.get('detach'):
        raise ValueError('Parallel checkpoint requires its own detached phase contract')
    if not connection.get('fresh_check_succeeded') or not 0 <= time.time() - connection.get('checked_at_unix', 0) <= 600:
        raise ValueError('Fresh Modal evidence required')
    if not connection.get('evidence_sha256') or not connection.get('full_lifecycle_bound_verified'):
        raise ValueError('Full lifecycle evidence required')
    if resources.get('retries') != 0 or resources.get('max_containers') != 1:
        raise ValueError('Single container and zero retries required')
    for key in ('timeout_s', 'cpu', 'memory_gib', 'stop_allowance_s'):
        if money(resources.get(key, 0)) <= 0:
            raise ValueError('Positive resource bound required: ' + key)
    if money(rates.get('cpu_usd_s')) <= 0 or money(rates.get('memory_gib_usd_s')) <= 0:
        raise ValueError('Current CPU/memory rates required')
    duration = (money(resources['timeout_s']) + money(resources.get('startup_allowance_s', 120))
        + money(resources['stop_allowance_s']))
    rate = (money(resources['cpu']) * money(rates['cpu_usd_s'])
        + money(resources['memory_gib']) * money(rates['memory_gib_usd_s']))
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
        if any(job['id'] == 'checkpoint' for job in grant['jobs']):
            raise ValueError('Duplicate checkpoint phase')
        active = [job for job in grant['jobs'] if job['status'] != 'closed']
        if any(job['id'] != 'mel' for job in active):
            raise ValueError('Checkpoint may only overlap the independent mel phase')
        job = dict(id='checkpoint', phase='checkpoint', status='reserved',
            reserved_upper_usd=str(upper), reported_actual_usd='0', live_upper_usd='0',
            rate_upper_usd_s=str(rate), other_upper_usd=str(money(resources.get('other_upper_usd', 0))),
            resources=resources, contract_sha256=contract.get('contract_sha256'),
            evidence_sha256=connection['evidence_sha256'], reserved_at_unix=time.time(),
            parallel_with='mel' if any(job['id'] == 'mel' for job in active) else None,
            parallel_reason='independent Hugging Face checkpoint download; no shared raw/mel output files')
        grant['jobs'].append(job)
        totals(grant)
        event(ledger, 'cloud_preparation_parallel_checkpoint_reserved', grant_id=GRANT,
            job_id='checkpoint', upper_usd=str(upper), parallel_with=job['parallel_with'])
        return job

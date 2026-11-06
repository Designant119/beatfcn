"""Separate Owner-reviewed >=120M grant;never spends historical reservations.

No implicit grant is created. A missing dollar cap or review blocks reservation.
All timestamps/costs are conservative upper bounds,not claimed settled billing.
"""
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from .bounded_ledger import locked_ledger
from .hn2_contract import contract_digest,validate_connection


def money(v):
    d=Decimal(str(v))
    if not d.is_finite() or d<0:raise ValueError('Invalid money')
    return d


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def event(ledger,kind,**fields):
    ledger.setdefault('owner_candidate_events',[]).append(dict(kind=kind,recorded_at_utc=datetime.now(timezone.utc).isoformat(),**fields))


def totals(grant):
    jobs=grant.setdefault('jobs',[])
    used=sum((max(money(j['reserved_upper_usd']),money(j['live_upper_usd']),money(j['reported_actual_usd'])) for j in jobs),Decimal(0))
    cap=money(grant['new_usd_cap'])
    grant.update(committed_upper_usd=str(used),remaining_usd=str(max(cap-used,Decimal(0))),
        actual_reported_usd=str(sum((money(j['reported_actual_usd']) for j in jobs),Decimal(0))),
        stop_required=used>cap or any(j.get('stop_required') and j['status']!='closed' for j in jobs))


def authorize_owner_candidate(path,grant_id,cap_usd,*,owner_evidence_path,root):
    """Call only after an explicit Owner dollar-cap answer;not from a deadline."""
    cap=money(cap_usd);evidence=Path(root)/owner_evidence_path
    if cap<=0 or not evidence.is_file():raise ValueError('Explicit positive cap and Owner evidence required')
    with locked_ledger(path) as ledger:
        grant=ledger.setdefault('conditional_candidate_authorizations',{}).get(grant_id)
        if not grant:raise ValueError('Unknown Owner candidate directive')
        if grant.get('owner_authorized'):
            raise ValueError('Preserve existing grant;new instruction needs explicit amendment')
        grant.update(owner_authorized=True,new_usd_cap=str(cap),owner_cap_evidence=owner_evidence_path,
            owner_cap_evidence_sha256=digest(evidence),funding='new explicit Owner cap;not historical80USD or probe6USD',
            cost_question_pending=False,jobs=[],paid_dispatched=False)
        totals(grant);event(ledger,'candidate_grant_authorized',grant_id=grant_id,cap_usd=str(cap),evidence=owner_evidence_path)


def candidate_gate(contract,root):
    root=Path(root);errors=[]
    expected_contract=contract_digest(contract)
    if contract.get('execution_stage')!='cuda_health_only' or contract.get('quality_updates_allowed')!=0:
        errors.append('This budget entry permits discarded CUDA health only')
    if contract.get('contract_sha256',expected_contract)!=expected_contract:errors.append('Derived contract digest mismatch')
    if int(contract.get('actual_trainable_parameters',0))<120_000_000:errors.append('Actual trainable parameters below120M')
    if not contract.get('candidate_id') or contract.get('grant_id')!='owner-modal120m-20261030':errors.append('Candidate/directive identity missing')
    for field in ('independent_review_receipt','local_gate_receipt'):
        path=root/contract.get(field,'')
        if not path.is_file():errors.append(field+' missing');continue
        value=json.loads(path.read_text())
        if value.get('contract_sha256')!=expected_contract:errors.append(field+' complete contract differs')
        if value.get('passed') is not True or value.get('candidate_id')!=contract.get('candidate_id'):
            errors.append(field+' not passed for this candidate')
        if value.get('source_sha256')!=contract.get('source_sha256'):
            errors.append(field+' source set differs from launch source')
        if value.get('data_manifest_sha256')!=contract.get('data_manifest_sha256'):
            errors.append(field+' data identity differs from launch data')
        if field=='independent_review_receipt':
            if value.get('unresolved_blockers'):errors.append('Independent review blockers remain')
            if value.get('source_sha256')!=contract.get('source_sha256'):errors.append('Review source set differs from launch source')
            if value.get('data_manifest_sha256')!=contract.get('data_manifest_sha256'):errors.append('Review data identity differs')
            if not value.get('independent_original_reports'):errors.append('Independent review originals absent')
            for report in value.get('independent_original_reports',[]):
                p=root/report.get('path','')
                if not p.is_file() or digest(p)!=report.get('sha256'):
                    errors.append('Independent review original missing or drifted')
    for name,expected in contract.get('source_sha256',{}).items():
        p=root/name
        if not p.is_file() or digest(p)!=expected:errors.append('Frozen source drift '+name)
    if not contract.get('source_sha256'):errors.append('Missing source closure')
    for name,expected in contract.get('data_manifest_sha256',{}).items():
        p=root/name
        if not p.is_file() or digest(p)!=expected:errors.append('Data identity drift '+name)
    if not contract.get('data_manifest_sha256'):errors.append('Missing data manifest identity')
    if contract.get('density_rule')!='all-events-1s-gt12-drag-only-v2':errors.append('Wrong Owner data rule')
    if contract.get('evaluate_every')!=250:errors.append('250-applied-update progress required')
    return errors


def reserve_owner_candidate(path,job_id,contract,root,resources,rates,connection):
    errors=candidate_gate(contract,root)
    if errors:raise ValueError('; '.join(errors))
    validate_connection(connection,root)
    if resources!=contract.get('resources'):raise ValueError('Resource arguments differ from reviewed contract')
    if any(money(rates[k])!=money(contract['frozen_rates'][k]) for k in ('gpu_usd_s','cpu_usd_s','memory_gib_usd_s')):
        raise ValueError('Rates differ from complete reviewed contract')
    if not connection.get('fresh_check_succeeded') or not 0<=time.time()-connection.get('checked_at_unix',0)<=600:
        raise ValueError('Fresh account/app/rate evidence required')
    if not connection.get('evidence_sha256') or not connection.get('full_lifecycle_bound_verified'):
        raise ValueError('Full lifecycle bound and current evidence required')
    if resources.get('retries',0)!=0 or resources.get('max_containers',1)!=1:raise ValueError('Single container/zero retries required')
    if any(money(resources[k])<=0 for k in ('timeout_s','cpu','memory_gib')):
        raise ValueError('Positive hard resource limits required')
    if any(money(rates[k])<=0 for k in ('gpu_usd_s','cpu_usd_s','memory_gib_usd_s')):
        raise ValueError('Current positive billing rates required')
    duration=money(resources['timeout_s'])+money(resources.get('startup_allowance_s',120))+money(resources.get('stop_allowance_s',30))
    if duration<=0 or money(resources.get('stop_allowance_s',0))<=0:raise ValueError('Positive bounded lifecycle/stop allowance required')
    rate=money(rates['gpu_usd_s'])+money(resources['cpu'])*money(rates['cpu_usd_s'])+money(resources['memory_gib'])*money(rates['memory_gib_usd_s'])
    other=money(resources.get('other_upper_usd',0));upper=rate*duration+other
    grant_id=contract['grant_id']
    with locked_ledger(path) as ledger:
        grant=ledger['conditional_candidate_authorizations'][grant_id]
        if not grant.get('owner_authorized') or grant.get('new_usd_cap') is None:raise ValueError('Owner dollar cap pending;no paid dispatch')
        if digest(Path(root)/grant['owner_cap_evidence'])!=grant['owner_cap_evidence_sha256']:raise ValueError('Owner authorization evidence drift')
        totals(grant)
        if grant['stop_required'] or upper>money(grant['remaining_usd']):raise ValueError('Owner candidate cap exhausted')
        if any(j['id']==job_id for j in grant['jobs']):raise ValueError('Duplicate attempt;no automatic retries')
        if any(j['status']!='closed' for j in grant['jobs']):raise ValueError('Only one active paid candidate job')
        job=dict(id=job_id,candidate_id=contract['candidate_id'],status='reserved',reserved_upper_usd=str(upper),live_upper_usd='0',reported_actual_usd='0',
            rate_upper_usd_s=str(rate),other_upper_usd=str(other),resources=resources,stop_required=False,contract_sha256=contract.get('contract_sha256'),
            evidence_sha256=connection['evidence_sha256'],reserved_at_unix=time.time())
        grant['jobs'].append(job);totals(grant);event(ledger,'candidate_reserved',grant_id=grant_id,job_id=job_id,upper_usd=str(upper))
        return job


def record_owner_candidate_usage(path,job_id,*,elapsed_s_since_dispatch,app_id,reported_actual_usd=None,closed=False,stop_evidence=None,
                                grant_id='owner-modal120m-20261030'):
    with locked_ledger(path) as ledger:
        grant=ledger['conditional_candidate_authorizations'][grant_id];job=next(j for j in grant['jobs'] if j['id']==job_id)
        if job['status']=='closed':raise ValueError('Closed attempt cannot resume')
        if not app_id or job.get('app_id',app_id)!=app_id:raise ValueError('Actual unique app identity required')
        elapsed=money(elapsed_s_since_dispatch)
        if elapsed<money(job.get('elapsed_s_since_dispatch',0)):raise ValueError('Meter cannot go backwards')
        actual=money(job['reported_actual_usd']) if reported_actual_usd is None else money(reported_actual_usd)
        if actual<money(job['reported_actual_usd']):raise ValueError('Reported cost cannot go backwards')
        if closed and not stop_evidence:raise ValueError('Actual app-stop evidence required')
        upper=max(money(job['live_upper_usd']),actual,elapsed*money(job['rate_upper_usd_s'])+money(job['other_upper_usd']))
        stop=upper+money(job['resources']['stop_allowance_s'])*money(job['rate_upper_usd_s'])>=money(job['reserved_upper_usd'])
        job.update(app_id=app_id,elapsed_s_since_dispatch=str(elapsed),reported_actual_usd=str(actual),live_upper_usd=str(upper),
            stop_required=bool(stop),status='closed' if closed else 'stopping' if stop else 'running')
        if closed:job['stop_evidence']=stop_evidence
        grant['paid_dispatched']=True;totals(grant);event(ledger,'candidate_usage',job_id=job_id,elapsed_s=str(elapsed),upper_usd=str(upper),status=job['status'])
        return dict(job=job,grant=grant)

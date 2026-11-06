"""Scoped $6 probe envelope within the existing cumulative campaign ledger."""
from decimal import Decimal, ROUND_CEILING
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import time

from .campaign_budget import locked_ledger, refresh_totals, reservation_cost


POOL = 'owner-cloud-probes-20261029'
CAP = Decimal('6.00')
ALLOWED_PURPOSES = {'hypothesis_test', 'causal_control', 'repair_probe', 'small_reproduction'}


def money(value):
    value=Decimal(str(value))
    if not value.is_finite() or value<0:raise ValueError('Cost must be finite and nonnegative')
    return value


def _event(ledger,kind,**fields):
    ledger.setdefault('cloud_probe_events',[]).append(dict(kind=kind,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),**fields))


def _pool_totals(pool):
    jobs=pool['jobs']
    committed=sum((max(money(j['reserved_upper_usd']),money(j['live_upper_usd']),
                       money(j['reported_actual_usd'])) for j in jobs),Decimal(0))
    reported=sum((money(j['reported_actual_usd']) for j in jobs),Decimal(0))
    live=sum((max(money(j['live_upper_usd']),money(j['reported_actual_usd'])) for j in jobs),Decimal(0))
    pool.update(committed_upper_usd=str(committed),reported_actual_usd=str(reported),
        live_upper_usd=str(live),remaining_for_reservations_usd=str(max(Decimal(0),CAP-committed)),
        budget_exhausted=committed>=CAP,cap_breached=live>CAP,
        stop_required=live>=CAP or any(j.get('stop_required',False) and j['status']!='closed' for j in jobs))


def authorize_probe_pool(path):
    """Ringfence $6 from existing headroom; never add $6 to the total approval."""
    with locked_ledger(path) as ledger:
        pools=ledger.setdefault('cloud_probe_pools',{})
        if POOL in pools:
            if money(pools[POOL]['cap_usd'])!=CAP:raise ValueError('Existing grant amount differs')
            return pools[POOL]
        if POOL in ledger['planned_stages']:raise ValueError('Envelope already exists without grant metadata')
        refresh_totals(ledger)
        if money(ledger['unallocated_margin_usd'])<CAP:raise ValueError('Existing campaign headroom cannot fund the grant')
        pool=dict(id=POOL,cap_usd=str(CAP),funding='existing_unallocated_campaign_headroom',
            scope='Bounded hypothesis/control/repair/reproduction probes only; no final scaling',
            jobs=[],cloud_jobs_dispatched=0)
        _pool_totals(pool);pools[POOL]=pool
        ledger['planned_stages'][POOL]=dict(reserved_upper_usd=float(CAP),scope=pool['scope'],
            reservation_kind='authorization_pool_not_a_dispatched_job',children_accounted_within_pool=True)
        ledger['cloud_dispatch_policy']=dict(mode='small_probes_only',pool_id=POOL,
            scaling_authorized=False,legacy_launches_authorized=False)
        _event(ledger,'owner_authorized_probe_pool',pool_id=POOL,cap_usd=str(CAP),
            source='Owner Yukoval direct instruction 2026-10-29',approved_total_increase_usd='0.00')
        refresh_totals(ledger)
        return pool


def cloud_probe_gate(contract,root):
    """Local evidence + mechanism contract; deliberately no scaling quality gates."""
    from scripts.validate_mechanism_reviews import validate_successor, validate_review, load_registry
    root=Path(root)
    rows=load_registry(root)
    errors=validate_successor(contract,{r['id'] for r in rows})
    for row in rows:
        if row['id'] not in contract.get('predecessors',[]):continue
        path=root/row['review']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['review_sha256']:
            errors.append('Predecessor review identity changed')
        errors.extend(validate_review(json.loads(path.read_text())))
    if contract.get('experiment_class')!='cloud_probe':errors.append('Wrong entry: cloud_probe required')
    if contract.get('purpose') not in ALLOWED_PURPOSES:errors.append('Probe purpose not authorized')
    if contract.get('scaling_run') or contract.get('capacity_scale_run'):
        errors.append('Scaling cannot spend the probe grant')
    if not contract.get('single_intervention'):errors.append('Single intervention must be explicit')
    local=root/contract.get('local_gate_receipt','')
    if not local.is_file():errors.append('Actual local gate receipt missing')
    else:
        result=json.loads(local.read_text())
        if not (result.get('health_passed') is True or result.get('passed') is True):
            errors.append('Local gate has not passed')
    if not contract.get('baseline_identity') or not contract.get('source_sha256'):
        errors.append('Baseline and source identities required')
    for field in ('baseline_identity','source_sha256'):
        for name,expected in contract.get(field,{}).items():
            path=root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                errors.append('Frozen identity changed: '+name)
    return errors


def scaling_gate(contract,root,*,scaling_authorized=False):
    """Separate entry; this $6 grant never grants scaling permission."""
    root=Path(root);errors=[]
    if contract.get('experiment_class')!='scaling':errors.append('Wrong entry: scaling required')
    if not scaling_authorized:errors.append('Scaling authorization absent; probe grant is not scaling authorization')
    if contract.get('primary_claim') not in ('capacity','defect_repair'):
        errors.append('Declare capacity or defect_repair as the primary claim')
    for key in ('matched_control','frozen_new_components','numerical_resource_receipt','preregistered_acceptance'):
        if not contract.get(key):errors.append('Scaling prerequisite missing: '+key)
    acceptance=contract.get('preregistered_acceptance',{})
    if not isinstance(acceptance,dict) or not all(k in acceptance for k in
            ('primary_metric','baseline_value','minimum_gain','maximum_regressions','fixed_endpoint')):
        errors.append('Scaling acceptance must freeze metric, baseline, gain, regression bounds and endpoint')
    numerical=root/contract.get('numerical_resource_receipt','')
    if not numerical.is_file():errors.append('Actual scaling numerical/resource receipt missing')
    required={'per_type','empty_windows','reference_masked','hold_tails'}
    if not required<=set(contract.get('required_reports',[])):errors.append('Scaling reports are incomplete')
    receipt=root/contract.get('scaling_entry_receipt','')
    if not receipt.is_file():errors.append('Separate scaling entry receipt missing')
    else:
        data=json.loads(receipt.read_text());gates=data.get('gates',{})
        required_gates={'purpose_isolated','matched_local_evidence','new_components_frozen','numerical_resources_checked','acceptance_registered'}
        if (data.get('experiment_class')!='scaling' or data.get('passed') is not True or
                not required_gates<=set(gates) or not all(v is True for v in gates.values())):
            errors.append('Scaling-specific entry gate not passed')
    return errors


def reserve_probe(path,job_id,contract,root,resources,rates,connection,*,other_upper_usd):
    errors=cloud_probe_gate(contract,root)
    if errors:raise ValueError('; '.join(errors))
    if not connection.get('fresh_check_succeeded') or not 0<=time.time()-connection.get('checked_at_unix',0)<=600:
        raise ValueError('Fresh rates/app/billing checks required')
    if not connection.get('full_lifecycle_bound_verified') or not connection.get('evidence_sha256'):
        raise ValueError('Full lifecycle and rate evidence required')
    stop=money(resources.get('stop_allowance_s',0))
    if stop<=0:raise ValueError('Positive external-stop allowance required')
    bounded={**resources,'timeout_s':float(money(resources['timeout_s'])+stop)}
    cost=(money(reservation_cost(bounded,rates))+money(other_upper_usd)).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    duration=money(bounded['timeout_s'])+money(bounded.get('startup_allowance_s',120))
    rate=(cost-money(other_upper_usd))/duration
    with locked_ledger(path) as ledger:
        pool=ledger['cloud_probe_pools'][POOL];_pool_totals(pool)
        if pool['stop_required'] or pool['cap_breached']:raise ValueError('Probe budget stop is active')
        if any(j['id']==job_id for j in pool['jobs']):raise ValueError('Duplicate attempt; no silent retries')
        if any(j['status']!='closed' for j in pool['jobs']):raise ValueError('One active cloud probe at a time')
        if cost<=0 or cost>money(pool['remaining_for_reservations_usd']):raise ValueError('Six-dollar probe budget exhausted')
        job=dict(id=job_id,status='reserved',reserved_upper_usd=str(cost),reported_actual_usd='0',live_upper_usd='0',
            rate_upper_usd_s=str(rate),other_upper_usd=str(money(other_upper_usd)),stop_allowance_s=str(stop),
            resources=resources,contract_id=contract['id'],evidence_sha256=connection['evidence_sha256'])
        pool['jobs'].append(job);_pool_totals(pool)
        _event(ledger,'probe_reserved',pool_id=POOL,job_id=job_id,reserved_upper_usd=str(cost))
        return job


def record_probe_usage(path,job_id,*,elapsed_s_since_dispatch,reported_actual_usd=None,
                       closed=False,stop_evidence=None,app_id=None,billing_evidence_sha256=None):
    """Call on every heartbeat/stage/exit. Record overruns honestly, then stop."""
    with locked_ledger(path) as ledger:
        pool=ledger['cloud_probe_pools'][POOL]
        job=next(j for j in pool['jobs'] if j['id']==job_id)
        was_closed=job['status']=='closed'
        if was_closed and not billing_evidence_sha256:
            raise ValueError('Closed attempt cannot restart; late billing evidence required')
        if app_id and job.get('app_id',app_id)!=app_id:raise ValueError('App identity changed')
        if not (app_id or job.get('app_id')):raise ValueError('Actual app identity required for metering')
        elapsed=money(elapsed_s_since_dispatch)
        if elapsed<money(job.get('elapsed_s_since_dispatch',0)):raise ValueError('Elapsed meter cannot go backwards')
        reported=money(job['reported_actual_usd']) if reported_actual_usd is None else money(reported_actual_usd)
        if reported<money(job['reported_actual_usd']):raise ValueError('Reported charges cannot go backwards')
        if closed and not (stop_evidence or was_closed):raise ValueError('Actual app-stop evidence required')
        if job['status']=='reserved':pool['cloud_jobs_dispatched']+=1
        if app_id:job['app_id']=app_id
        live=max(money(job['live_upper_usd']),elapsed*money(job['rate_upper_usd_s'])+money(job['other_upper_usd']),reported)
        stop=live+money(job['stop_allowance_s'])*money(job['rate_upper_usd_s'])>=money(job['reserved_upper_usd'])
        job.update(elapsed_s_since_dispatch=str(elapsed),reported_actual_usd=str(reported),live_upper_usd=str(live),
            stop_required=bool(stop),status='closed' if closed or was_closed else 'stopping' if stop else 'running')
        if closed:job['stop_evidence']=stop_evidence
        if billing_evidence_sha256:job['billing_evidence_sha256']=billing_evidence_sha256
        _pool_totals(pool)
        ledger['planned_stages'][POOL]['accounted_upper_usd']=str(max(CAP,money(pool['committed_upper_usd'])))
        refresh_totals(ledger)
        _event(ledger,'probe_usage',job_id=job_id,elapsed_s_since_dispatch=str(elapsed),reported_actual_usd=str(reported),
            live_upper_usd=str(live),status=job['status'],stop_required=pool['stop_required'])
        return dict(job=job,pool=pool)

"""Single cumulative ledger for bounded paid stages; reservations survive failures."""
from contextlib import contextmanager
from decimal import Decimal
from datetime import datetime,timezone
import fcntl
import json
import math
from pathlib import Path
import time


@contextmanager
def locked_ledger(path):
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text())
        yield ledger
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)


def reservation_cost(resources, rates):
    cpu = Decimal(str(resources['cpu']))
    memory = Decimal(str(resources['memory_gib']))
    duration = Decimal(str(resources['timeout_s'])) + Decimal(str(resources.get('startup_allowance_s', 120)))
    if any(not x.is_finite() or x <= 0 for x in (cpu, memory, duration)):
        raise ValueError('resource bounds must be finite and positive')
    hourly = cpu * Decimal(rates['cpu_hour_cost']) + memory * Decimal(rates['mem_gib_hour_cost'])
    if resources.get('gpu'):
        hourly += Decimal(rates['gpu_hour_cost_' + resources['gpu'].lower()])
    if not hourly.is_finite() or hourly <= 0:
        raise ValueError('invalid billing rates')
    if resources.get('retries', 0) != 0 or resources.get('max_containers', 1) != 1:
        raise ValueError('this campaign requires zero retries and one container')
    return float(hourly * duration / 3600)


def accounted_cost(stage):
    """Reservations remain immutable; verified closed-app bounds are separate evidence."""
    reconciliation=stage.get('reconciliation')
    return reconciliation['accounted_upper_usd'] if reconciliation else stage['reserved_usd']


def refresh_totals(ledger):
    used={s['name'] for s in ledger['paid_stages_dispatched']}
    pending=sum(max(s['reserved_upper_usd'],float(s.get('accounted_upper_usd',0)))
                for name,s in ledger['planned_stages'].items() if name not in used)
    upper=(ledger['historical_accounted_upper_usd']+ledger['planned_other_reserve_usd']+pending+
           sum(accounted_cost(s) for s in ledger['paid_stages_dispatched']))
    ledger['campaign_committed_upper_usd']=upper
    ledger['unallocated_margin_usd']=ledger['approved_total_usd']-upper


def reconcile_closed_stages(path, lifecycles, billing, rates, connection, *, evidence):
    """Bound old, stopped, single-container apps by full app lifetime at resource caps.

    Keep the startup allowance, require all occupied hourly billing intervals and
    at least one hour after the final interval, and add a billing-delay cushion.
    This is a conservative resource bound, not a claim that invoices are final.
    Recent, running, ambiguous or unobserved stages keep their full reservation.
    """
    now=time.time()
    if not connection.get('fresh_check_succeeded') or not 0<=now-connection.get('checked_at_unix',0)<=600:
        raise ValueError('fresh billing and rates required for reconciliation')
    if not 0<=now-lifecycles.get('checked_at_unix',0)<=600:
        raise ValueError('fresh lifecycle evidence required')
    if not evidence or not all(evidence.values()):raise ValueError('evidence fingerprints required')
    by_app={r['app_id']:r for r in lifecycles['apps']}
    if len(by_app)!=len(lifecycles['apps']):raise ValueError('duplicate lifecycle app')
    cutoff=math.floor(connection['checked_at_unix']/3600)*3600-3600
    charges={}
    for row in billing:
        interval=datetime.fromisoformat(row['interval_start']).replace(tzinfo=timezone.utc).timestamp()
        cost=Decimal(str(row['cost']))
        if not cost.is_finite() or cost<0:raise ValueError('invalid reported cost')
        key=(row['object_id'],interval)
        if key in charges:raise ValueError('duplicate app billing interval')
        charges[key]=float(cost)
    changes=[]
    with locked_ledger(path) as ledger:
        stages=ledger['paid_stages_dispatched']
        if len({s['app_id'] for s in stages})!=len(stages):raise ValueError('apps shared by stages require manual reconciliation')
        for stage in stages:
            app=by_app.get(stage['app_id'],{})
            start,stop=app.get('created_at',0),app.get('stopped_at',0)
            if not app.get('ok') or app.get('app_state')!='APP_STATE_STOPPED' or not 0<start<stop<=cutoff:continue
            resource=stage['resources']
            if resource.get('max_containers',1)!=1 or resource.get('retries',0)!=0:continue
            intervals=list(range(int(start//3600)*3600,int(math.ceil(stop/3600))*3600,3600))
            if any((stage['app_id'],i) not in charges for i in intervals):continue
            billed=sum(cost for (app_id,_),cost in charges.items() if app_id==stage['app_id'])
            original_duration=resource['timeout_s']+resource.get('startup_allowance_s',120)
            rate=max(stage['reserved_usd'],reservation_cost(resource,rates))/original_duration
            lifetime_upper=rate*(stop-start+resource.get('startup_allowance_s',120))
            upper=max(lifetime_upper,billed*1.1+.02)
            before=accounted_cost(stage)
            stage['reconciliation']=dict(method='closed_app_resource_lifetime_upper_v1',
                observed_at_unix=now,app_id=stage['app_id'],app_created_at=start,app_stopped_at=stop,
                billed_compute_usd=billed,billing_complete_before_unix=cutoff,
                rate_upper_usd_s=rate,lifetime_resource_upper_usd=lifetime_upper,
                accounted_upper_usd=upper,original_reserved_usd=stage['reserved_usd'],evidence=evidence,
                limitation='Includes startup allowance and reporting cushion; not a settled invoice. Other reserves unchanged.')
            changes.append(dict(name=stage['name'],before=before,after=upper,reported_compute=billed))
        refresh_totals(ledger)
        result=dict(changes=changes,campaign_committed_upper_usd=ledger['campaign_committed_upper_usd'],
                    unallocated_margin_usd=ledger['unallocated_margin_usd'],approved_total_usd=ledger['approved_total_usd'])
    return result


def reserve(path, name, resources, rates, connection, *, app_id, source_sha256):
    age = time.time() - connection.get('checked_at_unix', 0)
    if not connection.get('fresh_check_succeeded') or not 0 <= age <= 600:
        raise ValueError('fresh app, rate and billing reads required before paid dispatch')
    amount = reservation_cost(resources, rates)
    with locked_ledger(path) as ledger:
        if ledger.get('cloud_dispatch_policy', {}).get('mode') == 'small_probes_only':
            raise ValueError('Current Owner authorizes the small-probe pool only; use its scoped reservation guard')
        stages = ledger['paid_stages_dispatched']
        if any(s['name'] == name for s in stages):
            raise ValueError('stage already reserved; inspect its exact handle, do not dispatch twice')
        # Keep the rest of the planned training/evaluation envelope available.
        used_names = {s['name'] for s in stages} | {name}
        pending = sum(r['reserved_upper_usd'] for key, r in ledger['planned_stages'].items() if key not in used_names)
        upper = (ledger['historical_accounted_upper_usd'] + ledger['planned_other_reserve_usd'] + pending +
                 sum(accounted_cost(s) for s in stages) + amount)
        if upper > ledger['approved_total_usd']:
            raise ValueError('cumulative campaign budget exhausted')
        stage = {'name': name, 'resources': resources, 'reserved_usd': amount, 'status': 'reserved',
                 'app_id': app_id, 'source_sha256': source_sha256, 'reserved_at': time.time()}
        stages.append(stage)
        ledger['campaign_committed_upper_usd'] = upper
        ledger['unallocated_margin_usd'] = ledger['approved_total_usd'] - upper
    return stage


def update_stage(path, name, **fields):
    forbidden = {'reserved_usd', 'name', 'resources', 'source_sha256','reconciliation'}
    if forbidden & fields.keys():
        raise ValueError('cannot rewrite a reservation through a status update')
    with locked_ledger(path) as ledger:
        matches = [s for s in ledger['paid_stages_dispatched'] if s['name'] == name]
        if len(matches) != 1:
            raise ValueError('reserved stage missing or ambiguous')
        matches[0].update(fields)

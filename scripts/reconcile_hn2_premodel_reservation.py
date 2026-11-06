"""Propose/reconcile a stopped pre-model GPU-entry path failure using its conservative meter.

Default is read-only. It never calls Modal, declares an invoice, or creates funds.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

from beatfcn.train.bounded_ledger import locked_ledger
from beatfcn.train.owner_admission_budget import event, totals


GRANT = 'owner-modal120m-20261030'


def proposal(ledger, receipt, apps):
    grant = ledger['conditional_candidate_authorizations'][GRANT]
    job = next(j for j in grant['jobs'] if j['id'] == receipt['job_id'])
    if (job['status'] != 'closed' or not receipt.get('stop_verified')
            or not receipt.get('accounting_closed') or not receipt.get('cleanup', {}).get('passed')):
        raise ValueError('Actual stop, closed meter and verified cleanup all required')
    if job.get('reservation_reconciliation'):
        raise ValueError('Preserve an already reconciled reservation')
    proof = receipt['supervisor']['stop_evidence']
    if proof != job['stop_evidence'] or proof['app_id'] != job['app_id']:
        raise ValueError('Stop proof differs from the closed ledger')
    row = next(r for r in apps if r['app_id'] == job['app_id'])
    if row['state'] != 'stopped' or str(row['tasks']) != '0':
        raise ValueError('Read-only postrun app evidence must prove stopped/tasks0')
    calls = receipt.get('cloud_calls', [])
    if (receipt['job_id'] != 'hn2-138m-health-20261030-05'
            or [c['stage'] for c in calls] != ['cpu_asset_admission', 'cuda_admission_and_discarded_health_only']):
        raise ValueError('This reconciliation is restricted to run05 before model execution')
    if (receipt.get('status') != 'failed' or receipt.get('failure', {}).get('type') != 'ValueError'
            or receipt.get('failure', {}).get('reason') != 'Private asset prefix differs from owned run'):
        raise ValueError('Unexpected predecessor failure')
    original = Decimal(job['reserved_upper_usd'])
    meter = max(Decimal(job['live_upper_usd']), Decimal(job['reported_actual_usd']))
    retained = meter.quantize(Decimal('.01'), rounding=ROUND_CEILING)
    if retained <= 0 or retained >= original:
        raise ValueError('No conservative unspent reservation is available')
    return dict(job_id=job['id'], app_id=job['app_id'],
        original_reserved_upper_usd=str(original), closed_live_upper_usd=str(meter),
        retained_closed_cost_upper_usd=str(retained), released_unspent_reservation_usd=str(original-retained),
        statement='Conservative closed cost upper, rounded UP;not an invoice or refund')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', default='reports/rhythm_v2_budget.json')
    p.add_argument('--receipt', default='reports/council-modal120m-c26/runtime-launch.json')
    p.add_argument('--apps', default='reports/council-modal120m-c26/postrun-apps.json')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    apps = json.loads(Path(args.apps).read_text())
    if args.apply:
        with locked_ledger(args.ledger) as ledger:
            value = proposal(ledger, receipt, apps)
            job = next(j for j in ledger['conditional_candidate_authorizations'][GRANT]['jobs']
                       if j['id'] == value['job_id'])
            evidence = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
                        for n in (args.receipt, args.apps)}
            job['original_reserved_upper_usd'] = job['reserved_upper_usd']
            job['reserved_upper_usd'] = value['retained_closed_cost_upper_usd']
            job['reservation_reconciliation'] = dict(**value, evidence_sha256=evidence)
            totals(ledger['conditional_candidate_authorizations'][GRANT])
            event(ledger, 'closed_pre_model_path_failure_reservation_reconciled', **value, evidence_sha256=evidence)
    else:
        value = proposal(json.loads(Path(args.ledger).read_text()), receipt, apps)
    print(json.dumps(dict(**value, applied=args.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

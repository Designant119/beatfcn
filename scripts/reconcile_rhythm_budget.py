"""Review or apply closed-app resource bounds without changing approved budget."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from beatfcn.train.campaign_budget import reconcile_closed_stages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    reports=ROOT/'reports';ledger=reports/'rhythm_v2_budget.json'
    paths={name:reports/file for name,file in {
        'lifecycles':'rhythm_v2_stage_lifecycles.json','billing':'rhythm_v2_billing.json',
        'rates':'rhythm_v2_rates.json','connection':'rhythm_v2_connection_status.json'}.items()}
    docs={key:json.loads(path.read_text()) for key,path in paths.items()}
    evidence={f'{key}_sha256':hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    evidence.update(billing_documentation='https://modal.com/docs/guide/billing',
        resource_caps='Single function call per app; zero retries; max_containers=1; CPU and memory maximums match ledger and dispatch definitions.')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    with tempfile.TemporaryDirectory() as directory:
        if args.apply:
            before=reports/f'rhythm_v2_budget_before_reconcile_{stamp}.json'
            before.write_bytes(ledger.read_bytes());evidence['before_ledger_file']=str(before.relative_to(ROOT))
            target=ledger
        else:
            target=Path(directory)/'dry-run.json';target.write_bytes(ledger.read_bytes())
        result=reconcile_closed_stages(target,docs['lifecycles'],docs['billing'],docs['rates'],docs['connection'],evidence=evidence)
    result.update(applied=args.apply,method='Full stopped-app lifetime at configured max resource rate plus startup allowance; max with reported costs plus 10% and $0.02. Only apps stopped before a full-hour billing interval plus one-hour delay buffer.',
        reserved_usd_fields_unchanged=True,other_reserves_unchanged=True,
        scope='Reconciled resource upper bounds, not final invoices. Recent apps retain full reservation.',evidence=evidence)
    out=reports/f'rhythm_v2_budget_reconciliation_{stamp}.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(out.relative_to(ROOT)),**result},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

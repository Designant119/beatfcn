"""Check, reserve, then explicitly launch one reviewed HN2 lifecycle.

Without --launch this performs local checks only. No Modal import, upload,
deployment or execution can precede the review/Owner-ledger/budget checks.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from beatfcn.train.hn2_stop_supervisor import OwnedStopSupervisor
from beatfcn.train.hn2_output_recovery import recover_output
from beatfcn.train.hn2_contract import contract_digest
from scripts.hn2_result_poll import wait_existing_call
from beatfcn.train.hn2_safe_runtime import cleanup_remote_prefixes,settle_owned_stop,GRANT,ConstructionWatchdog,local_launch_gate,rooted,write_json


ROOT=Path(__file__).resolve().parents[1]


def fresh_stopped_evidence(root,app_id,timeout=20):
    """A cancellation request is not proof that the app has stopped."""
    command=[str(Path(root)/'.venv/bin/modal'),'app','list','--json']
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        return None
    values=json.loads(result.stdout)
    row=next((v for v in values if v.get('app_id')==app_id),None)
    if row is None or row.get('state')!='stopped' or str(row.get('tasks'))!='0':
        return None
    return dict(app_id=app_id,state='stopped',tasks=0,checked_at_unix=time.time(),
        cli_stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())


def launch_checked(root,envelope,*,upload_data=False):
    """The first cloud-capable import is after the sole ledger reservation."""
    from beatfcn.train.owner_admission_budget import reserve_owner_candidate,record_owner_candidate_usage
    root=Path(root);contract=envelope['contract'];resources=contract['resources']
    if not upload_data and not contract.get('remote_assets_already_uploaded'):
        raise ValueError('Freeze asset availability or explicitly request upload before reservation/build')
    ledger=rooted(root,contract.get('ledger_path','reports/rhythm_v2_budget.json'))
    started=time.time();job_id=contract['run_id']
    reservation_contract={**contract,'contract_sha256':contract_digest(contract)}
    reservation=reserve_owner_candidate(ledger,job_id,reservation_contract,root,
        resources,envelope['rates'],envelope['connection'])
    # Everything above is local and must succeed before this import.
    import modal
    from beatfcn.cloud import hn2_safe_app as hn2_app
    deadline=started+resources['startup_allowance_s']+resources['timeout_s']+resources['stop_allowance_s']
    soft_deadline=deadline-resources['stop_allowance_s']
    state=dict(app_id=None,call=None,stop_reason=None);closed=threading.Event();stop=threading.Event()
    construction=ConstructionWatchdog(deadline,resources['stop_allowance_s'])
    receipt_path=rooted(root,contract.get('runtime_receipt_path','reports/modal-120m-20261030/runtime-launch.json'))
    receipt=dict(candidate_id=contract['candidate_id'],job_id=job_id,reservation=reservation,
        started_at_unix=started,deadline_unix=deadline,cloud_calls=[],status='reserved')
    receipt_lock=threading.RLock()
    def save_receipt():
        with receipt_lock:write_json(receipt_path,receipt)
    save_receipt()
    def cancel_owned():
        call=state['call']
        if call is not None:
            try:call.cancel(terminate_containers=True)
            except Exception as exc:
                receipt['cancel_error']=type(exc).__name__
    def stop_cli(app_id):
        subprocess.run([str(root/'.venv/bin/modal'),'app','stop',app_id],
            cwd=root,capture_output=True,text=True,timeout=10)
    def record_usage(app_id,proof):
        return record_owner_candidate_usage(ledger,job_id,elapsed_s_since_dispatch=time.time()-started,
            app_id=app_id,closed=proof is not None,stop_evidence=proof)
    def publish(snapshot):
        receipt['supervisor']=snapshot;save_receipt()
    def recover_and_cleanup():
        evidence=asyncio.run(recover_output(hn2_app.data,job_id,modal.exception.NotFoundError))
        if evidence:
            receipt['recovered_evidence']=evidence;save_receipt()
        return asyncio.run(cleanup_remote_prefixes(hn2_app.data,job_id,modal.exception.NotFoundError))
    supervisor=OwnedStopSupervisor(deadline,resources['stop_allowance_s'],
        app_id=lambda:hn2_app.app.app_id,cancel=cancel_owned,stop_app=stop_cli,
        verify=lambda app_id:fresh_stopped_evidence(root,app_id),record=record_usage,
        cleanup=recover_and_cleanup,
        publish=publish).start()
    def check():
        supervisor.check()
    def await_call(call):
        from modal._functions import _Invocation
        state['call']=call
        value=wait_existing_call(call,check=check,soft_deadline=soft_deadline,
            poll_code=_Invocation.poll_function.__code__)
        state['call']=None
        return value
    try:
        with modal.enable_output():
            with hn2_app.app.run(detach=False,name='hn2-'+job_id):
                state['app_id']=hn2_app.app.app_id
                if not state['app_id']:raise RuntimeError('Actual app identity missing')
                check();receipt.update(status='app_started',app_id=state['app_id']);save_receipt()
                asset_prefix='hn2-assets/'+job_id;asset_root='/data/'+asset_prefix
                if upload_data:
                    upload=json.loads(rooted(root,contract['upload_manifest']).read_text())
                    names=set(upload['files'])|set(contract['data_manifest_sha256'])|{contract['upload_manifest']}
                    with hn2_app.data.batch_upload(force=False) as context:
                        for name in sorted(names):
                            check();context.put_file(str(rooted(root,name)),asset_prefix+'/'+name)
                    receipt['uploaded_files']=len(names);save_receipt()
                elif not contract.get('remote_assets_already_uploaded'):
                    raise ValueError('Explicit --upload-data or frozen remote asset availability required')
                kwargs=dict(contract_json=envelope['contract_json'],authorization_json=envelope['authorization_json'],
                    review_json=envelope['review_json'],local_gate_json=envelope['local_gate_json'],asset_root=asset_root)
                check()
                call=hn2_app.verify_assets.with_options(timeout=max(1,min(1200,int(soft_deadline-time.time())))).spawn(**kwargs)
                receipt['cloud_calls'].append(dict(stage='cpu_asset_admission',call_id=call.object_id));save_receipt()
                asset=await_call(call)
                if asset['receipt']['passed'] is not True:raise ValueError('CPU remote asset admission failed')
                check()
                function=hn2_app.train.with_options(gpu='H100',cpu=(resources['cpu'],resources['cpu']),
                    memory=(resources['memory_gib']*1024,resources['memory_gib']*1024),
                    timeout=max(1,min(resources['timeout_s'],int(soft_deadline-time.time()))),retries=0,max_containers=1,scaledown_window=2)
                call=function.spawn(**kwargs,asset_receipt_sha256=asset['sha256'],dispatch_deadline_unix=deadline)
                receipt['cloud_calls'].append(dict(stage='cuda_admission_and_discarded_health_only',call_id=call.object_id));save_receipt()
                receipt['result']=await_call(call);receipt['status']='compute_returned';save_receipt()
    except BaseException as exc:
        receipt.update(status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)))
        supervisor.request_stop('Main exception')
        raise
    finally:
        supervisor.request_stop('CLI finally')
        supervisor.finished.wait(resources['stop_allowance_s'])
        snapshot=dict(supervisor.state)
        receipt.update(supervisor=snapshot,stop_verified=snapshot['stop_verified'],
            accounting_closed=snapshot['accounting_closed'],cleanup=snapshot.get('cleanup'),
            elapsed_s_since_dispatch=time.time()-started,
            billing_scope='Conservative live upper;not a settled invoice')
        save_receipt()
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--authorization',help='Optional exact summary of the authoritative ledger grant')
    parser.add_argument('--launch',action='store_true',help='Reserve full lifecycle BEFORE any Modal operation')
    parser.add_argument('--upload-data',action='store_true')
    args=parser.parse_args(argv)
    if args.upload_data and not args.launch:parser.error('--upload-data requires --launch')
    envelope=local_launch_gate(ROOT,args.contract,args.authorization,verify_assets=args.launch)
    if not args.launch:
        print(json.dumps(dict(status='local_checks_passed_no_cloud_calls',candidate_id=envelope['contract']['candidate_id'],
            grant_id=GRANT,cloud_calls=0,reservation_performed=False)),flush=True)
        return
    result=launch_checked(ROOT,envelope,upload_data=args.upload_data)
    print(json.dumps(dict(status=result['status'],stop_verified=result['stop_verified'])),flush=True)


if __name__=='__main__':main()

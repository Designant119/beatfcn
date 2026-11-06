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

from beatfcn.train.hn2_admission_runtime import cleanup_remote_prefixes,settle_owned_stop,GRANT,ConstructionWatchdog,local_launch_gate,rooted,write_json


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
    from beatfcn.train.owner_candidate_budget import reserve_owner_candidate,record_owner_candidate_usage
    root=Path(root);contract=envelope['contract'];resources=contract['resources']
    if not upload_data and not contract.get('remote_assets_already_uploaded'):
        raise ValueError('Freeze asset availability or explicitly request upload before reservation/build')
    ledger=rooted(root,contract.get('ledger_path','reports/rhythm_v2_budget.json'))
    started=time.time();job_id=contract['run_id']
    reservation_contract={**contract,'contract_sha256':hashlib.sha256(envelope['contract_json'].encode()).hexdigest()}
    reservation=reserve_owner_candidate(ledger,job_id,reservation_contract,root,
        resources,envelope['rates'],envelope['connection'])
    # Everything above is local and must succeed before this import.
    import modal
    from beatfcn.cloud import hn2_admission_app as hn2_app
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
    stop_lock=threading.RLock()
    def settle():
        with stop_lock:
            if receipt.get('safe_to_exit') and receipt.get('cleanup',{}).get('passed'):return receipt['stop_result']
            if receipt.get('safe_to_exit'):
                result=receipt['stop_result']
            else:
                result=None
            def stop_app(app_id):
                result=subprocess.run([str(root/'.venv/bin/modal'),'app','stop',app_id],
                    cwd=root,capture_output=True,text=True,timeout=20)
                receipt['stop_cli_returncode']=result.returncode
            def record(app_id,evidence):
                return record_owner_candidate_usage(ledger,job_id,
                    elapsed_s_since_dispatch=time.time()-started,app_id=app_id,
                    closed=evidence is not None,stop_evidence=evidence)
            if result is None:
                result=settle_owned_stop(state['app_id'],cancel=cancel_owned,stop=stop_app,
                    verify=lambda app_id:fresh_stopped_evidence(root,app_id),record=record)
            receipt.update(stop_result=result,safe_to_exit=result['safe_to_exit'],
                stop_verified=result['stop_verified'],stopped_evidence=result['evidence'])
            if result['safe_to_exit']:
                try:
                    receipt['cleanup']=asyncio.run(cleanup_remote_prefixes(hn2_app.data,job_id,modal.exception.NotFoundError))
                except Exception as exc:
                    receipt['cleanup']=dict(passed=False,failure=type(exc).__name__)
            save_receipt()
            return {**result,'safe_to_exit':result['safe_to_exit'] and receipt.get('cleanup',{}).get('passed',False)}
    def meter():
        last_stop=-float('inf')
        while not closed.is_set():
            try:
                tick=construction.tick(hn2_app.app.app_id,time.time())
                if tick['app_id'] and not state['app_id']:
                    state['app_id']=tick['app_id'];receipt['app_id']=tick['app_id']
                    receipt['app_detected_during_construction_at_unix']=time.time();save_receipt()
                if tick['meter_due'] and not receipt.get('safe_to_exit'):
                    result=record_owner_candidate_usage(ledger,job_id,
                        elapsed_s_since_dispatch=time.time()-started,app_id=state['app_id'])
                    if result['job']['stop_required']:
                        state['stop_reason']='budget_stop_required';stop.set()
                if tick['stop_required'] or stop.is_set():
                    state['stop_reason']=state['stop_reason'] or 'independent_lifecycle_cutoff';stop.set()
                    if state['app_id'] and time.time()-last_stop>=5:
                        last_stop=time.time()
                        if settle()['safe_to_exit']:return
                # No os._exit: an unverified app must retain a live stop monitor.
                if tick['hard_stop_required']:
                    receipt['hard_deadline_reached_stop_pending']=not receipt.get('safe_to_exit')
                    save_receipt()
            except BaseException as exc:
                state['stop_reason']='meter_failure:'+type(exc).__name__;stop.set();cancel_owned()
                if state['app_id'] and time.time()-last_stop>=5:
                    last_stop=time.time()
                    if settle()['safe_to_exit']:return
            closed.wait(1)
    meter_thread=threading.Thread(target=meter,daemon=False)
    meter_thread.start()
    def check():
        if stop.is_set() or time.time()>=soft_deadline:
            raise TimeoutError(state['stop_reason'] or 'Independent lifecycle cutoff')
    def await_call(call):
        state['call']=call
        while True:
            check()
            try:
                value=call.get(timeout=min(10,max(.1,soft_deadline-time.time())))
                state['call']=None
                return value
            except modal.exception.FunctionTimeoutError:
                raise
            except modal.exception.TimeoutError:
                continue
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
                call=hn2_app.verify_assets.spawn(**kwargs)
                receipt['cloud_calls'].append(dict(stage='cpu_asset_admission',call_id=call.object_id));save_receipt()
                asset=await_call(call)
                if asset['receipt']['passed'] is not True:raise ValueError('CPU remote asset admission failed')
                check()
                function=hn2_app.train.with_options(gpu='H100',cpu=(resources['cpu'],resources['cpu']),
                    memory=(resources['memory_gib']*1024,resources['memory_gib']*1024),
                    timeout=resources['timeout_s'],retries=0,max_containers=1,scaledown_window=2)
                call=function.spawn(**kwargs,asset_receipt_sha256=asset['sha256'],dispatch_deadline_unix=deadline)
                receipt['cloud_calls'].append(dict(stage='cuda_admission_and_discarded_health_only',call_id=call.object_id));save_receipt()
                receipt['result']=await_call(call);receipt['status']='compute_returned';save_receipt()
    except BaseException as exc:
        receipt.update(status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)))
        cancel_owned()
        raise
    finally:
        stop.set()
        result=settle()
        if result['safe_to_exit']:
            closed.set();meter_thread.join(timeout=2)
        # If verification failed, the daemon remains a live stop observer while
        # this CLI is alive; the reservation stays open and reports the failure.
        receipt.update(elapsed_s_since_dispatch=time.time()-started,stop_reason=state['stop_reason'],
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

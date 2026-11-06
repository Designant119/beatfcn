"""I03: prepare the full fixed feature working set before consuming batches."""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import save, source_digest

REPO = training.REPO
PLAN = REPO/'bench/rhythm-v2/i03-materialization-v1.json'
INVENTORY = REPO/'reports/d12_materialization.json'
HELPER = REPO/'scripts/materialize_features.py'
SEQUENCE = REPO/'inputs/i02_audio_sequence.json'
BASELINE = REPO/'reports/i02_current.json'
ROOT = '/data/evolution/capability-evolution-20261024/materialization-v1'
app = modal.App('beatfcn-materialization-i03')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = training.image
for local, remote in [(Path(__file__), '/audit/materialization_probe_app.py'), (PLAN, '/audit/i03-plan.json'),
                       (HELPER, '/audit/materialize_features.py'), (INVENTORY, '/audit/inventory.json'),
                       (SEQUENCE, '/audit/sequence.json'), (BASELINE, '/audit/expected-payloads.json')]:
    image = image.add_local_file(str(local), remote)


@app.function(image=image, cpu=(4,4), memory=(16384,16384), volumes={'/data':data},
              timeout=300, retries=0, max_containers=1, scaledown_window=2)
def probe(expected_source, expected_script, expected_plan):
    import beatfcn
    import torch
    from beatfcn.train.data import read_npy
    from beatfcn.train.pool import SongPool
    torch.set_num_threads(4)
    sys.path.insert(0, '/audit')
    from materialize_features import materialize, local_paths
    started = time.perf_counter()
    if source_digest(Path(beatfcn.__file__).parent) != expected_source:
        raise ValueError('source changed')
    for file, sha in [('/audit/materialization_probe_app.py',expected_script),('/audit/i03-plan.json',expected_plan)]:
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
            raise ValueError('registered source changed')
    plan=json.loads(Path('/audit/i03-plan.json').read_text())
    for file,key in [('materialize_features.py','helper_sha256'),('inventory.json','inventory_sha256'),
                     ('sequence.json','sequence_sha256'),('expected-payloads.json','baseline_sha256')]:
        if hashlib.sha256((Path('/audit')/file).read_bytes()).hexdigest()!=plan[key]:
            raise ValueError('registered input changed')
    output=Path(ROOT)/'probe.json'
    if output.exists():raise ValueError('I03 exists; inspect its original handle')
    inventory=json.loads(Path('/audit/inventory.json').read_text())
    seq=json.loads(Path('/audit/sequence.json').read_text())
    expected=json.loads(Path('/audit/expected-payloads.json').read_text())['identities']
    report=dict(complete=False,source_sha256=expected_source,script_sha256=expected_script,
                plan_sha256=expected_plan,optimizer_updates=0,gpu_requested=False,phase='prepare')
    def record():
        report['function_wall_s']=time.perf_counter()-started
        save(output,report);data.commit()
    record()
    try:
        with tempfile.TemporaryDirectory(prefix='beatfcn-local-features-') as folder:
            report['materialization']=materialize(inventory,'/data',folder,
                deadline=time.perf_counter()+plan['maximum_preparation_s'],workers=4)
            report['phase']='local_replay';record()
            consumed=[];loaded=set();local_read_s=[]
            def load(sha,mel=False):
                if time.perf_counter()-started>280:raise TimeoutError('I03 function reserve')
                t=time.perf_counter()
                paths=local_paths(folder,inventory,sha,mel=mel)
                if any(str(p).startswith('/data/') for p in paths):raise ValueError('remote path in hot reader')
                arrays=[read_npy(p,chunk=256<<20) for p in paths]
                local_read_s.append(time.perf_counter()-t);loaded.add(sha)
                return dict(zip((9,16,24),arrays[:3])),arrays[3],None
            pool=SongPool(load,(9,16,24),device='cpu',capacity=40,loaders=4,mel=False)
            queue=deque();position=0;begin=time.perf_counter()
            for index in range(len(seq)):
                while len(queue)<16 and position<len(seq):
                    pool.request(seq[position]);queue.append((position,seq[position]));position+=1
                current,keys=queue.popleft()
                if current!=index:raise ValueError('sample order changed')
                for key in keys:pool.get(key)
                pool.release(keys);consumed.append(keys)
            pool.pool.shutdown(wait=True)
            report.update(local_replay_s=time.perf_counter()-begin,local_read_total_s=sum(local_read_s),
                same_sample_order=consumed==seq,consumed_batches=len(consumed),pool_stats=dict(pool.stats),
                local_only_reads=True,local_loaded_songs=len(loaded))
            # Independent numerical-payload hashes are outside the measured hot loop.
            del pool
            failures=[];verify_begin=time.perf_counter()
            for sha in sorted(loaded):
                if time.perf_counter()-started>280:raise TimeoutError('I03 verification reserve')
                identity=[]
                for p in local_paths(folder,inventory,sha):
                    array=read_npy(p,chunk=256<<20)
                    identity.append(dict(shape=list(array.shape),dtype=array.dtype.str,bytes=array.nbytes,
                        sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest()))
                    del array
                if identity!=expected.get(sha):failures.append(sha)
            report.update(payload_identity_failures=failures,payload_verify_s=time.perf_counter()-verify_begin,
                complete=True,phase='complete',limitations=plan['limitations'])
            report['gate_passed']=(not failures and report['same_sample_order'] and len(consumed)==1024
                and report['materialization']['wall_s']<=plan['maximum_preparation_s']
                and report['local_replay_s']<=plan['maximum_local_replay_s']
                and pool_wait(report)<=plan['maximum_future_wait_s'])
            record()
        return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';record();raise


def pool_wait(report):
    return report['pool_stats']['wait_s']


@app.local_entrypoint()
def main():
    from beatfcn.train.campaign_budget import reserve,update_stage
    plan=json.loads(PLAN.read_text());reg=json.loads((REPO/'reports/i03_registration.json').read_text())
    source=source_digest(REPO/'beatfcn');script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_sha=hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if (source,script,plan_sha)!=tuple(reg[k] for k in ('source_sha256','script_sha256','plan_sha256')):
        raise ValueError('I03 registration changed')
    ledger=REPO/'reports/rhythm_v2_budget.json';stage='i03-materialization'
    reserve(ledger,stage,plan['resources'],json.loads((REPO/'reports/rhythm_v2_rates.json').read_text()),
        json.loads((REPO/'reports/rhythm_v2_connection_status.json').read_text()),app_id=app.app_id,source_sha256=source)
    try:
        call=probe.spawn(source,script,plan_sha)
        update_stage(ledger,stage,status='running',function_call_id=call.object_id)
        result=call.get();save(REPO/'reports/i03_probe.json',result)
        update_stage(ledger,stage,status='complete' if result['complete'] else 'failed',
            function_wall_s=result['function_wall_s'],result_path=f'{ROOT}/probe.json',ended_at=time.time())
        print(json.dumps({k:result.get(k) for k in ['complete','gate_passed','function_wall_s','local_replay_s','pool_stats']}))
    except BaseException as exc:
        update_stage(ledger,stage,status='observation_error',error=f'{type(exc).__name__}: {exc}')
        raise

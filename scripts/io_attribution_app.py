"""D11: attribute the original CPU replay's wall time; no candidate or GPU."""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import threading
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import save, source_digest

REPO = training.REPO
PLAN = REPO / 'bench/rhythm-v2/d11-io-attribution-v1.json'
SEQUENCE = REPO / 'inputs/i02_audio_sequence.json'
ROOT = '/data/evolution/capability-evolution-20261024/io-attribution-v1'
app = modal.App('beatfcn-io-attribution-d11')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (training.image.add_local_file(__file__, '/audit/io_attribution_app.py')
         .add_local_file(str(PLAN), '/audit/d11-plan.json')
         .add_local_file(str(SEQUENCE), '/audit/i02-sequence.json'))


def cpu_state():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(process_cpu_s=usage.ru_utime + usage.ru_stime, max_rss=usage.ru_maxrss)
    for name in ('cpu.stat', 'cpu.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result[name] = path.read_text()
    return result


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), volumes={'/data': data},
              timeout=110, retries=0, max_containers=1, scaledown_window=2)
def diagnose(expected_source, expected_script, expected_plan):
    import beatfcn
    import torch
    from beatfcn.train.data import read_npy
    from beatfcn.train.pool import SongPool
    torch.set_num_threads(4)
    started = time.perf_counter()
    if source_digest(Path(beatfcn.__file__).parent) != expected_source:
        raise ValueError('source changed')
    for path, sha in [('/audit/io_attribution_app.py', expected_script), ('/audit/d11-plan.json', expected_plan)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('registered identity changed')
    plan = json.loads(Path('/audit/d11-plan.json').read_text())
    raw = Path('/audit/i02-sequence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['sequence_sha256']:
        raise ValueError('sequence changed')
    sequence = json.loads(raw)[:plan['batches']]
    output = Path(ROOT) / 'attribution.json'
    if output.exists():
        raise ValueError('D11 result exists; inspect its original handle')
    lock = threading.Lock()
    identities, loads = {}, []
    totals = defaultdict(float)
    report = dict(complete=False, source_sha256=expected_source, script_sha256=expected_script,
                  plan_sha256=expected_plan, sequence_sha256=plan['sequence_sha256'],
                  optimizer_updates=0, gpu_requested=False, progress=[], cpu_before=cpu_state())

    def load(sha, mel=False):
        begin = time.perf_counter()
        base = Path('/data/taps') / plan['taps_key'] / sha
        arrays, file_times = [], []
        for name in [f'L{k}.npy' for k in plan['layers']] + ['S20.npy']:
            t = time.perf_counter()
            arrays.append(read_npy(base / name, chunk=plan['read_chunk_bytes']))
            file_times.append(dict(name=name, read_s=time.perf_counter() - t))
        t = time.perf_counter()
        identity = [dict(shape=list(a.shape), dtype=a.dtype.str, bytes=a.nbytes,
                         sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest()) for a in arrays]
        hash_s = time.perf_counter() - t
        with lock:
            if sha in identities and identities[sha] != identity:
                raise ValueError('source changed between loads')
            identities[sha] = identity
            loads.append(dict(sha=sha, files=file_times, hash_s=hash_s, total_s=time.perf_counter() - begin))
        return dict(zip(plan['layers'], arrays[:-1])), arrays[-1], None

    pool = SongPool(load, tuple(plan['layers']), device='cpu', capacity=40, loaders=4, mel=False)

    def record():
        t = time.perf_counter()
        with lock:
            report['identities'], report['loads'] = dict(identities), list(loads)
        report['phase_s'] = dict(totals)
        report['pool_stats'] = dict(pool.stats)
        report['function_wall_s'] = time.perf_counter() - started
        save(output, report)
        totals['report_write_s'] += time.perf_counter() - t
        t = time.perf_counter()
        data.commit()
        totals['volume_commit_s'] += time.perf_counter() - t

    record()
    # Exclude the first publication from the measured replay, as I02 did.
    totals.clear()
    queue, next_index = deque(), 0
    begin = time.perf_counter()
    try:
        for index in range(len(sequence)):
            if time.perf_counter() - started > 95:
                raise TimeoutError('D11 processing reserve')
            t = time.perf_counter()
            while len(queue) < 16 and next_index < len(sequence):
                keys = sequence[next_index]
                pool.request(keys)
                queue.append((next_index, keys))
                next_index += 1
            totals['request_s'] += time.perf_counter() - t
            current, keys = queue.popleft()
            if current != index:
                raise ValueError('sequence changed')
            t = time.perf_counter()
            for key in keys:
                pool.get(key)
            totals['get_s'] += time.perf_counter() - t
            t = time.perf_counter()
            pool.release(keys)
            totals['release_s'] += time.perf_counter() - t
            t = time.perf_counter()
            time.sleep(plan['simulated_compute_s'])
            totals['actual_sleep_s'] += time.perf_counter() - t
            t = time.perf_counter()
            memory = pool.telemetry()
            totals['telemetry_s'] += time.perf_counter() - t
            report['consumed_batches'] = index + 1
            if (index + 1) % 128 == 0:
                report['progress'].append(dict(batches=index + 1, wall_s=time.perf_counter() - begin,
                                               phase_s=dict(totals), cpu=cpu_state(), memory=memory))
                record()
        t = time.perf_counter()
        pool.pool.shutdown(wait=True)
        totals['shutdown_s'] += time.perf_counter() - t
        elapsed = time.perf_counter() - begin
        report.update(complete=True, loop_wall_s=elapsed, loop_phase_s=dict(totals),
                      unattributed_s=elapsed-sum(totals.values()), cpu_after=cpu_state(),
                      expected_sleep_s=len(sequence)*plan['simulated_compute_s'],
                      limitations=plan['limitations'])
        record()
        return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        record()
        raise


@app.local_entrypoint()
def main():
    from beatfcn.train.campaign_budget import reserve, update_stage
    plan = json.loads(PLAN.read_text())
    reg = json.loads((REPO / 'reports/d11_registration.json').read_text())
    source = source_digest(REPO / 'beatfcn')
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if (source, script, plan_sha) != tuple(reg[k] for k in ('source_sha256', 'script_sha256', 'plan_sha256')):
        raise ValueError('D11 registration changed')
    ledger = REPO / 'reports/rhythm_v2_budget.json'
    stage = 'd11-io-attribution'
    reserve(ledger, stage, plan['resources'], json.loads((REPO / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((REPO / 'reports/rhythm_v2_connection_status.json').read_text()),
            app_id=app.app_id, source_sha256=source)
    try:
        call = diagnose.spawn(source, script, plan_sha)
        update_stage(ledger, stage, status='running', function_call_id=call.object_id)
        result = call.get()
        save(REPO / 'reports/d11_attribution.json', result)
        update_stage(ledger, stage, status='complete' if result['complete'] else 'failed',
                     function_wall_s=result['function_wall_s'], result_path=f'{ROOT}/attribution.json', ended_at=time.time())
        print(json.dumps({k: result.get(k) for k in ('complete', 'loop_wall_s', 'loop_phase_s', 'unattributed_s')}))
    except BaseException as exc:
        update_stage(ledger, stage, status='observation_error', error=f'{type(exc).__name__}: {exc}')
        raise

"""Recycle bounded MPS worker processes while retaining committed song caches."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.acquire_local_mert_weights import ROOT, save
from scripts.probe_local_mert_full import sha


def run(path):
    plan = json.loads(path.read_text())
    output = ROOT/plan['report']
    if output.exists():
        raise FileExistsError('Inspect active worker coordinator instead of restarting')
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    prior = json.loads((ROOT/plan['previous_report']).read_text())
    assert prior['execution_complete']
    records = copy.deepcopy(prior['records'])
    known = {row['audio_sha256'] for row in records}
    assert len(known) == len(records)
    for row in records:
        for name, identity in row['files'].items():
            file = ROOT/row['path']/name
            assert file.stat().st_size == identity['bytes'] and sha(file) == identity['sha256']
    original = json.loads((ROOT/plan['original_cache_plan']).read_text())
    directories = {p.name for p in (ROOT/original['output_root']).iterdir() if p.is_dir()}
    if directories != known:
        raise ValueError('Unreceipted original cache directory; inspect instead of overwriting')
    sources = {x['audio_sha256']:x for x in original['audios']}
    pending = [x for x in original['audios'] if x['audio_sha256'] not in known]
    root = ROOT/plan['worker_plan_root']
    root.mkdir(parents=True, exist_ok=False)
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        plan_sha256=sha(path), records=records, workers=[], planned_songs=len(original['audios']),
        committed_songs=len(records), committed_bytes=sum(sum(v['bytes'] for v in x['files'].values()) for x in records),
        previous_report=plan['previous_report'], optimizer_updates=0, new_cloud_compute_calls=0, new_cloud_spend_usd=0,
        started_at_unix=time.time(), absolute_deadline_unix=plan['absolute_deadline_unix'],
        current_worker_pid=None, max_songs_per_process=plan['songs_per_worker'])
    start = time.monotonic()

    def record():
        report.update(wall_s=time.monotonic()-start, observed_at_unix=time.time(), committed_songs=len(report['records']))
        save(output, report)

    record()
    try:
        for index, offset in enumerate(range(0, len(pending), plan['songs_per_worker'])):
            remaining = plan['absolute_deadline_unix']-time.time()
            if remaining < 30:
                raise TimeoutError('Original end-to-end cache deadline reached; committed songs retained')
            subset = pending[offset:offset+plan['songs_per_worker']]
            worker = copy.deepcopy(original)
            worker.update(id=f'D96-cache-worker-v2-{index:03}',
                report=f'reports/d96_cache_worker_v2_{index:03}.json',
                output_root=f'inputs/d96_mert2_features_v2/worker{index:03}',
                audios=subset, reuse_resource_outputs=False,
                timeout_s=min(plan['per_worker_timeout_s'], remaining),
                cache_bytes_limit=original['cache_bytes_limit']-report['committed_bytes'])
            worker['source_sha256'] = dict(plan['worker_source_sha256'])
            if index == 0:
                anchor = prior['records'][-1]
                worker['parity_reference'] = dict(record=anchor, source=sources[anchor['audio_sha256']])
                for name in ('L12.npy', 'L23.npy'):
                    worker['source_sha256'][anchor['path']+'/'+name] = anchor['files'][name]['sha256']
            worker_path = root/f'worker{index:03}.json'
            save(worker_path, worker)
            command = [original['environment']+'/bin/python', '-u', '-m', 'scripts.cache_local_mert_features',
                       '--plan', str(worker_path)]
            item = dict(index=index, plan=str(worker_path.relative_to(ROOT)), report=worker['report'],
                requested_audio=[x['audio_sha256'] for x in subset], timeout_s=worker['timeout_s'],
                started_at_unix=time.time(), complete=False)
            report['workers'].append(item)
            with (ROOT/f'reports/d96_cache_worker_v2_{index:03}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item['pid'] = child.pid
                report['current_worker_pid'] = child.pid
                record()
                try:
                    item['exit_code'] = child.wait(timeout=worker['timeout_s'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    item.update(exit_code=child.returncode, error='Per-worker deadline exceeded; no retry')
            item.update(complete=True, ended_at_unix=time.time())
            child_report = ROOT/worker['report']
            if child_report.exists():
                data = json.loads(child_report.read_text())
                for row in data['records']:
                    if row['audio_sha256'] in known or row['audio_sha256'] not in item['requested_audio']:
                        raise ValueError('Duplicate/unrequested committed song')
                    for name, identity in row['files'].items():
                        file = ROOT/row['path']/name
                        assert file.stat().st_size == identity['bytes'] and sha(file) == identity['sha256']
                    known.add(row['audio_sha256'])
                    report['records'].append(row)
                    report['committed_bytes'] += sum(f['bytes'] for f in row['files'].values())
                item['post_cleanup_memory'] = data.get('post_cleanup_memory')
                item['fresh_process_feature_parity'] = data.get('fresh_process_feature_parity')
            else:
                data = {}
            report['current_worker_pid'] = None
            record()
            if item['exit_code'] != 0 or not data.get('complete'):
                raise RuntimeError('Worker failed; stop without automatic retry: '+str(data.get('error', item)))
            assert report['committed_bytes'] <= original['cache_bytes_limit']
            print(json.dumps(dict(worker=index, committed_songs=len(known), total=len(sources))), flush=True)
        assert known == sources.keys()
        report.update(complete=True, execution_complete=True)
        record()
    except BaseException as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        record()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    run(parser.parse_args().plan)

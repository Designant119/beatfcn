"""Bounded public metadata collection, without chart downloads or cloud compute."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    identity = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['code_sha256']
    root = ROOT / plan['output_root']
    root.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / plan['report']
    if report_path.exists():
        raise FileExistsError('Use a new plan for another collection; preserve the prior receipt')
    start = time.monotonic()
    lock = threading.Lock()
    bounds = plan['bounds']
    state = dict(schema='beatfcn.public_catalog_collection.v1', plan_sha256=identity,
                 pid=os.getpid(), started_at_unix=time.time(), complete=False,
                 total_bytes=0, sources={}, training_started=False, cloud_compute_started=False)
    save(report_path, state)

    def collect(source):
        name = source['name']
        folder = root / name
        folder.mkdir(exist_ok=True)
        result = dict(pages=[], downloaded_rows=0, unique_rows=0, complete=False,
                      reported_counts=[], state='collecting')
        records = {}

        def publish():
            with lock:
                state['sources'][name] = copy.deepcopy(result)
                save(report_path, state)

        publish()
        for number in range(source['max_pages']):
            remaining = bounds['wall_s'] - (time.monotonic() - start)
            with lock:
                byte_remaining = bounds['total_bytes'] - state['total_bytes']
            if remaining <= 0 or byte_remaining < bounds['page_bytes'] * len(plan['sources']):
                result.update(state='bounded_stop', error='wall time or byte bound')
                break
            params = dict(source['params'])
            page = number + source['first_page']
            params['page'] = page
            url = source['url'] + '?' + urlencode(params)
            attempts = []
            for attempt in range(bounds.get('transport_retries', 0) + 1):
                target = folder / (f'{page:04d}.json' if attempt == 0 else f'{page:04d}.attempt{attempt + 1}.json')
                headers = target.with_suffix('.headers')
                tick = time.monotonic()
                remaining = bounds['wall_s'] - (tick - start)
                with lock:
                    byte_remaining = bounds['total_bytes'] - state['total_bytes']
                if remaining <= 0 or byte_remaining < bounds['page_bytes'] * len(plan['sources']):
                    break
                proc = subprocess.run(
                    ['curl', '-sSL', '--fail', '--connect-timeout', '10', '--max-time',
                     str(min(bounds['request_s'], max(1, int(remaining)))),
                     '--max-filesize', str(bounds['page_bytes']), '-A', 'BeatFCN-data-audit/1.0',
                     '--dump-header', str(headers), '--output', str(target), url],
                    capture_output=True, text=True, timeout=min(bounds['request_s'] + 5, remaining + 5))
                size = target.stat().st_size if target.exists() else 0
                receipt = dict(page=page, url=url, returncode=proc.returncode, bytes=size,
                               file=str(target.relative_to(ROOT)), attempt=attempt + 1,
                               wall_s=time.monotonic() - tick, received_at_unix=time.time())
                if target.exists():
                    receipt['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
                if proc.returncode:
                    receipt['error'] = proc.stderr[-500:]
                attempts.append(receipt)
                with lock:
                    state['total_bytes'] += size
                if proc.returncode not in (28, 35, 52, 56) or attempt == bounds.get('transport_retries', 0):
                    break
                time.sleep(min(3 * (attempt + 1), max(0, bounds['wall_s'] - (time.monotonic() - start))))
            if not attempts:
                result.update(state='bounded_stop', error='wall time or byte bound before request')
                break
            receipt = dict(attempts[-1], attempts=attempts)
            if proc.returncode:
                result.update(state='failed', error=receipt['error'])
                result['pages'].append(receipt)
                break
            try:
                doc = json.loads(target.read_text())
                rows = doc[source['rows_key']]
                total = int(doc[source['count_key']])
                assert isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
                assert total >= 0
                for row in rows:
                    records[str(row['id'])] = row
                result['downloaded_rows'] += len(rows)
                result['unique_rows'] = len(records)
                result['reported_counts'].append(total)
                receipt['rows'] = len(rows)
                # Both providers publish count; no inference from a short page.
                finished = (number + 1) * source['page_size'] >= total
                if not rows and not finished:
                    raise ValueError('Empty page before advertised end of catalogue')
                if finished:
                    result.update(complete=len(records) >= total,
                                  state='complete' if len(records) >= total else 'snapshot_drift',
                                  pagination_finished=True)
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                receipt['error'] = f'{type(exc).__name__}: {exc}'
                result.update(state='failed', error=receipt['error'])
            result['pages'].append(receipt)
            publish()
            if result['state'] != 'collecting':
                break
            time.sleep(min(bounds['between_pages_s'], max(0, bounds['wall_s'] - (time.monotonic() - start))))
        if result['state'] == 'collecting':
            result.update(state='bounded_stop', error='page bound reached')
        output = folder / 'catalog.jsonl'
        output.write_text(''.join(json.dumps(records[k], ensure_ascii=False) + '\n' for k in sorted(records)))
        result.update(catalog=str(output.relative_to(ROOT)),
                      catalog_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        if name == 'mania':
            charts = {str(b['id']): b for r in records.values() for b in r.get('beatmaps', [])
                      if b.get('mode_int') == 3 and b.get('cs') == 4
                      and b.get('status') in ('ranked', 'approved')}
            counts = Counter(str(int(b['difficulty_rating'])) for b in charts.values())
            result['coverage'] = dict(unique_four_key_charts=len(charts),
                                      native_star_floor_histogram=dict(sorted(counts.items())))
        else:
            import sys
            sys.path.insert(0, str(ROOT))
            from beatfcn.ingest.expansion import declared_burden, tag_of, phira_acquisition_eligible
            eligible = [r for r in records.values() if phira_acquisition_eligible(r)]
            result['coverage'] = dict(standard_reviewed_candidates=len(eligible),
                native_tags=dict(Counter(tag_of(r) for r in eligible)),
                conservative_numeric_hints=dict(Counter(declared_burden(r)[0] for r in eligible)),
                conflicting_numeric_labels=sum(declared_burden(r)[1] for r in eligible))
        result['counts_are_metadata_only'] = True
        publish()
        return result

    with ThreadPoolExecutor(max_workers=len(plan['sources'])) as pool:
        futures = [pool.submit(collect, source) for source in plan['sources']]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                state.setdefault('errors', []).append(f'{type(exc).__name__}: {exc}')
    state.update(complete=not state.get('errors') and all(r['complete'] for r in state['sources'].values()),
                 ended_at_unix=time.time(), wall_s=time.monotonic() - start)
    save(report_path, state)
    print(json.dumps(dict(complete=state['complete'], bytes=state['total_bytes'],
                         sources={k:dict(state=v['state'], unique_rows=v['unique_rows'],
                                         coverage=v.get('coverage')) for k,v in state['sources'].items()})))


if __name__ == '__main__':
    main()

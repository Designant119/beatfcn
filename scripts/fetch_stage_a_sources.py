"""Read existing Volume files locally. Never creates an App or invokes a Function.

Only native chart/metadata/canon are saved. Audio and artwork remain in the archive
buffer. Archive and source hashes are checked against the frozen development set.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'inputs/phira_stage_a_development')
    args = parser.parse_args()
    import modal
    import yaml
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    rows = [r for r in json.loads((ROOT / 'bench/capability-v1/manifest.json').read_text())['rows'] if r['mode'] == 'phira']
    args.output.mkdir(parents=True, exist_ok=True)
    lock, transferred = threading.Lock(), [0]
    limit = 512 * 1024 * 1024

    def read(path):
        buf = io.BytesIO()
        for chunk in volume.read_file(path):
            with lock:
                transferred[0] += len(chunk)
                if transferred[0] > limit:
                    raise ValueError('read-only transfer limit exceeded')
            buf.write(chunk)
        return buf.getvalue()

    def fetch(row):
        cid = str(row['native_id'])
        meta_path = args.output / f'{cid}.source.json'
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if (hashlib.sha256((args.output / f'{cid}.chart.json').read_bytes()).hexdigest() == old['source_sha256']
                    and hashlib.sha256((args.output / f'{cid}.canon.json').read_bytes()).hexdigest() == old['canon_sha256']
                    and json.loads((args.output / f'{cid}.info.json').read_text()).get('offset', 0) == old['info_offset_s']
                    and old['audio_sha256'] == row['audio_sha256']):
                return old
        canon_bytes = read(f'canon/phira/{cid}.json')
        canon = json.loads(canon_bytes)
        if canon['audio_sha256'] != row['audio_sha256']:
            raise ValueError('canonical audio differs from frozen development manifest')
        archive = read(f'raw/phira/zips/{cid}.zip')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names = {n.lower().rsplit('/', 1)[-1]: n for n in z.namelist()}
            info_name = names.get('info.yml') or names.get('info.yaml')
            info = yaml.safe_load(z.read(info_name).decode('utf-8-sig'))
            member = names[str(info['chart']).lower().rsplit('/', 1)[-1]]
            if z.getinfo(member).file_size > 64 * 1024 * 1024:
                raise ValueError('native chart exceeds local preparation bound')
            raw = z.read(member)
            if hashlib.sha256(raw).hexdigest() != canon['source_sha256']:
                raise ValueError('archive chart differs from canonical source')
            music = names[str(info['music']).lower().rsplit('/', 1)[-1]]
            if z.getinfo(music).file_size > 128 * 1024 * 1024:
                raise ValueError('audio identity check exceeds local bound')
            if hashlib.sha256(z.read(music)).hexdigest() != row['audio_sha256']:
                raise ValueError('archive audio differs from frozen development manifest')
        meta = {'chart_id': row['chart_id'], 'source_sha256': canon['source_sha256'],
                'canon_sha256': hashlib.sha256(canon_bytes).hexdigest(), 'audio_sha256': row['audio_sha256'],
                'archive_sha256': hashlib.sha256(archive).hexdigest(), 'archive_bytes': len(archive),
                'volume': 'beatfcn-data', 'remote_archive': f'raw/phira/zips/{cid}.zip',
                'chart_member': member, 'info_offset_s': info.get('offset', 0),
                'remote_compute_started': False}
        (args.output / f'{cid}.chart.json').write_bytes(raw)
        (args.output / f'{cid}.canon.json').write_bytes(canon_bytes)
        (args.output / f'{cid}.info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(fetch, row): row for row in rows}
        for future in as_completed(pending):
            row = pending[future]
            try:
                record = future.result()
                records.append(record)
                print(json.dumps({'chart_id': row['chart_id'], 'status': 'verified', 'bytes': record['archive_bytes']}), flush=True)
            except Exception as exc:
                failures.append({'chart_id': row['chart_id'], 'error': f'{type(exc).__name__}:{exc}'})
                print(json.dumps(failures[-1]), flush=True)
    previous_path = args.output / 'manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    result = {'schema': 'stage_a_readonly_sources_v1', 'expected': len(rows), 'verified': len(records),
              'transferred_bytes': transferred[0], 'transfer_limit_bytes': limit,
              'total_transferred_bytes': transferred[0] + previous.get('total_transferred_bytes', previous.get('transferred_bytes', 0)),
              'remote_compute_started': False, 'records': sorted(records, key=lambda r: r['chart_id']), 'failures': failures}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    if failures:
        raise SystemExit('source fetch incomplete; failures recorded, no compute started')


if __name__ == '__main__':
    main()

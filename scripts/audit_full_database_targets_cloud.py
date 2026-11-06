"""Audit immutable target batch markers against the exact parsed manifest."""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/target-audit-v1.json'
REMOTE_BATCHES = 'full-database-v1/batches'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def read_json(volume, remote):
    payload = b''.join([chunk async for chunk in volume.read_file.aio(remote)])
    return json.loads(payload)


async def main():
    if DEST.exists():
        raise FileExistsError('Target audit already exists')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    parsed = [row for row in rows if row.get('status') == 'parsed']
    expected = [row['chart_id'] for row in parsed]
    expected_batches = (len(expected) + 255) // 256
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    entries = await volume.listdir.aio(REMOTE_BATCHES)
    markers = await asyncio.gather(*(read_json(volume, entry.path) for entry in entries
                                     if entry.path.endswith('.json')))
    if len(markers) != expected_batches:
        raise ValueError(f'Target markers {len(markers)} != expected {expected_batches}')
    markers.sort(key=lambda value: value['batch'])
    if [value['batch'] for value in markers] != list(range(expected_batches)):
        raise ValueError('Noncontiguous or duplicate target batch numbers')
    manifest_sha = digest(MANIFEST)
    planned = []
    success = []
    failures = []
    statistics = Counter()
    types_before = Counter()
    types_after = Counter()
    split_audio = {}
    by_split = Counter()
    parsed_by_id = {row['chart_id']: row for row in parsed}
    for marker in markers:
        if marker.get('manifest_sha256') != manifest_sha:
            raise ValueError('Target batch belongs to a different manifest')
        planned.extend(marker['planned'])
        for row in marker['successes']:
            success.append(row['chart_id'])
            stats = row['statistics']
            statistics['native_events'] += stats['native_events']
            statistics['filtered_events'] += stats['filtered_events']
            statistics['removed_drag'] += stats['removed_drag']
            statistics['qualifying_windows'] += stats['qualifying_windows']
            statistics['clock_verified'] += bool(stats['clock_verified'])
            for key, count in stats['types_before'].items():types_before[key] += count
            for key, count in stats['types_after'].items():types_after[key] += count
            split_audio[row['chart_id']] = row
            by_split[parsed_by_id[row['chart_id']]['split']] += 1
        failures.extend(marker['permanent_failures'])
    missing = sorted(set(expected) - set(success))
    unexpected = sorted(set(success) - set(expected))
    duplicate = [chart for chart, count in Counter(success).items() if count > 1]
    result = dict(schema='beatfcn.full-database-target-audit.v1',
        manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=manifest_sha,
        batches=len(markers), expected_batches=expected_batches,
        expected_parsed=len(expected), success=len(success), permanent_failures=failures,
        missing=missing, unexpected=unexpected, duplicate_chart_ids=duplicate,
        statistics=dict(statistics), types_before=dict(types_before), types_after=dict(types_after),
        by_split=dict(by_split), cloud_compute_calls=0,
        limitations=['Remote batch markers are checked; target arrays are not re-downloaded in this audit.',
            'Target receipt files include canonical SHA and array hashes at write time.',
            'A later content/hash repair pass must verify uploaded arrays before model training.'])
    result['complete'] = bool(not missing and not unexpected and not duplicate and not failures
        and len(success) == len(expected)
        and statistics['native_events'] == statistics['filtered_events'] + statistics['removed_drag']
        and statistics['clock_verified'] == len(success))
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('missing', 'unexpected', 'permanent_failures')}, ensure_ascii=False, indent=2))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())

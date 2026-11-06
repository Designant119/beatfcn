"""Read-only two-chart probe of cloud catalogue rows through target preparation."""
import asyncio
import json
from pathlib import Path
import tempfile

import modal

from beatfcn.train.full_database import prepare_chart


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/target-sample-probe-v1.json'


async def read_json_from_volume(volume, remote):
    payload = b''.join([chunk async for chunk in volume.read_file.aio(remote)])
    return json.loads(payload)


async def download(volume, remote, output):
    payload = b''.join([chunk async for chunk in volume.read_file.aio(remote)])
    output.write_bytes(payload)
    return output


async def main():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    choices = {}
    for row in rows:
        if row.get('kept') and row.get('status') == 'parsed' and row['source'] not in choices:
            if row['source'] == 'osu' or row.get('mode') == 'phira':
                choices[row['source']] = row
    if set(choices) != {'osu', 'phira'}:
        raise ValueError('Could not select one kept row per source')
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    records = []
    with tempfile.TemporaryDirectory(prefix='beatfcn-full-db-probe-') as temporary:
        temporary = Path(temporary)
        for source, row in choices.items():
            remote = f"canon/{ 'osu' if source == 'osu' else 'phira' }/{row['native_id']}.json"
            local = await download(volume, remote, temporary / f"{row['native_id']}.json")
            target_root = temporary / 'targets'
            result = prepare_chart(row, local, target_root)
            mel_remote = f"mel10/{row['audio_sha256']}.json"
            mel = await read_json_from_volume(volume, mel_remote)
            expected_frames = int(row['audio_sec'] * 100 + .999999)
            records.append(dict(chart_id=row['chart_id'], source=source, split=row['split'],
                canonical_remote=remote, mel_metadata_remote=mel_remote,
                target_statistics=result['statistics'], mel_metadata=mel,
                mel_frames_match=mel['frames'] == expected_frames,
                expected_frames=expected_frames,
                receipt_files=result['files'], cloud_compute_calls=0))
    report = dict(schema='beatfcn.full-database-target-sample-probe-v1', readonly=True,
        manifest=MANIFEST.relative_to(ROOT).as_posix(), records=records,
        complete=len(records) == 2, cloud_compute_calls=0,
        limitations=['Two catalogue examples only; not a corpus conversion or quality result.',
            'Cloud bytes were read to the local process but no Modal compute function was called.',
            'Raw OSU/Phira archive contents still require bulk hash/zip-member verification.'])
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())

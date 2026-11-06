"""Build exact unique-audio MERT jobs from the full catalogue and volume listings."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
MEL = ROOT / 'inputs/full-database-cloud-v1/mel-metadata.jsonl'
REPORTS = ROOT / 'reports/full-database-cloud-v1'
DEST = ROOT / 'inputs/full-database-cloud-v1/mert-jobs.jsonl'
RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-jobs-receipt-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_names(name, *, stem=False):
    value = json.loads((REPORTS / name).read_text())
    records = value if isinstance(value, list) else value['records']
    if not (isinstance(value, list) or value.get('complete')):
        raise ValueError('Incomplete inventory: ' + name)
    if stem:
        return {Path(record['path']).with_suffix('').name for record in records}
    return {Path(record['path']).name for record in records}


def main():
    if DEST.exists():
        raise FileExistsError('MERT job output already exists')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    mel = {json.loads(line)['audio_sha256']: json.loads(line) for line in MEL.read_text().splitlines()}
    osu_files = load_names('raw-osu-audio.json')
    phira_files = load_names('raw-phira-zips.json')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['audio_sha256']].append(row)
    jobs = []
    missing = []
    for audio, values in sorted(grouped.items()):
        parsed = [row for row in values if row.get('status') == 'parsed']
        if not parsed:
            continue
        splits = {row['split'] for row in values}
        if len(splits) != 1:
            missing.append(dict(audio_sha256=audio, reason='split_conflict', splits=sorted(splits)))
            continue
        representative = parsed[0]
        candidates = []
        if representative['source'] == 'osu':
            path = f"raw/osu/audio/{audio}.{representative['audio_ext']}"
            if Path(path).name not in osu_files:
                missing.append(dict(audio_sha256=audio, reason='raw_audio_missing', path=path))
                continue
            candidates.append(dict(kind='audio', path=path))
        else:
            ids = sorted({row['native_id'] for row in parsed if row['source'] == 'phira'})
            paths = [f'raw/phira/zips/{native}.zip' for native in ids]
            if any(Path(path).name not in phira_files for path in paths):
                missing.append(dict(audio_sha256=audio, reason='raw_archive_missing', paths=paths))
                continue
            candidates.extend(dict(kind='zip', path=path) for path in paths)
        if audio not in mel:
            missing.append(dict(audio_sha256=audio, reason='mel_receipt_missing'))
            continue
        jobs.append(dict(audio_sha256=audio, split=representative['split'],
            source=representative['source'], audio_sec=representative['audio_sec'],
            expected_mel_frames=mel[audio]['frames'],
            chart_ids=sorted(row['chart_id'] for row in parsed),
            parsed_chart_count=len(parsed),
            kept_chart_count=sum(bool(row.get('kept')) for row in parsed),
            candidates=candidates))
    lines = [json.dumps(job, ensure_ascii=False) for job in jobs]
    DEST.write_text('\n'.join(lines) + '\n')
    report = dict(schema='beatfcn.full-database-mert-jobs.v1', cloud_compute_calls=0,
        output=str(DEST), output_sha256=digest(DEST), jobs=len(jobs),
        unique_audio=len(jobs), missing=missing,
        by_source=dict(
            osu=sum(job['source'] == 'osu' for job in jobs),
            phira=sum(job['source'] == 'phira' for job in jobs)),
        by_split=dict(
            train=sum(job['split'] == 'train' for job in jobs),
            val=sum(job['split'] == 'val' for job in jobs),
            test=sum(job['split'] == 'test' for job in jobs)),
        manifest_sha256=digest(MANIFEST), mel_metadata_sha256=digest(MEL),
        limitations=['Catalogue/listing jobs only; raw content hashes are verified separately.',
            'A job does not authorize a paid Modal dispatch or establish MERT throughput.',
            'All parseable charts are represented; kept counts remain admission metadata.'])
    report['complete'] = not missing and len(jobs) == 10898
    RECEIPT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

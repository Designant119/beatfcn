"""Bind full mel receipts to exact catalogue audio and verify their clock domain."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
METADATA = ROOT / 'inputs/full-database-cloud-v1/mel-metadata.jsonl'
DOWNLOAD = ROOT / 'reports/full-database-cloud-v1/mel-metadata-receipt.json'
DEST = ROOT / 'reports/full-database-cloud-v1/mel-metadata-audit-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    download = json.loads(DOWNLOAD.read_text())
    if not download.get('complete'):
        raise ValueError('Mel metadata download is incomplete')
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    metadata = {}
    for line in METADATA.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value['audio_sha256'] in metadata:
            raise ValueError('Duplicate mel metadata audio')
        metadata[value['audio_sha256']] = value
    groups = defaultdict(list)
    for row in rows:
        groups[row['audio_sha256']].append(row)
    scopes = {}
    differences = []
    missing = {}
    for scope, predicate in (
        ('all', lambda values: True),
        ('parsed', lambda values: any(row.get('status') == 'parsed' for row in values)),
        ('kept', lambda values: any(row.get('status') == 'parsed' and row.get('kept') for row in values))):
        selected = {audio: values for audio, values in groups.items() if predicate(values)}
        absent = sorted(set(selected) - set(metadata))
        exact_duration = 0
        in_clock_domain = 0
        absent_duration = 0
        for audio, values in selected.items():
            if audio not in metadata:
                continue
            receipt = metadata[audio]
            catalogue = next((row['audio_sec'] for row in values if row.get('audio_sec') is not None), None)
            if catalogue is None:
                absent_duration += 1
                continue
            catalogue = float(catalogue)
            if receipt['audio_sec'] == catalogue:
                exact_duration += 1
            # frames = 100 * ceil(decoded_samples/960). The catalogue duration is
            # the source-codec duration, at or below the next 40ms decode ceiling.
            ceiling = receipt['frames'] / 100
            difference = float(catalogue) - ceiling
            in_clock_domain += -1e-9 <= difference <= 0.040000001
            if scope == 'parsed':
                differences.append(dict(audio_sha256=audio, catalogue_audio_sec=catalogue,
                    decode_ceiling_sec=ceiling, difference_s=difference))
        scopes[scope] = dict(unique_audio=len(selected), metadata_present=len(selected) - len(absent),
            metadata_missing=len(absent), exact_catalogue_duration=exact_duration,
            within_40ms_decode_ceiling=in_clock_domain,
            rows_without_catalogue_duration=absent_duration)
        missing[scope] = absent
    catalogue_audio = set(groups)
    report = dict(schema='beatfcn.full-database-mel-metadata-audit-v1',
        manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=digest(MANIFEST),
        metadata=str(METADATA.relative_to(ROOT)), metadata_sha256=digest(METADATA),
        download_receipt_sha256=digest(DOWNLOAD), metadata_records=len(metadata),
        scopes=scopes, metadata_not_in_catalogue=sorted(set(metadata) - catalogue_audio),
        catalogue_without_metadata=sorted(catalogue_audio - set(metadata)),
        parsed_difference_summary=dict(
            minimum=min(row['difference_s'] for row in differences),
            maximum=max(row['difference_s'] for row in differences),
            negative=sum(row['difference_s'] < -1e-9 for row in differences),
            over_40ms=sum(row['difference_s'] > .040000001 for row in differences)),
        codec_counts=dict(Counter(value['codec'] for value in metadata.values())),
        frames_total=sum(value['frames'] for value in metadata.values()),
        all_frames_divisible_by_four=all(value['frames'] % 4 == 0 for value in metadata.values()),
        cloud_compute_calls=0,
        limitations=['JSON receipts only; .npy array bytes and numerical values were not read.',
            'The 40ms ceiling checks the physical clock domain, not mel numerical correctness.',
            'Metadata outside the catalogue is retained but not treated as training data.'])
    report['complete'] = (scopes['parsed']['metadata_missing'] == 0
        and scopes['kept']['metadata_missing'] == 0
        and scopes['parsed']['within_40ms_decode_ceiling'] == scopes['parsed']['unique_audio']
        and scopes['kept']['within_40ms_decode_ceiling'] == scopes['kept']['unique_audio']
        and report['all_frames_divisible_by_four'])
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
        if key not in ('metadata_not_in_catalogue', 'catalogue_without_metadata')},
        ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

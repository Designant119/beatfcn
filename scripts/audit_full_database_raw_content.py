"""Verify raw-content receipt coverage and identities without any compute call."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
REPORT = ROOT / 'reports/full-database-cloud-v1/raw-content-receipt.json'
CONTENT = ROOT / 'inputs/full-database-cloud-v1/raw-content.jsonl'
DEST = ROOT / 'reports/full-database-cloud-v1/raw-content-audit-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads(REPORT.read_text())
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    parsed = [row for row in rows if row.get('status') == 'parsed']
    expected_audio = {row['audio_sha256'] for row in parsed}
    expected_osu_archives = {row['audio_sha256'] + '.' + row['audio_ext']
                             for row in parsed if row['source'] == 'osu'}
    expected_phira_archives = {row['native_id'] + '.zip'
                               for row in parsed if row['source'] == 'phira'}
    values = [json.loads(line) for line in CONTENT.read_text().splitlines() if line.strip()]
    if len({value['audio_sha256'] for value in values}) != len(values):
        raise ValueError('Duplicate audio hash in raw-content output')
    audio_map = {value['audio_sha256']: value for value in values}
    split_by_audio = {row['audio_sha256']: row['split'] for row in rows}
    for value in values:
        value['split'] = split_by_audio[value['audio_sha256']]
    missing = sorted(expected_audio - set(audio_map))
    unexpected = sorted(set(audio_map) - expected_audio)
    bad = [value for value in values if not value.get('verified')]
    bad_split = sorted({row['split'] for row in rows if row['audio_sha256'] in unexpected})
    osu_archives = {Path(value['archive']).name for value in values if value['source'] == 'osu'}
    phira_archives = {Path(value['archive']).name for value in values if value['source'] == 'phira'}
    split_conflicts = []
    for key in ('audio_sha256', 'song_group'):
        groups = {}
        for row in rows:
            groups.setdefault(row[key], set()).add(row['split'])
        conflicts = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
        split_conflicts.append(dict(key=key, groups=len(conflicts), examples=list(conflicts.items())[:10]))
    result = dict(schema='beatfcn.full-database-raw-content-audit.v1',
        source_report=str(REPORT.relative_to(ROOT)), source_report_sha256=digest(REPORT),
        content=str(CONTENT.relative_to(ROOT)), content_sha256=digest(CONTENT),
        expected=dict(parsed_rows=len(parsed), unique_audio=len(expected_audio),
            osu_archives=len(expected_osu_archives), phira_archives=len(expected_phira_archives)),
        observed=dict(records=len(values), verified=sum(bool(value['verified']) for value in values),
            bad_records=len(bad), osu_archives=len(osu_archives), phira_archives=len(phira_archives)),
        missing_audio=missing, unexpected_audio=unexpected,
        unexpected_audio_splits=bad_split,
        split_conflicts=split_conflicts,
        by_source=dict(Counter(value['source'] for value in values)),
        by_split=dict(Counter(value.get('split', 'unknown') for value in values)),
        source_receipt_complete=report.get('complete'), cloud_compute_calls=0)
    result['complete'] = bool(result['source_receipt_complete']
        and not missing and not unexpected and not bad
        and all(group['groups'] == 0 for group in split_conflicts)
        and osu_archives == expected_osu_archives
        and phira_archives == expected_phira_archives)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('unexpected_audio', 'missing_audio', 'split_conflicts')},
        ensure_ascii=False, indent=2))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

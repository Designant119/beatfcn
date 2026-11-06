"""Catalogue-to-cloud readiness audit without claiming content hashes verified."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'inputs/full-database-cloud-v1'
REPORTS = ROOT / 'reports/full-database-cloud-v1'
MANIFEST = BASE / 'manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
DEST = REPORTS / 'readiness-audit-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inventory(name, legacy=False):
    value = json.loads((REPORTS / name).read_text())
    records = value if isinstance(value, list) else value['records']
    if not (isinstance(value, list) or value.get('complete')):
        raise ValueError(f'Incomplete inventory: {name}')
    return records


def grouping(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len({row['chart_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate chart_id in full catalogue')
    canon_records = [*load_inventory('canon-osu.json'), *load_inventory('canon-phira.json')]
    canonical = {(Path(record['path']).parts[1], Path(record['path']).stem): record
                 for record in canon_records}
    osu_raw = {Path(record['path']).with_suffix('').name: record
               for record in load_inventory('raw-osu-audio.json', legacy=True)}
    phira_archives = {Path(record['path']).stem: record
                      for record in load_inventory('raw-phira-zips.json', legacy=True)}
    mel_records = load_inventory('mel10.json')
    mel_suffixes = defaultdict(set)
    for record in mel_records:
        mel_suffixes[Path(record['path']).stem].add(Path(record['path']).suffix)
    mel_pairs = {key for key, value in mel_suffixes.items() if value == {'.json', '.npy'}}

    statuses = Counter()
    reasons = Counter()
    by_source_split = defaultdict(Counter)
    missing = []
    for row in rows:
        source = row['source']
        native = row['native_id']
        audio = row['audio_sha256']
        has_canonical = (source, native) in canonical
        has_raw = audio in osu_raw if source == 'osu' else native in phira_archives
        has_mel = audio in mel_pairs
        parsed = row.get('status') == 'parsed'
        kept = bool(row.get('kept'))
        statuses['rows'] += 1
        statuses['parsed'] += parsed
        statuses['kept'] += kept
        statuses['kept_parsed'] += kept and parsed
        statuses['canonical_present'] += has_canonical
        statuses['raw_archive_present'] += has_raw
        statuses['mel_pair_present'] += has_mel
        statuses['parsed_all_inputs'] += parsed and has_canonical and has_raw and has_mel
        statuses['kept_all_inputs'] += kept and parsed and has_canonical and has_raw and has_mel
        if parsed and not (has_canonical and has_raw and has_mel):
            missing.append(dict(chart_id=row['chart_id'], source=source, split=row.get('split'),
                canonical=has_canonical, raw_archive=has_raw, mel_pair=has_mel))
        for reason in row.get('quarantine_reasons') or []:
            reasons[reason] += 1
        by_source_split[(source, row.get('split'))]['rows'] += 1
        by_source_split[(source, row.get('split'))]['parsed'] += parsed
        by_source_split[(source, row.get('split'))]['kept'] += kept
        by_source_split[(source, row.get('split'))]['ready'] += kept and parsed and has_canonical and has_raw and has_mel

    split_by_audio = grouping(rows, 'audio_sha256')
    split_by_song = grouping(rows, 'song_group')
    audio_multi_split = {key: sorted({row['split'] for row in values}, key=str)
                         for key, values in split_by_audio.items() if len({row['split'] for row in values}) > 1}
    song_multi_split = {key: sorted({row['split'] for row in values}, key=str)
                        for key, values in split_by_song.items() if len({row['split'] for row in values}) > 1}

    used_canonical = {(row['source'], row['native_id']) for row in rows}
    used_osu_audio = {row['audio_sha256'] for row in rows if row['source'] == 'osu'}
    used_phira_archives = {row['native_id'] for row in rows if row['source'] == 'phira'}
    used_audio = {row['audio_sha256'] for row in rows}
    report = dict(
        schema='beatfcn.full-database-readiness-audit-v1',
        manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=digest(MANIFEST),
        inventory_sha256={name: digest(REPORTS / name) for name in (
            'canon-osu.json', 'canon-phira.json', 'raw-osu-audio.json',
            'raw-phira-zips.json', 'mel10.json')},
        counts=dict(statuses),
        quarantine_reasons=dict(reasons),
        by_source_split={f'{source}|{split}': dict(values)
            for (source, split), values in sorted(by_source_split.items(), key=lambda item: str(item[0]))},
        unique_audio=len(split_by_audio), unique_song_groups=len(split_by_song),
        audio_with_kept=sum(any(row.get('kept') for row in values) for values in split_by_audio.values()),
        audio_all_not_kept=sum(not any(row.get('kept') for row in values) for values in split_by_audio.values()),
        split_isolation=dict(audio_multi_split=len(audio_multi_split), song_multi_split=len(song_multi_split)),
        split_isolation_examples=dict(audio=list(audio_multi_split.items())[:20],
            song=list(song_multi_split.items())[:20]),
        missing_parsed_inputs=missing,
        inventory_extras=dict(
            canonical_not_in_catalogue=len(set(canonical) - used_canonical),
            osu_audio_not_in_catalogue=len(set(osu_raw) - used_osu_audio),
            phira_archives_not_in_catalogue=len(set(phira_archives) - used_phira_archives),
            mel_pairs_not_in_catalogue=len(mel_pairs - used_audio)),
        ready_definition='kept+parsed+canonical listing+raw archive listing+mel json/npy pair+offset decision v2; not a content hash or playability proof.',
        content_verification=dict(
            canonical_hashes=False, raw_audio_hashes=False,
            phira_zip_member_hashes=False, mel_array_values=False,
            note='Filename/listing matches only. Content receipts require the cloud preparation worker.'),
        cloud_compute_calls=0,
        limitations=[
            'Readonly listing audit; no file content, canonical SHA, raw audio SHA, ZIP member hash or mel values were read.',
            'Phira audio is matched by native package filename, so package version/member mismatches remain possible.',
            'Kept is catalogue admission, not product playability or model readiness.',
            'Unsupported formats intentionally remain in the denominator and cannot be target-materialized by the current parser.'])
    report['complete'] = bool(statuses['rows'] == 24048 and statuses['kept_all_inputs'] == statuses['kept_parsed']
        and not missing and not audio_multi_split and not song_multi_split)
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
        if key not in ('inventory_sha256', 'split_isolation_examples')}, ensure_ascii=False, indent=2))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

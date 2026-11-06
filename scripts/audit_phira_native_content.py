"""Read-only native-object accounting; suspicious fields require review, never note deletion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.phira import parse_phira

VERSION = 'phira-native-object-accounting-isfake-only-v2'
SOURCE_ROOTS = (
    'inputs/data_expansion_v1/corpus/raw/phira/zips',
    'inputs/d34_plain_packages/raw',
    'inputs/d56_phira_085/raw',
    'inputs/phira_spatial_20261024',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_chart(path):
    with zipfile.ZipFile(path) as archive:
        names = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            key = member.filename.lower().rsplit('/', 1)[-1]
            if key in names:
                raise ValueError(f'ambiguous archive basename: {key}')
            names[key] = member.filename
        info_name = names.get('info.yml') or names.get('info.yaml')
        if info_name is None:
            raise ValueError('missing info.yml')
        info_bytes = archive.read(info_name)
        info = yaml.safe_load(info_bytes.decode('utf-8-sig'))
        if not isinstance(info, dict):
            raise ValueError('info must be a mapping')
        chart_name = names[str(info.get('chart', '')).lower().rsplit('/', 1)[-1]]
        chart_bytes = archive.read(chart_name)
    return info, chart_bytes, dict(info_sha256=digest(info_bytes),
        source_sha256=digest(chart_bytes), chart_member=chart_name)


def inspect_chart(raw, info_offset=None):
    """Use existing parser semantics; static anomalies are evidence, not rendering claims."""
    doc = json.loads(raw.decode('utf-8-sig'))
    chart = parse_phira(raw, info_offset_sec=info_offset)
    is_rpe = chart.source_format == 'phira_rpe'
    playable = {event.event_id for event in chart.events}
    fakes = {event.event_id for event in chart.fakes}
    raw_ids, expected_fakes, nonfake_ids, unknown_fake_ids = set(), set(), set(), set()
    signals = defaultdict(list)
    counts = Counter()
    for line_index, line in enumerate(doc['judgeLineList']):
        if not isinstance(line, dict):
            continue  # Existing parser records bad_line as a quarantine issue.
        sides = [('notes', '')] if is_rpe else [('notesAbove', 'above'), ('notesBelow', 'below')]
        line_ids = set()
        counts['lines'] += 1
        for key, side in sides:
            for index, note in enumerate(line.get(key) or []):
                event_id = (f'rpe:{line_index}:{index}' if is_rpe
                            else f'pgr:{line_index}:{side}:{index}')
                raw_ids.add(event_id)
                line_ids.add(event_id)
                if not isinstance(note, dict):
                    continue
                fake = note.get('isFake', 0) if is_rpe else 0
                if type(fake) not in (int, bool) or fake not in (0, 1):
                    unknown_fake_ids.add(event_id)
                    continue
                if fake:
                    expected_fakes.add(event_id)
                    continue
                nonfake_ids.add(event_id)
                for field in ('alpha', 'speed', 'size', 'visibleTime'):
                    if field not in note:
                        continue
                    value = note[field]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        signals['invalid_' + field].append(event_id)
                    elif value <= 0 and (field != 'visibleTime' or value == 0):
                        signals['nonpositive_' + field].append(event_id)
                # A moving/hidden judgement line alone is not a note exclusion.
                if is_rpe and line.get('attachUI') not in (None, ''):
                    signals['nonfake_on_attached_ui'].append(event_id)
        if not line_ids:
            counts['no_note_lines'] += 1
        elif not line_ids & nonfake_ids and line_ids <= expected_fakes:
            counts['fake_only_lines'] += 1
        if line.get('attachUI') not in (None, ''):
            counts['attached_ui_lines'] += 1
    checks = dict(
        all_objects_accounted=raw_ids == playable | fakes,
        explicit_fakes_separated=expected_fakes == fakes and not expected_fakes & playable,
        nonfake_objects_preserved=nonfake_ids == playable,
        disjoint_outputs=not playable & fakes,
        unambiguous_fake_flags=not unknown_fake_ids,
    )
    # Exact duplicate objects are preserved by DATA_RULES; they are not dirty by themselves.
    issues = [issue.to_json() for issue in chart.issues
              if issue.severity == 'quarantine' and issue.code != 'duplicate_object']
    # Visual properties do not establish playability. Only the native isFake
    # flag separates targets from fakes; these observations cannot block a chart.
    status = ('parse_or_accounting_quarantine' if issues or not all(checks.values())
              else 'static_screen_only')
    counts.update(raw_objects=len(raw_ids), canonical_events=len(playable),
                  canonical_fakes=len(fakes), explicit_fakes=len(expected_fakes))
    return dict(status=status, source_format=chart.source_format, counts=dict(counts),
        checks=checks, canonical_types=dict(Counter(event.type for event in chart.events)),
        fake_types=dict(Counter(event.type for event in chart.fakes)),
        separated_fake_ids=sorted(fakes), raw_ids_missing_from_outputs=sorted(raw_ids - playable - fakes),
        unknown_fake_ids=sorted(unknown_fake_ids), review_signals={},
        visual_observations=dict(signals), note_selection_policy='explicit_isFake_only', parser_issues=issues,
        native_playability_verified=False, training_ready=False, removed_nonfake_events=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Choose a new report; evidence is immutable.')
    packages = defaultdict(list)
    for folder in SOURCE_ROOTS:
        for path in sorted((ROOT / folder).glob('*.zip')):
            packages[digest(path.read_bytes())].append(str(path.relative_to(ROOT)))
    rows = []
    for package_sha, paths in packages.items():
        row = dict(archive_sha256=package_sha, paths=paths,
            native_ids=sorted({Path(path).stem for path in paths}), training_ready=False)
        try:
            info, raw, identity = read_chart(ROOT / paths[0])
            row.update(identity)
            row.update(inspect_chart(raw, info.get('offset')))
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, zipfile.BadZipFile) as exc:
            row.update(status='unsupported_or_invalid_package',
                       error=f'{type(exc).__name__}: {exc}'[:250])
        rows.append(row)
    totals = Counter()
    for row in rows:
        totals.update(row.get('counts', {}))
    record = dict(id='D60', schema=VERSION, complete=True,
        source_sha256={name: digest((ROOT / name).read_bytes()) for name in (
            'scripts/audit_phira_native_content.py', 'beatfcn/events/phira.py',
            'beatfcn/events/canonical.py')},
        summary=dict(archive_files=sum(map(len, packages.values())), unique_archive_bytes=len(rows),
            unique_native_ids=len({i for row in rows for i in row['native_ids']}),
            status_counts=dict(Counter(row['status'] for row in rows)), object_counts=dict(totals),
            all_accounting_checks_pass=sum(all(row.get('checks', {}).values()) for row in rows if row.get('checks'))),
        records=rows, optimizer_updates=0, new_cloud_spend_usd=0,
        scope='Local packages only; package ID is not a verified join to current catalog or stored canonical bytes.',
        limitations=['Static fields do not prove native playability or special-gameplay absence.',
                    'Transparency, movement, size, visibility and attached UI never imply fake notes or block admission.',
                    'No animation or drag-density heuristic removes notes; all original bytes remain unchanged.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()

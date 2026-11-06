"""Local structural admission audit of pinned Phigros sources; no clock repair."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from beatfcn.events.canonical import UnsupportedFormat, build_model_view
from beatfcn.events.controls import raw_controls
from beatfcn.events.phira import parse_phira
from beatfcn.ingest import pipeline
from beatfcn.ingest.coverage import native_tier, policy_identity
from beatfcn.ingest.expansion import incremental_splits


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rhythm_hash(events):
    # Sort without a set: equal rhythm notes at different positions still count
    # as multiple playable notes. Spatial variants may share the complete hash.
    values = sorted((e.onset_us, e.end_us, e.type) for e in events)
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def audio_metadata(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name,sample_rate,channels,duration:format=duration',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=10)
    doc = json.loads(result.stdout)
    streams = doc.get('streams', [])
    if len(streams) != 1:
        raise ValueError('expected exactly one selected audio stream')
    stream = streams[0]
    duration = float(stream.get('duration') or doc.get('format', {}).get('duration'))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid metadata duration')
    return dict(codec=stream['codec_name'], sample_rate=int(stream['sample_rate']),
                channels=int(stream['channels']), audio_sec=duration,
                duration_basis='container metadata; decoded sample count still pending')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    start = time.monotonic()
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'input identity changed: {name}')
    receipt = json.loads((ROOT / plan['receipt']).read_text())
    raw = (ROOT / plan['raw_root']).resolve()
    source_files = {item['path']: item for item in receipt['files']}
    catalog = [json.loads(line) for line in (ROOT / plan['catalog']).read_text().splitlines()]
    old = [json.loads(line) for name in plan['frozen_manifests']
           for line in (ROOT / name).read_text().splitlines()]
    policy = json.loads((ROOT / plan['difficulty_policy']).read_text())
    out = ROOT / plan['output_root']
    if (out / 'manifest/candidates.jsonl').exists():
        raise FileExistsError('completed admission audit already exists; do not overwrite')
    report = dict(schema='beatfcn.phigros_structural_admission.v1', complete=False,
        plan_sha256=sha(args.plan), sources_verified=0, processed=0, records=[], audio={},
        optimizer_updates=0, new_paid_compute_usd=0, training_ready=False,
        limitations=plan['limitations'])

    def save():
        report['wall_s'] = time.monotonic() - start
        write(ROOT / plan['result'], report)

    def deadline():
        if time.monotonic() - start > plan['audit_deadline_s']:
            raise TimeoutError('registered audit deadline')

    save()
    try:
        for name, item in source_files.items():
            deadline()
            path = (raw / name).resolve()
            if raw not in path.parents or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError(f'raw source identity changed: {name}')
            report['sources_verified'] += 1
        additions = []
        for item in catalog:
            deadline()
            native_id = hashlib.sha256(item['source_chart_id'].encode()).hexdigest()[:24]
            row = dict(chart_id=f'phigros_resource:{native_id}', source='phigros_resource',
                native_id=native_id, name=item['title'], title=item['title'], artist=item['artist'], mode='phira',
                source_chart_id=item['source_chart_id'], source_song_id=item['source_song_id'],
                chart_path=item['chart_path'], audio_path=item['audio_path'],
                source_sha256=source_files[item['chart_path']]['sha256'],
                audio_sha256=source_files[item['audio_path']]['sha256'],
                native_difficulty=item['native_difficulty'], native_level=item['native_level'],
                difficulty_tier=native_tier('phira', item['native_difficulty'], policy),
                difficulty_policy_sha256=policy_identity(policy),
                difficulty_evidence='pinned source info table and exact chart path',
                difficulty_review_status='native_metadata_verified_burden_review_pending',
                structural_audit_pass=False, technical_audit_pass=False, training_ready=False,
                feature_cache_status='not_encoded', clock_calibration='pending_source_specific_evidence',
                visibility_audit='pending; no artistic drag removed', quarantined=True)
            try:
                audio_path = item['audio_path']
                if audio_path not in report['audio']:
                    report['audio'][audio_path] = audio_metadata(raw / audio_path)
                row.update(report['audio'][audio_path])
                chart_bytes = (raw / item['chart_path']).read_bytes()
                original = json.loads(chart_bytes)
                raw_count = sum(len(line.get(side) or []) for line in original.get('judgeLineList', [])
                                for side in ('notesAbove', 'notesBelow'))
                chart = parse_phira(chart_bytes, source_name=item['chart_path'])
                reasons = sorted({issue.code for issue in chart.issues if issue.severity == 'quarantine'})
                reasons += pipeline.chart_validation(chart, round(row['audio_sec'] * 1e6))
                if raw_count != len(chart.events) + len(chart.fakes):
                    reasons.append('raw_to_canonical_note_count_mismatch')
                view = build_model_view('phira', (), chart.events)
                target = [event for group in view.target_groups for event in group.events]
                if len(target) != len(chart.events):
                    reasons.append('canonical_to_model_note_count_mismatch')
                reasons += [issue.code for issue in view.quarantine]
                row.update(status='parsed', source_format=chart.source_format,
                    format_version=chart.format_version, native_offset_sec=str(chart.offset_sec),
                    raw_notes=raw_count, canonical_notes=len(chart.events), fakes=len(chart.fakes),
                    model_notes=len(target), structural_reasons=sorted(set(reasons)),
                    warnings=sorted({issue.code for issue in chart.issues if issue.severity != 'quarantine'}),
                    types=dict(Counter(event.type for event in chart.events)),
                    pure_rhythm_hash=rhythm_hash(target) if not view.quarantine and len(target) == len(chart.events) else None,
                    burden=raw_controls('phira', target, [(0, round(row['audio_sec'] * 1e6))]),
                    onset_us_range=[min((e.onset_us for e in chart.events), default=None),
                                    max((e.onset_us for e in chart.events), default=None)],
                    maximum_group_size=max((len(group.events) for group in view.target_groups), default=0),
                    structural_audit_pass=not reasons)
                canonical = pipeline.chart_json(chart, dict(chart_id=row['chart_id'],
                    audio_sha256=row['audio_sha256'], raw_chart_path=str(raw / item['chart_path'])),
                    dict(stage='structural_only', clock_calibration='pending',
                         applied_clock_shift_us=0, structural_reasons=row['structural_reasons']))
                write(out / 'canon' / (native_id + '.json'), canonical)
            except (UnsupportedFormat, ValueError, OSError, subprocess.SubprocessError) as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
            additions.append(row)
            report['processed'] = len(additions)
            if len(additions) % 50 == 0:
                report['records'] = additions
                save()
                print(json.dumps(dict(processed=len(additions), audio=len(report['audio']),
                                      wall_s=round(time.monotonic() - start, 2))), flush=True)
        additions = incremental_splits(old, additions)
        duplicate_groups = defaultdict(list)
        for row in additions:
            if row.get('pure_rhythm_hash'):
                duplicate_groups[(row['audio_sha256'], row['pure_rhythm_hash'])].append(row)
        for group in duplicate_groups.values():
            ids = sorted(row['chart_id'] for row in group)
            for row in group:
                row['source_duplicate_ids'] = ids
                row['source_duplicate_of'] = ids[0] if row['chart_id'] != ids[0] else None
        report['records'] = additions
        report['summary'] = dict(raw_charts=len(catalog), independent_audio=len(report['audio']),
            parsed=sum(row['status'] == 'parsed' for row in additions),
            structural_pass=sum(row['structural_audit_pass'] for row in additions),
            split_conflicts=sum(row['split_conflict'] for row in additions),
            source_duplicate_charts=sum(bool(row.get('source_duplicate_of')) for row in additions),
            training_ready_charts=0, clock_calibrated_charts=0,
            raw_tiers=dict(Counter(row['difficulty_tier'] for row in additions)),
            structural_unique_training_tiers=dict(Counter(row['difficulty_tier'] for row in additions
                if row['structural_audit_pass'] and row['split'] == 'train' and not row.get('source_duplicate_of'))),
            structural_reasons=dict(Counter(code for row in additions for code in row.get('structural_reasons', []))),
            errors=[dict(chart_id=row['chart_id'], error=row['error']) for row in additions if row['status'] == 'error'])
        manifest = out / 'manifest/candidates.jsonl'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in additions))
        report.update(complete=True, manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest))
        save()
        print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    except Exception as exc:
        report.update(complete=False, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()

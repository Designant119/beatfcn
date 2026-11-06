"""Compare frozen Phira packages, native object semantics and canonical targets.

This is a CPU audit. It neither repairs source objects nor certifies the audio
decoder clock, rendering, feature cache or training readiness.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

import yaml

from beatfcn.events.canonical import Event
from beatfcn.events.phira import parse_phira
from beatfcn.ingest.phira_policy import phira_stratum
from scripts.audit_phira_native_content import inspect_chart, read_chart

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*2**20), b''):
            digest.update(data)
    return digest.hexdigest()


def signatures(events):
    return Counter(e.signature() for e in events)


def audio_identity(path, info, plan):
    """Hash the exact compressed audio bytes used by the existing ingest path."""
    with zipfile.ZipFile(path) as archive:
        names = {n.filename.lower().rsplit('/', 1)[-1]: n for n in archive.infolist() if not n.is_dir()}
        member = names.get(str(info.get('music') or '').lower().rsplit('/', 1)[-1])
        if member is None or not 0 < member.file_size <= plan['max_audio_member_bytes']:
            raise ValueError('Missing or oversized music member')
        digest = hashlib.sha256()
        count = 0
        with archive.open(member) as stream:
            for data in iter(lambda: stream.read(8*2**20), b''):
                count += len(data)
                if count > plan['max_audio_member_bytes']:
                    raise ValueError('Audio member exceeds byte limit')
                digest.update(data)
        if count != member.file_size:
            raise ValueError('Audio archive member length differs')
        return dict(audio_sha256=digest.hexdigest(), audio_bytes=count, audio_member=member.filename)


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    incremental = plan.get('incremental_receipt', False)
    if output.exists() and not incremental:
        raise FileExistsError(output)
    panel_path = ROOT/plan['panel']
    if sha(panel_path) != plan['panel_sha256']:
        raise ValueError('Frozen cohort changed')
    panel = json.loads(panel_path.read_text())
    candidates = {c['chart_id']: c for s in panel['records'] for c in s['candidate_charts'] if c['mode'] == 'phira'}
    receipt_path = ROOT/plan['receipt']
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    canonical_receipt_path = ROOT/plan['canonical_receipt']
    canonical_receipt = json.loads(canonical_receipt_path.read_text())
    if ((not incremental and (not receipt['complete'] or receipt['failures']))
            or not canonical_receipt['complete']
            or sha(ROOT/plan['transfer_plan']) != receipt['plan_sha256']):
        raise ValueError('Complete verified transfer receipts required')
    files = {r['path']: r for r in receipt['files']}
    canonical_files = {r['path']: r for r in canonical_receipt['files']}
    transfer_plan = json.loads((ROOT/plan['transfer_plan']).read_text())
    expected_files = {r['path'] for r in transfer_plan['files']}
    if (len(expected_files) != len(candidates) or not set(files) <= expected_files
            or not incremental and len(files) != len(candidates)):
        raise ValueError('Package/candidate cardinality differs')
    source_paths = [plan_path, panel_path, ROOT/plan['transfer_plan'], canonical_receipt_path, Path(__file__).resolve(),
                    ROOT/'scripts/audit_phira_native_content.py', ROOT/'beatfcn/events/phira.py',
                    ROOT/'beatfcn/events/canonical.py', ROOT/'beatfcn/ingest/phira_policy.py']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    if output.exists():
        report = json.loads(output.read_text())
        if report['source_sha256'] != source_hashes:
            raise ValueError('Incremental native audit source or code changed')
        if report['complete']:
            raise ValueError('Audit already complete; do not repeat it')
    else:
        report = dict(id=plan['id'], complete=False, training_ready=False, optimizer_updates=0,
                      new_cloud_compute_calls=0, source_sha256=source_hashes,
                      records=[], started_at_unix=time.time(), receipt_snapshots=[], passes=[])
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    snapshot = output.with_name(output.stem+'.receipt-'+receipt_sha[:12]+'.json')
    if not snapshot.exists():
        snapshot.write_bytes(receipt_bytes)
    report['receipt_snapshots'].append(dict(path=str(snapshot.relative_to(ROOT)), sha256=receipt_sha,
                                          files=len(files), complete=receipt['complete']))
    previous_records = {r['chart_id']: r for r in report['records']}
    count_before = len(previous_records)
    started = time.monotonic()

    def save():
        report['wall_s'] = time.monotonic()-started
        temp = output.with_suffix('.partial')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    for cid, candidate in sorted(candidates.items()):
        if time.monotonic()-started > plan['timeout_s']:
            save()
            raise TimeoutError('Native audit time bound reached')
        native_id = cid.split(':')[1]
        source = files.get(f'raw/phira/zips/{native_id}.zip')
        if source is None:
            continue
        if cid in previous_records:
            if previous_records[cid]['archive_sha256'] != source['sha256']:
                raise ValueError('Previously audited package changed')
            continue
        canonical = canonical_files[f'canon/phira/{native_id}.json']
        path, canonical_path = ROOT/source['local'], ROOT/canonical['local']
        for p, entry in ((path, source), (canonical_path, canonical)):
            if not p.resolve().is_relative_to((ROOT/'inputs/d66_local_cohort/source').resolve()):
                raise ValueError('Source escapes frozen cache')
            if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
                raise ValueError(f'File differs from receipt: {p}')
        row = dict(chart_id=cid, split=candidate['split'], stratum=candidate['stratum'],
                   archive_sha256=source['sha256'], source_identity_passed=False,
                   native_static_candidate=False, training_ready=False)
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if (sum(m.file_size for m in members) > plan['max_archive_uncompressed_bytes']
                        or any(m.file_size > plan['max_chart_member_bytes'] for m in members
                               if m.filename.lower().endswith(('.json', '.yml', '.yaml')))):
                    raise ValueError('Native archive exceeds registered CPU preparation limits')
            info, raw, identity = read_chart(path)
            audio = audio_identity(path, info, plan)
            doc = json.loads(canonical_path.read_text())
            chart = parse_phira(raw, info_offset_sec=info.get('offset'))
            semantic = inspect_chart(raw, info.get('offset'))
            expected = signatures(Event.from_json(e) for e in doc['events'])
            actual = signatures(chart.events)
            expected_fake = signatures(Event.from_json(e) for e in doc.get('fakes', []))
            actual_fake = signatures(chart.fakes)
            native_stratum = phira_stratum(dict(info, difficulty_label=info.get('level'), source='phira'))
            checks = dict(source_bytes_match=identity['source_sha256'] == doc['source_sha256'],
                          audio_bytes_match=audio['audio_sha256'] == doc['audio_sha256'] == candidate['audio_sha256'],
                          playable_multiset_match=actual == expected,
                          fake_multiset_match=actual_fake == expected_fake,
                          source_format_match=chart.source_format == doc['source_format'],
                          native_stratum_match=native_stratum['key'] == candidate['stratum'])
            source_passed = all(checks.values())
            row.update(identity, **audio, checks=checks, source_identity_passed=source_passed,
                       package_info_id=info.get('id'), package_level=info.get('level'),
                       package_difficulty=info.get('difficulty'), package_offset=info.get('offset'),
                       native_stratum=native_stratum,
                       missing_playable_objects=sum((expected-actual).values()),
                       unexpected_playable_objects=sum((actual-expected).values()),
                       native_semantics=semantic,
                       native_static_candidate=source_passed and semantic['status'] == 'static_screen_only',
                       audio_decoded_clock_verified=False, native_render_verified=False)
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, zipfile.BadZipFile) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'[:400]
        report['records'].append(row)
        if len(report['records']) % 25 == 0:
            save()
    report['summary'] = dict(packages=len(report['records']),
        source_identity_passed=sum(r['source_identity_passed'] for r in report['records']),
        native_static_candidates=sum(r['native_static_candidate'] for r in report['records']),
        statuses=dict(Counter(r.get('native_semantics', {}).get('status', 'error') for r in report['records'])),
        mismatched_checks=dict(Counter(k for r in report['records'] for k, v in r.get('checks', {}).items() if not v)),
        errors=sum('error' in r for r in report['records']),
        removed_nonfake_events=0, training_admitted=0)
    report['passes'].append(dict(new_records=len(report['records'])-count_before,
                                wall_s=time.monotonic()-started, ended_at_unix=time.time()))
    report.update(complete=(len(report['records']) == len(candidates)
                            and receipt['complete'] and not receipt['failures']), limitations=[
        'Static native checks do not prove renderer playability, complete musical alignment or feature correctness.',
        'Visual metadata is informational: only explicit isFake separates target notes from fakes.',
        'Audio content hash matches compressed bytes only; decoded duration and offset remain separate gates.',
        'Native stack depth is not an exclusion; song/target dedup and clock integrity are separate checks.'])
    save()
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())

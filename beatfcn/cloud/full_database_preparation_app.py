"""In-container full-database preparation: no local-machine corpus traffic."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import modal
import numpy as np


DATA = '/data'
REMOTE_ROOT = 'full-database-v1'
BATCH_SIZE = 256
CPU_WORKERS = 8
AUDIO_SUFFIXES = {'.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus'}
MAX_ARCHIVE_BYTES = 512 * 2**20
app = modal.App('beatfcn-full-database-preparation')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('torchaudio==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('transformers==4.53.2', 'numpy==2.2.6', 'safetensors==0.5.3')
         .add_local_python_source('beatfcn'))


def verify_beatfcn_sources(contract):
    import beatfcn
    package_root = Path(beatfcn.__file__).resolve().parent.parent
    for name, expected in contract['source_sha256'].items():
        if not name.startswith('beatfcn/'):
            continue
        path = package_root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Uploaded source differs: ' + name)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text('\n'.join(json.dumps(value, ensure_ascii=False) for value in records)
        + ('\n' if records else ''))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_path(row):
    return Path(DATA) / 'canon' / row['source'] / (row['native_id'] + '.json')


def contract_rows(contract):
    verify_beatfcn_sources(contract)
    if contract.get('output_root') != REMOTE_ROOT:
        raise ValueError('Preparation output root changed')
    rows = [json.loads(line) for line in Path(DATA, contract['manifest']).read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get('status') == 'parsed']


def marker_path(batch):
    return Path(DATA) / REMOTE_ROOT / 'batches' / f'{batch:04d}.json'


def load_markers():
    root = Path(DATA) / REMOTE_ROOT / 'batches'
    if not root.exists():
        return {}
    result = {}
    for path in root.glob('*.json'):
        value = json.loads(path.read_text())
        result[value['batch']] = value
    return result


def validate_target_marker(marker, batch, rows, manifest_sha):
    """Validate immutable batch accounting before any target is skipped."""
    if marker.get('batch') != batch or marker.get('manifest_sha256') != manifest_sha:
        raise ValueError(f'Existing target batch {batch} belongs to another manifest or index')
    planned = [row['chart_id'] for row in rows]
    if marker.get('planned') != planned:
        raise ValueError(f'Existing target batch {batch} planned IDs differ')
    successes = marker.get('successes', [])
    failures = marker.get('permanent_failures', [])
    success_ids = [value['chart_id'] for value in successes]
    failure_ids = [value['chart_id'] for value in failures]
    if (marker.get('success_count') != len(successes)
            or marker.get('permanent_failure_count') != len(failures)
            or len(set(success_ids)) != len(success_ids)
            or len(set(failure_ids)) != len(failure_ids)
            or set(success_ids) & set(failure_ids)
            or sorted(success_ids + failure_ids) != sorted(planned)):
        raise ValueError(f'Existing target batch {batch} has inconsistent outcome accounting')
    return successes


def process_batch(rows, batch, manifest_sha):
    from beatfcn.train.full_database import prepare_chart
    successes = []
    failures = []
    started = time.monotonic()

    def prepare(row):
        remote = canonical_path(row)
        try:
            result = prepare_chart(row, remote, Path(DATA) / REMOTE_ROOT / 'targets')
            return dict(chart_id=row['chart_id'], target_identity=Path(result['target_path']).parent.name,
                canonical_sha256=result['canonical_sha256'], statistics=result['statistics'])
        except Exception as exc:
            return dict(chart_id=row['chart_id'], error=f'{type(exc).__name__}: {exc}')

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        results = list(pool.map(prepare, rows))
    for result in results:
        if 'error' in result:
            failures.append(result)
        else:
            successes.append(result)
    marker = dict(schema='beatfcn.full-database-target-batch.v1', batch=batch,
        manifest_sha256=manifest_sha, planned=[row['chart_id'] for row in rows],
        successes=successes, permanent_failures=failures,
        success_count=len(successes), permanent_failure_count=len(failures),
        wall_s=time.monotonic() - started, cloud_compute_calls=1,
        worker_count=CPU_WORKERS, execution='modal_container')
    write_json(marker_path(batch), marker)
    data.commit()
    return marker


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(32768, 32768),
              timeout=4 * 3600, retries=0, max_containers=1, scaledown_window=2)
def prepare_targets(run_id, contract_json):
    contract = json.loads(contract_json)
    rows = contract_rows(contract)
    manifest_sha = hashlib.sha256(Path(DATA, contract['manifest']).read_bytes()).hexdigest()
    batches = [rows[index:index + BATCH_SIZE] for index in range(0, len(rows), BATCH_SIZE)]
    markers = load_markers()
    if not set(markers).issubset(set(range(len(batches)))):
        raise ValueError('Existing target markers contain out-of-range batches')
    row_by_id = {row['chart_id']: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError('Parsed manifest contains duplicate chart IDs')
    audit = []
    reused = 0
    # v6 reopened batches 0-68 and content-hashed new batches 69-88 at creation.
    # A later resume binds that evidence explicitly instead of paying for the same I/O again.
    reuse_through = contract.get('target_resume_verified_through_batch', -1)
    for batch, value in markers.items():
        successes = validate_target_marker(value, batch, batches[batch], manifest_sha)
        if batch <= reuse_through:
            reused += len(successes)
        else:
            for saved in successes:
                audit.append((saved, row_by_id[saved['chart_id']]))
    started = time.monotonic()

    def audit_target(item):
        from beatfcn.train.full_database import prepare_chart
        saved, row = item
        result = prepare_chart(row, canonical_path(row), Path(DATA) / REMOTE_ROOT / 'targets')
        identity = Path(result['target_path']).parent.name
        if identity != saved['target_identity'] or result['canonical_sha256'] != saved['canonical_sha256']:
            raise ValueError('Committed target identity differs from its batch marker: ' + row['chart_id'])
        return row['chart_id']

    # A marker is not sufficient evidence: reopen every prior receipt and hash its arrays.
    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        audited = list(pool.map(audit_target, audit))
    initial_batches = set(markers)
    for batch, rows_in_batch in enumerate(batches):
        if batch not in markers:
            markers[batch] = process_batch(rows_in_batch, batch, manifest_sha)
        print(dict(stage='target', batch=batch, total=len(batches),
            successes=markers[batch]['success_count'], failures=markers[batch]['permanent_failure_count']), flush=True)
    complete = set(markers) == set(range(len(batches)))
    work_charts = len(audited) + sum(value['success_count'] for batch, value in markers.items()
        if batch not in initial_batches)
    result = dict(schema='beatfcn.full-database-target-preparation.v1', run_id=run_id,
        manifest_sha256=manifest_sha, parsed_rows=len(rows), planned_batches=len(batches),
        completed_batches=sorted(markers), prior_targets_reopened=len(audited),
        prior_targets_reused_from_verified_run=reused,
        target_resume_verified_through_batch=reuse_through,
        resume_evidence=contract.get('recovery', {}), complete=complete,
        success_count=sum(value['success_count'] for value in markers.values()),
        permanent_failure_count=sum(value['permanent_failure_count'] for value in markers.values()),
        wall_s=time.monotonic() - started, execution='modal_container',
        worker_count=CPU_WORKERS, new_or_reopened_charts=work_charts,
        charts_per_second=work_charts / max(1e-9, time.monotonic() - started))
    write_json(Path(DATA) / REMOTE_ROOT / 'reports' / 'target-preparation-v1.json', result)
    data.commit()
    return result


def read_hash(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_osu(row):
    audio = row['audio_sha256']
    path = Path(DATA) / 'raw/osu/audio' / (audio + '.' + row['audio_ext'])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    digest_value = read_hash(path)
    return [dict(audio_sha256=audio, source='osu', archive=str(path.relative_to(DATA)),
        archive_member=None, archive_bytes=path.stat().st_size, payload_bytes=path.stat().st_size,
        payload_sha256=digest_value, verified=digest_value == audio)]


def verify_phira(native, rows):
    audio_ids = sorted({row['audio_sha256'] for row in rows})
    path = Path(DATA) / 'raw/phira/zips' / (native + '.zip')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    archive_bytes = path.stat().st_size
    if archive_bytes > MAX_ARCHIVE_BYTES:
        raise ValueError('Oversized Phira archive: ' + native)
    expected = set(audio_ids)
    found = {}
    with zipfile.ZipFile(path, 'r') as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            if member.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError('Oversized Phira audio member: ' + member.filename)
            hasher = hashlib.sha256()
            size = 0
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b''):
                    hasher.update(chunk)
                    size += len(chunk)
            digest_value = hasher.hexdigest()
            if size != member.file_size:
                raise ValueError('Truncated Phira audio member: ' + member.filename)
            if digest_value in expected and digest_value not in found:
                # Identical duplicate members are one audio identity; retain the first provenance.
                found[digest_value] = (member.filename, size, digest_value)
    result = []
    for audio in audio_ids:
        member, size, payload_sha = found.get(audio, (None, None, None))
        result.append(dict(audio_sha256=audio, source='phira',
            archive=str(path.relative_to(DATA)), archive_member=member,
            archive_bytes=archive_bytes, payload_bytes=size,
            payload_sha256=payload_sha, verified=member is not None))
    return result


def unique_osu_rows(rows):
    """One content hash per shared OSU audio; conflicting paths are not guessed away."""
    by_audio = {}
    for row in rows:
        if row['source'] != 'osu':
            continue
        audio = row['audio_sha256']
        prior = by_audio.get(audio)
        if prior is not None and prior.get('audio_ext') != row.get('audio_ext'):
            raise ValueError('Shared OSU audio has conflicting extensions: ' + audio)
        by_audio[audio] = prior or row
    return [by_audio[audio] for audio in sorted(by_audio)]


def plan_raw_verification(rows):
    """Select one deterministic physical source per audio and retain all provenance."""
    osu = unique_osu_rows(rows)
    assigned = {row['audio_sha256'] for row in osu}
    phira_candidates = {}
    provenance = {}
    for row in rows:
        audio = row['audio_sha256']
        if row['source'] == 'osu':
            path = f"raw/osu/audio/{audio}.{row['audio_ext']}"
        else:
            path = f"raw/phira/zips/{row['native_id']}.zip"
        entries = provenance.setdefault(audio, {})
        entries[(row['source'], path)] = dict(source=row['source'], path=path)
        if row['source'] == 'phira' and audio not in assigned:
            phira_candidates.setdefault(audio, {})[row['native_id']] = row
    phira = {}
    for audio, candidates in phira_candidates.items():
        native = min(candidates)
        phira.setdefault(native, []).append(candidates[native])
    phira = {native: sorted(values, key=lambda row: row['audio_sha256'])
             for native, values in sorted(phira.items())}
    covered = assigned | set(phira_candidates)
    expected = {row['audio_sha256'] for row in rows}
    if covered != expected:
        raise ValueError('Raw verification plan does not cover every unique parsed audio')
    return osu, phira, {audio: [entries[key] for key in sorted(entries)]
                        for audio, entries in provenance.items()}


PROGRESS_CHUNK = 128


def completed_report(report_path, destination, manifest_sha):
    path = Path(report_path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('manifest_sha256') != manifest_sha:
        raise ValueError('Existing content report belongs to another manifest')
    if not value.get('complete'):
        return None
    if not Path(destination).is_file() or digest(destination) != value.get('sha256'):
        raise ValueError('Completed content report does not match its final JSONL')
    return value


def load_partial_records(path, expected, manifest_sha):
    path = Path(path)
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        audio = value.get('audio_sha256')
        if (audio not in expected or not value.get('verified')
                or value.get('manifest_sha256') != manifest_sha):
            continue
        if audio in result:
            raise ValueError('Duplicate audio in resumable content receipts: ' + audio)
        result[audio] = value
    return result


def attach_raw_provenance(record, provenance):
    sources = []
    for entry in provenance[record['audio_sha256']]:
        selected = entry['source'] == record['source'] and entry['path'] == record['archive']
        sources.append(dict(entry, selected=selected, content_reread=selected))
    record['provenance'] = sources
    record['verification_scope'] = 'one_selected_physical_source_per_unique_audio'
    return record


def classify_raw_unverified(records, rows):
    """Audit failures may complete only for audio already excluded by manifest quarantine."""
    rows_by_audio = {}
    for row in rows:
        rows_by_audio.setdefault(row['audio_sha256'], []).append(row)
    allowed = {audio for audio, values in rows_by_audio.items()
               if all(row.get('quarantined') or row.get('kept') is False for row in values)}
    quarantined = []
    unexpected = []
    for record in records:
        if record.get('verified'):
            continue
        audio = record['audio_sha256']
        if audio not in allowed:
            unexpected.append(audio)
            continue
        quarantined.append(dict(audio_sha256=audio, source=record['source'],
            archive=record['archive'], reason='raw_content_not_verified',
            manifest_quarantined=True, kept=False))
    return quarantined, sorted(unexpected)


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(4096, 4096),
              timeout=4 * 3600, retries=0, max_containers=1, scaledown_window=2)
def verify_raw(run_id, contract_json):
    contract = json.loads(contract_json)
    rows = contract_rows(contract)
    manifest_sha = hashlib.sha256(Path(DATA, contract['manifest']).read_bytes()).hexdigest()
    osu, phira, provenance = plan_raw_verification(rows)
    destination = Path(DATA) / REMOTE_ROOT / 'raw-content.jsonl'
    report_path = Path(DATA) / REMOTE_ROOT / 'reports' / 'raw-content-receipt-v1.json'
    prior = completed_report(report_path, destination, manifest_sha)
    if prior is not None:
        return prior
    partial = Path(DATA) / REMOTE_ROOT / 'raw-content.partial.jsonl'
    progress_path = Path(DATA) / REMOTE_ROOT / 'reports' / 'raw-content-progress-v1.json'
    expected = {row['audio_sha256'] for row in rows}
    records_by_audio = load_partial_records(partial, expected, manifest_sha)
    reused = len(records_by_audio)
    new_records = []
    failures = []
    started = time.monotonic()
    units = [('osu', row) for row in osu if row['audio_sha256'] not in records_by_audio]
    for native, values in phira.items():
        pending = [row for row in values if row['audio_sha256'] not in records_by_audio]
        if pending:
            units.append(('phira', (native, pending)))

    def task(unit):
        kind, value = unit
        try:
            if kind == 'osu':
                return verify_osu(value), None
            native, values = value
            return verify_phira(native, values), None
        except Exception as exc:
            identity = value.get('chart_id') if kind == 'osu' else value[0]
            return None, dict(unit=kind, identity=identity,
                error=f'{type(exc).__name__}: {exc}')

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        for offset in range(0, len(units), PROGRESS_CHUNK):
            for result, failure in pool.map(task, units[offset:offset + PROGRESS_CHUNK]):
                if failure:
                    failures.append(failure)
                    continue
                for record in result:
                    record = attach_raw_provenance(record, provenance)
                    record['manifest_sha256'] = manifest_sha
                    records_by_audio[record['audio_sha256']] = record
                    new_records.append(record)
            write_jsonl(partial, sorted(records_by_audio.values(),
                key=lambda value: value['audio_sha256']))
            write_json(progress_path, dict(schema='beatfcn.full-database-raw-progress.v1',
                manifest_sha256=manifest_sha, complete=False, planned_audio=len(expected),
                verified_audio=len(records_by_audio), new_audio_this_run=len(new_records),
                reused_audio=reused, failures=failures, worker_count=CPU_WORKERS,
                wall_s=time.monotonic() - started))
            data.commit()
            print(dict(stage='raw', verified=len(records_by_audio), total=len(expected),
                new=len(new_records), failures=len(failures)), flush=True)
    records = sorted(records_by_audio.values(), key=lambda value: value['audio_sha256'])
    write_jsonl(destination, records)
    record_ids = [value['audio_sha256'] for value in records]
    provenance_complete = all(sum(entry['selected'] for entry in record['provenance']) == 1
                              for record in records)
    quarantined_content, unexpected_content = classify_raw_unverified(records, rows)
    report = dict(schema='beatfcn.full-database-raw-content-v2', run_id=run_id,
        manifest_sha256=manifest_sha, output=str(destination), sha256=digest(destination),
        parsed_rows=len(rows), unique_audio=len(expected),
        duplicate_chart_rows=len(rows) - len(expected),
        cross_source_audio=sum(len({entry['source'] for entry in values}) > 1
                               for values in provenance.values()),
        selected_osu_audio=len(osu), selected_phira_archives=len(phira),
        alternate_physical_sources_not_reread=sum(len(values) - 1 for values in provenance.values()),
        verification_scope='one selected physical source per unique audio; all provenance retained; alternates not reread',
        records=len(records), verified=sum(value['verified'] for value in records),
        all_content_verified=not quarantined_content and not unexpected_content,
        quarantined_content_count=len(quarantined_content),
        quarantined_content=quarantined_content,
        unquarantined_content_failures=unexpected_content,
        completion_scope='all parsed audio audited; content failure allowed only for pre-quarantined kept=false audio',
        failures=failures, complete=not failures and not unexpected_content
            and len(records) == len(expected) and len(set(record_ids)) == len(record_ids)
            and provenance_complete,
        wall_s=time.monotonic() - started, execution='modal_container',
        worker_count=CPU_WORKERS, reused_audio=reused, new_audio_this_run=len(new_records),
        audio_per_second=len(new_records) / max(1e-9, time.monotonic() - started),
        payload_bytes_per_second=sum((value['payload_bytes'] or 0) for value in new_records)
            / max(1e-9, time.monotonic() - started))
    write_json(report_path, report)
    data.commit()
    return report


def verify_mel_file(data_root, audio):
    """Hash and numerically verify one array against its decoded-frame receipt."""
    root = Path(data_root)
    metadata = json.loads((root / 'mel10' / (audio + '.json')).read_text())
    expected_frames = metadata.get('frames')
    if (type(expected_frames) is not int or expected_frames <= 0 or expected_frames % 4
            or metadata.get('hop_sec') != .01 or metadata.get('dtype') != 'float16'):
        raise ValueError('Invalid mel clock receipt: ' + audio)
    path = root / 'mel10' / (audio + '.npy')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    finite = bool(np.isfinite(value).all())
    return dict(audio_sha256=audio, remote=str(path.relative_to(root)),
        bytes=path.stat().st_size, sha256=read_hash(path), dtype=str(value.dtype),
        shape=list(value.shape), c_contiguous=bool(value.flags.c_contiguous),
        finite=finite, expected_frames=expected_frames,
        verified=value.dtype == np.float16 and value.shape == (expected_frames, 128)
            and finite and value.flags.c_contiguous)


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(4096, 4096),
              timeout=4 * 3600, retries=0, max_containers=1, scaledown_window=2)
def verify_mel(run_id, contract_json):
    contract = json.loads(contract_json)
    rows = contract_rows(contract)
    manifest_sha = hashlib.sha256(Path(DATA, contract['manifest']).read_bytes()).hexdigest()
    expected = {row['audio_sha256'] for row in rows}
    destination = Path(DATA) / REMOTE_ROOT / 'mel-array-content.jsonl'
    report_path = Path(DATA) / REMOTE_ROOT / 'reports' / 'mel-array-content-receipt-v1.json'
    prior = completed_report(report_path, destination, manifest_sha)
    if prior is not None:
        return prior
    partial = Path(DATA) / REMOTE_ROOT / 'mel-array-content.partial.jsonl'
    progress_path = Path(DATA) / REMOTE_ROOT / 'reports' / 'mel-array-content-progress-v1.json'
    records_by_audio = load_partial_records(partial, expected, manifest_sha)
    reused = len(records_by_audio)
    new_records = []
    failures = []
    started = time.monotonic()

    def verify(audio):
        try:
            return verify_mel_file(DATA, audio), None
        except Exception as exc:
            return None, dict(audio_sha256=audio, error=f'{type(exc).__name__}: {exc}')

    pending = sorted(expected - set(records_by_audio))
    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        for offset in range(0, len(pending), PROGRESS_CHUNK):
            for result, failure in pool.map(verify, pending[offset:offset + PROGRESS_CHUNK]):
                if failure:
                    failures.append(failure)
                else:
                    result['manifest_sha256'] = manifest_sha
                    records_by_audio[result['audio_sha256']] = result
                    new_records.append(result)
            write_jsonl(partial, sorted(records_by_audio.values(),
                key=lambda value: value['audio_sha256']))
            write_json(progress_path, dict(schema='beatfcn.full-database-mel-progress.v1',
                manifest_sha256=manifest_sha, complete=False, planned_audio=len(expected),
                verified_audio=len(records_by_audio), new_audio_this_run=len(new_records),
                reused_audio=reused, failures=failures, worker_count=CPU_WORKERS,
                wall_s=time.monotonic() - started))
            data.commit()
            print(dict(stage='mel', verified=len(records_by_audio), total=len(expected),
                new=len(new_records), failures=len(failures)), flush=True)
    records = sorted(records_by_audio.values(), key=lambda value: value['audio_sha256'])
    write_jsonl(destination, records)
    report = dict(schema='beatfcn.full-database-mel-array-content-v1', run_id=run_id,
        manifest_sha256=manifest_sha, output=str(destination), sha256=digest(destination),
        parsed_audio=len(expected), records=len(records),
        verified=sum(value['verified'] for value in records), failures=failures,
        bytes=sum(value['bytes'] for value in records),
        complete=not failures and len(records) == len(expected)
            and all(value['verified'] for value in records),
        wall_s=time.monotonic() - started, execution='modal_container',
        worker_count=CPU_WORKERS, reused_audio=reused, new_audio_this_run=len(new_records),
        audio_per_second=len(new_records) / max(1e-9, time.monotonic() - started),
        payload_bytes_per_second=sum(value['bytes'] for value in new_records)
            / max(1e-9, time.monotonic() - started))
    write_json(report_path, report)
    data.commit()
    return report


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(4096, 4096),
              timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def download_checkpoint(run_id, contract_json):
    from huggingface_hub import snapshot_download
    from beatfcn.audio.mert2 import REPO, REVISION, validate_local_source
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    destination = Path(DATA) / REMOTE_ROOT / 'mert2-checkpoint'
    started = time.monotonic()
    receipt = destination / 'receipt.json'
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if not value.get('complete') or value.get('revision') != REVISION:
            raise ValueError('Existing checkpoint receipt is incomplete or different')
        validate_local_source(destination)
        expected_files = set(value.get('files', {}))
        actual_files = {path.name for path in destination.iterdir()
                        if path.is_file() and path.name != 'receipt.json'}
        if actual_files != expected_files:
            raise ValueError('Existing checkpoint file closure differs from its receipt')
        for name, info in value['files'].items():
            path = destination / name
            if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
                raise ValueError('Existing checkpoint content differs from its receipt: ' + name)
        return value
    if not destination.exists():
        temporary = Path(DATA) / REMOTE_ROOT / 'mert2-checkpoint.partial'
        # snapshot_download reuses its on-disk metadata and completed files across a reconnect.
        snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(temporary),
            local_dir_use_symlinks=False, max_workers=CPU_WORKERS)
        validate_local_source(temporary)
        temporary.rename(destination)
    else:
        # A disconnect after rename but before receipt creation must not force a re-download.
        validate_local_source(destination)
    result = dict(schema='beatfcn.full-database-mert-checkpoint.v1', remote_prefix=str(destination),
        repo=REPO, revision=REVISION, complete=True,
        files={path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
               for path in destination.iterdir() if path.is_file()},
        wall_s=time.monotonic() - started, execution='modal_container',
        worker_count=CPU_WORKERS,
        payload_bytes_per_second=sum(path.stat().st_size for path in destination.iterdir()
            if path.is_file()) / max(1e-9, time.monotonic() - started))
    write_json(receipt, result)
    data.commit()
    return result

"""Cloud corpus preparation primitives; raw sources and splits remain immutable.

These functions prepare data, never optimize a model. Each completed item has a
content receipt. Missing or quarantined items stay in the catalogue denominator.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.audio.local_mel_cache import complete_song_mel
from beatfcn.train.data import chart_from_canon, model_shift_us
from beatfcn.train.event_density_drag_filter import RULE, dense_event_drag_mask


SCHEMA = 'beatfcn.full-database.v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Source path escapes its volume')
    return result


def event_arrays(document, row):
    """Native/model clocks retained separately; no quantization or deduplication."""
    native = chart_from_canon(document)
    if native.mode not in ('phira', 'mania_4k'):
        raise ValueError('Unsupported game mode')
    shift = model_shift_us(row, True) if row.get('offset_decision') else 0
    events = [RhythmEvent.from_event(e, native.mode) for e in native.events]
    raw = np.asarray([(e.onset_us, e.end_us, TYPES.index(e.type), e.lane)
                      for e in events], dtype=np.int64).reshape(-1, 4)
    # Apply the Owner predicate to the original complete chart, once. Holds
    # contribute a head only; simultaneous duplicate objects count separately.
    removed, intervals = dense_event_drag_mask(raw[:, 0], raw[:, 2])
    if native.mode == 'mania_4k' and removed.any():
        raise ValueError('Mania contains unexpected drag objects')
    clean = raw[~removed].copy()
    clean[:, :2] += shift
    non_drag = raw[:, 2] != TYPES.index('drag')
    shifted_non_drag = raw[non_drag].copy()
    shifted_non_drag[:, :2] += shift
    if not np.array_equal(clean[clean[:, 2] != TYPES.index('drag')], shifted_non_drag):
        raise AssertionError('Non-drag multiset changed')
    return raw, clean, removed, dict(
        native_events=len(raw), filtered_events=len(clean), removed_drag=int(removed.sum()),
        qualifying_windows=len(intervals), model_shift_us=shift,
        types_before=dict(Counter(TYPES[k] for k in raw[:, 2])),
        types_after=dict(Counter(TYPES[k] for k in clean[:, 2])),
        clock_verified=bool(row.get('offset_decision', {}).get('decision_version') == 2
                            or row.get('clock_calibration') == 'verified'))


def prepare_chart(row, canonical, output_root, *, canonical_sha256=None):
    """Materialize all chart objects even when admission still has blockers."""
    canonical = Path(canonical)
    source_sha = digest(canonical)
    if canonical_sha256 and source_sha != canonical_sha256:
        raise ValueError('Canonical source identity changed')
    document = json.loads(canonical.read_text())
    if document.get('audio_sha256') not in (None, row.get('audio_sha256')):
        raise ValueError('Canonical and catalogue audio differ')
    identity = safe_key(row['chart_id'] + ':' + source_sha)
    folder = Path(output_root) / 'targets' / identity
    receipt = folder / 'receipt.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        expected_shift = model_shift_us(row, True) if row.get('offset_decision') else 0
        identity_matches = (
            prior['canonical_sha256'] == source_sha and prior['rule'] == RULE
            and prior.get('chart_id') == row.get('chart_id')
            and prior.get('audio_sha256') == row.get('audio_sha256')
            and prior.get('split') == row.get('split')
            and prior.get('statistics', {}).get('model_shift_us') == expected_shift)
        if not identity_matches:
            raise ValueError('Existing target receipt belongs to another contract or manifest row')
        if set(prior['files']) != {'native.npy', 'filtered.npy', 'removed_drag.npy'}:
            raise ValueError('Existing target receipt has an incomplete file closure')
        for name, info in prior['files'].items():
            path = folder / name
            if not path.is_file() or path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
                raise ValueError('Previously committed target changed')
        return prior
    raw, clean, removed, stats = event_arrays(document, row)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in [('native.npy', raw), ('filtered.npy', clean), ('removed_drag.npy', removed)]:
        temporary = folder / (name + '.partial')
        with temporary.open('wb') as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(folder / name)
    result = dict(schema=SCHEMA, chart_id=row['chart_id'], mode=document['mode'],
        audio_sha256=row.get('audio_sha256'), split=row.get('split'),
        canonical_sha256=source_sha, rule=RULE, statistics=stats,
        target_path=str(folder / 'filtered.npy'), raw_target_path=str(folder / 'native.npy'),
        files={p.name:dict(bytes=p.stat().st_size, sha256=digest(p))
               for p in folder.glob('*.npy')},
        targets_prepared=True, training_ready=False,
        remaining_checks=['audio_clock', 'feature_identity', 'metadata_admission', 'split_integrity'])
    write_json(receipt, result)
    return result


def locate_audio(audio_sha, candidates, root, temporary_root):
    """Resolve original bytes by content hash, including Phira ZIP members."""
    if not re.fullmatch(r'[0-9a-f]{64}', audio_sha):
        raise ValueError('Invalid audio identity')
    temp = Path(temporary_root) / (audio_sha + '.audio')
    if temp.exists() and digest(temp) == audio_sha:
        return temp
    for candidate in candidates:
        path = checked_path(root, candidate['path'])
        if not path.is_file():
            continue
        if candidate['kind'] == 'audio':
            if digest(path) == audio_sha:
                return path
        elif candidate['kind'] == 'zip':
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or Path(member.filename).suffix.lower() not in (
                            '.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus'):
                        continue
                    if member.file_size > 512 * 2**20:
                        raise ValueError('Audio member exceeds explicit 512MiB item capacity')
                    temp.parent.mkdir(parents=True, exist_ok=True)
                    partial = temp.with_suffix('.partial')
                    hasher = hashlib.sha256()
                    with archive.open(member) as source, partial.open('wb') as target:
                        for chunk in iter(lambda: source.read(2**20), b''):
                            hasher.update(chunk)
                            target.write(chunk)
                    if hasher.hexdigest() == audio_sha:
                        partial.replace(temp)
                        return temp
                    partial.unlink()
    raise FileNotFoundError('No original audio bytes match ' + audio_sha)


def existing_feature_receipt(folder, encoder):
    """Validate a previously committed feature folder without rewriting it."""
    receipt = folder / 'receipt.json'
    if not receipt.exists():
        return None
    value = json.loads(receipt.read_text())
    if value['metadata']['revision'] != encoder.revision:
        raise ValueError('Committed MERT revision differs')
    for name, info in value['files'].items():
        if digest(folder / name) != info['sha256']:
            raise ValueError('Committed feature changed')
    return value


def decoded_audio(source, root, temporary_root):
    """Locate and decode original audio bytes; no feature or optimizer work."""
    from beatfcn.ingest.pipeline import decode
    audio = source['audio_sha256']
    original = locate_audio(audio, source['candidates'], root, temporary_root)
    wave = decode(str(original))
    if not np.isfinite(wave).all() or not len(wave):
        raise ValueError('Invalid decoded audio')
    return audio, original, wave


def feature_payloads(wave, encoder, *, mel=None, phase_callback=None):
    """Frozen GPU/MEL feature equations, retaining their native physical clocks."""
    if phase_callback is not None:
        phase_callback('gpu_encode_start')
    features = encoder.encode(wave)
    if phase_callback is not None:
        phase_callback('gpu_encode_end')
    if mel is None:
        mel = complete_song_mel(wave)
    expected = (len(wave) + 959) // 960
    if features['layers'].shape != (2, expected, 1024) or mel['mel'].shape != (expected * 4, 128):
        raise ValueError('MERT/mel physical clocks disagree')
    return features, mel, expected


def commit_feature(audio, original, output_root, temporary_root, features, mel, expected):
    folder = Path(output_root) / 'features' / audio
    folder.mkdir(parents=True, exist_ok=True)
    values = {'L12.npy': features['layers'][0], 'L23.npy': features['layers'][1],
              'times_s.npy': features['times_s'], 'intervals_s.npy': features['intervals_s'],
              'mel10_complete.npy': mel['mel']}
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite output ' + name)
        temporary = folder / (name + '.partial')
        with temporary.open('wb') as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        temporary.replace(folder / name)
    result = dict(schema=SCHEMA, audio_sha256=audio, decoded_samples=features['metadata']['real_samples'],
        frames=expected, metadata=features['metadata'], backend='cuda_bf16',
        files={name:dict(bytes=(folder/name).stat().st_size, sha256=digest(folder/name)) for name in values},
        optimizer_updates=0, wall_s=None, complete=True)
    write_json(folder / 'receipt.json', result)
    if original.is_relative_to(Path(temporary_root)):
        original.unlink()
    return result


def prepare_audio(source, root, output_root, temporary_root, encoder):
    """Called on a GPU worker. Frozen encoding, zero optimizer updates."""
    audio = source['audio_sha256']
    folder = Path(output_root) / 'features' / audio
    prior = existing_feature_receipt(folder, encoder)
    if prior is not None:
        return prior
    started = time.monotonic()
    audio, original, wave = decoded_audio(source, root, temporary_root)
    features, mel, expected = feature_payloads(wave, encoder)
    result = commit_feature(audio, original, output_root, temporary_root, features, mel, expected)
    result['wall_s'] = time.monotonic() - started
    write_json(folder / 'receipt.json', result)
    return result

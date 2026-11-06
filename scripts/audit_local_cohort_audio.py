"""Bounded, resumable local decode and clock audit of explicitly frozen songs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import torch

from beatfcn.audio.preprocess import CELL_SAMPLES, HOP, SR
from beatfcn.events.canonical import Event
from beatfcn.ingest.offset import flux_envelope
from beatfcn.ingest.pipeline import mel10
from beatfcn.train.data import model_shift_us
from beatfcn.train.local_audio_audit import chart_audio_report, tensor_health

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def materialize(song, cache):
    """Extract only the verified audio member, under its content-addressed name."""
    source = ROOT/song['source_path']
    if sha(source) != song['source_sha256']:
        raise ValueError('Original audio/package hash changed')
    if not song.get('archive_member'):
        if source.stat().st_size != song['audio_bytes'] or sha(source) != song['audio_sha256']:
            raise ValueError('Original compressed audio identity changed')
        return source
    destination = cache/'audio'/(song['audio_sha256']+'.audio')
    if destination.exists():
        if sha(destination) != song['audio_sha256'] or destination.stat().st_size != song['audio_bytes']:
            raise ValueError('Materialized audio differs from verified native source')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial')
    with zipfile.ZipFile(source) as archive:
        member = archive.getinfo(song['archive_member'])
        if member.file_size != song['audio_bytes'] or member.file_size > 512*2**20:
            raise ValueError('Audio archive member size changed')
        with archive.open(member) as reader, partial.open('wb') as writer:
            shutil.copyfileobj(reader, writer, 8*2**20)
    if sha(partial) != song['audio_sha256'] or partial.stat().st_size != song['audio_bytes']:
        raise ValueError('Materialized audio identity mismatch')
    partial.replace(destination)
    return destination


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output, cache = ROOT/plan['output'], ROOT/plan['cache']
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen clock input changed: '+name)
    identities = {**plan['input_sha256'], str(plan_path.relative_to(ROOT)): sha(plan_path)}
    for name in ('scripts/audit_local_cohort_audio.py', 'beatfcn/train/local_audio_audit.py',
                 'beatfcn/ingest/pipeline.py', 'beatfcn/audio/preprocess.py',
                 'beatfcn/ingest/offset.py', 'beatfcn/train/data.py', 'beatfcn/events/canonical.py'):
        identities[name] = sha(ROOT/name)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Local ffmpeg missing')
    identities['ffmpeg_binary'] = sha(Path(ffmpeg))
    if output.exists():
        report = json.loads(output.read_text())
        if report['source_sha256'] != identities or report['complete']:
            raise ValueError('Clock audit changed or already completed')
    else:
        report = dict(id=plan['id'], complete=False, training_ready=False, source_sha256=identities,
                      audio=[], records=[], passes=[], optimizer_updates=0, cloud_compute_calls=0,
                      labels_changed=False, torch_version=torch.__version__,
                      ffmpeg_version=subprocess.run([ffmpeg, '-version'], check=True, capture_output=True,
                                                    text=True).stdout.splitlines()[0],
                      interpretation='Decoded bounds are mechanical checks; acoustic scores are not ground truth.')
    manifest = {r['chart_id']: r for r in map(json.loads, (ROOT/plan['manifest']).read_text().splitlines())}
    receipt = json.loads((ROOT/plan['canonical_receipt']).read_text())
    if not receipt['complete'] or receipt['failures']:
        raise ValueError('Complete canonical receipt required')
    canonical = {r['path']: r for r in receipt['files']}
    done = {r['audio_sha256'] for r in report['audio']}
    if len(done) != len(report['audio']) or len({r['audio_sha256'] for r in plan['songs']}) != len(plan['songs']):
        raise ValueError('Duplicate audio entries')
    started = time.monotonic()
    processed_before = len(done)

    def save():
        report['summary'] = dict(audio=len(report['audio']), charts=len(report['records']),
            decoded_ok=sum(a['status'] == 'ok' for a in report['audio']),
            errors=sum(a['status'] != 'ok' for a in report['audio']),
            duration_mismatch_charts=sum(not r['duration_matches_rounded_manifest'] for r in report['records']),
            native_bounds_charts=sum(bool(r['native_bounds']['count']) for r in report['records']),
            model_bounds_charts=sum(bool(r['model_bounds']['count']) for r in report['records']),
            legacy_tail_head_events=sum(r['model_tail']['heads_in_omitted_tail'] for r in report['records']),
            legacy_tail_hold_ends=sum(r['model_tail']['hold_ends_in_omitted_tail'] for r in report['records']),
            by_mode=dict(Counter(r['mode'] for r in report['records'])))
        partial = output.with_suffix('.partial')
        partial.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        partial.replace(output)

    for song in plan['songs']:
        audio_sha = song['audio_sha256']
        if audio_sha in done:
            continue
        if time.monotonic()-started > plan['timeout_s']:
            break
        if shutil.disk_usage(ROOT).free < plan['minimum_free_bytes']:
            raise RuntimeError('Registered local storage reserve reached')
        try:
            raw_path = materialize(song, cache)
            # A time limit bounds decoded output even if a malformed source claims
            # a short duration. Hitting the bound is an error, never truncation admission.
            decoded = subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-threads', '1',
                '-i', str(raw_path), '-t', str(plan['maximum_audio_sec']), '-vn', '-f', 'f32le',
                '-ac', '1', '-ar', str(SR), 'pipe:1'], check=True, capture_output=True,
                timeout=plan['decode_timeout_s'])
            wave = np.frombuffer(decoded.stdout, dtype='<f4').copy()
            samples = len(wave)
            if samples < CELL_SAMPLES or samples >= plan['maximum_audio_sec']*SR:
                raise ValueError('Decoded samples outside registered limit')
            wave_health = tensor_health(wave)
            mel = mel10(wave)
            mel_health = tensor_health(mel)
            if mel.shape != (samples//CELL_SAMPLES*CELL_SAMPLES//HOP, 128):
                raise ValueError('Production mel and waveform clocks differ')
            env = flux_envelope(mel)
            mel_path = cache/'mel10_legacy'/(audio_sha+'.npy')
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            mel_partial = mel_path.with_suffix('.partial')
            with mel_partial.open('wb') as stream:
                np.save(stream, np.ascontiguousarray(mel), allow_pickle=False)
            mel_partial.replace(mel_path)
            chart_records = []
            for cid in song['chart_ids']:
                row = manifest[cid]
                source = canonical[f"canon/{row['source']}/{row['native_id']}.json"]
                canonical_path = ROOT/source['local']
                if sha(canonical_path) != source['sha256'] or row['audio_sha256'] != audio_sha:
                    raise ValueError('Canonical clock/audio identity changed')
                doc = json.loads(canonical_path.read_text())
                if doc['audio_sha256'] != audio_sha or doc['chart_id'] != cid:
                    raise ValueError('Canonical/audio join mismatch')
                events = tuple(Event.from_json(e) for e in doc['events'])
                chart_records.append(chart_audio_report(row, events, env, samples, model_shift_us(row, True)))
            report['records'].extend(chart_records)
            report['audio'].append(dict(status='ok', audio_sha256=audio_sha,
                raw_path=str(raw_path.relative_to(ROOT)), decoded_samples=samples,
                decoded_duration_sec=samples/SR, waveform=wave_health, mel=mel_health,
                mel_path=str(mel_path.relative_to(ROOT)), mel_sha256=sha(mel_path),
                c_contiguous=True, legacy_trimmed_samples=samples % CELL_SAMPLES,
                full_tail_feature_verified=False))
            del wave, decoded, mel, env
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
            report['audio'].append(dict(status='error', audio_sha256=audio_sha,
                error=f'{type(exc).__name__}: {exc}'[:500], chart_ids=song['chart_ids']))
        save()
    report['complete'] = len(report['audio']) == len(plan['songs'])
    report['passes'].append(dict(new_audio=len(report['audio'])-processed_before,
        wall_s=time.monotonic()-started, ended_at_unix=time.time()))
    save()
    print(json.dumps(dict(complete=report['complete'], **report['summary'], last_pass=report['passes'][-1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())

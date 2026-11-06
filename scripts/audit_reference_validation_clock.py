"""Bounded local clock evidence for the distinct, already-held-out reference pair.

This prepares evidence only; it does not admit samples or alter any training tape.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import torch

from beatfcn.audio.local_mel_cache import complete_song_mel
from beatfcn.audio.preprocess import SR
from beatfcn.events.phira import parse_phira
from beatfcn.ingest.offset import flux_envelope
from beatfcn.train.data import model_shift_us
from beatfcn.train.local_audio_audit import chart_audio_report, tensor_health
from scripts.audit_phira_native_content import read_chart
from scripts.run_local_song_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pp = ROOT/'bench/rhythm-v2/reference-validation-clock-v1.json'
    plan = json.loads(pp.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError('Inspect existing result before any retry')
    for name, digest in plan['inputs_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen clock input changed: '+name)
    root = (ROOT/plan['cache']).resolve()
    if not root.is_relative_to(ROOT) or root.exists():
        raise ValueError('A new dedicated local cache is required')
    if shutil.disk_usage(ROOT).free < plan['minimum_free_bytes']:
        raise RuntimeError('Local storage reserve reached')
    root.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    corpus = list(map(json.loads, (ROOT/plan['corpus']).read_text().splitlines()))
    rows = {r['chart_id']: r for r in corpus}
    selected = [rows[c] for c in plan['chart_ids']]
    audio_ids = {r['audio_sha256'] for r in selected}
    if len(audio_ids) != 1 or any(r['split'] != 'val' for r in selected):
        raise ValueError('Expected one already-held-out song')
    related = [r for r in corpus if r['audio_sha256'] in audio_ids or
               r.get('song_group') in {s['song_group'] for s in selected}]
    if any(r['split'] != 'val' for r in related):
        raise ValueError('Existing audio/song component crosses splits')
    original = json.loads((ROOT/plan['source_audit']).read_text())
    evidence = {f"phira:{r['id']}": r for r in original['records']}
    first = selected[0]
    info = evidence[first['chart_id']]['audio']
    package = ROOT/plan['packages'][first['chart_id']]
    raw_path = root/'original.audio'
    with zipfile.ZipFile(package) as archive:
        member = archive.getinfo(info['audio_member'])
        if member.file_size != info['audio_bytes'] or member.file_size > 32 << 20:
            raise ValueError('Unexpected audio archive member size')
        raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != first['audio_sha256']:
        raise ValueError('Native compressed audio identity differs')
    raw_path.write_bytes(raw)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Local ffmpeg unavailable')
    decoded = subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-threads', '1',
        '-i', str(raw_path), '-t', str(plan['maximum_audio_s']), '-vn', '-f', 'f32le',
        '-ac', '1', '-ar', str(SR), 'pipe:1'], check=True, capture_output=True, timeout=30)
    wave = np.frombuffer(decoded.stdout, dtype='<f4').copy()
    if not 393 <= len(wave) < plan['maximum_audio_s']*SR:
        raise ValueError('Decoded length outside frozen bounds')
    health = tensor_health(wave)
    mel = complete_song_mel(wave)
    np.savez(root/'mel10-complete.npz', **mel)
    # Diagnostic alignment uses only real audio frames, never padded silence.
    env = flux_envelope(mel['mel'][mel['valid']])
    records = []
    for row in selected:
        if time.monotonic()-started > plan['maximum_wall_s']:
            raise TimeoutError('Frozen clock audit deadline')
        native_info, chart_bytes, identity = read_chart(ROOT/plan['packages'][row['chart_id']])
        if identity != evidence[row['chart_id']]['source_identity']:
            raise ValueError('Native chart identity differs')
        chart = parse_phira(chart_bytes, info_offset_sec=native_info.get('offset'))
        if len(chart.events) != row['n_events'] or len(chart.fakes) != row['n_fakes']:
            raise ValueError('Native and manifest event accounting differs')
        record = chart_audio_report(row, chart.events, env, len(wave), model_shift_us(row, True))
        record.update(package_offset_s=native_info.get('offset'),
            canonical_offset_s=float(chart.offset_sec), source_identity=identity)
        records.append(record)
    result = dict(id=plan['id'], complete=True, records=records,
        decoded_samples=len(wave), decoded_duration_s=len(wave)/SR, waveform=health,
        mel=tensor_health(mel['mel']), mel_valid_frames=int(mel['valid'].sum()),
        mel_padding_samples=int(mel['padding_samples']),
        audio_sha256=first['audio_sha256'],
        same_component_chart_ids=[r['chart_id'] for r in related],
        registered_component_split='val', label_changes=0, training_ready=False,
        native_playability_verified=False, acoustic_alignment_verified=False,
        added_usable_reference_validation_songs=0, optimizer_updates=0, new_cloud_compute_calls=0,
        source_sha256={**plan['inputs_sha256'], str(pp.relative_to(ROOT)): sha(pp), 'ffmpeg_binary': sha(ffmpeg)},
        artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in (raw_path, root/'mel10-complete.npz')},
        wall_s=time.monotonic()-started,
        limitations=['No Pupu features or native render/playability admission.',
            'Stored component checks do not establish perceptual duplicate detection against all music.',
            'IN16+ and AT16+ differ in difficulty; reference effects cannot be attributed to style alone.',
            'Peak offsets are diagnostics and do not justify correcting either chart.'])
    write_json(output, result)
    print(json.dumps(dict(complete=True, duration_s=result['decoded_duration_s'],
        bounds=[r['model_bounds']['count'] for r in records], wall_s=result['wall_s'])))


if __name__ == '__main__':
    main()

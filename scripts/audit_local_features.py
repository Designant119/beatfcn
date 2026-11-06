"""Incrementally admit already downloaded frozen features and prepare complete mel."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from beatfcn.audio.local_mel_cache import complete_song_mel, legacy_prefix_comparison
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_features import song_summary
from scripts.audit_phigros_clock import decode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    identity = sha(plan_path)
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen feature input changed: {name}')
    out = ROOT/plan['report']
    code = {p: sha(ROOT/p) for p in ['scripts/audit_local_features.py', 'beatfcn/train/local_features.py',
        'beatfcn/audio/local_mel_cache.py', 'beatfcn/audio/preprocess.py', 'scripts/audit_phigros_clock.py']}
    report = json.loads(out.read_text()) if out.exists() else dict(id=plan['id'], plan_sha256=identity,
        code_sha256=code, records=[], attempts=[], failures=[], complete=False, optimizer_updates=0,
        cloud_compute_calls=0, scales=plan['scales'], training_ready=False)
    if report['plan_sha256'] != identity or report['code_sha256'] != code or report['failures']:
        raise ValueError('Audit identity changed or unresolved failure remains')
    if report['complete']:
        raise ValueError('Audit already completed; do not repeat it')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with np.load(ROOT/plan['scales'], allow_pickle=False) as archive:
        scales = {key: archive[key].copy() for key in archive.files}
    for layer in (9, 16, 24):
        scale = scales[f'scale_{layer}']
        if scale.shape != (1024,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid quantization scales')
    panel = json.loads((ROOT/plan['panel']).read_text())
    songs = [r for r in panel['songs'] if r['initial_panel']]
    clock = {r['audio_sha256']: r for r in json.loads((ROOT/plan['clock']).read_text())['audio']}
    probes = {r['audio_sha256']: r for r in json.loads((ROOT/plan['complete_mel_probe']).read_text())['records']}
    # Snapshot mutable transfer receipts; each admitted file is hashed against
    # the exact completed-file record, never inferred from path existence.
    known = {}
    attempt = dict(started_at_unix=time.time(), receipts=[], new_songs=0)
    for name in plan['receipts']:
        path = ROOT/name
        if not path.exists():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        folder = ROOT/'reports/d67_feature_receipt_snapshots'
        folder.mkdir(exist_ok=True)
        snapshot = folder/(path.stem+'-'+digest[:16]+'.json')
        if not snapshot.exists():
            snapshot.write_bytes(data)
        attempt['receipts'].append(dict(path=str(snapshot.relative_to(ROOT)), sha256=digest))
        for record in json.loads(data).get('files', []):
            if record['local'] in known and known[record['local']]['sha256'] != record['sha256']:
                raise ValueError('Transfer receipts disagree about cache bytes')
            known[record['local']] = record
    report['attempts'].append(attempt)

    def save():
        done = {r['audio_sha256'] for r in report['records']}
        report.update(updated_at_unix=time.time(), expected_songs=len(songs), completed_songs=len(done),
            missing_audio=[r['audio_sha256'] for r in songs if r['audio_sha256'] not in done],
            complete=len(done) == len(songs) and not report['failures'])
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(out)

    seen = {r['audio_sha256'] for r in report['records']}
    save()
    for song in songs:
        audio = song['audio_sha256']
        if audio in seen or not all(path in known for path in song['paths'].values()):
            continue
        if time.monotonic()-started >= plan['max_wall_s']:
            break
        try:
            if shutil.disk_usage(ROOT).free < plan['minimum_free_bytes']:
                raise ValueError('System disk reserve reached')
            source = clock[audio]
            samples = source['decoded_samples']
            layers, stats, hashes = [], {}, {}
            for layer in (9, 16, 24):
                name = song['paths'][f'L{layer}.npy']
                p = ROOT/name
                if p.stat().st_size != known[name]['bytes'] or sha(p) != known[name]['sha256']:
                    raise ValueError('Downloaded feature differs from receipt')
                a = np.load(p, mmap_mode='r', allow_pickle=False)
                if a.dtype != np.int8 or a.shape != (samples//960, 8, 1024):
                    raise ValueError(f'Pupu shape/clock mismatch: {a.shape}, {samples}')
                health = tensor_health(a)
                health.update(c_contiguous=bool(a.flags.c_contiguous),
                              saturation_fraction=float(np.mean((a == -128) | (a == 127))))
                stats[f'L{layer}'] = health
                hashes[name] = known[name]['sha256']
                layers.append(a)
            legacy_path = ROOT/source['mel_path']
            if sha(legacy_path) != source['mel_sha256']:
                raise ValueError('Legacy mel changed after clock audit')
            legacy = np.load(legacy_path, mmap_mode='r', allow_pickle=False)
            if audio in probes:
                p = probes[audio]
                if sha(ROOT/p['artifact']) != p['artifact_sha256']:
                    raise ValueError('Complete-mel probe cache changed')
                with np.load(ROOT/p['artifact'], allow_pickle=False) as archive:
                    complete = {k: archive[k].copy() for k in archive.files}
                reused_probe = True
            else:
                raw = ROOT/source['raw_path']
                if sha(raw) != audio:
                    raise ValueError('Original audio identity changed')
                wave = decode(raw, 45)
                if len(wave) != samples:
                    raise ValueError('Decoder sample count differs from prior audit')
                complete = complete_song_mel(wave)
                del wave
                reused_probe = False
            parity = legacy_prefix_comparison(legacy, complete)
            if not parity['interior_exact_equal']:
                raise ValueError('Complete mel interior differs from baseline recipe')
            mel = complete['mel']
            valid = complete['valid']
            if (int(complete['decoded_samples']) != samples or not mel.flags.c_contiguous
                    or not np.array_equal(valid, np.arange(len(mel))*240 < samples)
                    or int((complete['frame_end_samples']-complete['frame_start_samples']).sum()) != samples):
                raise ValueError('Complete mel sample coverage failed')
            dest = ROOT/'inputs/d66_local_cohort/decoded/mel10_complete'
            dest.mkdir(exist_ok=True)
            mp = dest/(audio+'.npy')
            if mp.exists():
                if not np.array_equal(np.load(mp, allow_pickle=False), mel):
                    raise ValueError('Previously prepared complete mel differs')
            else:
                np.save(mp, mel, allow_pickle=False)
            summary = song_summary(layers[-1], scales['scale_24'])
            dest = ROOT/'inputs/d66_local_cohort/song_summary'
            dest.mkdir(exist_ok=True)
            sp = dest/(audio+'.npz')
            if sp.exists():
                with np.load(sp, allow_pickle=False) as previous:
                    if not all(np.array_equal(previous[k], v) for k, v in summary.items()):
                        raise ValueError('Previously prepared summary differs')
            else:
                np.savez(sp, **summary)
            for p in (mp, sp):
                hashes[str(p.relative_to(ROOT))] = sha(p)
            record = dict(audio_sha256=audio, decoded_samples=samples,
                duration_us=samples*1_000_000//24000, paths=song['paths'], sha256=hashes,
                tensors=stats, complete_mel=str(mp.relative_to(ROOT)), song_summary=str(sp.relative_to(ROOT)),
                mel=tensor_health(mel), summary=tensor_health(summary['values']),
                valid_mel_frames=int(valid.sum()), invalid_padding_frames=int((~valid).sum()),
                missing_pupu_tail_samples=samples % 960, legacy_prefix_parity=parity,
                reused_complete_mel_probe=reused_probe, passed=True,
                interpretation='Array and clock evidence only; masks/model wiring and musical quality are separate gates.')
            report['records'].append(record)
            attempt['new_songs'] += 1
            save()
        except Exception as exc:
            report['failures'].append(dict(audio_sha256=audio, error=f'{type(exc).__name__}: {exc}'))
            save()
            raise
    attempt.update(ended_at_unix=time.time(), wall_s=time.monotonic()-started)
    save()
    print(json.dumps(dict(complete=report['complete'], completed_songs=report['completed_songs'],
                         expected_songs=len(songs), new_songs=attempt['new_songs'], wall_s=attempt['wall_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

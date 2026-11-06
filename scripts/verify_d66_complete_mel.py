"""Check the isolated complete-tail mel recipe on real decoded source songs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.audio.local_mel_cache import complete_song_mel, legacy_prefix_comparison
from beatfcn.train.local_audio_audit import tensor_health
from scripts.audit_phigros_clock import decode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'reports/d66_audio_clock_v1.json'
    report = json.loads(source.read_text())
    if not report['complete'] or report['summary']['errors']:
        raise ValueError('Complete successful decode audit required')
    output = ROOT/'reports/d66_complete_mel_probe_v1.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = {r['chart_id']: r for r in map(json.loads,
        (ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl').read_text().splitlines())}
    charts = {rows[r['chart_id']]['audio_sha256']: rows[r['chart_id']] for r in report['records']}
    selected = {}
    for audio in report['audio']:
        if audio['legacy_trimmed_samples']:
            row = charts[audio['audio_sha256']]
            selected.setdefault((row['mode'], row['codec']), audio)
    results = []
    for (mode, codec), audio in sorted(selected.items()):
        path = ROOT/audio['raw_path']
        if sha(path) != audio['audio_sha256']:
            raise ValueError('Original music identity changed')
        wave = decode(path, 45)
        if len(wave) != audio['decoded_samples']:
            raise ValueError('Decoder sample clock changed')
        complete = complete_song_mel(wave)
        legacy_path = ROOT/audio['mel_path']
        if sha(legacy_path) != audio['mel_sha256']:
            raise ValueError('Audited legacy mel changed')
        legacy = np.load(legacy_path, allow_pickle=False)
        parity = legacy_prefix_comparison(legacy, complete)
        conversion = np.ascontiguousarray(np.asfortranarray(legacy))
        c_order_exact = np.array_equal(conversion, legacy)
        ends, starts, valid = (complete[k] for k in ('frame_end_samples', 'frame_start_samples', 'valid'))
        intervals_ok = (starts[0] == 0 and ends[-1] == len(wave)
            and np.array_equal(ends[:-1], starts[1:]) and np.array_equal(valid, ends > starts)
            and int((ends-starts).sum()) == len(wave))
        cache = ROOT/'inputs/d66_local_cohort/decoded/mel10_complete_probe'
        cache.mkdir(parents=True, exist_ok=True)
        artifact = cache/(audio['audio_sha256']+'.npz')
        np.savez(artifact, **complete)
        results.append(dict(mode=mode, codec=codec, audio_sha256=audio['audio_sha256'],
            decoded_samples=len(wave), padding_samples=complete['padding_samples'],
            waveform=tensor_health(wave), mel=tensor_health(complete['mel']),
            valid_frames=int(valid.sum()), total_frames=len(valid),
            last_valid_frame_samples=int((ends-starts)[valid][-1]),
            intervals_cover_each_real_sample_once=bool(intervals_ok),
            fortran_to_c_values_unchanged=bool(c_order_exact), prefix_parity=parity,
            artifact=str(artifact.relative_to(ROOT)), artifact_sha256=sha(artifact),
            passed=bool(intervals_ok and c_order_exact and parity['interior_exact_equal'])))
    result = dict(id='D66-complete-mel-real-probe-v1', complete=True,
        source_report_sha256=sha(source), code_sha256={p: sha(ROOT/p) for p in (
            'scripts/verify_d66_complete_mel.py', 'beatfcn/audio/local_mel_cache.py',
            'beatfcn/audio/preprocess.py', 'scripts/audit_phigros_clock.py')},
        records=results, passed=bool(results) and all(r['passed'] for r in results),
        training_activated=False, optimizer_updates=0, cloud_compute_calls=0,
        limitations=['Real recipe/clock probe only; the full corpus has not been converted.',
            'The reflected legacy boundary legitimately changes; its final four frames are not a parity claim.',
            'Pupu partial-cell availability and model padding-mask consumption are still unverified.'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], audio=len(results),
        modes_and_codecs=[(r['mode'], r['codec']) for r in results],
        max_interior_difference=max(r['prefix_parity']['max_abs_difference'] for r in results))))


if __name__ == '__main__':
    main()

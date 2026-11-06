"""Independent audio-clock diagnostics for external charts and the rejected preview."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import Event, lane_conflicts
from beatfcn.events.osu_mania import parse_osu
from beatfcn.ingest.pipeline import decode, mel10
from beatfcn.model.benchmark import audio_features, music_metrics, product_metrics
from beatfcn.model.evaluation import score_events

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'artifacts/chart-preview-20261025'
RECORD = ROOT / 'reports/condition_v1_benchmark/records/412740394827aa998f539b5f.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_health(values):
    a = np.asarray(values)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(a).sum()),
                inf=int(np.isinf(a).sum()), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean(dtype=np.float64)), std=float(a.std(dtype=np.float64)))


def inspect(events, truth, features, wave, duration_us):
    runs = [(0, duration_us)]
    music, music_coverage = music_metrics(events, runs, features)
    product, product_coverage = product_metrics(events, runs, 'mania_4k')
    seconds = int(np.ceil(duration_us / 1e6))
    rms = [float(np.sqrt(np.mean(wave[k * 24000:(k + 1) * 24000] ** 2))) for k in range(seconds)]
    median_rms = float(np.median(rms))
    groups = sorted({e.onset_us for e in events})
    gaps = []
    for a, b in zip(groups, groups[1:]):
        if b - a < 2_000_000:
            continue
        span = wave[round(a * .024):round(b * .024)]
        gap_rms = float(np.sqrt(np.mean(span ** 2))) if len(span) else None
        held = sum(max(0, min(b, e.end_us) - max(a, e.onset_us)) for e in events if e.is_hold)
        gaps.append(dict(start_s=a / 1e6, end_s=b / 1e6, seconds=(b - a) / 1e6,
                         rms_relative_to_song_median=gap_rms / median_rms if median_rms and gap_rms is not None else None,
                         hold_lane_seconds=held / 1e6,
                         reference_heads_inside=sum(a < e.onset_us < b for e in truth)))
    counts = [sum(k * 1e6 <= e.onset_us < (k + 1) * 1e6 for e in events) for k in range(seconds)]
    truth_counts = [sum(k * 1e6 <= e.onset_us < (k + 1) * 1e6 for e in truth) for k in range(seconds)]
    return dict(notes=len(events), holds=sum(e.is_hold for e in events), groups=len(groups),
        music=music, music_coverage=music_coverage, product=product, product_coverage=product_coverage,
        reference_reconstruction=score_events(events, truth, 'mania_4k'),
        lane_conflicts=lane_conflicts(events),
        invalid_bounds=[e.to_json() for e in events if not 0 <= e.onset_us <= e.end_us <= duration_us
                        or e.is_hold and e.onset_us == e.end_us],
        internal_head_gaps_ge2s=gaps, max_internal_head_gap_s=max(np.diff(groups), default=0) / 1e6,
        leading_head_gap_s=groups[0] / 1e6 if groups else None,
        trailing_head_gap_s=(duration_us - groups[-1]) / 1e6 if groups else None,
        per_second=dict(note_count=counts, reference_count=truth_counts, waveform_rms=rms),
        peak_notes_per_fixed_second=max(counts, default=0),
        gap_caveat='A gap or a difference from one human chart is not automatically an error; inspect audio and holds.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chart', action='append', default=[], help='LABEL=path/to/generated.osu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    meta = json.loads((PREVIEW / 'manifest.json').read_text())
    assert sha(PREVIEW / 'audio.mp3') == meta['audio_sha256']
    record = json.loads(RECORD.read_text())
    # This is the existing fixed codec-clock convention from the rejected preview,
    # not a new shift fitted against a prediction or target.
    correction = meta['source_clock_correction_ms'] * 1000
    truth = [replace(Event.from_json(e), onset_us=e['onset_us'] + correction,
                     end_us=e['end_us'] + correction) for e in record['truth_events']]
    original = parse_osu((PREVIEW / 'original.osu').read_bytes())
    assert not original.quarantined
    expected = sorted((e['onset_us'] // 1000 + correction // 1000,
                       e['end_us'] // 1000 + correction // 1000, e['type'], e['location']['lane'])
                      for e in record['generated_events'])
    actual = sorted((e.onset_us // 1000, e.end_us // 1000, e.type, e.location.lane) for e in original.events)
    assert actual == expected, 'original export and saved predictions disagree'
    wave = decode(str(PREVIEW / 'audio.mp3'))
    duration = round(len(wave) / 24000 * 1e6)
    mel = mel10(wave)
    features = audio_features(mel, duration)
    folder = ROOT / 'inputs/d47_external_baseline'
    folder.mkdir(parents=True, exist_ok=True)
    # Immutable audio features used by both baseline and subsequent reads.
    cache = folder / 'mel10.npy'
    if cache.exists():
        assert np.array_equal(np.load(cache), mel)
    else:
        np.save(cache, mel)
    charts = {'previous_model': list(original.events), 'human_reference': truth}
    sources = {'audio': str(PREVIEW / 'audio.mp3'), 'old_record': str(RECORD)}
    for entry in args.chart:
        label, path = entry.split('=', 1)
        if label in charts:
            raise ValueError('duplicate label')
        parsed = parse_osu(Path(path).read_bytes())
        if parsed.quarantined:
            raise ValueError(f'invalid external export: {parsed.issues}')
        charts[label] = list(parsed.events)
        sources[label] = path
    result = dict(id='D47-external-mania-observation', completed_at_unix=time.time(),
        duration_s=duration / 1e6, source_clock_correction_us=correction,
        source_sha256={k: sha(v) for k, v in sources.items()},
        waveform=tensor_health(wave), mel=tensor_health(mel), mel_file_sha256=sha(cache),
        charts={k: inspect(v, truth, features, wave, duration) for k, v in charts.items()},
        inference_updates=0, evaluation_wall_s=time.time() - started,
        caveats=['One already rejected song; no population or unseen-data claim for external pretraining.',
                 'This is a different complete pretrained system, not a controlled BeatFCN architecture ablation.',
                 'Reference reconstruction is descriptive; the requested difficulties differ.',
                 'Audio-only flux metrics are onset proxies, not musical-quality certificates.',
                 'No Phira, completion, reference control or BeatFCN promotion is established.'])
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {f: v[f] for f in ('notes', 'holds', 'max_internal_head_gap_s', 'peak_notes_per_fixed_second')}
                      for k, v in result['charts'].items()}))


if __name__ == '__main__':
    main()

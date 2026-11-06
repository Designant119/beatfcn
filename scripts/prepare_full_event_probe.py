"""Prepare real Pupu inputs for the full-sized E-D implementation probe."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.canonical import Event, build_model_view
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventState, EventVocabulary, encode_events, teacher_sequence
from beatfcn.ingest.phira_policy import phira_stratum, screen_metadata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(a):
    a = np.asarray(a)
    f = a.astype(np.float64)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(f).sum()),
                inf=int(np.isinf(f).sum()), min=float(f.min()) if f.size else None,
                max=float(f.max()) if f.size else None, mean=float(f.mean()) if f.size else None,
                std=float(f.std()) if f.size else None)


def main():
    output = ROOT / 'inputs/diagnostic_panels/d58-full-event'
    if output.exists():
        raise FileExistsError(output)
    paths = [ROOT / 'inputs/r09_positive_pairs/catalog.pt',
             ROOT / 'inputs/r08_local_real_probe/pca/i8_9-16-24.npz',
             ROOT / 'inputs/data_scale_v2/catalog_v1/phira/catalog.jsonl',
             Path(__file__).resolve(), ROOT / 'beatfcn/model/full_event_codec.py',
             ROOT / 'beatfcn/events/canonical.py', ROOT / 'beatfcn/ingest/phira_policy.py']
    catalog = torch.load(paths[0], map_location='cpu', weights_only=False)
    scales = np.load(paths[1])
    metadata = {str(r['id']): r for r in map(json.loads, paths[2].read_text().splitlines())}
    # These already-frozen real pairs cover both modes and all four Phira types.
    selections = [('r09-pair-0000', 0), ('r09-pair-0128', 1)]
    arrays, records = {}, []
    for index, (pair_id, side) in enumerate(selections):
        entry = next(p for p in catalog['train_pairs'] if p['pair_id'] == pair_id)['entries'][side]
        row, example = entry['row'], entry['example']
        if row['split'] != 'train':
            raise ValueError('Only training data may enter this fitting probe')
        if row['mode'] == 'phira' and screen_metadata(metadata[row['native_id']])['status'] != 'eligible_metadata_only':
            raise ValueError('Probe chart is excluded by current metadata screening')
        c0, duration = int(example.c0_us), int(entry['request']['audio_duration_us'])
        request = EventRequest(row['mode'], c0, min(c0 + 8_000_000, duration), duration)
        native = [Event.from_json(e) for e in entry['truth_events']]
        view = build_model_view(row['mode'], (), native)
        if view.quarantine:
            raise ValueError(f'Invalid current model view: {view.quarantine}')
        events = [RhythmEvent.from_event(e, row['mode']) for group in view.target_groups for e in group.events]
        tokens = encode_events(events, request)
        sequence = teacher_sequence(tokens, request)
        features = ROOT / 'inputs/r09_positive_pairs/taps/v4_i8_L9-16-24' / row['audio_sha256']
        tap_files = [features / f'L{layer}.npy' for layer in (9, 16, 24)]
        paths.extend(tap_files)
        raw = [np.load(p, mmap_mode='r') for p in tap_files]
        if any(a.dtype != np.int8 or a.shape[1:] != (8, 1024) for a in raw) or len({len(a) for a in raw}) != 1:
            raise ValueError('Real frontend cache shape/dtype mismatch')
        n = len(raw[0])
        left, right = max(0, c0 // 40_000 - 100), min(n, c0 // 40_000 + 300)
        local = np.stack([np.asarray(a[left:right], np.float32) * scales[f'scale_{layer}']
                          for layer, a in zip((9, 16, 24), raw)])
        # Pure audio only: recompute half-second summaries from raw layer 24,
        # never reuse P1's chart/controls-conditioned song-stream tokens.
        bins = ((np.arange(n, dtype=np.int64) * 40_000 + 20_000) // 500_000).astype(int)
        count = np.bincount(bins)
        song = np.zeros((len(count), 1024), np.float32)
        for block in range(len(count)):
            lo, hi = np.searchsorted(bins, (block, block + 1))
            song[block] = np.asarray(raw[-1][lo:hi], np.float32).mean((0, 1)) * scales['scale_24']
        local_times = ((np.arange(left, right) * 40_000 + 20_000 - c0) / 1e6).astype(np.float32)
        centers = np.bincount(bins, weights=np.arange(n) * .04 + .02) / count
        song_times = (centers - c0 / 1e6).astype(np.float32)
        # Explicit unknown interval: no unobserved chart events or target counts.
        reference = np.zeros((1, 16), np.float32)
        reference[0, 1] = (request.end_us - request.start_us) / 1_200_000_000
        reference[0, 15] = 1
        controls = np.zeros((8, 6), np.float32)
        applicable = [1, 1, 1, 1, 1, int(row['mode'] == 'mania_4k'), int(row['mode'] == 'phira'), int(row['mode'] == 'phira')]
        controls[:, 3] = applicable
        inputs = dict(local=local, local_times=local_times, local_valid=np.ones(right-left, bool),
            song=song, song_times=song_times, song_valid=np.ones(len(song), bool),
            reference=reference, reference_times=np.zeros(1, np.float32), reference_valid=np.ones(1, bool),
            controls=controls, mode=np.asarray(0 if row['mode'] == 'mania_4k' else 1, np.int64))
        for name, a in inputs.items():
            arrays[f'{index}:input:{name}'] = a
        for name, a in sequence.items():
            arrays[f'{index}:sequence:{name}'] = a
        health = {k: stats(v) for k, v in {**inputs, **{f'target_{k}': v for k, v in sequence.items()}}.items()}
        if any(v['nan'] or v['inf'] for v in health.values()):
            raise ValueError('Nonfinite real tensor')
        state = EventState(request)
        for token in tokens:
            state.apply(token)
        if tuple(sorted(state.events)) != tuple(sorted(events)):
            raise ValueError('Target multiset was altered')
        records.append(dict(index=index, pair_id=pair_id, side=side, chart_id=row['chart_id'],
            mode=row['mode'], split='train', audio_sha256=row['audio_sha256'], request=asdict(request),
            source_core_start_us=c0, local_cells=[left, right], full_song_cells=n,
            feature_clock='P1 v4 cell centers: i*40ms+20ms; retained, not recalibrated by this probe',
            clock_evidence=row.get('offset_decision'),
            controls='all unknown; applicable mode flags only; no hidden target-derived control values',
            visible_reference='explicit unknown interval only; no hidden target events',
            native_target_count=len(native), model_target_count=len(events), tokens=len(tokens),
            target_types=dict(Counter(e.type for e in events)), target_events=[asdict(e) for e in events],
            quantized_events=len(view.quantized_from),
            phira_stratum=phira_stratum(metadata[row['native_id']]) if row['mode'] == 'phira' else None,
            tensors=health, native_package_identity_reverified=False))
    output.mkdir(parents=True)
    np.savez_compressed(output / 'inputs.npz', **arrays)
    manifest = dict(id='D58-full-event-real-inputs', records=records,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
        tensor_sha256=sha(output/'inputs.npz'), vocabulary=asdict(EventVocabulary()),
        model_updates=0, new_cloud_spend_usd=0, scope='Real existing training features for implementation/fitting only; not a new corpus admission or benchmark.')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(charts=len(records), events=sum(r['model_target_count'] for r in records),
        tokens=sum(r['tokens'] for r in records), input_files=len(manifest['input_sha256'])), ensure_ascii=False))


if __name__ == '__main__':
    main()

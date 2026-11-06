"""Decoded-clock evidence, without correcting labels from acoustic peaks."""
from collections import Counter

import numpy as np

from beatfcn.audio.preprocess import CELL_SAMPLES, HOP, SR
from beatfcn.ingest.offset import estimate_offset


def tensor_health(array):
    array = np.asarray(array)
    result = dict(shape=list(array.shape), dtype=str(array.dtype),
                  nan=int(np.isnan(array).sum()), inf=int(np.isinf(array).sum()))
    if not array.size or result['nan'] or result['inf']:
        raise ValueError(f'Invalid audio tensor: {result}')
    result.update(min=float(array.min()), max=float(array.max()),
                  mean=float(array.mean(dtype=np.float64)), std=float(array.std(dtype=np.float64)))
    return result


def decoded_bounds(events, samples, shift_us=0):
    """Use integer sample time, not the rounded manifest duration."""
    bad = []
    for event in events:
        head, tail = event.onset_us + shift_us, event.end_us + shift_us
        if head < 0 or head * SR >= samples * 1_000_000 or tail * SR > samples * 1_000_000:
            bad.append(event.event_id)
    return dict(count=len(bad), examples=bad[:20])


def tail_coverage(events, samples, shift_us=0):
    covered_samples = samples // CELL_SAMPLES * CELL_SAMPLES
    heads, tails = [], []
    for event in events:
        head, tail = event.onset_us + shift_us, event.end_us + shift_us
        if covered_samples * 1_000_000 <= head * SR < samples * 1_000_000:
            heads.append(event.event_id)
        if event.is_hold and covered_samples * 1_000_000 < tail * SR <= samples * 1_000_000:
            tails.append(event.event_id)
    return dict(legacy_covered_samples=covered_samples,
                omitted_samples=samples-covered_samples,
                omitted_ms=(samples-covered_samples)*1000/SR,
                final_cell_valid_samples=samples % CELL_SAMPLES or CELL_SAMPLES,
                legacy_mel_frames=covered_samples//HOP,
                complete_mel_frames=((samples+CELL_SAMPLES-1)//CELL_SAMPLES)*CELL_SAMPLES//HOP,
                heads_in_omitted_tail=len(heads), hold_ends_in_omitted_tail=len(tails),
                head_examples=heads[:20], hold_end_examples=tails[:20])


def acoustic_clock(events, envelope, shift_us=0):
    """Measure selected note heads, not acoustic onset ground truth or playability."""
    def estimate(heads, env):
        result = estimate_offset(heads, env=env)
        result.pop('agree_counts', None)
        return result

    heads = {name: np.unique([(e.onset_us+shift_us)/1e6 for e in events
                             if name == 'all' or e.type in types])
             for name, types in [('all', ()), ('tap_hold', ('tap', 'hold')),
                                 ('drag', ('drag',)), ('flick', ('flick',))]}
    whole = {name: estimate(values, envelope) for name, values in heads.items()}
    thirds = []
    for index in range(3):
        lo, hi = len(envelope)*index//3, len(envelope)*(index+1)//3
        thirds.append(estimate(heads['tap_hold']-lo*HOP/SR, envelope[lo:hi]))
    reliable = lambda r: r.get('status') == 'ok' and r['conf'] >= 3 and not r['edge']
    span = (max(r['delta_ms'] for r in thirds)-min(r['delta_ms'] for r in thirds)
            if all(reliable(r) for r in thirds) else None)
    return dict(whole=whole, tap_hold_thirds=thirds,
                reliable_thirds=sum(map(reliable, thirds)), thirds_span_ms=span,
                acoustic_alignment_verified=False,
                interpretation='Peak agreement is diagnostic; weak evidence or drag mismatch does not imply dirty labels.')


def chart_audio_report(row, events, envelope, samples, shift_us):
    duration_error_ms = (samples/SR-row['audio_sec'])*1000
    native = decoded_bounds(events, samples)
    shifted = decoded_bounds(events, samples, shift_us)
    return dict(chart_id=row['chart_id'], mode=row['mode'], split=row['split'],
                types=dict(Counter(e.type for e in events)),
                model_shift_us=shift_us, decoded_duration_error_ms=duration_error_ms,
                duration_matches_rounded_manifest=round(samples/SR, 4) == row['audio_sec'],
                native_bounds=native, model_bounds=shifted,
                native_tail=tail_coverage(events, samples),
                model_tail=tail_coverage(events, samples, shift_us),
                native_acoustic=acoustic_clock(events, envelope),
                model_acoustic=acoustic_clock(events, envelope, shift_us),
                label_changes=0, training_ready=False)

"""CPU capability diagnostics; reference ranges are descriptive, never quality certificates.

No chart labels enter audio feature extraction. All times use the existing model/audio
clock; never optimize an alignment shift against a generated chart. Missing, constant,
empty and unsupported measurements stay null and retain their denominators.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import math
from pathlib import Path
import numpy as np
from .control_eval import spearman
from ..events.controls import union

VERSION = 'beatfcn.capability_benchmark.v2'
HOP_US = 10_000
BLOCK_US = 2_000_000
SHIFT_US = (137_000, 313_000, 617_000)
REQUEST_KINDS = ('from_scratch', 'middle_holes', 'reference')


def metric_source_sha256():
    digest = hashlib.sha256()
    root = Path(__file__).parent.parent
    for name in ('model/evaluation.py', 'model/benchmark.py', 'model/control_eval.py', 'model/drag_benchmark.py',
                 'model/edit_utility.py', 'events/reference_policy.py', 'ingest/offset.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def audio_features(mel, duration_us):
    """Existing 10 ms Pupu log-mel cache -> fixed audio-only diagnostic features.

    This reuses the ingestion flux recipe, not a learned onset oracle. Mean normalized
    log-mel is an energy PROXY, not waveform RMS. Peaks have 10 ms resolution. Strong
    peaks are the upper quartile of detected peak strengths (ties retained).
    """
    from ..ingest.offset import flux_envelope, PEAK_MIN
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != 128 or len(m) < 51 or not np.isfinite(m).all():
        raise ValueError('audio benchmark requires finite [T>=51,128] mel10')
    # Ingestion truncates decoded waveforms to whole 40 ms cells (not padding).
    tail_us = duration_us - len(m) * HOP_US
    if duration_us <= 0 or not -50 <= tail_us < 40_050:
        raise ValueError('mel10 duration does not match the evaluation audio clock')
    env = flux_envelope(m)
    peak = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) &
                         (env[1:-1] > PEAK_MIN)) + 1
    peak = peak[peak * HOP_US < duration_us]
    strong = peak[env[peak] >= np.quantile(env[peak], .75)] if len(peak) else peak
    raw_flux = np.concatenate([[0.], np.maximum(m[1:] - m[:-1], 0).sum(axis=1)])
    return {'recipe': 'mel10_positive_flux_v1_offset_peaks_v1_strong_q75', 'hop_us': HOP_US,
            'duration_us': int(duration_us), 'mel_sha256': hashlib.sha256(m.tobytes()).hexdigest(),
            'coverage_end_us': min(duration_us, len(m) * HOP_US), 'uncovered_tail_us': max(0, tail_us),
            'onsets_us': peak * HOP_US, 'strong_onsets_us': strong * HOP_US,
            'flux': raw_flux, 'energy_proxy': m.mean(axis=1),
            'energy_definition': 'mean normalized log-mel, not waveform RMS'}


def _inside(t, runs):
    return any(a <= t < b for a, b in runs)


def _nearest_distances(query, reference):
    q, r = np.asarray(query), np.sort(reference)
    if not len(r):
        return np.full(len(q), np.inf)
    k = np.searchsorted(r, q)
    return np.minimum(abs(q - r[np.clip(k - 1, 0, len(r) - 1)]),
                      abs(q - r[np.clip(k, 0, len(r) - 1)]))


def _matched_count(left, right, tolerance):
    left, right = sorted(set(left)), sorted(set(right))
    i = j = matches = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j] - tolerance:
            i += 1
        elif right[j] < left[i] - tolerance:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def _count_blocks(events, runs):
    starts = [k * BLOCK_US for a, b in runs for k in range(-(-a // BLOCK_US), b // BLOCK_US)]
    times = np.sort([e.onset_us for e in events])
    counts = [np.searchsorted(times, a + BLOCK_US) - np.searchsorted(times, a) for a in starts]
    return starts, np.asarray(counts, dtype=float)


def music_metrics(events, runs, audio):
    names = [f'anchor_{ms}ms_{x}' for ms in (20, 50, 100)
             for x in ('precision', 'strong_recall', 'shift_excess')]
    out = {name: None for name in names}
    out.update(density_flux_spearman=None, density_energy_proxy_spearman=None)
    counts = {'notes': 0, 'onset_groups': 0, 'audio_peaks': 0, 'strong_peaks': 0, 'full_blocks': 0}
    if audio is None:
        return out, {**counts, 'status': 'missing_audio'}
    uncovered = sum(_inside(e.onset_us, runs) and e.onset_us >= audio.get('coverage_end_us', audio['duration_us'])
                    for e in events)
    runs = [(a, min(b, audio.get('coverage_end_us', audio['duration_us']))) for a, b in runs
            if a < audio.get('coverage_end_us', audio['duration_us'])]
    counts['notes_outside_audio_coverage'] = uncovered
    notes = [e.onset_us for e in events if _inside(e.onset_us, runs)]
    groups = sorted(set(notes))
    peaks = [int(t) for t in audio['onsets_us'] if _inside(t, runs)]
    strong = [int(t) for t in audio['strong_onsets_us'] if _inside(t, runs)]
    counts.update(notes=len(notes), onset_groups=len(groups), audio_peaks=len(peaks), strong_peaks=len(strong))
    near = _nearest_distances(notes, peaks)
    # Recall is one-to-one on distinct groups: neither chords nor nearby audio
    # peaks can multiply the credit for one musical onset.
    for ms in (20, 50, 100):
        precision = float(np.mean(near <= ms * 1000)) if notes and peaks else None
        out[f'anchor_{ms}ms_precision'] = precision
        out[f'anchor_{ms}ms_strong_recall'] = _matched_count(strong, groups, ms * 1000) / len(strong) if strong else None
        if precision is not None:
            controls = []
            # Rotate notes within EACH editable interval: preserve density and permissions.
            for shift in SHIFT_US:
                shifted = [a + (t - a + shift) % (b - a) for t in notes for a, b in runs if a <= t < b]
                controls.append(float(np.mean(_nearest_distances(shifted, peaks) <= ms * 1000)))
            out[f'anchor_{ms}ms_shift_excess'] = precision - float(np.mean(controls))
    starts, counts_per_block = _count_blocks(events, runs)
    counts['full_blocks'] = len(starts)
    if len(starts) >= 10:
        for key, field in (('flux', 'density_flux_spearman'), ('energy_proxy', 'density_energy_proxy_spearman')):
            block_audio = [float(np.mean(audio[key][a // HOP_US:(a + BLOCK_US) // HOP_US])) for a in starts]
            out[field] = spearman(counts_per_block, block_audio)
    return out, {**counts, 'status': 'ok' if peaks else 'no_detected_onsets'}


def beat_grid_metrics(events, runs, beats_us):
    """Optional AUDIO-only local beat intervals; no global BPM or best-fit phase search.

    Caller must retain the beat estimator identity. These remain diagnostics: estimated
    beats can be wrong. Subdivisions are reported separately, never selected by score.
    """
    beats = np.asarray(beats_us, dtype=float)
    if beats.ndim != 1 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError('beat times must be finite and strictly increasing')
    errors = {n: [] for n in (1, 2, 3, 4)}
    for t in sorted({e.onset_us for e in events if _inside(e.onset_us, runs)}):
        k = int(np.searchsorted(beats, t, side='right')) - 1
        if not 0 <= k < len(beats) - 1:
            continue
        length = beats[k + 1] - beats[k]
        phase = (t - beats[k]) / length
        for n in errors:
            errors[n].append(abs(phase - round(phase * n) / n) * length / 1000)
    return {f'subdivision_{n}': {'covered_groups': len(v), 'median_error_ms': float(np.median(v)) if v else None,
                                'within_20ms': float(np.mean(np.asarray(v) <= 20)) if v else None}
            for n, v in errors.items()}


def _entropy(counter, normalizer=None):
    total = sum(counter.values())
    if not total:
        return None
    p = np.asarray(list(counter.values()), float) / total
    h = float(-np.sum(p * np.log2(p)))
    return h / normalizer if normalizer else h


def _sequences(events, runs, mode):
    """Simultaneous mania notes form one lane bitmask; no arbitrary within-chord order."""
    sequences, iois = [], []
    for a, b in runs:
        groups = {}
        for e in events:
            if a <= e.onset_us < b:
                groups.setdefault(e.onset_us, []).append(e)
        times = sorted(groups)
        iois.extend(np.diff(times).tolist())
        if mode == 'mania_4k':
            sequences.append([sum(1 << lane for lane in {e.location.lane for e in groups[t]}) for t in times])
    return sequences, iois


def _ngrams(sequences, n):
    return Counter(tuple(seq[i:i + n]) for seq in sequences for i in range(len(seq) - n + 1))


def product_metrics(events, runs, mode, observed=()):
    notes = [e for e in events if _inside(e.onset_us, runs)]
    groups = Counter(e.onset_us for e in notes)
    seq, iois = _sequences(notes, runs, mode)
    _, density = _count_blocks(notes, runs)
    out = {'hold_fraction': sum(e.is_hold for e in notes) / len(notes) if notes else None,
           'chord_note_fraction': sum(groups[e.onset_us] > 1 for e in notes) / len(notes) if notes else None,
           'density_notes_per_second': len(notes) / (sum(b - a for a, b in runs) / 1e6) if runs else None,
           'density_block_cv': float(density.std() / density.mean()) if len(density) >= 2 and density.mean() else None,
           'ioi_median_ms': float(np.median(iois) / 1000) if iois else None,
           'same_lane_min_gap_ms': None, 'release_to_repress_min_ms': None,
           'rapid_same_lane_fraction': None, 'hand_onset_imbalance': None,
           'lane_bigram_entropy': None, 'lane_trigram_entropy': None}
    counts = {'notes': len(notes), 'onset_groups': len(groups), 'iois': len(iois), 'same_lane_pairs': 0,
              'rapid_threshold_ms': 100, 'lane_applicable': mode == 'mania_4k'}
    if mode == 'mania_4k':
        gaps, releases = [], []
        target_ids = {id(e) for e in notes}
        for lane in range(4):
            lane_events = sorted([e for e in [*notes, *observed] if e.location.lane == lane], key=lambda e:e.onset_us)
            for left, right in zip(lane_events, lane_events[1:]):
                # Include context at a seam, but never join two separate editable holes.
                if id(right) in target_ids and (id(left) not in target_ids or
                        any(a <= left.onset_us <= right.onset_us < b for a, b in runs)):
                    gaps.append((right.onset_us - left.onset_us) / 1000)
                    if left.is_hold:
                        releases.append((right.onset_us - left.end_us) / 1000)
        counts['same_lane_pairs'] = len(gaps)
        out.update(same_lane_min_gap_ms=min(gaps) if gaps else None,
                   release_to_repress_min_ms=min(releases) if releases else None,
                   rapid_same_lane_fraction=float(np.mean(np.asarray(gaps) < 100)) if gaps else None,
                   hand_onset_imbalance=abs(sum(e.location.lane < 2 for e in notes) -
                                           sum(e.location.lane >= 2 for e in notes)) / len(notes) if notes else None)
        for n, label in ((2, 'bigram'), (3, 'trigram')):
            out[f'lane_{label}_entropy'] = _entropy(_ngrams(seq, n), n * math.log2(15))
    return out, counts


def _js(a, b):
    if not sum(a.values()) or not sum(b.values()):
        return None
    keys = sorted(a.keys() | b.keys())
    p, q = (np.array([c[k] for k in keys], dtype=float) / sum(c.values()) for c in (a, b))
    m = (p + q) / 2
    return float(sum(np.sum(x[x > 0] * np.log2(x[x > 0] / m[x > 0])) for x in (p, q)) / 2)


def compare_vocabulary(pred, truth, runs, mode):
    ps, pi = _sequences(pred, runs, mode)
    ts, ti = _sequences(truth, runs, mode)
    # Include <10ms and >4000ms tails; no discarded out-of-range IOIs.
    edges = np.geomspace(10_000, 4_000_000, 25)
    hist = lambda x: Counter(np.searchsorted(edges, x, side='right').tolist())
    return {'ioi_js_bits': _js(hist(pi), hist(ti)),
            'lane_bigram_js_bits': _js(_ngrams(ps, 2), _ngrams(ts, 2)) if mode == 'mania_4k' else None,
            'lane_trigram_js_bits': _js(_ngrams(ps, 3), _ngrams(ts, 3)) if mode == 'mania_4k' else None}


def capability_metrics(pred, truth, request, audio=None, *, audio_error=None):
    runs = union((r['start_us'], r['end_us']) for r in request['regions'] if r['state'] == 'editable')
    from ..events.canonical import Event
    observed = [Event.from_json(e) for e in request.get('observed_events', [])]
    result = {'version': VERSION, 'scope': 'generated additions in editable union',
              'audio': {'status': 'missing_audio', 'reason': audio_error}, 'generated': {}, 'human': {}}
    if audio is not None:
        result['audio'] = {k: v for k, v in audio.items()
                           if k not in ('onsets_us', 'strong_onsets_us', 'flux', 'energy_proxy', 'beats_us')}
        result['audio']['status'] = 'ok'
    for label, events in (('generated', pred), ('human', truth)):
        musical, mc = music_metrics(events, runs, audio)
        product, pc = product_metrics(events, runs, request['mode'], observed)
        result[label] = {**musical, **product}
        result[f'{label}_coverage'] = {'music': mc, 'product': pc}
        if audio is not None and 'beats_us' in audio:
            result[f'{label}_grid'] = beat_grid_metrics(events, runs, audio['beats_us'])
    result['reference_distances'] = compare_vocabulary(pred, truth, runs, request['mode'])
    return result


def scorecard(records, *, expected_modes=('mania_4k', 'phira'), expected_kinds=REQUEST_KINDS):
    """One song gets one vote, including across seeds. Failed songs stay in coverage.

    5–95% human ranges describe this sample, with >=5 unique songs required; they
    are not tuned pass thresholds. No global score or overall quality-pass is emitted.
    """
    cells = []
    keys = set((r['mode'], r.get('request_kind', 'from_scratch'), r.get('stratum', 'unstratified')) for r in records)
    for mode in expected_modes:
        for kind in expected_kinds:
            if not any(m == mode and k == kind for m, k, _ in keys):
                keys.add((mode, kind, 'unstratified'))
    for mode, kind, stratum in sorted(keys):
        rs = [r for r in records if (r['mode'], r.get('request_kind', 'from_scratch'),
                                    r.get('stratum', 'unstratified')) == (mode, kind, stratum)]
        unique = {r['song'] for r in rs}
        eligible = [r for r in rs if r.get('complete') and not r.get('legality') and r.get('benchmark') and
                    r['benchmark']['generated_coverage']['product']['notes'] > 0]
        metrics = {}
        names = sorted({k for r in rs for k in r.get('benchmark', {}).get('generated', {})})
        for name in names:
            by_song = {'generated': {}, 'human': {}}
            for label in by_song:
                for r in eligible if label == 'generated' else rs:
                    value = r.get('benchmark', {}).get(label, {}).get(name)
                    if value is not None and np.isfinite(value):
                        by_song[label].setdefault(r['song'], []).append(value)
            g, h = ({s: float(np.mean(v)) for s, v in by_song[label].items()} for label in ('generated', 'human'))
            paired = sorted(g.keys() & h.keys())
            delta = np.array([g[s] - h[s] for s in paired])
            ci = None
            if len(delta) >= 5:
                boot = np.random.default_rng(0).choice(delta, (1000, len(delta)), replace=True).mean(axis=1)
                ci = np.quantile(boot, [.025, .975]).tolist()
            interval = np.quantile(list(h.values()), [.05, .95]).tolist() if len(h) >= 5 else None
            metrics[name] = {'generated_mean': float(np.mean(list(g.values()))) if g else None,
                             'generated_songs': len(g), 'human_songs': len(h),
                             'human_p05_p95': interval,
                             'fraction_in_human_range': float(np.mean([interval[0] <= v <= interval[1] for v in g.values()]))
                             if interval is not None and g else None,
                             'paired_songs': len(paired), 'paired_delta_ci95': ci,
                             'verdict': 'descriptive_only' if g else 'not_measured'}
        distances = {}
        for name in ('ioi_js_bits', 'lane_bigram_js_bits', 'lane_trigram_js_bits'):
            by_song = {}
            for r in eligible:
                value = r['benchmark'].get('reference_distances', {}).get(name)
                if value is not None:
                    by_song.setdefault(r['song'], []).append(value)
            values = [float(np.mean(v)) for v in by_song.values()]
            distances[name] = {'mean':float(np.mean(values)) if values else None, 'songs':len(values),
                               'verdict':'reference_distance_only'}
        cells.append({'mode': mode, 'request_kind': kind, 'stratum': stratum,
                      'status': 'not_run' if not rs else 'uncalibrated', 'requests': len(rs), 'songs': len(unique),
                      'completed': sum(bool(r.get('complete')) for r in rs),
                      'illegal': sum(bool(r.get('legality')) for r in rs),
                      'missing_audio': sum(r.get('benchmark', {}).get('audio', {}).get('status') != 'ok' for r in rs),
                      'empty_or_unscored': sum(r.get('benchmark', {}).get('generated_coverage', {}).get('product', {}).get('notes', 0) == 0 for r in rs),
                      'eligible_nonempty': len(eligible), 'metrics': metrics, 'reference_distances':distances})
        cells[-1]['empty_target_silent'] = sum(r.get('complete', False) and not r.get('legality') and
                                              r.get('scores', {}).get('empty_target_silent', False) for r in rs)
        cells[-1]['empty_target_requests'] = sum(r.get('scores', {}).get('empty_target', False) for r in rs)
    return {'schema': VERSION, 'status': 'not_calibrated', 'cells': cells,
            'policy': 'No overall pass: human ranges are descriptive; missing, failed and empty outputs retain coverage.'}

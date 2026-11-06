"""Reference-free best-of-N selection for sampled charts.

Never reads the human target. Uses only what a product request has: the
generated events, the request (and its density curve, if specified) and the
model's own per-frame onset head. A selected sample is a decoding readout,
reported separately from single-sample model capability.
"""
import math

import numpy as np

SHORT_GAP_US = 40_000
DEFAULT_WEIGHTS = dict(onset=1.0, short_gap=4.0, density=0.5, incomplete=10.0)


def _onsets(events):
    return np.unique(np.asarray([e['onset_us'] for e in events], np.int64))


def short_gap_fraction(events):
    onsets = _onsets(events)
    if len(onsets) < 2:
        return 0.
    return float((np.diff(onsets) < SHORT_GAP_US).mean())


def onset_alignment(events, onset_probs, onset_times_s, start_us, *, tolerance_s=.03):
    """Mean onset-head probability at the generated heads minus the window mean.

    Positive = heads sit where the model's audio head hears onsets. Returns 0
    for empty outputs (the expected-count term handles emptiness).
    """
    probs = np.asarray(onset_probs, np.float64)
    times = np.asarray(onset_times_s, np.float64)
    onsets = _onsets(events)
    if not len(onsets) or not len(probs):
        return 0.
    rel = (onsets - start_us) / 1e6
    hits = []
    for t in rel:
        near = np.abs(times - t) <= tolerance_s
        hits.append(probs[near].max() if near.any() else 0.)
    return float(np.mean(hits) - probs.mean())


def expected_count(onset_probs, onset_times_s, *, frame_s=None):
    """Onset-head expected head count: peak picking over the probability curve."""
    probs = np.asarray(onset_probs, np.float64)
    if len(probs) < 3:
        return float(probs.sum())
    peaks = (probs[1:-1] >= probs[:-2]) & (probs[1:-1] > probs[2:]) & (probs[1:-1] >= .5)
    return float(peaks.sum())


def density_mismatch(events, request, curve):
    """Mean |requested - produced| heads/s over specified curve segments.

    curve: list of dict(start_us, end_us, density) in absolute microseconds;
    None or empty means unspecified and contributes 0.
    """
    if not curve:
        return 0.
    onsets = _onsets(events)
    errors = []
    for segment in curve:
        lo, hi = max(segment['start_us'], request.start_us), min(segment['end_us'], request.end_us)
        if hi <= lo:
            continue
        produced = ((onsets >= lo) & (onsets < hi)).sum() / ((hi - lo) / 1e6)
        errors.append(abs(produced - float(segment['density'])))
    return float(np.mean(errors)) if errors else 0.


def score_candidate(prediction, request, *, onset_probs=None, onset_times_s=None,
                    curve=None, weights=None):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    events = prediction.get('events') or []
    parts = dict(
        incomplete=0. if prediction.get('complete') and not prediction.get('failure') else 1.,
        short_gap=short_gap_fraction(events),
        density=density_mismatch(events, request, curve),
        onset=0., count=0.)
    if onset_probs is not None and onset_times_s is not None:
        parts['onset'] = onset_alignment(events, onset_probs, onset_times_s, request.start_us)
        if not curve:
            # Without a density request, the audio head's own expected count is
            # the reference for "how many notes does this music call for".
            seconds = (request.end_us - request.start_us) / 1e6
            expected = expected_count(onset_probs, onset_times_s) / seconds
            parts['density'] = abs(len(_onsets(events)) / seconds - expected)
    score = (w['onset'] * parts['onset'] - w['short_gap'] * parts['short_gap']
             - w['density'] * math.log1p(parts['density']) - w['incomplete'] * parts['incomplete'])
    return score, parts


def select_best(predictions, request, **kwargs):
    """Return (index, scores, parts) of the highest reference-free score."""
    scored = [score_candidate(p, request, **kwargs) for p in predictions]
    scores = [s for s, _ in scored]
    best = int(np.argmax(scores)) if scores else -1
    return best, scores, [p for _, p in scored]


# ---------------------------------------------------------------- whole song
CORE_US = 8_000_000


class _Span:
    __slots__ = ('start_us', 'end_us')

    def __init__(self, start_us, end_us):
        self.start_us, self.end_us = int(start_us), int(end_us)


def score_song(state, *, weights=None, core_us=CORE_US):
    """Reference-free whole-song score of one rollout state (``generate_complete_songs`` song).

    Per-SONG, not per-core, selection: cores are chained (a hold started in core
    i is the next core's ``known_carry``), so splicing cores from different
    samples could orphan or duplicate hold carries. The song score is the sum
    of ``score_candidate`` over its generated cores (each core's events by
    onset; onset probabilities from the core's optional ``probe`` record, else
    the onset term is skipped and density is unscored; a core ``condition``
    record with ``segments`` is scored against that requested curve), minus the incomplete
    weight for every core that was never generated. Reads only the state's
    descriptor, generated events and probes -- never a human target.
    """
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    events = state.get('events') or []
    cores = state.get('cores') or []
    total, sums, onset_cores = 0., {}, 0
    for core in cores:
        lo, hi = core['start_us'], core['end_us']
        probe = core.get('probe') or {}
        probs, times = probe.get('onset_probs'), probe.get('onset_times_s')
        onset_cores += probs is not None and times is not None
        prediction = dict(events=[e for e in events if lo <= e['onset_us'] < hi],
                          complete=core.get('complete', False), failure=None)
        # A core conditioned on an explicit request (local_song_rollout condition_provider,
        # e.g. charter_intent_density) is scored against that request; otherwise unspecified.
        curve = (core.get('condition') or {}).get('segments') or None
        score, parts = score_candidate(prediction, _Span(lo, hi), onset_probs=probs, onset_times_s=times,
                                       curve=curve, weights=w)
        total += score
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.) + v
    expected = max(1, math.ceil(int(state['duration_us']) / core_us))
    missing = max(0, expected - len(cores))
    total -= w['incomplete'] * missing
    song_complete = bool(state.get('complete')) and not state.get('failure')
    return total, dict(song_complete=song_complete, cores=len(cores), missing_cores=missing,
                       onset_term_cores=onset_cores, **{'sum_'+k: v for k, v in sums.items()})


def select_best_song(states, **kwargs):
    """(index, scores, parts) over N rollout states of the SAME song.

    Lexicographic: any complete, legal song beats any incomplete one; then the
    highest ``score_song``; ties keep the lowest index (lowest seed).
    """
    scored = [score_song(s, **kwargs) for s in states]
    if not scored:
        return -1, [], []
    best = max(range(len(scored)), key=lambda i: (scored[i][1]['song_complete'], scored[i][0], -i))
    return best, [s for s, _ in scored], [p for _, p in scored]

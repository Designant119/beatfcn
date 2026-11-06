"""Counterfactual density samples: edit a real target to match a requested curve.

The same audio/window is paired with several targets at different densities.
Unlike target-derived curves, the curve is now the intervention: the model must
read it to predict which target variant follows.
"""
from collections import defaultdict
import math

from beatfcn.model.event_canvas import RhythmEvent

CELL_US = 500_000
MIN_GAP_US = 50_000


def _is_hold(event):
    return event.type == 'hold'


def _lane_free(events, lane, onset, mode):
    if mode != 'mania_4k':
        return True
    for event in events:
        if event.lane != lane:
            continue
        end = event.end_us if _is_hold(event) else event.onset_us
        if event.onset_us <= onset <= end:
            return False
    return True


def _interval_free(events, lane, onset, end):
    """Match the codec's half-open hold interval, including future heads."""
    for event in events:
        if event.lane != lane:
            continue
        if event.onset_us == onset:
            return False
        if _is_hold(event) and event.onset_us <= onset < event.end_us:
            return False
        if end > onset and onset <= event.onset_us < end:
            return False
    return True


def _add_taps(events, count, cell_start, cell_end, mode, duration_us, context):
    added = []
    if count <= 0:
        return added
    span = max(1, cell_end - cell_start)
    for index in range(count):
        onset = cell_start + int((index + .5) * span / count)
        onset = (onset // 1000) * 1000
        if onset >= cell_end or onset >= duration_us:
            continue
        if mode == 'mania_4k':
            placed = False
            for shift in range(0, 8):
                candidate = onset + shift * 1000
                if candidate >= cell_end:
                    break
                for lane in range(4):
                    if _lane_free(context + added, lane, candidate, mode):
                        added.append(RhythmEvent(candidate, candidate, 'tap', lane))
                        placed = True
                        break
                if placed:
                    break
        else:
            added.append(RhythmEvent(onset, onset, 'tap', -1))
    return added


def _remove_events(events, count):
    if count <= 0:
        return list(events)
    priority = {'tap': 0, 'flick': 1, 'drag': 2, 'hold': 3}
    ordered = sorted(events, key=lambda event: (priority.get(event.type, 9), event.onset_us))
    removed = set(id(event) for event in ordered[:count])
    return [event for event in events if id(event) not in removed]


def edit_events_to_density(events, start_us, end_us, rate, mode, duration_us):
    """Return (edited events, actual heads/s per 0.5 s cell)."""
    if rate < 0:
        raise ValueError('Density must be nonnegative')
    cells = max(1, math.ceil((end_us - start_us) / CELL_US))
    buckets = defaultdict(list)
    for event in events:
        if start_us <= event.onset_us < end_us:
            buckets[(event.onset_us - start_us) // CELL_US].append(event)
    out = []
    actual = []
    for cell in range(cells):
        lo = start_us + cell * CELL_US
        hi = min(end_us, lo + CELL_US)
        current = buckets[cell]
        desired = int(round(rate * (hi - lo) / 1e6))
        if desired < len(current):
            kept = _remove_events(current, len(current) - desired)
        else:
            kept = list(current)
        kept = kept + _add_taps(kept, max(0, desired - len(kept)), lo, hi, mode, duration_us, out + kept)
        out.extend(kept)
        actual.append(len(kept) / ((hi - lo) / 1e6))
    out = sanitize_mania(out, mode)
    return out, actual


def _cell_density(events, start_us, cell_end):
    """Real heads/s inside one CELL_US window (no synthetic invention)."""
    span = max(1, cell_end - start_us)
    count = sum(1 for event in events if start_us <= event.onset_us < cell_end)
    return count / (span / 1e6)


def _subdivision_candidates(onsets, lo, hi, duration_us, min_gap_us):
    """Midpoints of adjacent onset pairs that land in [lo, hi), widest first.

    Both halves must stay >= min_gap_us, so a split never creates a gap shorter
    than the native low tail (1st/5th percentile ~40/52-64 ms).
    """
    found = []
    for a, b in zip(onsets, onsets[1:]):
        middle = (a + b) // 2 // 1000 * 1000
        if (lo <= middle < hi and middle < duration_us
                and middle - a >= min_gap_us and b - middle >= min_gap_us):
            found.append((b - a, a, middle))
    found.sort(key=lambda item: (-item[0], item[1]))
    return [(a, middle) for _, a, middle in found]


def scaled_count(count, multiplier):
    """Heads kept in one cell: round half up, and >=1 for a nonempty cell at multiplier>0.

    Python's round() is banker's rounding: at x0.5 a one-head cell became 0
    (round(0.5)) and a five-head cell 2 (round(2.5)), so sparse windows were
    systematically emptied. Half-up is unbiased for the common .5 case, and a
    nonempty cell keeping one head preserves isolated musical accents: thinning
    removes redundancy inside dense cells, silence is the separate x0 treatment.
    Sparse windows therefore thin less than requested; the training curve is
    measured from the achieved target, so this never mislabels a sample.
    """
    if count <= 0 or multiplier <= 0:
        return 0
    return max(1, math.floor(count * float(multiplier) + .5))


def edit_events_to_multiplier(events, start_us, end_us, multiplier, mode,
                              duration_us, *, seed=0, min_gap_us=MIN_GAP_US,
                              carry=(), avoid_same_lane=False):
    """Rescale a window's density by a relative multiplier while keeping human content.

    Down-sampling draws a uniform subset of the cell's own events, so the type mix
    (tap/hold/drag/flick) and local rhythm survive. Up-sampling subdivides the
    widest gaps between existing onsets (8ths -> 16ths), copying the type of the
    earlier neighbour, and never creates a gap below min_gap_us. The former
    +-20 ms jittered duplicates produced 21-40% sub-40 ms gaps (native ~1%) and
    taught free generation to emit clusters of taps a few ms apart. A cell whose
    gaps are all too short to split keeps fewer events than requested; the
    returned per-cell density is the honest achieved value. A cell that is empty
    in the source stays empty.

    ``carry`` holds the request's authorized incoming holds (onset before
    start_us). They are never edited or emitted, but in mania they occupy their
    lane until their tail, so an inserted head/hold may not land on a held key.

    ``avoid_same_lane`` (mania; default off = byte-identical): an inserted note prefers a key not used by its two
    neighbouring onsets. Off, the source's own key comes first, and x1.5 / x2 targets hold about 9-11x the native jack rate.
    """
    if multiplier < 0:
        raise ValueError('Density multiplier must be nonnegative')
    import random
    cells = max(1, math.ceil((end_us - start_us) / CELL_US))
    buckets = defaultdict(list)
    for event in events:
        if start_us <= event.onset_us < end_us:
            buckets[(event.onset_us - start_us) // CELL_US].append(event)
    out, actual = [], []
    for cell in range(cells):
        lo = start_us + cell * CELL_US
        hi = min(end_us, lo + CELL_US)
        current = sorted(buckets[cell], key=lambda event: (event.onset_us, event.type))
        desired = scaled_count(len(current), multiplier)
        # Cap the upsampled cell: unbounded duplication on dense sections made
        # targets long enough to OOM an 80 GB H100 at microbatch 4.
        desired = min(desired, max(len(current), 6))
        if not current:
            kept = []
        elif multiplier <= 0:
            kept = []
        elif desired < len(current):
            rng = random.Random((seed * 1_000_003) ^ (start_us // CELL_US + cell))
            kept = sorted(rng.sample(current, desired), key=lambda event: event.onset_us)
        elif desired == len(current):
            kept = list(current)
        else:
            kept = list(current)
            rng = random.Random((seed * 7_919) ^ (cell * 131 + 17))
            window = [event for event in events if start_us <= event.onset_us < end_us]
            while len(kept) < desired:
                # Whole-window context: a split may pair this cell's last head
                # with the next cell's first head. Future cells keep all their
                # human heads when multiplier >= 1, so they are safe neighbours.
                context = window + out + kept
                onsets = sorted({event.onset_us for event in context})
                added = None
                for left, onset in _subdivision_candidates(onsets, lo, hi, duration_us, min_gap_us):
                    sources = [event for event in context if event.onset_us == left]
                    source = sources[rng.randrange(len(sources))]
                    end = onset + max(0, source.end_us - source.onset_us) if _is_hold(source) else onset
                    if end > duration_us:
                        continue
                    lane = source.lane
                    if mode == 'mania_4k':
                        # Include untouched future cells: a new hold must not cover
                        # a human head that will only be added to `out` later.
                        candidates = [lane] + [other for other in range(4) if other != lane]
                        if avoid_same_lane:
                            later = [t for t in onsets if t > onset]
                            around = {e.lane for e in context if e.onset_us == left or (later and e.onset_us == later[0])}
                            rng.shuffle(candidates)
                            candidates.sort(key=lambda candidate: candidate in around)
                        occupied = list(carry) + list(events) + out + kept
                        lane = next((candidate for candidate in candidates
                                     if _interval_free(occupied, candidate, onset, end)), None)
                        if lane is None:
                            continue
                    added = RhythmEvent(onset, end, source.type, lane)
                    break
                if added is None:
                    break
                kept.append(added)
        kept.sort(key=lambda event: (event.onset_us, event.type, event.lane))
        out.extend(kept)
        actual.append(len(kept) / ((hi - lo) / 1e6))
    out.sort(key=lambda event: (event.onset_us, event.type, event.lane))
    return out, actual


def sanitize_mania(events, mode):
    """Drop lane conflicts that the mania codec cannot express.

    Jittered duplicates can put a hold behind an earlier note in the same lane
    (or two notes on one onset/lane), which the teacher sequence rejects with
    `Illegal mania:<pattern>`. Keep the earliest note and clip hold tails.
    """
    if mode != 'mania_4k':
        return list(events)
    lanes = defaultdict(list)
    for event in events:
        lane = event.lane if event.lane in range(4) else None
        if lane is not None:
            lanes[lane].append(event)
    seen = set()
    out = []
    for lane, lane_events in lanes.items():
        lane_events.sort(key=lambda event: (event.onset_us, _is_hold(event)))
        for index, event in enumerate(lane_events):
            key = (event.onset_us, lane)
            if key in seen:
                continue
            seen.add(key)
            end = event.end_us
            if _is_hold(event):
                following = lane_events[index+1].onset_us if index+1 < len(lane_events) else None
                if following is not None and end >= following:
                    end = following - 1000
                if end <= event.onset_us:
                    continue
            out.append(RhythmEvent(event.onset_us, end, event.type, lane))
    out.sort(key=lambda event: (event.onset_us, event.type, event.lane))
    return out


def real_density_curve(events, start_us, end_us):
    """Per-cell heads/s of the untouched human window, used as the curve baseline."""
    cells = max(1, math.ceil((end_us - start_us) / CELL_US))
    curve = []
    for cell in range(cells):
        lo = start_us + cell * CELL_US
        hi = min(end_us, lo + CELL_US)
        curve.append(dict(start_us=lo, end_us=hi,
                          density=_cell_density(events, lo, hi)))
    return curve


def achieved_density_segments(events, start_us, end_us, *, cell_us=2_000_000):
    """Heads/s of the ACTUAL target per coarse cell, as caller-style segments.

    Counts head onsets in [start_us, end_us) per half-open cell anchored at
    start_us (a 2 s cell is the sum of its four 0.5 s editor cells). The last
    cell may be shorter; its density uses its own span. Coarse on purpose: the
    curve tells the model how busy each region is, not where the onsets are.
    """
    if type(cell_us) is not int or cell_us < CELL_US or cell_us % CELL_US:
        raise ValueError('Coarse density cells must be a positive multiple of the editor cell')
    if not 0 <= start_us < end_us:
        raise ValueError('Invalid density interval')
    segments = []
    for lo in range(start_us, end_us, cell_us):
        hi = min(end_us, lo + cell_us)
        count = sum(1 for event in events if lo <= event.onset_us < hi)
        segments.append(dict(start_us=int(lo), end_us=int(hi), density=count / ((hi - lo) / 1e6)))
    return segments

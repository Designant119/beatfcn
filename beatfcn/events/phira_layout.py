"""Rule-based two-thumb position layout for Phira drafts (ROADMAP item 7, v1; no model, deterministic).

The rhythm model writes onset / type / tail only. This lays the notes out on ONE static judge line in RPE x
units (screen width -675..675) for two-thumb play:
  - two hand zones (left -560..-60, right 60..560); a chord of two puts one note in each zone, mirrored-ish;
  - a hand that is holding stays busy until the hold ends (the other hand takes the notes meanwhile);
  - fast singles (gap < STREAM_GAP_US) alternate hands; slower singles alternate too, but a long rest
    lets the hand jump to a new spot in its zone (outer / middle / inner cycle);
  - consecutive drags of one hand slide smoothly (small steps), flicks stay near the hand's spot;
  - a hand never moves faster than MAX_SPEED (x units per second) between its own notes, and two notes
    within NEAR_US never land closer than MIN_SEP apart.
Returns [{id, t_us, end_us, type, x}] in time order (the format of rhythm_preview.display_positions).
"""
from bisect import bisect_left
from collections import defaultdict
import random

LEFT, RIGHT = (-560.0, -60.0), (60.0, 560.0)
BUSY_NPS, CALM_NPS = 7.0, 2.0
BUSY_SPAN, CALM_SPAN = 300.0, 560.0
INNER = 40.0
STREAM_GAP_US = 220_000
REST_GAP_US = 600_000
MAX_SPEED = 2600.0          # x units per second for one thumb
MIN_SEP = 180.0
NEAR_US = 60_000
SPOTS = (0.15, 0.5, 0.85)   # fraction across a zone, outer->inner for the left hand


def _zone(hand):
    return LEFT if hand == 0 else RIGHT


def _spot(hand, k):
    lo, hi = _zone(hand)
    f = SPOTS[k % len(SPOTS)]
    return lo + (hi - lo) * (f if hand == 0 else 1 - f)


def _clamp(hand, x):
    lo, hi = _zone(hand)
    return max(lo, min(hi, x))


def _span_of(onsets, t):
    """Half-width the hands use around time t: narrow when the next and previous second are busy."""
    lo, hi = bisect_left(onsets, t-1_000_000), bisect_left(onsets, t+1_000_000)
    nps = (hi-lo)/2
    f = min(1., max(0., (nps-CALM_NPS)/(BUSY_NPS-CALM_NPS)))
    return CALM_SPAN+(BUSY_SPAN-CALM_SPAN)*f


def density_layout(events, *, seed=0):
    """Two hands at a time; zones follow local density; a hand may repeat its spot; chords are near-mirrored."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, e in enumerate(events):
        groups[int(e['onset_us'])].append((i, e))
    onsets = sorted(groups)
    x_of = [-180.0, 180.0]
    t_of = [None, None]
    busy = [-1, -1]
    last, out = 1, []

    def place(hand, t, span, kind):
        gap = None if t_of[hand] is None else t-t_of[hand]
        sign = -1 if hand == 0 else 1
        lo, hi = INNER, span
        x = abs(x_of[hand])
        if gap is None or gap >= REST_GAP_US:
            x = rng.uniform(lo, hi)
        elif kind == 'drag':
            x += rng.uniform(-80, 80)
        elif gap < STREAM_GAP_US:
            x += rng.choice((-1, 1))*rng.uniform(0, 90)
        else:
            x += rng.choice((-1, 1))*rng.uniform(40, 220)
        x = min(hi, max(lo, x))
        if gap:
            reach = MAX_SPEED*gap/1e6
            x = min(abs(x_of[hand])+reach, max(abs(x_of[hand])-reach, x))
        return sign*x

    for t in onsets:
        group = sorted(groups[t], key=lambda ie: (ie[1]['type'] != 'hold', ie[0]))
        span = _span_of(onsets, t)
        free = [h for h in (0, 1) if busy[h] <= t]
        if len(group) >= 2:
            hands = [0, 1]+[None]*(len(group)-2)
        elif len(free) == 1:
            hands = free
        else:
            gap = None if t_of[last] is None else t-t_of[last]
            hand = last if gap is not None and gap >= STREAM_GAP_US and rng.random() < .3 else 1-last
            hands = [hand]
        for (i, e), hand in zip(group, hands):
            if hand is None:
                x = rng.uniform(-span/2, span/2)
            else:
                x = place(hand, t, span, e['type'])
                x_of[hand], t_of[hand] = x, t
                if e['type'] == 'hold':
                    busy[hand] = int(e['end_us'])
            out.append(dict(id=f'event:{i}', t_us=int(e['onset_us']), end_us=int(e['end_us']), type=e['type'], x=round(float(x), 1)))
        if len(group) == 1:
            last = hands[0]
    out.sort(key=lambda r: (r['t_us'], r['x']))
    return out


def two_thumb_layout(events, *, seed=0):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, e in enumerate(events):
        groups[int(e['onset_us'])].append((i, e))
    state = [dict(x=_spot(0, 1), t=None, busy=-1, k=1, dir=1), dict(x=_spot(1, 1), t=None, busy=-1, k=1, dir=-1)]
    last_hand, placed, out = 1, [], []

    def move(hand, t, kind):
        s = state[hand]
        gap = None if s['t'] is None else t - s['t']
        if gap is None or gap >= REST_GAP_US:
            s['k'] = (s['k'] + rng.choice((1, 2))) % len(SPOTS)
            target = _spot(hand, s['k']) + rng.uniform(-40, 40)
        elif kind == 'drag':
            target = s['x'] + s['dir'] * rng.uniform(40, 90)
        elif gap < STREAM_GAP_US:
            target = s['x'] + s['dir'] * rng.uniform(60, 140)
        else:
            target = s['x'] + s['dir'] * rng.uniform(120, 260)
        lo, hi = _zone(hand)
        if target < lo or target > hi:
            s['dir'] = -s['dir']
            target = s['x'] + s['dir'] * abs(target - s['x'])
        target = _clamp(hand, target)
        if gap is not None and gap > 0:
            reach = MAX_SPEED * gap / 1e6
            target = max(s['x'] - reach, min(s['x'] + reach, target))
        return target

    def separate(x, t, hand):
        for (pt, px) in placed:
            if abs(pt - t) <= NEAR_US and abs(px - x) < MIN_SEP:
                x = px + (MIN_SEP if hand == 1 else -MIN_SEP)
        return _clamp(hand, x)

    for t in sorted(groups):
        group = sorted(groups[t], key=lambda ie: (ie[1]['type'] != 'hold', ie[0]))
        free = [h for h in (0, 1) if state[h]['busy'] <= t]
        if len(group) >= 2:
            hands = [0, 1] + [None] * (len(group) - 2)
        else:
            if len(free) == 1:
                hands = free
            else:
                hands = [1 - last_hand]
        xs = []
        for (i, e), hand in zip(group, hands):
            if hand is None:  # >= 3 simultaneous: spread across the middle
                x = rng.uniform(-200, 200)
                hand_used = 0 if x < 0 else 1
            else:
                x = move(hand, t, e['type'])
                hand_used = hand
            x = separate(x, t, hand_used)
            xs.append(x)
            s = state[hand_used]
            s['x'], s['t'] = x, t
            if e['type'] == 'hold':
                s['busy'] = int(e['end_us'])
            placed.append((t, x))
            out.append(dict(id=f'event:{i}', t_us=int(e['onset_us']), end_us=int(e['end_us']), type=e['type'],
                            x=round(float(x), 1)))
        if len(group) == 1:
            last_hand = hands[0]
        placed = [(pt, px) for pt, px in placed if t - pt <= NEAR_US]
    out.sort(key=lambda r: (r['t_us'], r['x']))
    return out


def layout_report(layout):
    """Simple playability checks of a layout: hand speed and near-simultaneous separation."""
    by_t = sorted(layout, key=lambda r: r['t_us'])
    too_close = sum(1 for a, b in zip(by_t, by_t[1:]) if b['t_us'] - a['t_us'] <= NEAR_US and abs(b['x'] - a['x']) < MIN_SEP - 1)
    xs = [r['x'] for r in layout]
    return dict(notes=len(layout), near_too_close=too_close, x_min=min(xs), x_max=max(xs),
                left_share=round(sum(x < 0 for x in xs) / len(xs), 3))

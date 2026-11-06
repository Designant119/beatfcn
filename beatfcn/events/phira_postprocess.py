"""Playability clean-up of a Phira rhythm draft, after the model and before layout. No model, deterministic.

limit_fingers: at most N fingers at once. Extra chord members are dropped; a hold that would leave no free finger
ends just before the new note, or becomes a tap when that would leave it too short.
clean_types: a lone flick or hold between two taps inside a fast run (both gaps under FAST_GAP_US) becomes a tap.
Human charts have about .14 tap-other-tap alternations per note; independent per-note type sampling makes about .20.
"""
FAST_GAP_US = 250_000
RELEASE_US = 40_000
MIN_HOLD_US = 80_000
KEEP_ORDER = {'hold': 0, 'flick': 1, 'tap': 2, 'drag': 3}


def limit_fingers(events, max_fingers=2):
    groups = {}
    for e in events:
        groups.setdefault(int(e['onset_us']), []).append(dict(e))
    out, active = [], []
    report = dict(dropped=0, shortened=0, to_tap=0)
    for t in sorted(groups):
        active = [e for e in active if e['end_us'] > t]
        group = sorted(groups[t], key=lambda e: KEEP_ORDER.get(e['type'], 9))
        if len(group) > max_fingers:
            report['dropped'] += len(group)-max_fingers
            group = group[:max_fingers]
        while active and len(active)+len(group) > max_fingers:
            oldest = min(active, key=lambda e: e['onset_us'])
            active.remove(oldest)
            end = t-RELEASE_US
            if end-oldest['onset_us'] >= MIN_HOLD_US:
                oldest['end_us'] = end
                report['shortened'] += 1
            else:
                oldest['type'], oldest['end_us'] = 'tap', oldest['onset_us']
                report['to_tap'] += 1
        for e in group:
            out.append(e)
            if e['type'] == 'hold':
                active.append(e)
    return out, report


def clean_types(events):
    out = sorted((dict(e) for e in events), key=lambda e: e['onset_us'])
    onsets = sorted({e['onset_us'] for e in out})
    index = {t: k for k, t in enumerate(onsets)}
    kinds = {}
    for e in out:
        kinds.setdefault(e['onset_us'], []).append(e['type'])
    changed = 0
    for e in out:
        if e['type'] not in ('flick', 'hold') or len(kinds[e['onset_us']]) > 1:
            continue
        k = index[e['onset_us']]
        if not 0 < k < len(onsets)-1:
            continue
        a, b = onsets[k-1], onsets[k+1]
        if e['onset_us']-a < FAST_GAP_US and b-e['onset_us'] < FAST_GAP_US and kinds[a] == ['tap'] and kinds[b] == ['tap']:
            e['type'], e['end_us'] = 'tap', e['onset_us']
            changed += 1
    return out, dict(types_cleaned=changed)


def merge_fragments(events, min_gap_us=90_000):
    """Onsets closer than min_gap_us are not separate hits at this difficulty: keep the one holding the stronger note."""
    groups = {}
    for e in events:
        groups.setdefault(int(e['onset_us']), []).append(dict(e))
    rank = lambda g: min(KEEP_ORDER.get(e['type'], 9) for e in g)  # noqa: E731
    keep, dropped = [], 0
    for t in sorted(groups):
        if keep and t-keep[-1] < min_gap_us:
            dropped += len(groups[t]) if rank(groups[keep[-1]]) <= rank(groups[t]) else len(groups[keep[-1]])
            if rank(groups[t]) < rank(groups[keep[-1]]):
                keep[-1] = t
            continue
        keep.append(t)
    return [e for t in keep for e in groups[t]], dict(fragments_dropped=dropped)


def limit_holds(events, max_holds=1, min_hold_gap_us=150_000):
    """At most max_holds down at once, and no new hold right after another one starts: the later hold becomes a tap
    unless the earlier one can end first and stay long enough."""
    out = sorted((dict(e) for e in events), key=lambda e: (e['onset_us'], e['type'] != 'hold'))
    active, last_start, report = [], None, dict(holds_shortened=0, holds_to_tap=0)
    for e in out:
        if e['type'] != 'hold':
            continue
        t = e['onset_us']
        active = [h for h in active if h['end_us'] > t]
        if last_start is not None and t-last_start < min_hold_gap_us:
            e['type'], e['end_us'] = 'tap', t
            report['holds_to_tap'] += 1
            continue
        while len(active) >= max_holds:
            h = min(active, key=lambda x: x['onset_us'])
            active.remove(h)
            if t-RELEASE_US-h['onset_us'] >= MIN_HOLD_US:
                h['end_us'] = t-RELEASE_US
                report['holds_shortened'] += 1
            else:
                h['type'], h['end_us'] = 'tap', h['onset_us']
                report['holds_to_tap'] += 1
        active.append(e)
        last_start = t
    return out, report


def clean_chords(events):
    """A chord keeps one kind of special note: hold+tap, flick+tap, flick+flick and drag+drag are what charters write;
    a chord mixing different special notes turns the weaker ones into taps."""
    groups = {}
    for e in events:
        groups.setdefault(int(e['onset_us']), []).append(dict(e))
    out, changed = [], 0
    for t in sorted(groups):
        g = sorted(groups[t], key=lambda e: KEEP_ORDER.get(e['type'], 9))
        special = None
        for e in g:
            if e['type'] != 'tap':
                if special is not None and e['type'] != special:
                    e['type'], e['end_us'] = 'tap', e['onset_us']
                    changed += 1
                elif special is None:
                    special = e['type']
            out.append(e)
    return out, dict(chords_cleaned=changed)


PROFILES = {
    'relaxed': dict(max_fingers=2, max_holds=1, min_hold_gap_us=150_000, min_gap_us=90_000),
    'standard': dict(max_fingers=2, max_holds=2, min_hold_gap_us=100_000, min_gap_us=60_000),
    'expert': dict(max_fingers=4, max_holds=3, min_hold_gap_us=0, min_gap_us=30_000),
}


def playable(events, profile='relaxed'):
    """The whole clean-up in order. Returns events and one report; every rule is a preference a charter can change."""
    p = PROFILES[profile]
    report = {}
    events, r = merge_fragments(events, p['min_gap_us']); report.update(r)
    events, r = clean_chords(events); report.update(r)
    events, r = limit_holds(events, p['max_holds'], p['min_hold_gap_us']); report.update(r)
    events, r = limit_fingers(events, p['max_fingers']); report.update(r)
    events, r = clean_types(events); report.update(r)
    return events, report

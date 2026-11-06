"""Frozen counterfactual exposure, condition dropout, and per-sample conditions.

Exposure (``counterfactual-exposure-v2``): each epoch is a seeded permutation of
the train requests, so every request is seen once per epoch in a fresh order.
Each (request, epoch) gets one treatment (native or a density multiplier). The
treatment is a stratified draw: each request walks its own seeded shuffle of a
"deck" whose composition equals the configured mix, so every request receives
every treatment in exact proportion over one deck cycle and each epoch's
treatment frequencies match the mix up to sampling noise across requests.

Conditions: every sample, native or edited, is conditioned on ITS OWN actual
target -- a coarse density curve and the 8x6 window controls recomputed from
the (edited) target with the tape builder's label functions. Dropout is drawn
per (request, epoch) and frozen in the exposure: with ``all_probability`` every
condition is removed together (the unconditional mode used at evaluation and
for classifier-free guidance); otherwise the curve is hidden with
``curve_probability`` and each control field independently with
``control_probability``.
"""
from fractions import Fraction
import math

import numpy as np

from beatfcn.train.counterfactual_density import achieved_density_segments
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.event_conditions import CONTROL_FIELDS, encode_controls, training_labels

FORMAT = 'counterfactual-exposure-v2'
DEFAULT_TREATMENT_MIX = (
    dict(name='native', multiplier=None, probability=.50),
    dict(name='x0.5', multiplier=.5, probability=.15),
    dict(name='x1.5', multiplier=1.5, probability=.15),
    dict(name='x2.0', multiplier=2.0, probability=.15),
    dict(name='x0', multiplier=0., probability=.05),
)
DEFAULT_CONDITION_DROPOUT = dict(all_probability=.2, curve_probability=.3,
                                 control_probability=.3)
DEFAULT_CURVE_CELL_US = 2_000_000
MAX_DECK = 1000
_TREATMENT_TAG, _DROPOUT_TAG, _EPOCH_TAG, _EDIT_TAG, _VALIDATION_TAG = 101, 202, 303, 404, 505
_MODE_TAG = 707


def normalize_mix(mix=None):
    mix = [dict(item) for item in (mix or DEFAULT_TREATMENT_MIX)]
    names = [item['name'] for item in mix]
    if len(set(names)) != len(names) or not mix:
        raise ValueError('Treatment names must be unique and nonempty')
    for item in mix:
        p, m = item['probability'], item['multiplier']
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError('Treatment probabilities must lie in [0, 1]')
        if m is not None and not (isinstance(m, (int, float)) and m >= 0 and math.isfinite(m)):
            raise ValueError('Treatment multipliers must be finite and nonnegative, or None for native')
        item['multiplier'] = None if m is None else float(m)
        item['probability'] = float(p)
    if abs(sum(item['probability'] for item in mix) - 1) > 1e-9:
        raise ValueError('Treatment probabilities must sum to one')
    return mix


def treatment_deck(mix):
    """Smallest integer deck whose composition is the mix (None if impractically large)."""
    fractions = [Fraction(item['probability']).limit_denominator(MAX_DECK) for item in mix]
    if any(abs(float(f) - item['probability']) > 1e-9 for f, item in zip(fractions, mix)):
        return None
    length = math.lcm(*[f.denominator for f in fractions])
    if length > MAX_DECK:
        return None
    deck = []
    for index, fraction in enumerate(fractions):
        deck.extend([index] * int(fraction * length))
    return deck


def _rng(seed, *keys):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *[int(k) & 0xFFFFFFFF for k in keys]])


def draw_treatment(mix, deck, seed, request, epoch):
    """Index into ``mix`` for one (request, epoch); stratified when a deck exists."""
    if deck is None:
        p = np.asarray([item['probability'] for item in mix])
        return int(_rng(seed, _TREATMENT_TAG, request, epoch).choice(len(mix), p=p))
    cycle, position = divmod(int(epoch), len(deck))
    order = _rng(seed, _TREATMENT_TAG, request, cycle).permutation(len(deck))
    return deck[int(order[position])]


def normalize_dropout(dropout=None):
    dropout = dict(DEFAULT_CONDITION_DROPOUT, **(dropout or {}))
    if set(dropout) != set(DEFAULT_CONDITION_DROPOUT):
        raise ValueError('Unknown condition dropout field')
    for key, value in dropout.items():
        if not (isinstance(value, (int, float)) and 0 <= value <= 1):
            raise ValueError('Dropout probabilities must lie in [0, 1]')
        dropout[key] = float(value)
    return dropout


def draw_condition_dropout(dropout, seed, request, epoch):
    """Return (unconditional, curve_specified, specified control-row indices)."""
    rng = _rng(seed, _DROPOUT_TAG, request, epoch)
    draws = rng.random(2 + len(CONTROL_FIELDS))
    if draws[0] < dropout['all_probability']:
        return True, False, []
    curve = bool(draws[1] >= dropout['curve_probability'])
    fields = [i for i in range(len(CONTROL_FIELDS)) if draws[2 + i] >= dropout['control_probability']]
    return False, curve, fields


def edit_seed(seed, request, epoch):
    return int(_rng(seed, _EDIT_TAG, request, epoch).integers(0, 2**31 - 1))


def _song_chunked(order, train, song_of, chunk, rng):
    """Epoch order with I/O locality: each song's windows (shuffled) are cut into
    chunks of ``chunk`` and the chunks are shuffled. Every request still appears
    exactly once per epoch; only adjacency changes, so a loader item of ``chunk``
    requests usually reads one song's features instead of ``chunk`` songs."""
    songs = {}
    for i in order:
        songs.setdefault(song_of[train[i]], []).append(i)
    chunks, rest = [], []
    for members in songs.values():
        whole = len(members) - len(members) % chunk
        chunks.extend(members[k:k+chunk] for k in range(0, whole, chunk))
        rest.extend(members[whole:])
    # Remainders are packed into full-size mixed chunks so every chunk boundary
    # stays aligned with loader-item boundaries.
    chunks.extend(rest[k:k+chunk] for k in range(0, len(rest), chunk))
    return [i for j in rng.permutation(len(chunks)) for i in chunks[j]]


def normalize_mode_weights(mode_weights, modes):
    """{mode: nonnegative finite weight}; every train mode needs a weight."""
    weights = {str(k): float(v) for k, v in dict(mode_weights).items()}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Mode weights must be finite and nonnegative')
    missing = set(modes) - set(weights)
    if missing:
        raise ValueError(f'Mode weights missing for {sorted(missing)}')
    if not any(weights[m] > 0 for m in modes):
        raise ValueError('At least one train mode needs a positive weight')
    return dict(sorted(weights.items()))


def _weighted_positions(train, mode_of, weights, seed, epoch):
    """Train positions for one epoch: each request appears floor(w) times plus
    one more with probability frac(w), drawn per (request, epoch)."""
    u = _rng(seed, _MODE_TAG, epoch).random(len(train))
    positions = []
    for i, request in enumerate(train):
        w = weights[mode_of[request]]
        copies = int(math.floor(w)) + int(u[i] < w - math.floor(w))
        positions.extend([i] * copies)
    return np.asarray(positions, np.int64)


def build_exposure(train, *, steps, seed, mix=None, dropout=None, pool_identity=None,
                   requests_per_step=4, song_of=None, song_chunk=1, unconditional_native=False,
                   mode_weights=None, mode_of=None):
    """Exposure steps over ``train`` request indices, ``steps`` four-request groups.

    ``song_of`` (request index -> song key) with ``song_chunk`` > 1 keeps runs of
    ``song_chunk`` same-song windows adjacent (full-database feature reads are
    network-latency bound; measured 83% data wait at random order).

    ``mode_weights`` (optional, e.g. {'phira': 1.6, 'mania_4k': 1.0}; needs
    ``mode_of``: request index -> mode) re-weights each epoch deterministically:
    a request of weight w appears floor(w) times plus once more with
    probability frac(w) (w < 1 subsamples). Treatment/dropout/edit draws then
    key on the request's occurrence count (recorded as ``occurrences``) so
    repeats get fresh, still stratified draws; the realized mix is recorded in
    ``mode_mix``. With mode_weights=None the exposure is unchanged."""
    train = [int(index) for index in train]
    if not train or len(set(train)) != len(train):
        raise ValueError('Train requests must be unique and nonempty')
    if type(steps) is not int or steps < 1:
        raise ValueError('Exposure needs a positive integer number of steps')
    mix, dropout = normalize_mix(mix), normalize_dropout(dropout)
    deck = treatment_deck(mix)
    if mode_weights is not None:
        if mode_of is None:
            raise ValueError('mode_weights needs mode_of (request index -> mode)')
        mode_weights = normalize_mode_weights(mode_weights, {mode_of[r] for r in train})
    needed = steps * requests_per_step
    stream = []
    epoch = 0
    epoch_lengths = []
    while len(stream) < needed:
        rng = _rng(seed, _EPOCH_TAG, epoch)
        if mode_weights is None:
            order = rng.permutation(len(train))
        else:
            positions = _weighted_positions(train, mode_of, mode_weights, seed, epoch)
            order = positions[rng.permutation(len(positions))]
        if song_of is not None and song_chunk > 1:
            order = _song_chunked(order, train, song_of, int(song_chunk), rng)
        stream.extend((train[i], epoch) for i in order)
        epoch_lengths.append(len(order))
        epoch += 1
    stream = stream[:needed]
    occurrence = {}
    out = []
    for update in range(steps):
        group = stream[update*requests_per_step:(update+1)*requests_per_step]
        fields = dict(effective_update=update + 1, indices=[], epochs=[], rates=[],
                      treatments=[], edit_seeds=[], unconditional=[], curve_specified=[],
                      specified_condition_fields=[])
        if mode_weights is not None:
            fields['occurrences'] = []
        for request, request_epoch in group:
            draw_key = request_epoch
            if mode_weights is not None:
                draw_key = occurrence.get(request, 0)
                occurrence[request] = draw_key + 1
                fields['occurrences'].append(draw_key)
            choice = mix[draw_treatment(mix, deck, seed, request, draw_key)]
            unconditional, curve, specified = draw_condition_dropout(dropout, seed, request, draw_key)
            if unconditional_native and unconditional and choice['multiplier'] is not None:
                # With no condition the target must be what a human charter wrote:
                # edited targets (x0/x0.5/x1.5/x2) under an empty condition teach a
                # density prior that no request asked for (Phira drafts had 0.70x notes).
                choice = next(item for item in mix if item['multiplier'] is None)
            fields['indices'].append(request)
            fields['epochs'].append(request_epoch)
            fields['rates'].append(choice['multiplier'])
            fields['treatments'].append(choice['name'])
            fields['edit_seeds'].append(edit_seed(seed, request, draw_key))
            fields['unconditional'].append(unconditional)
            fields['curve_specified'].append(curve)
            fields['specified_condition_fields'].append(specified)
        out.append(fields)
    result = dict(format=FORMAT, pool_identity=pool_identity, seed=int(seed),
                train_requests=len(train), epochs=epoch, complete_epochs=needed // len(train),
                treatment_mix=mix, treatment_deck_length=None if deck is None else len(deck),
                treatment_assignment='stratified per-request deck' if deck else 'iid per (request, epoch)',
                condition_dropout=dropout, condition_fields=list(CONTROL_FIELDS),
                song_chunk=int(song_chunk) if song_of is not None else 1,
                unconditional_native=bool(unconditional_native),
                steps=out)
    if mode_weights is not None:
        result['complete_epochs'] = int(sum(np.cumsum(epoch_lengths) <= needed))
        result['treatment_assignment'] += ' (draws keyed on per-request occurrence)'
        result['mode_weights'] = mode_weights
        result['mode_mix'] = mode_mix(stream, train, mode_of, mode_weights)
    return result


def mode_mix(stream, train, mode_of, weights):
    """Realized per-mode share of exposure slots vs the train pool and the target."""
    pool = {}
    for request in train:
        pool[mode_of[request]] = pool.get(mode_of[request], 0) + 1
    slots = {}
    for request, _ in stream:
        slots[mode_of[request]] = slots.get(mode_of[request], 0) + 1
    target_total = sum(weights[m] * n for m, n in pool.items())
    modes = sorted(pool)
    return dict(train_requests={m: pool[m] for m in modes},
                train_fraction={m: pool[m] / len(train) for m in modes},
                target_fraction={m: weights[m] * pool[m] / target_total for m in modes},
                exposure_slots={m: slots.get(m, 0) for m in modes},
                exposure_fraction={m: slots.get(m, 0) / max(1, len(stream)) for m in modes},
                distinct_requests_seen={m: len({r for r, _ in stream if mode_of[r] == m}) for m in modes})


def exposure_summary(exposure):
    """Realized treatment and dropout frequencies of a frozen exposure."""
    slots = [(step, k) for step in exposure['steps'] for k in range(len(step['indices']))]
    total = len(slots)
    treatments = {}
    for step, k in slots:
        treatments[step['treatments'][k]] = treatments.get(step['treatments'][k], 0) + 1
    conditional = [(s, k) for s, k in slots if not s['unconditional'][k]]
    fields = len(CONTROL_FIELDS)
    extra = dict(mode_mix=exposure['mode_mix']) if exposure.get('mode_weights') else {}
    return dict(**extra, requests=total,
                treatment_fraction={k: v/total for k, v in sorted(treatments.items())},
                unconditional_fraction=sum(s['unconditional'][k] for s, k in slots)/total,
                curve_specified_fraction=sum(s['curve_specified'][k] for s, k in slots)/total,
                curve_hidden_given_conditional=(sum(not s['curve_specified'][k] for s, k in conditional)
                                                / max(1, len(conditional))),
                control_hidden_given_conditional=(1 - sum(len(s['specified_condition_fields'][k])
                    for s, k in conditional) / max(1, len(conditional)*fields)))


def _label_event(event, mode):
    """Adapter giving a RhythmEvent the canonical attributes the label code reads."""
    from types import SimpleNamespace
    return SimpleNamespace(onset_us=event.onset_us, end_us=event.end_us, type=event.type,
                           is_hold=event.type == 'hold',
                           location=SimpleNamespace(lane=event.lane if mode == 'mania_4k' else None))


def target_conditions(mode, events, request, *, carry=()):
    """Bands/raw values of the actual target, with incoming holds for occupancy."""
    scope = (int(request.start_us), int(request.end_us))
    label_events = [_label_event(e, mode) for e in (*carry, *events)]
    return training_labels(mode, label_events, scope)


def sample_conditions(mode, events, request, *, curve_specified, specified_fields,
                      render_cell_us=100_000, curve_cell_us=DEFAULT_CURVE_CELL_US, scale=16.):
    """Model condition inputs describing ``events`` (the actual target).

    Returns (inputs update dict with controls + intensity curve, audit dict).
    A hidden control keeps its applicability/scope columns, exactly like the
    unconditional rows used at evaluation and generation (encode_controls({})).
    """
    bands, raw = target_conditions(mode, events, request, carry=request.known_carry)
    keep = {CONTROL_FIELDS[i] for i in specified_fields}
    shown = {field: band for field, band in bands.items() if field in keep}
    segments = (achieved_density_segments(events, request.start_us, request.end_us,
                                          cell_us=curve_cell_us) if curve_specified else [])
    inputs = condition_inputs(mode, request, bands=shown, segments=segments,
                              render_cell_us=render_cell_us, scale=scale)
    audit = dict(bands=bands, raw=raw, shown_fields=sorted(shown), curve_segments=segments)
    return inputs, audit


def condition_inputs(mode, request, *, bands=None, segments=(), render_cell_us=100_000, scale=16.):
    """The single renderer of model condition inputs for training, evaluation and generation.

    controls: encode_controls (applicability + scope columns always present,
    values only for supplied ``bands``); curve: caller/achieved ``segments``
    resampled onto the fixed ``render_cell_us`` grid, gaps explicitly
    unspecified (valid=False). With no bands and no segments this is the
    unconditional request -- identical to a training sample whose conditions
    were all dropped.
    """
    controls = encode_controls(mode, dict(bands or {}), (int(request.start_us), int(request.end_us)),
                               origin_us=int(request.start_us))
    curve = grid_density_curve(list(segments), request.start_us, request.end_us,
                               cell_us=int(render_cell_us), scale=scale)
    return dict(controls=controls, **curve)


def unconditional_inputs(mode, request, *, render_cell_us=100_000, scale=16.):
    """Condition inputs of an unconditional (nothing specified) request."""
    return condition_inputs(mode, request, render_cell_us=render_cell_us, scale=scale)


def select_validation(records, *, count=200, seed=0):
    """Deterministic validation windows stratified by mode, then chart.

    Each mode gets an equal share of ``count`` (capped by availability, the
    remainder flowing to the other mode); within a mode every chart gets an
    equal round-robin share and windows are
    drawn uniformly without replacement inside the chart. No window is excluded
    for its content or any model outcome; empty (silent) windows are eligible.
    """
    by_mode = {}
    for index, row in enumerate(records):
        if row['split'] == 'val':
            by_mode.setdefault(row['mode'], {}).setdefault(row['chart_id'], []).append(index)
    if not by_mode:
        raise ValueError('No validation records')
    modes = sorted(by_mode)
    available = {m: sum(map(len, by_mode[m].values())) for m in modes}
    quota = {m: 0 for m in modes}
    remaining = min(count, sum(available.values()))
    while remaining:
        open_modes = [m for m in modes if quota[m] < available[m]]
        share = max(1, remaining // len(open_modes))
        for m in open_modes:
            take = min(share, available[m] - quota[m], remaining)
            quota[m] += take
            remaining -= take
            if not remaining:
                break
    chosen = []
    for m in modes:
        charts = sorted(by_mode[m])
        rng = _rng(seed, _VALIDATION_TAG, modes.index(m))
        pools = {c: [by_mode[m][c][i] for i in rng.permutation(len(by_mode[m][c]))] for c in charts}
        taken = {c: 0 for c in charts}
        left = quota[m]
        # Round-robin over a seeded chart order: equal per-chart shares.
        order = [charts[i] for i in rng.permutation(len(charts))]
        while left:
            progressed = False
            for c in order:
                if left and taken[c] < len(pools[c]):
                    chosen.append(pools[c][taken[c]])
                    taken[c] += 1
                    left -= 1
                    progressed = True
            if not progressed:
                break
    chosen.sort()
    summary = dict(count=len(chosen), seed=int(seed), modes={m: quota[m] for m in modes},
                   charts={m: len(by_mode[m]) for m in modes},
                   charts_covered=len({records[i]['chart_id'] for i in chosen}),
                   empty_windows=sum(records[i].get('event_count', 0) == 0 for i in chosen),
                   rule='equal per-mode share, round-robin per chart, uniform within chart; no outcome filter')
    return chosen, summary

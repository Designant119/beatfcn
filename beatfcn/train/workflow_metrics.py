"""Workflow-readiness metrics: how much charting work a model draft saves (whole song).

Chart Edit Rate (CER) is the chart analogue of HTER (human-targeted translation
edit rate): the minimum total cost of an edit script turning a *draft* into a
human *reference* chart, divided by the cost of charting the reference from
scratch (inserting every reference note). CER < 1 means starting from the
draft is cheaper than starting from an empty chart; ``1 - CER`` is reported as
the *estimated operation savings*. CER can exceed 1 (a draft full of junk costs
deletions on top of all insertions).

Edit operations (one draft note ``d`` paired with one reference note ``r``, or
unpaired notes):

* keep     |d.on - r.on| <= 20 ms, same type, same lane (mania), and for holds
           |d.end - r.end| <= 40 ms                                   cost 0
* retime   same type/lane, 20 ms < |d.on - r.on| <= 150 ms              0.4
           (an editor drag moves the whole object, so the hold tail is then
           judged by duration: |dur_d - dur_r| <= 40 ms)
* retype   onset within 20 ms, wrong type                              0.4
* relane   onset within 20 ms, same type, wrong mania column           0.4
           (not in the original list; a column drag is priced like a retime)
* tail     hold tail outside 40 ms (or a tap retyped into a hold needs
           its tail placed)                                            0.4
* delete   unpaired draft note                                         0.3
* insert   unpaired reference note (placing a note from scratch)       1.0

Strict mode (default) forbids compound timing+type/lane fixes (a note that is
both late and of the wrong type is delete+insert); ``compound=True`` allows
them additively. Phira CER is *rhythm only*: the model does not output Phira
positions (owner decision), so judge-line/x placement is neither scored nor
charged -- a from-scratch charter must place positions too, which CER omits.

Matcher: exact minimum-cost bipartite assignment. Pair costs are only defined
for |d.on - r.on| <= 150 ms, so the candidate graph is split into connected
components (union-find) and each component is solved exactly with the
shortest-augmenting-path Hungarian algorithm (O(k^3) worst case, k = component
size; components are local in time, so k is small in practice: <= ~350 notes
on real human-vs-human pairs). A time-monotone DP alignment would be faster
but is NOT exact: chords make optimal matchings cross (two notes at one onset
matched to two reference notes in either order), and different lanes/types
interleave in time. Optimality is checked against brute force in tests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

import numpy as np

DEFAULT_COSTS = dict(insert=1.0, delete=0.3, retime=0.4, retype=0.4, relane=0.4, tail=0.4)
EDIT_OPS = ('retime', 'retype', 'relane', 'tail')
DEFAULT_TOLERANCES = dict(keep_onset_us=20_000, tail_us=40_000, window_us=150_000)
SNAP_DIVISORS = (1, 2, 4, 8, 16, 3, 6)
SNAP_MAX_US = 30_000


# ------------------------------------------------------------------ notes
def as_notes(events, mode):
    """(onset_us, end_us, type, lane) tuples; lane is -1 for Phira (positions not scored)."""
    out = []
    for e in events:
        if isinstance(e, dict):
            on, end, typ, lane = e['onset_us'], e['end_us'], e['type'], e.get('lane', -1)
        elif isinstance(e, tuple):
            on, end, typ, lane = (e + (-1,))[:4] if len(e) == 3 else e
        else:
            on, end, typ = e.onset_us, e.end_us, e.type
            lane = getattr(e, 'lane', None)
            if lane is None:
                loc = getattr(e, 'location', None)
                lane = getattr(loc, 'lane', -1)
        out.append((int(on), int(end), str(typ), int(lane) if mode == 'mania_4k' else -1))
    return out


def _costs(costs):
    merged = dict(DEFAULT_COSTS)
    if costs:
        unknown = set(costs)-set(DEFAULT_COSTS)
        if unknown:
            raise ValueError(f'unknown cost keys {sorted(unknown)}')
        merged.update(costs)
    if any(v < 0 for v in merged.values()):
        raise ValueError('costs must be non-negative')
    return merged


def _tolerances(tolerances):
    merged = dict(DEFAULT_TOLERANCES)
    merged.update(tolerances or {})
    return merged


def pair_ops(d, r, *, tolerances=None, compound=False):
    """Edit operations needed to turn draft note d into reference note r, or None if not pairable."""
    tol = _tolerances(tolerances)
    dt = abs(d[0]-r[0])
    if dt > tol['window_us']:
        return None
    same_type, same_lane = d[2] == r[2], d[3] == r[3]
    d_hold, r_hold = d[2] == 'hold', r[2] == 'hold'
    ops = []
    if dt <= tol['keep_onset_us']:
        if not same_type:
            ops.append('retype')
        if not same_lane:
            ops.append('relane')
        if r_hold and (not d_hold or abs(d[1]-r[1]) > tol['tail_us']):
            ops.append('tail')
        return tuple(ops)
    if not (same_type and same_lane) and not compound:
        return None
    ops.append('retime')
    if not same_type:
        ops.append('retype')
    if not same_lane:
        ops.append('relane')
    if r_hold and (not d_hold or abs((d[1]-d[0])-(r[1]-r[0])) > tol['tail_us']):
        ops.append('tail')
    return tuple(ops)


# ------------------------------------------------------------- assignment
def hungarian(cost):
    """Exact min-cost assignment of every row (rows <= cols). Returns col index per row."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n > m:
        raise ValueError('hungarian needs rows <= cols')
    u, v = np.zeros(n+1), np.zeros(m+1)
    p, way = np.zeros(m+1, dtype=np.int64), np.zeros(m+1, dtype=np.int64)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minv = np.full(m+1, np.inf)
        used = np.zeros(m+1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0-1]-u[i0]-v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked))+1
            delta = masked[j1-1]
            idx = np.flatnonzero(used)
            u[p[idx]] += delta
            v[idx] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = np.full(n, -1, dtype=np.int64)
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result


def _candidate_pairs(draft, ref, costs, tolerances, compound):
    gain_cap = costs['delete']+costs['insert']
    if not draft or not ref:
        return []
    order = sorted(range(len(ref)), key=lambda j: ref[j][0])
    onsets = np.array([ref[j][0] for j in order], dtype=np.int64)
    window = _tolerances(tolerances)['window_us']
    pairs = []
    for i, d in enumerate(draft):
        lo = int(np.searchsorted(onsets, d[0]-window, 'left'))
        hi = int(np.searchsorted(onsets, d[0]+window, 'right'))
        for k in range(lo, hi):
            j = order[k]
            ops = pair_ops(d, ref[j], tolerances=tolerances, compound=compound)
            if ops is None:
                continue
            c = sum(costs[o] for o in ops)
            if c < gain_cap:          # otherwise delete+insert is at least as cheap
                pairs.append((i, j, c, ops))
    return pairs


def optimal_matching(draft, ref, *, costs=None, tolerances=None, compound=False):
    """Exact min-cost edit script. Returns list of (draft_index, ref_index, cost, ops)."""
    costs = _costs(costs)
    pairs = _candidate_pairs(draft, ref, costs, tolerances, compound)
    if not pairs:
        return []
    n = len(draft)
    parent = list(range(n+len(ref)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _, _ in pairs:
        a, b = find(i), find(n+j)
        if a != b:
            parent[a] = b
    groups = defaultdict(list)
    for p in pairs:
        groups[find(p[0])].append(p)
    gain_cap = costs['delete']+costs['insert']
    chosen = []
    for group in groups.values():
        if len(group) == 1:
            chosen.append(group[0])
            continue
        rows = sorted({p[0] for p in group})
        cols = sorted({p[1] for p in group})
        transpose = len(rows) > len(cols)
        if transpose:
            rows, cols = cols, rows
        ri = {x: k for k, x in enumerate(rows)}
        ci = {x: k for k, x in enumerate(cols)}
        # Weight (cost - delete - insert) < 0 for a useful pair; 0 = "leave both unpaired".
        matrix = np.zeros((len(rows), len(cols)))
        lookup = {}
        for p in group:
            a, b = (p[1], p[0]) if transpose else (p[0], p[1])
            matrix[ri[a], ci[b]] = p[2]-gain_cap
            lookup[(ri[a], ci[b])] = p
        for r_idx, c_idx in enumerate(hungarian(matrix)):
            p = lookup.get((r_idx, int(c_idx)))
            if p is not None:
                chosen.append(p)
    return chosen


def chart_edit_rate(draft, ref, mode, *, costs=None, tolerances=None, compound=False):
    """CER and operation breakdown for one draft against one reference chart."""
    costs = _costs(costs)
    draft, ref = as_notes(draft, mode), as_notes(ref, mode)
    matching = optimal_matching(draft, ref, costs=costs, tolerances=tolerances, compound=compound)
    ops = Counter()
    matched_cost = 0.
    for _, _, c, pair in matching:
        matched_cost += c
        ops.update(pair)
        if not pair:
            ops['keep'] += 1
    ops['delete'] = len(draft)-len(matching)
    ops['insert'] = len(ref)-len(matching)
    total = matched_cost+costs['delete']*ops['delete']+costs['insert']*ops['insert']
    scratch = costs['insert']*len(ref)
    cer = total/scratch if scratch > 0 else None
    return dict(cer=cer, savings=None if cer is None else 1.-cer, total_cost=total, scratch_cost=scratch,
                draft_notes=len(draft), reference_notes=len(ref), paired=len(matching),
                ops={k: int(ops.get(k, 0)) for k in ('keep',)+EDIT_OPS+('delete', 'insert')},
                op_cost={k: float(costs[k]*ops.get(k, 0)) for k in EDIT_OPS+('delete', 'insert')},
                costs=costs, compound=compound)


def cost_sensitivity(draft, ref, mode, *, factors=(.5, 2.), per_op=False, tolerances=None, compound=False):
    """CER with every non-insert cost scaled (insert fixed: a uniform scale cancels in CER)."""
    out = {}
    for f in factors:
        scaled = {k: v*f for k, v in DEFAULT_COSTS.items() if k != 'insert'}
        out[f'edits_x{f:g}'] = chart_edit_rate(draft, ref, mode, costs=scaled, tolerances=tolerances,
                                               compound=compound)['cer']
        if per_op:
            for op in ('delete',)+EDIT_OPS:
                out[f'{op}_x{f:g}'] = chart_edit_rate(draft, ref, mode, costs={op: DEFAULT_COSTS[op]*f},
                    tolerances=tolerances, compound=compound)['cer']
    return out


# ------------------------------------------------------------- timing grid
class UnusableTiming(ValueError):
    """A stop / teleport BPM (osu gimmick: BPM ~0 or absurd): no usable beat grid."""


MIN_BEAT_US, MAX_BEAT_US = 1_000.0, 60_000_000.0  # 60000 BPM .. 1 BPM (same rule as scripts/build_chart_timing.py)


def grid_from_timing(timing):
    """Charter timing (canonical TimingMap) -> list of (segment_start_us_audio, beat_us).

    Raises UnusableTiming for gimmick BPMs (callers treat the chart as having no grid)."""
    if timing is None:
        return None
    segments = []
    for s in timing.segments:
        start = (s.start_sec+timing.offset_sec)*1_000_000
        try:
            beat = float(60_000_000/s.bpm)
        except (OverflowError, ZeroDivisionError) as exc:
            raise UnusableTiming(f'unusable BPM {s.bpm!r} at {s.start_sec}s') from exc
        if not MIN_BEAT_US <= beat <= MAX_BEAT_US:
            raise UnusableTiming(f'beat {beat:.6g} us outside [{MIN_BEAT_US}, {MAX_BEAT_US}]')
        segments.append((float(start), beat))
    segments.sort()
    return segments


def snap_times(times_us, grid, *, divisors=SNAP_DIVISORS, max_snap_us=SNAP_MAX_US):
    """Snap to the nearest point of the union of beat subdivisions; leave alone if > max_snap_us away.

    The grid is anchored at each segment's (red line's) start; times before the
    first segment extend its BPM backwards (osu rule).
    """
    t = np.asarray(times_us, dtype=np.float64)
    if not len(t) or not grid:
        return t.astype(np.int64), np.zeros(len(t), dtype=bool)
    starts = np.array([g[0] for g in grid])
    beat = np.array([g[1] for g in grid])
    seg = np.clip(np.searchsorted(starts, t, 'right')-1, 0, len(grid)-1)
    local = (t-starts[seg])/beat[seg]
    best = np.full(len(t), np.inf)
    snapped = t.copy()
    for k in divisors:
        cand = starts[seg]+np.round(local*k)/k*beat[seg]
        d = np.abs(cand-t)
        better = d < best
        best[better] = d[better]
        snapped[better] = cand[better]
    moved = best <= max_snap_us
    result = np.where(moved, np.round(snapped), t).astype(np.int64)
    return result, moved


def snap_notes(notes, grid, **kw):
    """Snap heads and hold tails; a hold whose snapped tail would not follow its head keeps its raw tail."""
    if not notes:
        return [], dict(notes=0, heads_snapped=0, tails_snapped=0)
    heads, hm = snap_times([n[0] for n in notes], grid, **kw)
    tails, tm = snap_times([n[1] for n in notes], grid, **kw)
    out, tails_snapped = [], 0
    for n, h, e, moved in zip(notes, heads.tolist(), tails.tolist(), tm.tolist()):
        if n[2] == 'hold':
            if e <= h:
                e = n[1] if n[1] > h else h+1
            else:
                tails_snapped += bool(moved)
        else:
            e = h
        out.append((h, e, n[2], n[3]))
    return out, dict(notes=len(notes), heads_snapped=int(hm.sum()), tails_snapped=tails_snapped)


def grid_alignment(ref_onsets_us, chart_onsets_us, tolerance_us=1000):
    """Fraction of reference onsets within tolerance of a raw-chart onset (same audio clock check)."""
    a = np.sort(np.asarray(chart_onsets_us, dtype=np.int64))
    r = np.asarray(ref_onsets_us, dtype=np.int64)
    if not len(r) or not len(a):
        return 0.
    i = np.clip(np.searchsorted(a, r), 1, len(a)-1) if len(a) > 1 else np.zeros(len(r), dtype=np.int64)
    d = np.minimum(np.abs(a[i]-r), np.abs(a[np.maximum(i-1, 0)]-r))
    return float(np.mean(d <= tolerance_us))


# --------------------------------------------------------------- playability
def playability(events, mode, *, min_gap_us=40_000, chord_us=1000, max_stack=4):
    """Reference-free validity checks. Rates are violations per opportunity (see keys)."""
    notes = sorted(as_notes(events, mode))
    n = len(notes)
    if mode == 'mania_4k':
        dup = covered = jack = jack100 = release = 0
        by_lane = defaultdict(list)
        for x in notes:
            by_lane[x[3]].append(x)
        for lane_notes in by_lane.values():
            lane_notes.sort()
            holds = []
            for k, x in enumerate(lane_notes):
                if k and lane_notes[k-1][0] == x[0]:
                    dup += 1
                elif k and 0 < x[0]-lane_notes[k-1][0] < min_gap_us:
                    jack += 1
                if k and 0 < x[0]-lane_notes[k-1][0] < 100_000:
                    jack100 += 1  # same key again within 100 ms: ~absent in human charts (playability, 2026-11-02)
                if k and lane_notes[k-1][2] == 'hold' and 0 <= x[0]-lane_notes[k-1][1] < min_gap_us:
                    # next press on the key at (or < min_gap after) the previous hold's release: unplayable
                    # (osu autoplay misses it). Not caught by hold_covers_head, which needs end > onset
                    # (metric erratum 2026-11-02 21:20, owner pilot practice segment).
                    release += 1
                holds = [h for h in holds if h[1] > x[0]]
                if any(h[0] < x[0] for h in holds):
                    covered += 1
                if x[2] == 'hold':
                    holds.append(x)
        bad_lane = sum(1 for x in notes if not 0 <= x[3] < 4)
        return dict(notes=n, same_lane_overlap=dup, hold_covers_head=covered, jack_lt_40ms=jack,
                    bad_lane=bad_lane, same_lane_overlap_rate=_rate(dup, n), hold_covers_head_rate=_rate(covered, n),
                    jack_lt_40ms_rate=_rate(jack, n), jack_lt_100ms=jack100, jack_lt_100ms_rate=_rate(jack100, n),
                    hold_release_lt_40ms=release, hold_release_lt_40ms_rate=_rate(release, n),
                    any_violation_rate=_rate(dup+covered+jack+bad_lane, n),
                    any_violation_v2_rate=_rate(dup+covered+jack+bad_lane+release, n))
    groups = []
    for x in notes:
        if groups and x[0]-groups[-1][0][0] <= chord_us:
            groups[-1].append(x)
        else:
            groups.append([x])
    short = short_nondrag = 0
    for a, b in zip(groups, groups[1:]):
        if b[0][0]-a[0][0] < min_gap_us:
            short += 1
            if not (all(x[2] == 'drag' for x in a) and all(x[2] == 'drag' for x in b)):
                short_nondrag += 1
    stacks = sum(1 for g in groups if len(g) > max_stack)
    holds = [x for x in notes if x[2] == 'hold']
    zero = sum(1 for x in holds if x[1] <= x[0])
    gaps = max(len(groups)-1, 0)
    return dict(notes=n, onset_groups=len(groups), short_gap_lt_40ms=short, short_gap_lt_40ms_rate=_rate(short, gaps),
                short_gap_nondrag=short_nondrag, short_gap_nondrag_rate=_rate(short_nondrag, gaps),
                stack_gt4=stacks, stack_gt4_rate=_rate(stacks, len(groups)), holds=len(holds),
                zero_length_hold=zero, zero_length_hold_rate=_rate(zero, len(holds)))


def _rate(k, n):
    return k/n if n else None


def repair_hold_release(notes, mode, *, min_release_us=40_000, min_hold_us=60_000):
    """Mania playability repair: a hold whose release is closer than ``min_release_us`` to the next press of
    the same key is shortened to end ``min_release_us`` before it; if that leaves less than ``min_hold_us`` the
    hold becomes a tap. Phira is returned unchanged. Returns (notes, counts); notes are (onset, end, type, lane)."""
    if mode != 'mania_4k':
        return list(notes), dict(shortened=0, to_tap=0)
    notes = sorted(tuple(x) for x in notes)
    out, counts = list(notes), dict(shortened=0, to_tap=0)
    by_lane = defaultdict(list)
    for i, x in enumerate(notes):
        by_lane[x[3]].append(i)
    for idx in by_lane.values():
        for a, b in zip(idx, idx[1:]):
            on, end, typ, lane = out[a]
            nxt = out[b][0]
            if typ == 'hold' and nxt - end < min_release_us:
                new_end = nxt - min_release_us
                if new_end - on < min_hold_us:
                    out[a] = (on, on, 'tap', lane)
                    counts['to_tap'] += 1
                else:
                    out[a] = (on, new_end, typ, lane)
                    counts['shortened'] += 1
    return out, counts


# ------------------------------------------------------------------ density
def density_profile(draft, ref, duration_us, *, bin_us=1_000_000):
    bins = max(1, math.ceil(duration_us/bin_us))
    def hist(notes):
        h = np.zeros(bins)
        for x in notes:
            h[min(bins-1, max(0, x[0]//bin_us))] += 1
        return h
    a, b = hist(draft), hist(ref)
    pearson = None
    if a.std() > 0 and b.std() > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
    return dict(bin_us=bin_us, draft_notes_per_s=float(a.sum()/duration_us*1e6),
                reference_notes_per_s=float(b.sum()/duration_us*1e6),
                count_ratio=float(a.sum()/b.sum()) if b.sum() else None, pearson=pearson,
                mean_abs_bin_error=float(np.abs(a-b).mean()))


# ---------------------------------------------------------------- bootstrap
def bootstrap_mean(values_by_unit, *, levels=(.8, .95), samples=2000, seed=0):
    """Cluster bootstrap over units (songs). Each unit contributes the mean of its own values."""
    units = [float(np.mean(v)) for v in values_by_unit if len(v)]
    if not units:
        return dict(n_units=0, mean=None)
    x = np.asarray(units)
    out = dict(n_units=len(x), mean=float(x.mean()), median=float(np.median(x)))
    if len(x) < 2:
        out.update({f'ci{int(l*100)}': None for l in levels})
        return out
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(1)
    for level in levels:
        lo, hi = np.quantile(means, [(1-level)/2, (1+level)/2])
        out[f'ci{int(level*100)}'] = [float(lo), float(hi)]
    out['bootstrap_samples'] = samples
    return out


def stable_rank(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,)+parts)).encode()).hexdigest()


# ------------------------------------------------- charter timing: grid metrics
# "Charter timing" = the reference chart's own BPM/offset (red lines). A charter
# sets this up before placing notes, so it is a legitimate workflow input -- but
# every readout that uses it is labelled and never reported as model capability.
CHARTER_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)
ON_GRID_US = 5_000
CHARTER_SNAP_MAX_US = 30_000
SUBDIVISION_LABELS = tuple(f'1/{k}' for k in CHARTER_DIVISORS)+('off',)
JS_MAX_NATS = math.log(2.)


def js_nats(left, right):
    """Jensen-Shannon divergence (nats) with status, same convention as benchmark v2
    (requirement_diagnostics.js_with_status): both empty -> (0, 'both_empty'); one side
    empty -> (ln 2, 'one_side_empty')."""
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if not a.sum() and not b.sum():
        return 0., 'both_empty'
    if not a.sum() or not b.sum():
        return JS_MAX_NATS, 'one_side_empty'
    a, b = a/a.sum(), b/b.sum()
    m = (a+b)/2

    def kl(p):
        k = p > 0
        return float((p[k]*np.log(p[k]/m[k])).sum())
    return (kl(a)+kl(b))/2, 'defined'


def _segments(times_us, grid):
    t = np.asarray(times_us, dtype=np.float64)
    starts = np.array([g[0] for g in grid])
    beat = np.array([g[1] for g in grid])
    seg = np.clip(np.searchsorted(starts, t, 'right')-1, 0, len(grid)-1)
    return t, starts, beat, seg


def beat_length_at(times_us, grid):
    """Beat length (us) of the timing segment that governs each time (first segment extends backwards)."""
    t, _, beat, seg = _segments(times_us, grid)
    return beat[seg] if len(t) else np.zeros(0)


def subdivision_classes(times_us, grid, *, divisors=CHARTER_DIVISORS, tolerance_us=ON_GRID_US):
    """Per time: index into ``divisors`` of the COARSEST subdivision (1/k of the beat) with a grid
    point within ``tolerance_us``, or len(divisors) when off-grid. The next red line's own start
    also counts as an on-beat point (a note a hair before a BPM change sits on the new line)."""
    t, starts, beat, seg = _segments(times_us, grid)
    out = np.full(len(t), len(divisors), dtype=np.int64)
    if not len(t):
        return out
    local = (t-starts[seg])/beat[seg]
    for idx in range(len(divisors)-1, -1, -1):  # finest first, so coarser hits overwrite
        k = divisors[idx]
        d = np.abs(local*k-np.round(local*k))*beat[seg]/k
        out[d <= tolerance_us] = idx
    nxt = np.minimum(seg+1, len(grid)-1)
    near_next = (nxt > seg) & (np.abs(starts[nxt]-t) <= tolerance_us)
    out[near_next] = 0
    return out


def grid_profile(notes, grid, *, divisors=CHARTER_DIVISORS, tolerance_us=ON_GRID_US):
    """Distinct onsets (chord = one onset) classified by subdivision of the charter timing."""
    onsets = np.unique(np.asarray([n[0] for n in notes], dtype=np.int64))
    classes = subdivision_classes(onsets, grid, divisors=divisors, tolerance_us=tolerance_us)
    hist = np.bincount(classes, minlength=len(divisors)+1)
    labels = tuple(f'1/{k}' for k in divisors)+('off',)
    return dict(onsets=int(len(onsets)), on_grid_fraction=float(1-hist[-1]/len(onsets)) if len(onsets) else None,
                histogram={lab: int(c) for lab, c in zip(labels, hist)})


def chance_on_grid(beat_us, *, divisors=CHARTER_DIVISORS, tolerance_us=ON_GRID_US):
    """Fraction of a beat within ``tolerance_us`` of a 1/k point (k in divisors): the on-grid fraction of
    onsets placed uniformly at random. Dense unions at fast tempos make it large (~0.6 at 180 BPM)."""
    from fractions import Fraction
    points = sorted({Fraction(j, k) for k in divisors for j in range(k)})
    lo = [float(f)*beat_us-tolerance_us for f in points]
    hi = [float(f)*beat_us+tolerance_us for f in points]
    covered, end = 0., -math.inf
    for a, b in zip(lo, hi):  # merge intervals on the line, then wrap the last onto the first
        a = max(a, end)
        if b > a:
            covered += b-a
            end = b
    covered -= max(0., end-(lo[0]+beat_us))  # overlap of the wrapped last interval with the first
    return float(min(1., covered/beat_us))


def grid_comparison(draft, ref, grid, **kw):
    """On-grid fraction (onset within 5 ms of a 1/1..1/16 point) of draft and reference, and the JS
    divergence of their subdivision histograms (incl. the off-grid bin), on the reference's charter timing."""
    a, b = grid_profile(draft, grid, **kw), grid_profile(ref, grid, **kw)
    js, status = js_nats(list(a['histogram'].values()), list(b['histogram'].values()))
    diff = (abs(a['on_grid_fraction']-b['on_grid_fraction'])
            if a['on_grid_fraction'] is not None and b['on_grid_fraction'] is not None else None)
    ref_heads = np.unique(np.asarray([n[0] for n in ref], dtype=np.int64))
    chance = (float(np.mean([chance_on_grid(x, **kw) for x in beat_length_at(ref_heads, grid)]))
              if len(ref_heads) else None)
    return dict(draft_on_grid_fraction=a['on_grid_fraction'], reference_on_grid_fraction=b['on_grid_fraction'],
                chance_on_grid_fraction=chance, on_grid_abs_diff=diff, subdivision_js_nats=js, subdivision_js_status=status,
                draft_histogram=a['histogram'], reference_histogram=b['histogram'],
                draft_onsets=a['onsets'], reference_onsets=b['onsets'])


def _greedy_tp(a, b, tol_us):
    a, b = sorted(a), sorted(b)
    i = j = tp = 0
    while i < len(a) and j < len(b):
        d = a[i]-b[j]
        if abs(d) <= tol_us:
            tp, i, j = tp+1, i+1, j+1
        elif d < 0:
            i += 1
        else:
            j += 1
    return tp


def note_f1(draft, ref, mode, *, tol_us=50_000):
    """Whole-song one-to-one greedy F1 (benchmark v2 convention): ``note`` = onset+type(+lane for mania),
    ``onset`` = onset only. Both empty -> None."""
    draft, ref = as_notes(draft, mode), as_notes(ref, mode)
    total = len(draft)+len(ref)
    if not total:
        return dict(note_f1=None, onset_f1=None, tp_note=0, tp_onset=0, draft=0, reference=0)
    keys = {(n[2], n[3]) for n in draft+ref}
    tp = sum(_greedy_tp([n[0] for n in draft if (n[2], n[3]) == k], [n[0] for n in ref if (n[2], n[3]) == k], tol_us)
             for k in keys)
    tpo = _greedy_tp([n[0] for n in draft], [n[0] for n in ref], tol_us)
    return dict(note_f1=2*tp/total, onset_f1=2*tpo/total, tp_note=tp, tp_onset=tpo, draft=len(draft), reference=len(ref))


def charter_timing_snap(draft, ref, mode, grid, *, divisors=CHARTER_DIVISORS, max_snap_us=CHARTER_SNAP_MAX_US):
    """Readout ``charter_timing_snap``: every draft onset (and hold tail) is snapped to the nearest
    1/1..1/16 point of the reference's charter timing if within 30 ms (farther notes are kept as
    they are), then CER and note/onset F1@50 ms are rescored. Uses human timing: NEVER capability."""
    draft, ref = as_notes(draft, mode), as_notes(ref, mode)
    snapped, stats = snap_notes(draft, grid, divisors=divisors, max_snap_us=max_snap_us)
    raw_f1, snap_f1 = note_f1(draft, ref, mode), note_f1(snapped, ref, mode)
    cer = chart_edit_rate(snapped, ref, mode)
    return dict(label='charter_timing_snap', cer=cer['cer'], ops=cer['ops'],
                note_f1_50ms=snap_f1['note_f1'], onset_f1_50ms=snap_f1['onset_f1'],
                raw_note_f1_50ms=raw_f1['note_f1'], raw_onset_f1_50ms=raw_f1['onset_f1'],
                snap_stats=stats, divisors=list(divisors), max_snap_us=max_snap_us)


# ------------------------------------------------ pattern-level human likeness
# Profiles are additive count accumulators (pattern_profile -> merge_profiles), so the
# same code serves one whole song (workflow) and pooled 8 s windows (benchmark v2).
IOI_BEAT_VALUES = (1/16, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 3/4, 1., 1.5, 2., 4.)
IOI_BEAT_EDGES = (0.,)+tuple(math.sqrt(a*b) for a, b in zip(IOI_BEAT_VALUES, IOI_BEAT_VALUES[1:]))+(math.inf,)
IOI_MS_EDGES = (0., 10., 20., 30., 40., 60., 80., 120., 160., 250., 375., 500., 750., 1000., 2000., 4000., math.inf)
JACK_US = 200_000
DENSITY_BIN_US = 2_000_000
CHORD_US = 1_000
GROUP_SIZES = 4  # bins 1, 2, 3, 4+
RUN_EDGES = (1, 2, 3, 5, 9, 17)  # bins 1, 2, 3-4, 5-8, 9-16, 17+
PHIRA_KINDS = ('tap', 'hold', 'flick', 'drag')
MANIA_KINDS = ('tap', 'hold')


def kinds(mode):
    return MANIA_KINDS if mode == 'mania_4k' else PHIRA_KINDS


def onset_groups(notes, chord_us=CHORD_US):
    """Notes sorted by onset, grouped into simultaneous onsets (within ``chord_us`` of the group's first)."""
    groups = []
    for x in sorted(notes):
        if groups and x[0]-groups[-1][0][0] <= chord_us:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def _run_bin(length):
    return int(np.searchsorted(RUN_EDGES, length, 'right'))-1


def pattern_profile(notes, mode, *, grid=None, start_us=0, duration_us=None):
    """Additive pattern statistics of one chart (or window). ``grid`` (charter timing) switches IOIs
    to beat units; ``duration_us`` (song/window length from ``start_us``) fixes the density bins and
    the hold-occupancy denominator so draft and reference of one song share them."""
    notes = as_notes(notes, mode)
    k = kinds(mode)
    groups = onset_groups(notes)
    width = 4 if mode == 'mania_4k' else len(PHIRA_KINDS)
    prof = dict(mode=mode, notes=len(notes), groups=len(groups),
                type_counts=[sum(1 for n in notes if n[2] == t) for t in k],
                group_size=[0]*GROUP_SIZES, ioi_ms=[], ioi_beat=[] if grid else None,
                transition=[[0.]*width for _ in range(width)])
    for g in groups:
        prof['group_size'][min(len(g), GROUP_SIZES)-1] += 1
    heads = np.array([g[0][0] for g in groups], dtype=np.float64)
    if len(heads) > 1:
        gaps = np.diff(heads)
        prof['ioi_ms'] = (gaps/1000.).tolist()
        if grid:
            prof['ioi_beat'] = (gaps/beat_length_at(heads[:-1], grid)).tolist()
    key = (lambda n: n[3]) if mode == 'mania_4k' else (lambda n: PHIRA_KINDS.index(n[2]))
    for g, h in zip(groups, groups[1:]):
        w = 1./(len(g)*len(h))
        for a in g:
            for b in h:
                ka, kb = key(a), key(b)
                if 0 <= ka < width and 0 <= kb < width:
                    prof['transition'][ka][kb] += w
    if duration_us is None:
        lo = min((n[0] for n in notes), default=start_us)
        hi = max((max(n[0], n[1]) for n in notes), default=lo)
        start_us, duration_us = lo, max(hi-lo, 1)
    prof['span_us'] = int(duration_us)
    if mode == 'mania_4k':
        lanes = defaultdict(list)
        for n in notes:
            lanes[n[3]].append(n[0])
        prof['lane_counts'] = [len(lanes.get(lane, [])) for lane in range(4)]
        pairs = jacks = 0
        for times in lanes.values():
            d = np.diff(np.sort(np.asarray(times, dtype=np.int64)))
            d = d[d > 0]
            pairs += len(d)
            jacks += int((d < JACK_US).sum())
        prof.update(lane_pairs=pairs, jack_pairs=jacks,
                    hold_us=int(sum(max(n[1]-n[0], 0) for n in notes if n[2] == 'hold')), lane_time_us=4*int(duration_us))
        edges = np.arange(start_us, start_us+duration_us, DENSITY_BIN_US, dtype=np.float64)
        edges = np.append(edges, start_us+duration_us)
        widths = np.diff(edges)
        counts = np.histogram([n[0] for n in notes], edges)[0] if len(edges) > 1 else np.zeros(0)
        keep = widths >= DENSITY_BIN_US/2  # drop a short trailing bin
        prof['density_rates'] = (counts[keep]/(widths[keep]/1e6)).tolist()
    else:
        runs = dict(drag=[0]*len(RUN_EDGES), flick=[0]*len(RUN_EDGES))
        run_lengths = dict(drag=[], flick=[])
        for kind in runs:
            length = 0
            for g in groups+[[]]:
                if any(n[2] == kind for n in g):
                    length += 1
                elif length:
                    runs[kind][_run_bin(length)] += 1
                    run_lengths[kind].append(length)
                    length = 0
        prof.update(runs=runs, run_lengths=run_lengths)
    return prof


def merge_profiles(profiles):
    """Pool several profiles of one mode/IOI unit: counts add, value lists concatenate."""
    profiles = list(profiles)
    if not profiles:
        return None
    if len({p['mode'] for p in profiles}) != 1:
        raise ValueError('cannot merge profiles of different modes')
    concat = ('ioi_ms', 'density_rates')
    out = dict(mode=profiles[0]['mode'])
    # Beat-unit IOIs pool only when every profile had charter timing.
    out['ioi_beat'] = (None if any(p['ioi_beat'] is None for p in profiles)
                       else [x for p in profiles for x in p['ioi_beat']])
    for key, value in profiles[0].items():
        if key in out:
            continue
        if key in concat:
            out[key] = [x for p in profiles for x in p[key]]
        elif key == 'runs':
            out[key] = {kind: np.sum([p[key][kind] for p in profiles], axis=0).tolist() for kind in value}
        elif key == 'run_lengths':
            out[key] = {kind: [x for p in profiles for x in p[key][kind]] for kind in value}
        elif isinstance(value, list):
            out[key] = np.sum([p[key] for p in profiles], axis=0).tolist()
        else:
            out[key] = sum(p[key] for p in profiles)
    return out


def _ioi_ms_hist(prof):
    return np.histogram(np.asarray(prof['ioi_ms'], np.float64), IOI_MS_EDGES)[0]


def _ioi_beat_hist(prof):
    return np.histogram(np.asarray(prof['ioi_beat'], np.float64), IOI_BEAT_EDGES)[0]


def _frac(a, b):
    return a/b if b else None


def profile_values(prof):
    """Scalar summary of a profile (None where undefined, e.g. no notes)."""
    n, g = prof['notes'], prof['groups']
    sizes = np.arange(1, GROUP_SIZES+1)
    trans = np.asarray(prof['transition'])
    out = dict(type_mix=_frac(prof['type_counts'][0], n),
               ioi_ms=float(np.median(prof['ioi_ms'])) if prof['ioi_ms'] else None,
               ioi_beat=float(np.median(prof['ioi_beat'])) if prof['ioi_beat'] else None,
               same_key_transition=_frac(float(np.trace(trans)), float(trans.sum())),
               group_size_mean=_frac(float((np.asarray(prof['group_size'])*sizes).sum()), g),
               multi_note_fraction=_frac(sum(prof['group_size'][1:]), g))
    if prof['mode'] == 'mania_4k':
        lanes = np.asarray(prof['lane_counts'], np.float64)
        p = lanes[lanes > 0]/lanes.sum() if lanes.sum() else None
        rates = np.asarray(prof['density_rates'])
        out.update(jack_rate_200ms=_frac(prof['jack_pairs'], prof['lane_pairs']),
                   lane_balance_entropy=float(-(p*np.log(p)).sum()/math.log(4)) if p is not None else None,
                   hold_occupancy=_frac(prof['hold_us'], prof['lane_time_us']),
                   density_var_2s=float(rates.var()) if len(rates) > 1 and n else None,
                   density_cv_2s=float(rates.std()/rates.mean()) if len(rates) > 1 and rates.mean() > 0 else None)
    else:
        out.update(hold_fraction=_frac(prof['type_counts'][1], n),
                   drag_run_mean=float(np.mean(prof['run_lengths']['drag'])) if prof['run_lengths']['drag'] else None,
                   flick_run_mean=float(np.mean(prof['run_lengths']['flick'])) if prof['run_lengths']['flick'] else None)
    return out


# metric -> (scalar key in profile_values, distribution accessor for JS or None, meaning)
PROFILE_METRICS = dict(
    mania_4k=(
        ('chord_size', 'group_size_mean', lambda p: p['group_size'], 'notes per onset (1..4+); JS over the 1/2/3/4+ histogram'),
        ('jack_rate_200ms', 'jack_rate_200ms', None, 'same-lane consecutive onsets < 200 ms / same-lane consecutive pairs'),
        ('lane_balance', 'lane_balance_entropy', lambda p: p['lane_counts'], 'lane-usage entropy / ln 4 (1 = balanced); JS over lane counts'),
        ('lane_transition', 'same_key_transition', lambda p: np.ravel(p['transition']),
         'lane bigram between consecutive onsets (chord pairs weighted 1/(|a||b|)); value = same-lane share'),
        ('hold_occupancy', 'hold_occupancy', None, 'hold body time / (4 lanes x song length)'),
        ('density_var_2s', 'density_var_2s', None, 'variance of notes/s over 2 s bins (song length)'),
        ('density_cv_2s', 'density_cv_2s', None, 'coefficient of variation of notes/s over 2 s bins'),
        ('ioi_beat', 'ioi_beat', _ioi_beat_hist, 'inter-onset interval between distinct onsets in beats of the charter '
         'timing (only rows with timing); value = median, JS over 1/16..4 beat bins'),
        ('ioi_ms', 'ioi_ms', _ioi_ms_hist, 'same in ms (always; benchmark v2 gap bins)'),
        ('type_mix', 'type_mix', lambda p: p['type_counts'], 'type histogram; value = tap share'),
    ),
    phira=(
        ('type_transition', 'same_key_transition', lambda p: np.ravel(p['transition']),
         'type bigram between consecutive onsets; value = same-type share'),
        ('drag_run_length', 'drag_run_mean', lambda p: p['runs']['drag'],
         'consecutive onsets containing a drag; value = mean run, JS over 1/2/3-4/5-8/9-16/17+'),
        ('flick_run_length', 'flick_run_mean', lambda p: p['runs']['flick'], 'same for flick'),
        ('hold_fraction', 'hold_fraction', None, 'holds / notes'),
        ('simultaneous', 'multi_note_fraction', lambda p: p['group_size'],
         'notes per onset; value = share of onsets with >= 2 notes, JS over 1/2/3/4+'),
        ('ioi_beat', 'ioi_beat', _ioi_beat_hist, 'inter-onset interval between distinct onsets in beats of the charter '
         'timing (only rows with timing); value = median, JS over 1/16..4 beat bins'),
        ('ioi_ms', 'ioi_ms', _ioi_ms_hist, 'same in ms (always; benchmark v2 gap bins)'),
        ('type_mix', 'type_mix', lambda p: p['type_counts'], 'type histogram; value = tap share'),
    ))


def compare_profiles(draft, ref):
    """Per metric: draft value, reference value, |draft - reference|, and JS (nats) where a distribution exists."""
    if draft['mode'] != ref['mode']:
        raise ValueError('profiles must share the mode')
    a, b = profile_values(draft), profile_values(ref)
    beats = draft['ioi_beat'] is not None and ref['ioi_beat'] is not None
    out = {}
    for name, key, dist, _ in PROFILE_METRICS[draft['mode']]:
        if name == 'ioi_beat' and not beats:
            continue
        da, db = a[key], b[key]
        entry = dict(draft=da, reference=db, abs_diff=abs(da-db) if da is not None and db is not None else None)
        if dist is not None:
            entry['js_nats'], entry['js_status'] = js_nats(dist(draft), dist(ref))
        out[name] = entry
    return dict(metrics=out, charter_timing=beats, draft_notes=draft['notes'], reference_notes=ref['notes'])


def compare_patterns(draft, ref, mode, *, grid=None, duration_us=None):
    """Whole-song pattern-level human-likeness of ``draft`` vs one reference chart."""
    return compare_profiles(pattern_profile(draft, mode, grid=grid, duration_us=duration_us),
                            pattern_profile(ref, mode, grid=grid, duration_us=duration_us))

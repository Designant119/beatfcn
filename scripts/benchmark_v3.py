#!/usr/bin/env python3
"""Benchmark v3 scorer: W (workflow), H (human-likeness), M (music fidelity), hard constraints, paired compare.

Design source: reports/research-20261104/{B_quality_evaluation.md §5-6, D_human_chart_data.md Implications},
PLAN.md 2026-11-04 pre-registration. Reading guide: reports/evolve-20261104/benchmark_v3/README.md.

    build-floor   floor.json + human_reference.npz from D's non-test data and the tape (once; local, no model)
    score         one drafts file (consensus JSON, or seeds of drafts-samples.json) -> JSON + short table
    compare       a.json b.json -> 80% paired CIs per axis per mode and the PLAN dominance verdict
    noise         bootstrap noise of a scored JSON at given panel sizes, plus optional same-arm (A/A) false alarms
    admission     dose-response of each metric on held-out human charts (W, H, and M when mels are local)

Numpy only, no model, no GPU. Test-split charts are refused (chart JSON header checked before any notes are read).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from beatfcn.train.workflow_metrics import DEFAULT_COSTS, chart_edit_rate, playability  # noqa: E402

SCHEMA = 'benchmark-v3-1'
TYPES = ('tap', 'hold', 'flick', 'drag')  # beatfcn.events.canonical.TYPES (that import pulls torch; a test pins equality)
PANEL_SEED = 20261102  # run_workflow_benchmark.PANEL_SEED (a test pins equality)
SCAN_MS = tuple(range(-40, 41, 4))  # best-shift clock: same scan as random100/score_r100.py
DEFAULT_TAPE = '/Users/yukoval/beatfcn-backup/data/native-v2-tape'
DEFAULT_TIMING = '/Users/yukoval/beatfcn-backup/data/chart_timing-v2.json'
REPO = Path(os.environ.get('BEATFCN_REPO', '/Users/yukoval/yukoval-projects/beatfcn'))
DEFAULT_FLOOR = os.environ.get('BENCHMARK_V3_FLOOR', str(REPO/'reports/evolve-20261104/benchmark_v3/floor.json'))
EPSILON = .01  # PLAN: ties under .01 go to the cheaper
COST_TIE = .05  # PLAN: ties under 5% cost
CI_Z_LEVELS = (10, 90)  # 80% CI percentiles
LOG_RHO_EDGES = (math.log(1/2), math.log(1/1.5), math.log(1/1.25), math.log(1.25), math.log(1.5), math.log(2))
RHO_BINS = ('<.5', '.5-.67', '.67-.8', '.8-1.25', '1.25-1.5', '1.5-2', '>2')
CENTRAL_BIN = 3
NOTE_RATIO_BAND = (.85, 1.15)  # PLAN: median of the closest reference
NOTE_RATIO_SONG_BAND = (.7, 1.4)  # B §6 E5

# ---- H protocol. Frozen: changing any of these changes every H value (the hash below goes into each result).
H_PROTOCOL = dict(
    classifier='LOSO mean-difference LDA: w = inv(S)(mean_model_without_song - mean_human), S = human covariance '
               'from the pre-fit sample with shrinkage lambda; human sample never refit per arm',
    shrink=.1, z_bins=8, z_clip=6., ref_split='one chart per song_group, train split only; hash split fit/eval/admit',
    placement='AUC = mean over songs of P(score(eval human) < score(left-out draft)), ties .5')
CORE_COMMON = ['peak_over_avg', 'cv2', 'cv8', 'ac8', 'range8', 'break_share', 'hold_share', 'hold_occ', 'chord2', 'chord3',
               'fast', 'ioi_lt100', 'ioi_entropy']
CORE_MODE = dict(mania_4k=['p_stream', 'p_jump', 'p_hand', 'p_jack', 'p_chordjack', 'p_trill', 'p_roll', 'jack_lt200',
                           'jack_lt100', 'inner_share', 'lr_bias'],
                 phira=['tap_share', 'flick_share', 'drag_share', 'flick_run', 'drag_run', 'type_entropy'])
# type_switch is out: it reads the first note of a chord, so it changes with the storage order of notes (admission: invariance failed).
GRID_FEATURES = ['i1_1', 'i1_2', 'i1_3', 'i1_4', 'i1_6', 'i1_8', 'i_other', 'g1_1', 'g1_2', 'g1_4', 'g_triplet', 'g_fine',
                 'hold_beats']  # diagnostic set only: blown up by a few ms of clock offset (D §3, arm A)
STYLE_TRACK = dict(phira=['tap_share', 'hold_share', 'drag_share', 'flick_share', 'type_entropy', 'ioi_entropy'],
                   mania_4k=['p_jack', 'peak_over_avg', 'cv2', 'chord2', 'hold_share', 'p_jump'])
LOGF = {'nps_peak2', 'range8', 'hold_beats', 'n_seg', 'ioi_med_ms', 'drag_run', 'flick_run', 'bpm', 'hold_occ'}


def core_features(mode):
    return CORE_COMMON+CORE_MODE[mode]


def tf(k, v):
    return np.log1p(v) if k in LOGF else v


# ------------------------------------------------------------------ small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stable_int(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def primary_reference(song):
    """Same hash pick as run_workflow_benchmark.primary_reference (no selection on density or CER)."""
    rank = lambda r: hashlib.sha256(':'.join(map(str, (PANEL_SEED, 'primary', r['chart_id']))).encode()).hexdigest()
    return min(song['references'], key=rank)


def ci80(boot):
    return [float(np.percentile(boot, CI_Z_LEVELS[0])), float(np.percentile(boot, CI_Z_LEVELS[1]))]


def rng_for(*parts):
    return np.random.default_rng(stable_int(*parts) % (2**32))


def boot_mean(x, samples=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return dict(mean=None, ci80=None, n=0)
    m = x[np.random.default_rng(seed).integers(0, len(x), (samples, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci80=ci80(m), n=int(len(x)))


def ratio_score(cer, null, human):
    """W from per-song triples: ratio of means, 0 = null, 1 = human, not clipped here."""
    denom = np.mean(null)-np.mean(human)
    return float((np.mean(null)-np.mean(cer))/denom) if denom > 0 else float('nan')


def boot_ratio(cer, null, human, samples=2000, seed=0):
    cer, null, human = (np.asarray(x, float) for x in (cer, null, human))
    if not len(cer):
        return dict(W=None, W_raw=None, ci80=None, n=0)
    idx = np.random.default_rng(seed).integers(0, len(cer), (samples, len(cer)))
    b = (null[idx].mean(1)-cer[idx].mean(1))/np.maximum(null[idx].mean(1)-human[idx].mean(1), 1e-9)
    w = ratio_score(cer, null, human)
    return dict(W=float(np.clip(w, 0, 1)), W_raw=w, ci80=ci80(b), n=int(len(cer)))


def rho_bin(rho):
    return int(np.searchsorted(LOG_RHO_EDGES, math.log(max(rho, 1e-6))))


def count_lower_bound(rho, delete=DEFAULT_COSTS['delete'], insert=DEFAULT_COSTS['insert']):
    """Least possible CER for a draft with rho = draft notes / reference notes: B §5.e E3."""
    return max(insert*(1-rho), delete*(rho-1), 0.)


# ------------------------------------------------------------------ loading
def chart_split(tape, target_name):
    """The split from the chart's own JSON header. Fail closed if it cannot be read."""
    path = Path(tape)/target_name.replace('-targets.npz', '-chart.json')
    with open(path) as f:
        head = f.read(4096)
    m = re.search(r'"split":\s*"(\w+)"', head)
    if not m:
        raise ValueError(f'no split flag in {path}')
    return m.group(1)


def load_chart(tape, target_name, mode, *, check_split=True):
    """Notes [(on, end, type, lane)] of a chart. Refuses test-split charts before opening the notes."""
    target_name = Path(target_name).name
    if check_split and chart_split(tape, target_name) == 'test':
        raise PermissionError(f'test-split chart refused: {target_name}')
    with np.load(Path(tape)/target_name, allow_pickle=False) as z:
        return [(int(a), int(b), TYPES[int(k)], int(l) if mode == 'mania_4k' else -1)
                for a, b, k, l in zip(z['onset_us'], z['end_us'], z['kind'], z['lane'])]


def load_drafts(path, seed=None):
    """{song_id: state} from a consensus/drafts JSON, or from drafts-samples.json with ``seed``.
    A truncated samples file (C_product) only yields seed 0; other seeds raise."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if not text.startswith('{"0": ') or seed not in (0, '0'):
            raise ValueError(f'{path} is truncated or malformed and only seed 0 of a samples file can be recovered')
        return json.JSONDecoder().raw_decode(text, 6)[0]
    del text
    if data and all(re.fullmatch(r'\d+', k) for k in data):
        if seed is None:
            raise ValueError(f'{path} is a samples file: give --seeds')
        return data[str(seed)]
    if seed not in (None, 'none'):
        raise ValueError(f'{path} is not a samples file: do not give --seeds')
    return data


def draft_tuples(song, state):
    return [(int(e['onset_us']), int(e['end_us']), e['type'], int(e['lane']) if song['mode'] == 'mania_4k' else -1)
            for e in state['events']]


def load_panel(path):
    with open(path) as f:
        data = json.load(f)
    panel = data['panel'] if 'panel' in data else data
    readouts = {k: v for k, v in (data.get('readout_rows') or {}).items()}
    if 'draft_rows' in data:
        readouts['single_sample'] = dict(draft_rows=data['draft_rows'])
    return panel, readouts


def load_timing(path):
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        charts = json.load(f)['charts']
    return {k: v['segments'] for k, v in charts.items() if v.get('valid') and v.get('segments')}


def arrays(notes):
    if not len(notes):
        return (np.zeros(0, np.int64),)*4
    a = np.array([(x[0], x[1], TYPES.index(x[2]), x[3]) for x in notes], np.int64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# ------------------------------------------------------------------ beat clock
class BeatClock:
    """Maps time to beats over charter timing segments [start_us, beat_us, meter] (extrapolates at both ends)."""

    def __init__(self, segments):
        self.st = np.array([s[0] for s in segments], float)
        self.b = np.array([s[1] for s in segments], float)
        self.cum = np.concatenate([[0.], np.cumsum(np.diff(self.st)/self.b[:-1])])

    @property
    def median_beat(self):
        return float(np.median(self.b))

    def to_beats(self, t):
        t = np.asarray(t, float)
        i = np.clip(np.searchsorted(self.st, t, 'right')-1, 0, len(self.st)-1)
        return self.cum[i]+(t-self.st[i])/self.b[i]

    def to_time(self, x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(self.cum, x, 'right')-1, 0, len(self.st)-1)
        return self.st[i]+(x-self.cum[i])*self.b[i]


def warp_notes(notes, segs_src, segs_dst, duration_us):
    """Wrong-song null (B N3): human notes moved beat-for-beat onto another song's charter grid.
    Beats are octave-folded so the tempo ratio stays within sqrt(2)."""
    src, dst = BeatClock(segs_src), BeatClock(segs_dst)
    fold = 2.**round(math.log2(src.median_beat/dst.median_beat))
    if not len(notes):
        return []
    on = dst.to_time(src.to_beats([n[0] for n in notes])*fold)
    end = dst.to_time(src.to_beats([n[1] for n in notes])*fold)
    return [(int(a), int(max(a, b)), n[2], n[3]) for a, b, n in zip(on, end, notes) if 0 <= a <= duration_us]


def f1_at(draft_on, ref_on, tol=20_000):
    a, b = np.unique(draft_on), np.unique(ref_on)
    i = j = tp = 0
    while i < len(a) and j < len(b):
        d = a[i]-b[j]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return 2.*tp/(len(a)+len(b)) if tp else 0.


# ------------------------------------------------------------------ style features (verbatim from research D, feats.py)
FAST_US = 250_000
DIVS = (1, 2, 4, 3, 6, 8, 12, 16)
DNAME = {1: 'g1_1', 2: 'g1_2', 4: 'g1_4', 3: 'g1_3', 6: 'g1_6', 8: 'g1_8', 12: 'g1_12', 16: 'g1_16'}
IOIV = {'i1_1': 1., 'i1_2': .5, 'i1_3': 1/3, 'i1_4': .25, 'i1_6': 1/6, 'i1_8': .125}
def beat_at(t, segs):
    st = np.array([s[0] for s in segs], float); b = np.array([s[1] for s in segs], float)
    i = np.clip(np.searchsorted(st, t, 'right')-1, 0, len(st)-1)
    return st[i], b[i]

def _lag1(x):
    if len(x) < 3 or x.std() == 0: return 0.
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

def run_stat(flag):
    """mean length of True runs (0 if none)"""
    if not flag.any(): return 0.
    d = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
    return float((np.where(d == -1)[0]-np.where(d == 1)[0]).mean())

def features(on, end, kind, lane, mode, duration_us, segs=None):
    o = np.argsort(on, kind='stable'); on, end, kind, lane = on[o], end[o], kind[o], lane[o]
    n = len(on); f = {}
    if n < 2: return None
    span = max(int(on[-1]-on[0]), 1_000_000); span_s = span/1e6
    times, first, cnt = np.unique(on, return_index=True, return_counts=True)
    ng = len(times)
    f['n_notes'] = float(n); f['active_s'] = span_s; f['duration_s'] = duration_us/1e6
    f['nps'] = n/span_s; f['group_rate'] = ng/span_s
    f['log_nps'] = math.log(max(n/span_s, 1e-3))
    f['lead_share'] = float(on[0])/max(duration_us, 1); f['tail_share'] = max(duration_us-int(max(on[-1], end.max())), 0)/max(duration_us, 1)
    e2 = np.arange(on[0], on[-1]+2_000_001, 2_000_000); c2 = np.histogram(on, e2)[0]/2.
    f['nps_peak2'] = float(c2.max()); f['nps_p95_2'] = float(np.percentile(c2, 95)); f['peak_over_avg'] = float(c2.max()/max(c2.mean(), 1e-6))
    f['break_share'] = float((c2 == 0).mean()); f['cv2'] = float(c2.std()/max(c2.mean(), 1e-6))
    e8 = np.arange(on[0], on[-1]+8_000_001, 8_000_000); c8 = np.histogram(on, e8)[0]/8.
    f['cv8'] = float(c8.std()/max(c8.mean(), 1e-6)); f['ac8'] = _lag1(c8); f['range8'] = float((np.percentile(c8, 90)+.25)/(np.percentile(c8, 10)+.25))
    for nm, k in (('tap', 0), ('hold', 1), ('flick', 2), ('drag', 3)):
        if mode == 'phira' or k < 2: f[nm+'_share'] = float((kind == k).mean())
    h = kind == 1; hl = (end-on)[h]
    f['hold_occ'] = float(hl.sum())/span/(4 if mode == 'mania_4k' else 1)
    f['chord2'] = float((cnt >= 2).mean()); f['chord3'] = float((cnt >= 3).mean()); f['npg'] = float(n/ng)
    ioi = np.diff(times).astype(float)
    f['ioi_med_ms'] = float(np.median(ioi)/1000); f['ioi_lt60'] = float((ioi < 60_000).mean()); f['ioi_lt100'] = float((ioi < 100_000).mean())
    f['fast'] = float((ioi <= FAST_US).mean())
    pp = np.bincount(np.clip((np.log2(np.maximum(ioi, 1000)/1000)*2).astype(int), 0, 40)); pp = pp[pp > 0]/pp.sum(); f['ioi_entropy'] = float(-(pp*np.log(pp)).sum())
    # grid
    if segs:
        st, b = beat_at(times[:-1].astype(float), segs); bb = ioi/b
        for k, v in IOIV.items():
            f[k] = float((np.abs(bb-v)*b <= 8000).mean())
        f['i_other'] = max(0., 1-sum(f[k] for k in IOIV))
        st, b = beat_at(times.astype(float), segs); x = (times-st)/b
        left = np.ones(ng, bool); tot = {}
        for D in DIVS:
            dev = np.abs(x*D-np.round(x*D))*b/D
            hit = left & (dev <= 5000)
            tot[D] = float(hit.mean()); left &= ~hit
        for D in DIVS: f[DNAME[D]] = tot[D]
        f['g_off'] = float(left.mean()); f['g_triplet'] = tot[3]+tot[6]+tot[12]; f['g_fine'] = tot[8]+tot[16]
        f['bpm'] = 6e7/float(np.median(b)); f['n_seg'] = float(len(segs))
        if h.any():
            sth, bh = beat_at(on[h].astype(float), segs); f['hold_beats'] = float(np.median(hl/bh))
        else: f['hold_beats'] = 0.
    else:
        for k in list(IOIV)+['i_other', 'g_off', 'g_triplet', 'g_fine', 'bpm', 'n_seg', 'hold_beats']+list(DNAME.values()): f[k] = float('nan')
    if mode == 'mania_4k':
        m = np.zeros(ng, np.int64); idx = np.searchsorted(times, on); np.bitwise_or.at(m, idx, 1 << np.clip(lane, 0, 3))
        pc = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4])[m]
        fast = ioi <= FAST_US; sh = (m[:-1] & m[1:]) != 0; mx = np.maximum(pc[:-1], pc[1:]); nf = max(int(fast.sum()), 1)
        f['p_jack'] = float((fast & sh & (mx == 1)).sum()/nf); f['p_chordjack'] = float((fast & sh & (mx >= 2)).sum()/nf)
        f['p_stream'] = float((fast & ~sh & (mx == 1)).sum()/nf); f['p_jump'] = float((fast & ~sh & (mx == 2)).sum()/nf); f['p_hand'] = float((fast & ~sh & (mx >= 3)).sum()/nf)
        single = pc == 1
        if ng > 3:
            tr = fast[:-2] & fast[1:-1] & fast[2:] & single[:-3] & single[1:-2] & (m[:-3] == m[2:-1]) & (m[1:-2] == m[3:]) & (m[:-3] != m[1:-2])
            f['p_trill'] = float(tr.sum()/nf)
            lg = np.log2(np.maximum(m, 1)).astype(int); d = np.diff(lg)
            ro = fast[:-1] & fast[1:] & single[:-2] & single[1:-1] & single[2:] & (d[:-1]*d[1:] > 0) & (np.abs(d[:-1]) == 1) & (np.abs(d[1:]) == 1)
            f['p_roll'] = float(ro.sum()/nf)
        else: f['p_trill'] = f['p_roll'] = 0.
        pairs = jacks = j100 = 0; lc = np.zeros(4)
        for l in range(4):
            tl = np.sort(on[lane == l]); lc[l] = len(tl); d = np.diff(tl); d = d[d > 0]; pairs += len(d); jacks += int((d < 200_000).sum()); j100 += int((d < 100_000).sum())
        f['jack_lt200'] = jacks/max(pairs, 1); f['jack_lt100'] = j100/max(pairs, 1)
        p = lc/lc.sum(); f['lane_entropy'] = float(-(p[p > 0]*np.log2(p[p > 0])).sum()); f['lr_bias'] = float(abs(p[0]+p[1]-p[2]-p[3])); f['inner_share'] = float(p[1]+p[2])
    else:
        idx = np.searchsorted(times, on)
        for nm, k in (('flick', 2), ('drag', 3)):
            fl = np.zeros(ng, bool); fl[idx[kind == k]] = True
            f[nm+'_run'] = run_stat(fl); f[nm+'_groups'] = float(fl.mean())
        tk = kind[first]; f['type_switch'] = float((tk[1:] != tk[:-1]).mean())
        c = np.bincount(kind, minlength=4)/n; f['type_entropy'] = float(-(c[c > 0]*np.log2(c[c > 0])).sum())
    return f



# ------------------------------------------------------------------ hard-constraint checks
def rule_counts(notes, mode, duration_us):
    """Audio-free rule checks (B §2.1 H1-H5, §5.b B7 + workflow_metrics.playability)."""
    p = playability(notes, mode)
    on, end, kind, lane = arrays(notes)
    n = len(notes)
    out = dict(notes=n)
    hold = kind == 1 if n else np.zeros(0, bool)
    out['bad_hold_end'] = int((end[hold] <= on[hold]).sum()) if n else 0
    out['nonhold_with_tail'] = int((end[~hold] != on[~hold]).sum()) if n else 0
    out['out_of_song'] = int(((on < 0) | (end > duration_us+1_000_000)).sum()) if n else 0
    if mode == 'mania_4k':
        out.update(same_lane_overlap=p['same_lane_overlap'], hold_covers_head=p['hold_covers_head'], bad_lane=p['bad_lane'],
                   jack_lt_40ms=p['jack_lt_40ms'], jack_lt_100ms=p['jack_lt_100ms'], hold_release_lt_40ms=p['hold_release_lt_40ms'],
                   empty_lanes=4-len(set(lane.tolist())) if n else 4)
        gaps = np.concatenate([np.diff(np.sort(on[lane == k])) for k in range(4)]) if n else np.zeros(0)
        gaps = gaps[gaps > 0]
        out['min_jack_ms'] = float(gaps.min()/1000) if len(gaps) else None
    else:
        out.update(short_gap_lt_40ms=p['short_gap_lt_40ms'], short_gap_nondrag=p['short_gap_nondrag'], stack_gt4=p['stack_gt4'],
                   zero_length_hold=p['zero_length_hold'], onset_groups=p['onset_groups'])
        if n:  # B7: fingers busy at each onset group = simultaneous heads + live holds
            t, c = np.unique(on, return_counts=True)
            hs, he = np.sort(on[hold]), np.sort(end[hold])
            live = np.searchsorted(hs, t, 'left')-np.searchsorted(he, t, 'right')
            out['finger_load_gt4'] = int(((c+np.maximum(live, 0)) > 4).sum())
        else:
            out['finger_load_gt4'] = 0
    if n:
        t = np.sort(on)
        d = np.diff(t)
        out['short_drain'] = bool((t[-1]-t[0]-d[d >= 4_000_000].sum()) < 30_000_000)
    else:
        out['short_drain'] = True
    return out


HARD_RULES = dict(mania_4k=('same_lane_overlap', 'hold_covers_head', 'bad_lane', 'bad_hold_end', 'nonhold_with_tail', 'out_of_song', 'empty_lanes'),
                  phira=('zero_length_hold', 'bad_hold_end', 'nonhold_with_tail', 'out_of_song'))
RATE_RULES = dict(mania_4k=('jack_lt_40ms', 'jack_lt_100ms', 'hold_release_lt_40ms'),
                  phira=('short_gap_lt_40ms', 'short_gap_nondrag', 'stack_gt4', 'finger_load_gt4'))


# ------------------------------------------------------------------ M: music fidelity from mel onset strength
MEL_FRAME_US = 10_000
SILENCE_WINDOW_US = 50_000
LIFT_WINDOW_FRAMES = 3  # +-30 ms (B A1)
BEAT_SHIFTS = (-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8)
A2_CELL_US = 2_000_000


def find_mel(features_dir, audio_sha):
    root = Path(features_dir)
    for p in (root/audio_sha/'mel10_complete.npy', root/'features'/audio_sha/'mel10_complete.npy', root/f'{audio_sha}.npy'):
        if p.exists():
            return p
    return None


def audio_curves(mel):
    """Onset strength F (z-scored mean positive log-mel difference) with +-30 ms max, plus frame energy. mel [frames, bins], 10 ms."""
    mel = np.asarray(mel, np.float32)
    flux = np.maximum(np.diff(mel, axis=0, prepend=mel[:1]), 0).mean(1)
    flux = (flux-flux.mean())/max(float(flux.std()), 1e-6)
    star = np.max([np.roll(flux, k) for k in range(-LIFT_WINDOW_FRAMES, LIFT_WINDOW_FRAMES+1)], axis=0)
    return dict(flux=flux, star=star, energy=mel.mean(1).astype(np.float64))


def _mean_at(curve, t_us):
    idx = np.round((np.asarray(t_us, float)-MEL_FRAME_US/2)/MEL_FRAME_US).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < len(curve))]
    return float(curve[idx].mean()) if len(idx) else float('nan')


def onset_lift(star, onsets_us, segments=None):
    """A1: mean onset strength at the draft's onsets minus the same onsets moved by whole beats (grid-controlled).
    Without timing, falls back to multiples of 500 ms and says so (not grid-controlled)."""
    t = np.unique(np.asarray(onsets_us, np.int64))
    if len(t) < 2:
        return dict(lift=float('nan'), z=float('nan'), grid_controlled=bool(segments))
    real = _mean_at(star, t)
    if segments:
        clock = BeatClock(segments)
        beats = clock.to_beats(t)
        null = [_mean_at(star, clock.to_time(beats+k)) for k in BEAT_SHIFTS]
    else:
        null = [_mean_at(star, t+k*500_000) for k in BEAT_SHIFTS]
    null = np.array(null, float)
    return dict(lift=real-float(np.nanmean(null)), z=(real-float(np.nanmean(null)))/max(float(np.nanstd(null)), 1e-6),
                grid_controlled=bool(segments))


def _ranks(x):
    u, inv, c = np.unique(x, return_inverse=True, return_counts=True)
    cs = np.cumsum(c)
    return (cs-(c-1)/2.)[inv]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def density_energy(energy, onsets_us, duration_us):
    """A2: Spearman of note count and mean mel energy over 2 s cells."""
    cell = A2_CELL_US//MEL_FRAME_US
    cells = max(int(min(duration_us//A2_CELL_US, len(energy)//cell)), 0)
    if cells < 3:
        return float('nan')
    t = np.asarray(onsets_us, np.int64)
    counts = np.histogram(t, np.arange(cells+1)*A2_CELL_US)[0]
    return spearman(counts, energy[:cells*cell].reshape(cells, cell).mean(1))


def shifted_rolls(song_id, duration_us, n=4):
    """Audio roll offsets (frames) for the shifted-audio null: at least 2 s from zero, hash-chosen."""
    frames = max(int(duration_us//MEL_FRAME_US), 800)
    rng = rng_for('shifted-audio', song_id)
    return [int(x) for x in rng.integers(200, frames-200, n)]


def music_row(onset_sets, song, segments, features_dir, others):
    """A1 + A2 and both nulls per onset set. Returns None when the song has no mel."""
    path = find_mel(features_dir, song['audio_sha256'])
    if path is None:
        return None
    cur = audio_curves(np.load(path, mmap_mode='r', allow_pickle=False))
    other = []
    for sha in others:
        q = find_mel(features_dir, sha)
        if q is not None:
            other.append(audio_curves(np.load(q, mmap_mode='r', allow_pickle=False)))
    out = {}
    for name, on in onset_sets.items():
        base = onset_lift(cur['star'], on, segments)
        rolls = [onset_lift(np.roll(cur['star'], r), on, segments)['lift'] for r in shifted_rolls(song['song_id'], song['duration_us'])]
        wrong, wrong_a2 = [], []
        for c in other:
            star = np.resize(c['star'], len(cur['star']))
            wrong.append(onset_lift(star, on, segments)['lift'])
            wrong_a2.append(density_energy(np.resize(c['energy'], len(cur['energy'])), on, song['duration_us']))
        out[name] = dict(lift=base['lift'], lift_z=base['z'], grid_controlled=base['grid_controlled'],
                         shifted_audio=float(np.nanmean(rolls)), wrong_song=float(np.nanmean(wrong)) if wrong else None,
                         a2=density_energy(cur['energy'], on, song['duration_us']),
                         a2_wrong_song=float(np.nanmean(wrong_a2)) if wrong_a2 else None)
    return out


def near_silent(onset_us, energy, threshold):
    """requirement_diagnostics.near_silent (that module pulls torch)."""
    lo = max(0, math.ceil((onset_us-SILENCE_WINDOW_US-MEL_FRAME_US//2)/MEL_FRAME_US))
    hi = min(len(energy)-1, math.floor((onset_us+SILENCE_WINDOW_US-MEL_FRAME_US//2)/MEL_FRAME_US))
    return True if hi < lo else float(energy[lo:hi+1].max()) < threshold


def uncharted_regions(notes, duration_us, *, min_gap_us=4_000_000, margin_us=100_000):
    """requirement_diagnostics.uncharted_regions: >=4 s spans with no human note or hold body."""
    spans = sorted((int(n[0]), max(int(n[0]), int(n[1]))) for n in notes)
    regions, cursor, first = [], 0, True
    for onset, end in spans+[(int(duration_us), int(duration_us))]:
        if onset-cursor >= min_gap_us:
            lo = cursor if first else cursor+margin_us
            hi = onset if onset >= duration_us else onset-margin_us
            if hi > lo:
                regions.append((lo, hi))
        first = False
        cursor = max(cursor, end)
    return regions


def silence_rate(draft_onsets, ref_notes, duration_us, energy):
    threshold = float(np.percentile(energy, 2.)+1.)
    regions = uncharted_regions(ref_notes, duration_us)
    seconds = sum(b-a for a, b in regions)/1e6
    inside = [t for t in draft_onsets if any(a <= t < b for a, b in regions)]
    silent = sum(near_silent(int(t), energy, threshold) for t in inside)
    return dict(uncharted_s=seconds, true_silence_notes=int(silent), true_silence_events_per_s=silent/seconds if seconds else None)


# ------------------------------------------------------------------ H: pre-fit human reference + LOSO LDA
class HumanRef:
    """Fixed human sample for one mode: 'fit' sets the density-conditional z transform, mean and covariance;
    'eval' is what drafts are ranked against. Both are train-split only and never overlap by song_group."""

    def __init__(self, names, log_nps_fit, fit, log_nps_eval, evals):
        self.names = list(names)
        ln = np.asarray(log_nps_fit, float)
        q = np.quantile(ln, np.linspace(0, 1, H_PROTOCOL['z_bins']+1))
        self.table = []
        for j in range(len(self.names)):
            x = fit[:, j]
            sd = np.nanstd(x)
            cen, med, spread = [], [], []
            for b in range(H_PROTOCOL['z_bins']):
                m = (ln >= q[b]) & (ln <= q[b+1]) & ~np.isnan(x)
                if m.sum() < 5:
                    continue
                lo, hi = np.percentile(x[m], [25, 75])
                cen.append(ln[m].mean())
                med.append(np.median(x[m]))
                spread.append((hi-lo)/1.349)
            self.table.append((np.array(cen), np.array(med), np.maximum(spread, max(.25*sd, 1e-3))))
        self.fit_z = self.z(fit, ln)
        self.eval_z = self.z(evals, np.asarray(log_nps_eval, float))
        self.core = None

    def z(self, X, log_nps):
        X = np.atleast_2d(X)
        log_nps = np.atleast_1d(log_nps)
        out = np.zeros_like(X, dtype=float)
        for j, (c, m, s) in enumerate(self.table):
            out[:, j] = np.clip(np.nan_to_num((X[:, j]-np.interp(log_nps, c, m))/np.interp(log_nps, c, s)),
                                -H_PROTOCOL['z_clip'], H_PROTOCOL['z_clip'])
        return out

    def solver(self, idx):
        """(mean, inverse covariance, eval rows) on feature subset idx."""
        Z = self.fit_z[:, idx]
        S = np.atleast_2d(np.cov(Z.T))
        lam = H_PROTOCOL['shrink']
        return Z.mean(0), np.linalg.inv((1-lam)*S+lam*np.eye(len(idx))*np.trace(S)/len(idx)), self.eval_z[:, idx]


def loso_auc(Z, solver, counts=None):
    """AUC and per-song placement. Z [songs, features]; counts = bootstrap multiplicity per row (a left-out song
    leaves all of its copies out, so nothing leaks). Per-song placement values make it pairable between arms."""
    mu, A, E = solver
    n = len(Z)
    counts = np.ones(n, float) if counts is None else np.asarray(counts, float)
    active = counts > 0
    m = counts.sum()
    others = counts[active]
    if (m-others).min() <= 0:  # only one distinct song
        return float('nan'), np.full(n, np.nan)
    Za = Z[active]
    mean_wo = ((counts[:, None]*Z).sum(0)[None]-others[:, None]*Za)/(m-others)[:, None]
    W = (mean_wo-mu)@A
    self_score = (Za*W).sum(1)
    human = E@W.T
    u = np.full(n, np.nan)
    u[active] = ((human < self_score).mean(0)+.5*(human == self_score).mean(0))
    return float((others*u[active]).sum()/m), u


def dprime(auc):
    a = min(max(auc, .001), .999)
    return math.sqrt(2)*statistics.NormalDist().inv_cdf(a)


def h_from_auc(auc, auc_null):
    """1 at AUC .5, 0 at the null's AUC, clipped."""
    if auc is None or math.isnan(auc):
        return None
    return float(np.clip((auc_null-auc)/max(auc_null-.5, 1e-6), 0, 1))


def gap_score(gaps, rulers):
    """H_gap (B's H_fast): mean over tracked features of min(1, ruler/|gap|). 1 = all gaps within the human pair ruler."""
    return float(np.mean([min(1., r/max(abs(g), 1e-9)) for g, r in zip(gaps, rulers)]))


class Floor:
    def __init__(self, path):
        self.path = str(path)
        self.sha256 = sha256_file(path)
        with open(path) as f:
            self.data = json.load(f)
        self._ref = {}

    def ref(self, mode):
        if mode not in self._ref:
            info = self.data['H']['reference']
            with np.load(Path(self.path).parent/info['file'], allow_pickle=False) as z:
                self._ref[mode] = HumanRef(info['names'][mode], z[f'{mode}_fit_lognps'], z[f'{mode}_fit'], z[f'{mode}_eval_lognps'], z[f'{mode}_eval'])
        return self._ref[mode]

    def solver(self, mode, names):
        r = self.ref(mode)
        return r.solver([r.names.index(k) for k in names])

    def auc_null(self, mode):
        return float(self.data['H'][mode]['auc_null'])


# ------------------------------------------------------------------ scoring one song
def song_job(a):
    song, states, segs, tape, features_dir, others, need_rr = a
    mode, duration = song['mode'], int(song['duration_us'])
    refs_meta = song['references']
    refs = [load_chart(tape, r['target_path'], mode) for r in refs_meta]
    pid = primary_reference(song)['chart_id']
    p = next(i for i, r in enumerate(refs_meta) if r['chart_id'] == pid)
    ref_on = [np.array([n[0] for n in r], np.int64) for r in refs]
    # reference x reference CER (drop near-identical re-uploads like D)
    keep = list(range(len(refs)))
    rr = None
    if len(refs) >= 2 and need_rr:
        rr = np.zeros((len(refs), len(refs)))
        for i in range(len(refs)):
            for j in range(len(refs)):
                if i != j:
                    rr[i, j] = chart_edit_rate(refs[i], refs[j], mode)['cer']
        keep = []
        for i in range(len(refs)):
            if not any(min(rr[i, j], rr[j, i]) < .05 for j in keep):
                keep.append(i)
    per_seed = []
    for state in states:
        d = draft_tuples(song, state)
        on = np.array([n[0] for n in d], np.int64)
        curve = np.zeros((len(refs), len(SCAN_MS)))
        for i, r in enumerate(refs):
            for k, ms in enumerate(SCAN_MS):
                moved = [(a_+ms*1000, b_+ms*1000, t_, l_) for a_, b_, t_, l_ in d] if ms else d
                curve[i, k] = chart_edit_rate(moved, r, mode)['cer']
        c0 = SCAN_MS.index(0)
        exact, best = curve[:, c0], curve.min(1)
        shift = [SCAN_MS[int(np.argmin(curve[i]+1e-9*np.abs(np.array(SCAN_MS))))] for i in range(len(refs))]
        rho = [len(d)/max(len(r), 1) for r in refs]
        nearest = int(np.argmin([abs(math.log(max(x, 1e-6))) for x in rho]))
        loro_exact = None
        split_half = None
        if rr is not None and len(keep) >= 2:
            ex = []
            for j in keep:
                s = [i for i in keep if i != j]
                ex.append(float(exact[s].min()-rr[j, s].min()))
            loro_exact = float(np.mean(ex))
            if len(keep) >= 4:
                order = sorted(keep, key=lambda i: stable_int('half', refs_meta[i]['chart_id']))
                ha, hb = order[::2], order[1::2]
                split_half = [float(exact[hx].min()-np.mean([rr[j, hx].min() for j in hy])) for hx, hy in ((ha, hb), (hb, ha))]
        on_f, end_f, k_f, l_f = arrays(d)
        seg_arg = segs if segs else None
        x = features(on_f, end_f, k_f, l_f, mode, duration, seg_arg) if len(d) >= 2 else None
        per_seed.append(dict(
            notes=len(d), complete=bool(state.get('complete', True)), failure=state.get('failure'),
            legality=len(state.get('legality') or []), exact=exact.tolist(), best=best.tolist(), shift=shift,
            rho=rho, rho_closest=rho[nearest], nearest=nearest, loro_exact=loro_exact, split_half=split_half,
            f1_20_max=max(f1_at(on, h) for h in ref_on), f1_20_primary=f1_at(on, ref_on[p]),
            rules=rule_counts(d, mode, duration), features=x))
    r_on, r_end, r_k, r_l = arrays(refs[p])
    row = dict(song_id=song['song_id'], mode=mode, audio_sha256=song['audio_sha256'], duration_us=duration, K=len(refs), K_eff=len(keep),
               primary=pid, primary_index=p, seeds=per_seed, primary_features=features(r_on, r_end, r_k, r_l, mode, duration, segs or None))
    if features_dir:
        sets = {'ref': r_on.tolist()}
        for k, state in enumerate(states):
            sets[f'd{k}'] = [int(e['onset_us']) for e in state['events']]
        m = music_row(sets, song, segs or None, features_dir, others)
        if m is not None:
            row['music'] = m
            energy = audio_curves(np.load(find_mel(features_dir, song['audio_sha256']), mmap_mode='r', allow_pickle=False))['energy']
            row['silence'] = [silence_rate(sets[f'd{k}'], refs[p], duration, energy) for k in range(len(states))]
    return row


def run_jobs(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


# ------------------------------------------------------------------ aggregating
def _seed_mean(seeds, f):
    vals = [f(s) for s in seeds]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def song_metrics(row, wt):
    """Per-song, seed-averaged values that compare()/noise() bootstrap: (CER, null, human) triples, ratios, flags."""
    seeds, p = row['seeds'], row['primary_index']
    mode_t = wt['rho_table']
    def triple(clock, ref_idx):
        """W uses D's density-matched ruler and the central-bin null as constants; rho-conditioned values feed W_rho only."""
        cer, nu, hu, norm = [], [], [], []
        for s in seeds:
            i = s['nearest'] if ref_idx == 'nearest' else p
            b = rho_bin(s['rho'][i])
            c = s[clock][i]
            n = mode_t[clock]['null'][b]
            cer.append(c)
            nu.append(n)
            hu.append(mode_t[clock]['human'][b])
            lb = count_lower_bound(s['rho'][i])
            norm.append((c-lb)/max(n-lb, .1))
        return dict(cer=float(np.mean(cer)), null=mode_t[clock]['null'][CENTRAL_BIN], human=mode_t[clock]['human'][CENTRAL_BIN],
                    null_rho=float(np.mean(nu)), human_rho=float(np.mean(hu)), norm=float(np.mean(norm)))
    k = str(min(max(row['K_eff'], 1), 5))
    out = dict(song_id=row['song_id'], K=row['K'], K_eff=row['K_eff'], primary=row['primary'])
    for clock in ('exact', 'best'):
        out[f'single_{clock}'] = triple(clock, 'primary')
        out[f'nearest_{clock}'] = triple(clock, 'nearest')
        out[f'minref_{clock}'] = dict(cer=float(np.mean([min(s[clock]) for s in seeds])), null=wt['minref'][clock]['null'][k], human=wt['minref'][clock]['human'][k])
    out['rho_nearest'] = float(np.mean([s['rho'][s['nearest']] for s in seeds]))
    out['shift_ms_primary'] = float(np.mean([s['shift'][p] for s in seeds]))
    out['rho_primary'] = float(np.mean([s['rho'][p] for s in seeds]))
    out['rho_closest'] = float(np.mean([s['rho_closest'] for s in seeds]))
    out['notes'] = float(np.mean([s['notes'] for s in seeds]))
    out['f1_20_max'] = float(np.mean([s['f1_20_max'] for s in seeds]))
    out['f1_20_primary'] = float(np.mean([s['f1_20_primary'] for s in seeds]))
    out['loro_exact'] = _seed_mean(seeds, lambda s: s['loro_exact'])
    sh = [s['split_half'] for s in seeds if s['split_half']]
    out['split_half'] = np.mean(sh, 0).tolist() if sh else None
    out['complete'] = all(s['complete'] for s in seeds)
    out['failures'] = sum(s['failure'] is not None for s in seeds)
    out['legality'] = sum(s['legality'] for s in seeds)
    if 'music' in row:
        m = row['music']
        d = [m[f'd{i}'] for i in range(len(seeds))]
        out['music'] = dict(
            lift=float(np.nanmean([x['lift'] for x in d])), shifted_audio=float(np.nanmean([x['shifted_audio'] for x in d])),
            wrong_song=_seed_mean(d, lambda x: x['wrong_song']), a2=float(np.nanmean([x['a2'] for x in d])),
            a2_wrong_song=_seed_mean(d, lambda x: x['a2_wrong_song']), ref_lift=m['ref']['lift'], ref_a2=m['ref']['a2'],
            ref_shifted_audio=m['ref']['shifted_audio'], ref_wrong_song=m['ref']['wrong_song'], grid_controlled=m['ref']['grid_controlled'])
        out['true_silence_per_s'] = _seed_mean(row['silence'], lambda x: x['true_silence_events_per_s'])
    return out


def style_table(rows, names, rulers, ref):
    """Per-feature gap versus human spread on the same songs (draft minus primary reference)."""
    table = {}
    for k in names:
        j = ref.names.index(k)
        M = np.array([[tf(k, s['features'][k]) if s['features'] and s['features'].get(k) is not None else np.nan for s in r['seeds']] for r in rows], float)
        H = np.array([tf(k, r['primary_features'][k]) if r['primary_features'] and r['primary_features'].get(k) is not None else np.nan for r in rows], float)
        with np.errstate(all='ignore'):
            m = np.array([np.nanmean(x) if (~np.isnan(x)).any() else np.nan for x in M])
        d = m-H
        ok = ~np.isnan(d)
        z_model = np.array([np.nanmean([ref.z(_vec(s['features'], ref.names), s['features']['log_nps'])[0, j] for s in r['seeds'] if s['features']] or [np.nan]) for r in rows])
        z_human = np.array([ref.z(_vec(r['primary_features'], ref.names), r['primary_features']['log_nps'])[0, j] if r['primary_features'] else np.nan for r in rows])
        gap = boot_mean(d[ok], seed=stable_int('gap', k) % 2**31) if ok.any() else dict(mean=None, ci80=None)
        sd_seed = float(np.nanmean(np.nanstd(M[ok], 1))) if M.shape[1] > 1 else None
        if not ok.any():  # e.g. no feature value in any draft of a tiny panel
            table[k] = dict(model=None, human=None, gap=None, gap_ci80=None, z_gap=None, z_model=None, ruler=rulers.get(k), gap_over_ruler=None, sdr=None, sdr_z=None, sd_across_seeds=None)
            continue
        table[k] = dict(model=float(np.nanmean(m)), human=float(np.nanmean(H)), gap=gap['mean'], gap_ci80=gap['ci80'],
                        z_gap=float(np.nanmean(z_model-z_human)), z_model=float(np.nanmean(z_model)), ruler=rulers.get(k),
                        gap_over_ruler=abs(gap['mean'])/rulers[k] if rulers.get(k) and gap['mean'] is not None else None,
                        sdr=float(np.nanstd(m)/max(np.nanstd(H), 1e-9)), sdr_z=float(np.nanstd(z_model)/max(np.nanstd(z_human), 1e-9)),
                        sd_across_seeds=sd_seed)
    return table


def _vec(f, names):
    return np.array([tf(k, f.get(k, np.nan)) if f.get(k) is not None else np.nan for k in names], float)


def score_panel(songs, drafts_by_seed, *, tape, timing, floor, features_dir=None, workers=3, label='', cost=None, source=None, silence_rows=None):
    jobs = []
    by_mode = defaultdict(list)
    for s in songs:
        by_mode[s['mode']].append(s)
    for s in songs:
        pid = primary_reference(s)['chart_id']
        peers = [x['audio_sha256'] for x in by_mode[s['mode']] if x['song_id'] != s['song_id']]
        peers = sorted(peers, key=lambda x: stable_int('wrong', s['song_id'], x))
        if features_dir:
            peers = [x for x in peers if find_mel(features_dir, x) is not None]
        peers = peers[:3]
        states = [d[s['song_id']] for d in drafts_by_seed]
        jobs.append((s, states, timing.get(pid), tape, features_dir, peers, True))
    rows = run_jobs(song_job, jobs, workers)
    out = dict(schema=SCHEMA, label=label, source=source or {}, cost=cost, floor=dict(path=floor.path, sha256=floor.sha256),
               songs_sha256=hashlib.sha256(json.dumps(sorted(s['song_id'] for s in songs)).encode()).hexdigest(),
               h_protocol=H_PROTOCOL, modes={})
    for mode in sorted(by_mode):
        R = [r for r in rows if r['mode'] == mode]
        out['modes'][mode] = score_mode(R, mode, floor, silence_rows)
    return out


def _w_block(M, key):
    tr = lambda f: np.array([m[key][f] for m in M], float)
    bl = boot_ratio(tr('cer'), tr('null'), tr('human'), seed=stable_int(key) % 2**31)
    bl['cer'] = boot_mean(tr('cer'), seed=stable_int(key, 'c') % 2**31)
    bl['null'] = float(tr('null').mean())
    bl['human'] = float(tr('human').mean())
    if 'norm' in M[0][key]:
        bl['cer_norm'] = boot_mean(tr('norm'), seed=stable_int(key, 'n') % 2**31)
        bl['null_rho'] = float(tr('null_rho').mean())
        bl['human_rho'] = float(tr('human_rho').mean())
        bl['W_rho_raw'] = ratio_score(tr('cer'), tr('null_rho'), tr('human_rho'))
    return bl


def score_mode(R, mode, floor, silence_rows=None):
    wt = floor.data['W'][mode]
    M = [song_metrics(r, wt) for r in R]
    w = {}
    for clock in ('exact', 'best'):
        w[f'single_{clock}'] = _w_block(M, f'single_{clock}')
        w[f'nearest_{clock}'] = _w_block(M, f'nearest_{clock}')
        w[f'minref_{clock}'] = _w_block(M, f'minref_{clock}')
        by_k = {}
        for k in sorted({m['K_eff'] for m in M}):
            sub = [m for m in M if m['K_eff'] == k]
            by_k[str(k)] = dict(n=len(sub), cer=float(np.mean([m[f'minref_{clock}']['cer'] for m in sub])), human=sub[0][f'minref_{clock}']['human'], null=sub[0][f'minref_{clock}']['null'])
        w[f'minref_{clock}']['by_K'] = by_k
        dm = [m for m in M if rho_bin(m['rho_primary']) == CENTRAL_BIN]
        w[f'single_{clock}']['density_matched_subset'] = dict(n=len(dm), cer=float(np.mean([m[f'single_{clock}']['cer'] for m in dm])) if dm else None)
    lo = [m['loro_exact'] for m in M if m['loro_exact'] is not None]
    w['loro_excess_exact'] = boot_mean(lo, seed=11)
    sh = [m['split_half'] for m in M if m['split_half']]
    w['split_half_reliability'] = dict(n=len(sh), spearman=spearman([x[0] for x in sh], [x[1] for x in sh]) if len(sh) >= 5 else None)
    w['median_shift_ms'] = float(np.median(np.abs([m['shift_ms_primary'] for m in M])))
    w['rho_primary_median'] = float(np.median([m['rho_primary'] for m in M]))
    w['rho_nearest_median'] = float(np.median([m['rho_nearest'] for m in M]))
    # H
    ref = floor.ref(mode)
    core = core_features(mode)
    solver = floor.solver(mode, core)
    nseed = len(R[0]['seeds'])
    Z_all = np.full((nseed, len(R), len(ref.names)), np.nan)
    for i, r in enumerate(R):
        for k, s in enumerate(r['seeds']):
            if s['features']:
                Z_all[k, i] = ref.z(_vec(s['features'], ref.names), s['features']['log_nps'])[0]
    Z_all = np.nan_to_num(Z_all)  # a draft with <2 notes sits at the human median of nothing: complete/note-ratio catch it
    Z_ref = np.array([ref.z(_vec(r['primary_features'], ref.names), r['primary_features']['log_nps'])[0] for r in R])
    idx = [ref.names.index(k) for k in core]
    u = np.array([loso_auc(Z_all[k][:, idx], solver)[1] for k in range(nseed)])
    auc = float(u.mean(1).mean())
    boots = []
    rng = np.random.default_rng(stable_int('hboot', mode) % 2**31)
    for _ in range(400):
        counts = np.bincount(rng.integers(0, len(R), len(R)), minlength=len(R))
        v = [loso_auc(Z_all[k][:, idx], solver, counts)[0] for k in range(nseed)]
        boots.append(np.mean(v))
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    auc0 = floor.auc_null(mode)
    all_names = [k for k in ref.names]
    all_solver = floor.solver(mode, all_names)
    auc_all = float(np.mean([loso_auc(Z_all[k], all_solver)[0] for k in range(nseed)]))
    rulers = floor.data['H'][mode]['rulers']
    style = style_table(R, core, rulers, ref)
    track = STYLE_TRACK[mode]
    gaps = [style[k]['gap'] for k in track]
    h = dict(auc=auc, auc_ci80=ci80(boots), auc_sd=float(boots.std()), H=h_from_auc(auc, auc0), H_ci80=[h_from_auc(ci80(boots)[1], auc0), h_from_auc(ci80(boots)[0], auc0)],
             auc_null=auc0, dprime=dprime(auc), auc_all_features_diagnostic=auc_all, human_control_auc=loso_auc(Z_ref[:, idx], solver)[0],
             H_gap=gap_score(gaps, [rulers[k] for k in track]), n_songs=len(R), seeds=nseed, style_track=track)
    # constraints
    cons = constraints_block(R, M, mode, floor, silence_rows)
    # M
    music = music_block(M)
    # per-song rows for compare()/noise()
    for i, m in enumerate(M):
        m['h_u'] = float(u[:, i].mean())
        m['style'] = {k: float(np.nanmean([tf(k, s['features'][k]) if s['features'] and s['features'].get(k) is not None else np.nan for s in R[i]['seeds']])) for k in track}
        m['style_ref'] = {k: float(tf(k, R[i]['primary_features'][k])) for k in track}
        m['z'] = Z_all[:, i, :].round(4).tolist()
    return dict(n_songs=len(R), W=w, H=h, style_gap=style, constraints=cons, M=music, songs=M, feature_names=ref.names)


def constraints_block(R, M, mode, floor, silence_rows):
    hu = floor.data['rules'][mode]
    totals = defaultdict(float)
    notes = sum(s['notes'] for r in R for s in r['seeds'])
    nseed = len(R[0]['seeds'])
    for r in R:
        for s in r['seeds']:
            for k, v in s['rules'].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    totals[k] += float(v)
    hard = {k: int(totals[k]) for k in HARD_RULES[mode]}
    rates = {k: dict(count=int(totals[k]), per_1k_notes=1000*totals[k]/max(notes, 1), human_per_1k=hu['per_1k'].get(k)) for k in RATE_RULES[mode]}
    rate_ok = {k: v['per_1k_notes'] <= 2*(v['human_per_1k'] or 0)+1e-12 for k, v in rates.items()}
    ratio = float(np.median([m['rho_closest'] for m in M]))
    songs_in = float(np.mean([NOTE_RATIO_SONG_BAND[0] <= m['rho_closest'] <= NOTE_RATIO_SONG_BAND[1] for m in M]))
    silence = None
    if any('true_silence_per_s' in m for m in M):
        v = [m['true_silence_per_s'] for m in M if m.get('true_silence_per_s') is not None]
        silence = dict(true_silence_per_s=float(np.mean(v)) if v else None, n=len(v), source='mel energy (--features-dir)')
    elif silence_rows:
        v = [silence_rows[m['song_id']] for m in M if silence_rows.get(m['song_id']) is not None]
        silence = dict(true_silence_per_s=float(np.mean(v)) if v else None, n=len(v), source="drafts' own workflow-benchmark.json readout (min-CER reference)")
    short = sum(bool(s['rules']['short_drain']) for r in R for s in r['seeds'])/nseed
    out = dict(complete=float(np.mean([m['complete'] for m in M])), failures=int(sum(m['failures'] for m in M)),
               legality_events=int(sum(m['legality'] for m in M)), hard_rule_counts=hard, rate_rules=rates, rate_ok=rate_ok,
               note_ratio_median=ratio, note_ratio_band=list(NOTE_RATIO_BAND), songs_in_wide_band=songs_in, f1_20_max=boot_mean([m['f1_20_max'] for m in M], seed=3),
               f1_20_primary=boot_mean([m['f1_20_primary'] for m in M], seed=4), short_drain_songs=short, silence=silence)
    fails = []
    if out['complete'] < 1:
        fails.append('complete')
    if out['failures']:
        fails.append('failures')
    if out['legality_events']:
        fails.append('legality')
    fails += [f'{k}={v}' for k, v in hard.items() if v]
    fails += [f'{k} rate' for k, ok in rate_ok.items() if not ok]
    if not NOTE_RATIO_BAND[0] <= ratio <= NOTE_RATIO_BAND[1]:
        fails.append('note_ratio')
    out['failed'] = fails
    out['pass'] = not fails
    return out


def music_block(M):
    S = [m['music'] for m in M if 'music' in m]
    if not S:
        return dict(available=False, coverage=0, reason='no mel10_complete.npy for this panel; give --features-dir')
    a = lambda k: np.array([x[k] if x[k] is not None else np.nan for x in S], float)
    lift, sh, wr, ref = a('lift'), a('shifted_audio'), a('wrong_song'), a('ref_lift')
    def dci(x):
        return boot_mean(x, seed=13)
    d_shift, d_wrong = dci(lift-sh), dci(lift-wr)
    admitted = d_shift['ci80'][0] > 0 and bool(d_wrong['n']) and d_wrong['ci80'][0] > 0
    null = np.nanmean([sh, wr], axis=0)
    m_a1 = float((np.nanmean(lift)-np.nanmean(null))/max(np.nanmean(ref)-np.nanmean(null), 1e-6))
    a2, a2h, a2w = a('a2'), a('ref_a2'), a('a2_wrong_song')
    m_a2 = float(np.clip(1-np.nanmean(np.abs(a2-a2h))/max(np.nanmean(np.abs(a2h-a2w)), 1e-6), 0, 1))
    return dict(available=True, coverage=len(S), of=len(M), grid_controlled=all(x['grid_controlled'] for x in S), lift=dci(lift), ref_lift=dci(ref),
                shifted_audio=dci(sh), wrong_song=dci(wr), minus_shifted=d_shift, minus_wrong_song=d_wrong, admitted_vs_nulls=admitted,
                M_A1_raw=m_a1, M=float(np.clip(m_a1, 0, 1)) if admitted else None, a2=dci(a2), ref_a2=dci(a2h), a2_wrong_song=dci(a2w), M_A2_diagnostic=m_a2)


# ------------------------------------------------------------------ text table
def _f(x, d=3):
    return '  -  ' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'


def _ci(c, d=3):
    return '[ - ]' if not c else f'[{c[0]:.{d}f},{c[1]:.{d}f}]'


def text_table(doc):
    lines = [f"benchmark v3 | {doc['label']} | floor {doc['floor']['sha256'][:10]}"]
    for mode, m in doc['modes'].items():
        w, h, c, mm = m['W'], m['H'], m['constraints'], m['M']
        lines.append(f"== {mode} n={m['n_songs']} seeds={h['seeds']} rho median: primary {w['rho_primary_median']:.2f}, nearest {w['rho_nearest_median']:.2f}")
        lines.append('   W                    CER [80%]              CERnorm  null  human  W     raw [80%]             W_rho')
        for key, name in (('nearest_exact', 'nearest-dens exact *'), ('nearest_best', 'nearest-dens best'), ('single_exact', 'primary exact'), ('single_best', 'primary best'),
                          ('minref_exact', 'min-ref exact'), ('minref_best', 'min-ref best')):
            b = w[key]
            norm = b.get('cer_norm', {}).get('mean')
            lines.append(f"   {name:20s} {_f(b['cer']['mean'])} {_ci(b['cer']['ci80'])}  {_f(norm)}   {_f(b['null'])} {_f(b['human'])}  "
                         f"{_f(b['W'])} raw {_f(b['W_raw'])} {_ci(b['ci80'])}  {_f(b.get('W_rho_raw'))}")
        rel = w['split_half_reliability']
        lines.append(f"   LORO excess exact {_f(w['loro_excess_exact']['mean'])} {_ci(w['loro_excess_exact']['ci80'])} n={w['loro_excess_exact']['n']}; "
                     f"split-half Spearman {_f(rel['spearman'], 2)} n={rel['n']}; min-ref K={ {k: v['n'] for k, v in w['minref_exact']['by_K'].items()} }")
        lines.append(f"   H {_f(h['H'])} {_ci(h['H_ci80'])} | AUC {_f(h['auc'])} {_ci(h['auc_ci80'])} null {_f(h['auc_null'])} d' {_f(h['dprime'], 2)} | "
                     f"H_gap {_f(h['H_gap'])} | human control AUC {_f(h['human_control_auc'])} | all-feature AUC {_f(h['auc_all_features_diagnostic'])}")
        lines.append('   style            model  human  gap     z-gap  /ruler  SDR')
        for k in h['style_track']:
            s = m['style_gap'].get(k) or {}
            lines.append(f"   {k:16s} {_f(s.get('model'))}  {_f(s.get('human'))}  {_f(s.get('gap'))}  {_f(s.get('z_gap'), 2)}  {_f(s.get('gap_over_ruler'), 1)}  {_f(s.get('sdr'), 2)}")
        if mm['available']:
            lines.append(f"   M {_f(mm['M'])} coverage {mm['coverage']}/{mm['of']} | lift {_f(mm['lift']['mean'])} ref {_f(mm['ref_lift']['mean'])} shifted-audio {_f(mm['shifted_audio']['mean'])} "
                         f"wrong-song {_f(mm['wrong_song']['mean'])} | admitted {mm['admitted_vs_nulls']}")
        else:
            lines.append(f"   M unavailable: {mm['reason']}")
        sil = c['silence']
        lines.append(f"   constraints {'PASS' if c['pass'] else 'FAIL '+', '.join(c['failed'])} | complete {c['complete']:.2f} note ratio {c['note_ratio_median']:.2f} "
                     f"F1@20 {_f(c['f1_20_max']['mean'])} | silence {_f(sil['true_silence_per_s']) if sil else 'n/a'}")
    return '\n'.join(lines)


# ------------------------------------------------------------------ bootstrapped axes
class Side:
    """A scored JSON for one mode as arrays, ordered by song id, for weighted (bootstrap) evaluation."""

    def __init__(self, doc, mode):
        self.doc = doc
        self.mode = mode
        self.m = doc['modes'][mode]
        self.songs = sorted(self.m['songs'], key=lambda s: s['song_id'])
        self.ids = [s['song_id'] for s in self.songs]
        self.Z = np.array([s['z'] for s in self.songs], float).transpose(1, 0, 2)  # seeds, songs, features
        self.w = {k: {f: np.array([s[k][f] for s in self.songs], float) for f in ('cer', 'null', 'human')}
                  for k in ('single_exact', 'single_best', 'nearest_exact', 'nearest_best', 'minref_exact', 'minref_best')}
        self.norm = np.array([s['nearest_exact']['norm'] for s in self.songs], float)
        self.f1 = np.array([s['f1_20_max'] for s in self.songs], float)
        self.loro = np.array([np.nan if s['loro_exact'] is None else s['loro_exact'] for s in self.songs], float)
        self.track = self.m['H']['style_track']
        self.style = np.array([[s['style'][k] for k in self.track] for s in self.songs], float)
        self.style_ref = np.array([[s['style_ref'][k] for k in self.track] for s in self.songs], float)
        music = [s.get('music') or dict.fromkeys(('lift', 'ref_lift', 'shifted_audio', 'wrong_song')) for s in self.songs]
        self.music = {k: np.array([np.nan if x[k] is None else x[k] for x in music], float) for k in ('lift', 'ref_lift', 'shifted_audio', 'wrong_song')}
        self.music_ok = ~np.isnan(self.music['lift'])  # songs without a mel are left out of M on both arms
        self.has_music = bool(self.music_ok.any())


def eval_axes(side, floor, counts):
    c = np.asarray(counts, float)
    mean = lambda x: float((c*x).sum()/c.sum())
    out = {}
    for k, v in side.w.items():
        n, cer, hu = mean(v['null']), mean(v['cer']), mean(v['human'])
        raw = (n-cer)/max(n-hu, 1e-9)
        out[f'W_{k}'] = float(np.clip(raw, 0, 1))
        out[f'Wraw_{k}'] = float(raw)
        out[f'CER_{k}'] = cer
    out['CER_norm_nearest_exact'] = mean(side.norm)
    out['F1_20'] = mean(side.f1)
    ok = ~np.isnan(side.loro)
    out['LORO_excess'] = float((c[ok]*side.loro[ok]).sum()/c[ok].sum()) if c[ok].sum() else float('nan')
    ref = floor.ref(side.mode)
    core = core_features(side.mode)
    idx = [ref.names.index(k) for k in core]
    solver = floor.solver(side.mode, core) if not hasattr(side, '_solver') else side._solver
    side._solver = solver
    auc = float(np.mean([loso_auc(Z[:, idx], solver, c)[0] for Z in side.Z]))
    out['AUC'] = auc
    out['H'] = h_from_auc(auc, floor.auc_null(side.mode))
    rulers = floor.data['H'][side.mode]['rulers']
    gaps = ((side.style-side.style_ref)*c[:, None]).sum(0)/c.sum()
    out['H_gap'] = gap_score(gaps, [rulers[k] for k in side.track])
    if side.has_music and c[side.music_ok].sum():
        mm = lambda k: float(np.nansum(c[side.music_ok]*side.music[k][side.music_ok])/c[side.music_ok].sum())
        null = (mm('shifted_audio')+mm('wrong_song'))/2
        out['M'] = float(np.clip((mm('lift')-null)/max(mm('ref_lift')-null, 1e-6), 0, 1))
    return out


HIGHER_BETTER = {'Wraw_single_exact': 1, 'Wraw_single_best': 1, 'Wraw_nearest_exact': 1, 'Wraw_nearest_best': 1, 'Wraw_minref_exact': 1, 'Wraw_minref_best': 1, 'W_single_exact': 1, 'W_single_best': 1, 'W_nearest_exact': 1, 'W_nearest_best': 1, 'W_minref_exact': 1, 'W_minref_best': 1, 'H': 1, 'H_gap': 1, 'M': 1, 'F1_20': 1,
                 'AUC': -1, 'LORO_excess': -1, 'CER_norm_nearest_exact': -1, 'CER_single_exact': -1, 'CER_single_best': -1, 'CER_minref_exact': -1, 'CER_minref_best': -1, 'CER_nearest_exact': -1, 'CER_nearest_best': -1}
OBJECTIVE_AXES = ('W_nearest_exact', 'H', 'M')  # PLAN objectives; cost joins when given


def paired_axes(a, b, floor, samples=500, seed=0, subset=None):
    """Point differences and 80% paired CI (A - B) with songs resampled jointly and the classifier refit per resample."""
    n = len(a.ids)
    pool = np.arange(n) if subset is None else np.asarray(subset)
    base = np.zeros(n)
    np.add.at(base, pool, 1)
    pa, pb = eval_axes(a, floor, base), eval_axes(b, floor, base)
    rng = np.random.default_rng(seed)
    boots = defaultdict(list)
    for _ in range(samples):
        counts = np.bincount(pool[rng.integers(0, len(pool), len(pool))], minlength=n)
        ea, eb = eval_axes(a, floor, counts), eval_axes(b, floor, counts)
        for k in ea:
            if k in eb and ea[k] is not None and eb[k] is not None and not (math.isnan(ea[k]) or math.isnan(eb[k])):
                boots[k].append(ea[k]-eb[k])
    out = {}
    for k in pa:
        if k not in pb or k not in boots or pa[k] is None or pb[k] is None:
            continue
        out[k] = dict(a=pa[k], b=pb[k], diff=pa[k]-pb[k], ci80=ci80(boots[k]), n=int(len(pool)))
    return out


def axis_call(row, sign=1):
    """'A', 'B' or 'tie' under PLAN: beats only if the 80% paired CI excludes 0 and the difference is at least epsilon."""
    d, (lo, hi) = row['diff']*sign, sorted(x*sign for x in row['ci80'])
    if abs(d) < EPSILON:
        return 'tie'
    if lo > 0:
        return 'A'
    if hi < 0:
        return 'B'
    return 'tie'


def split_half_order(sa, sb):
    """B E8: on K>=4 songs, does the A-B difference (min-ref excess, exact clock) survive a change of references?"""
    ha = [s['split_half'] for s in sa.songs]
    hb = [s['split_half'] for s in sb.songs]
    d = np.array([[x[0]-y[0], x[1]-y[1]] for x, y in zip(ha, hb) if x and y], float)
    if len(d) < 5:
        return dict(n=len(d), halves_same_sign=None, spearman=None)
    return dict(n=len(d), diff_half0=float(d[:, 0].mean()), diff_half1=float(d[:, 1].mean()), halves_same_sign=bool(d[:, 0].mean()*d[:, 1].mean() > 0), spearman=spearman(d[:, 0], d[:, 1]))


def compare_docs(a, b, floor, *, cost_a=None, cost_b=None, axes=OBJECTIVE_AXES, samples=500):
    if a['songs_sha256'] != b['songs_sha256']:
        raise ValueError('different song sets: v3 only compares arms on the same panel (same songs, same reference counts)')
    for doc in (a, b):
        if doc['floor']['sha256'] != floor.sha256:
            raise ValueError(f"{doc['label']} was scored with another floor; rescore it")
    cost_a = a.get('cost') if cost_a is None else cost_a
    cost_b = b.get('cost') if cost_b is None else cost_b
    report = dict(a=a['label'], b=b['label'], modes={}, rules=dict(epsilon=EPSILON, cost_tie=COST_TIE, ci='80% paired song bootstrap, classifier refit per resample'))
    wins = {'A': 0, 'B': 0}
    calls = []
    for mode in sorted(a['modes']):
        sa, sb = Side(a, mode), Side(b, mode)
        rows = paired_axes(sa, sb, floor, samples=samples, seed=stable_int('cmp', mode) % 2**31)
        for k, r in rows.items():
            r['call'] = axis_call(r, HIGHER_BETTER.get(k, 1))
        mc = {k: rows[k]['call'] for k in axes if k in rows}
        f1 = rows['F1_20']
        constraint = dict(a_pass=a['modes'][mode]['constraints']['pass'], a_failed=a['modes'][mode]['constraints']['failed'],
                          b_pass=b['modes'][mode]['constraints']['pass'], b_failed=b['modes'][mode]['constraints']['failed'],
                          a_f1_no_harm=f1['ci80'][0] >= -EPSILON, b_f1_no_harm=-f1['ci80'][1] >= -EPSILON)
        ca, cb = a['modes'][mode]['constraints']['rate_rules'], b['modes'][mode]['constraints']['rate_rules']
        constraint['a_playability_not_worse'] = all(ca[k]['per_1k_notes'] <= cb[k]['per_1k_notes']*1.05+1e-9 for k in ca)
        constraint['b_playability_not_worse'] = all(cb[k]['per_1k_notes'] <= ca[k]['per_1k_notes']*1.05+1e-9 for k in cb)
        report['modes'][mode] = dict(axes=rows, objective_calls=mc, constraints=constraint, split_half=split_half_order(sa, sb))
        calls += list(mc.values())
    if cost_a is not None and cost_b is not None:
        if min(cost_a, cost_b) >= (1-COST_TIE)*max(cost_a, cost_b):
            cost_call = 'tie'
        else:
            cost_call = 'A' if cost_a < cost_b else 'B'
        report['cost'] = dict(a=cost_a, b=cost_b, call=cost_call)
        calls.append(cost_call)
    wins = {x: calls.count(x) for x in 'AB'}
    if wins['A'] and not wins['B']:
        verdict = 'A dominates B'
    elif wins['B'] and not wins['A']:
        verdict = 'B dominates A'
    elif wins['A'] and wins['B']:
        verdict = 'no dominance (trade-off)'
    else:
        verdict = 'tie (no cost given or equal cost: keep the incumbent)'
    report['verdict'] = verdict
    from scripts.pareto_admission import qualify_comparison
    return qualify_comparison(report, objectives=axes)


def compare_text(rep):
    lines = [f"A = {rep['a']}  |  B = {rep['b']}  |  diff = A-B, 80% paired CI, song bootstrap"]
    show = ('W_nearest_exact', 'W_nearest_best', 'W_single_exact', 'W_single_best', 'Wraw_nearest_exact', 'W_minref_exact', 'W_minref_best', 'CER_nearest_exact', 'CER_nearest_best', 'CER_single_exact', 'CER_norm_nearest_exact',
            'LORO_excess', 'H', 'AUC', 'H_gap', 'M', 'F1_20')
    for mode, m in rep['modes'].items():
        lines.append(f'== {mode}')
        for k in show:
            if k in m['axes']:
                r = m['axes'][k]
                star = ' <objective>' if k in OBJECTIVE_AXES else ''
                lines.append(f"  {k:22s} A {r['a']:.3f}  B {r['b']:.3f}  diff {r['diff']:+.3f} [{r['ci80'][0]:+.3f},{r['ci80'][1]:+.3f}]  -> {r['call']}{star}")
        c = m['constraints']
        lines.append(f"  constraints A {'pass' if c['a_pass'] else 'FAIL '+','.join(c['a_failed'])}; B {'pass' if c['b_pass'] else 'FAIL '+','.join(c['b_failed'])}; "
                     f"F1@20 no-harm A over B {c['a_f1_no_harm']}, B over A {c['b_f1_no_harm']}; playability not worse A {c['a_playability_not_worse']}, B {c['b_playability_not_worse']}")
        sh = m['split_half']
        if sh['n'] >= 5:
            lines.append(f"  reference split (K>=4, n={sh['n']}): A-B excess {sh['diff_half0']:+.3f} / {sh['diff_half1']:+.3f}, same sign {sh['halves_same_sign']}, per-song Spearman {_f(sh['spearman'], 2)}")
    if 'cost' in rep:
        lines.append(f"cost A {rep['cost']['a']} B {rep['cost']['b']} -> {rep['cost']['call']}")
    lines.append(f"VERDICT: {rep['verdict']}")
    return '\n'.join(lines)


AA_AXES = ('W_nearest_exact', 'W_nearest_best', 'H', 'AUC', 'H_gap')


def noise_report(doc, floor, sizes, other=None, reps=300, aa_docs=None, inner=150, aa_reps=60):
    """Refit-bootstrap SD and 80% half-width of each axis at a panel size per mode; A/A false alarms if a same-arm pair is given."""
    out = {}
    axes = ('W_nearest_exact', 'W_nearest_best', 'W_single_exact', 'W_minref_exact', 'H', 'AUC', 'H_gap', 'CER_nearest_exact')
    for mode in sorted(doc['modes']):
        side = Side(doc, mode)
        n = len(side.ids)
        out[mode] = {}
        os_ = Side(other, mode) if other else None
        for size in sizes:
            rng = np.random.default_rng(stable_int('noise', mode, size) % 2**31)
            vals, diffs = defaultdict(list), defaultdict(list)
            for _ in range(reps):
                counts = np.bincount(rng.integers(0, n, size), minlength=n)
                e = eval_axes(side, floor, counts)
                for k in axes:
                    vals[k].append(e[k])
                if os_ is not None:
                    e2 = eval_axes(os_, floor, counts)
                    for k in axes:
                        diffs[k].append(e[k]-e2[k])
            row = {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), halfwidth80=float(np.nanpercentile(v, 90)-np.nanpercentile(v, 10))/2) for k, v in vals.items()}
            if diffs:
                row['paired_diff'] = {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), halfwidth80=float(np.nanpercentile(v, 90)-np.nanpercentile(v, 10))/2) for k, v in diffs.items()}
            out[mode][str(size)] = row
        if aa_docs:
            sa, sb = Side(aa_docs[0], mode), Side(aa_docs[1], mode)
            aa = {}
            for size in sizes:
                rng = np.random.default_rng(stable_int('aa', mode, size) % 2**31)
                alarms = defaultdict(int)
                for r in range(aa_reps):
                    sub = rng.choice(n, min(size, n), replace=False)
                    rows = paired_axes(sa, sb, floor, samples=inner, seed=int(rng.integers(2**31)), subset=sub)
                    for k in AA_AXES:
                        if k in rows and axis_call(rows[k], HIGHER_BETTER[k]) != 'tie':
                            alarms[k] += 1
                aa[str(size)] = {k: alarms[k]/aa_reps for k in AA_AXES}
            out[mode]['aa_false_alarm'] = aa
    return out


# ------------------------------------------------------------------ corruptions (admission) and nulls
def _renote(on, end, kind, lane, mode, order=None):
    out = [(int(a), int(max(a, b)), TYPES[int(k)], int(l) if mode == 'mania_4k' else -1) for a, b, k, l in zip(on, end, kind, lane)]
    return sorted(out)


def corrupt(notes, name, dose, mode, seed, duration_us=None):
    """Return a corrupted copy. Doses: ms for jitter/shift, probability for lane/type, fraction for drop/insert, seconds for bars/loop."""
    rng = rng_for(name, dose, seed)
    on, end, kind, lane = arrays(notes)
    n = len(on)
    if n == 0 or name == 'control':
        return list(notes)
    dur = end-on
    duration_us = int(duration_us or (end.max()+1))
    if name == 'shift':
        return _renote(on+int(dose*1000), end+int(dose*1000), kind, lane, mode)
    if name == 'jitter':
        d = rng.integers(-int(dose*1000), int(dose*1000)+1, n)
        return _renote(on+d, on+d+dur, kind, lane, mode)
    if name == 'lane_shuffle':
        m = rng.random(n) < dose
        lane = lane.copy()
        lane[m] = rng.integers(0, 4, int(m.sum()))
        return _renote(on, end, kind, lane, mode)
    if name == 'lane_mirror':
        return _renote(on, end, kind, 3-lane if mode == 'mania_4k' else lane, mode)
    if name == 'jack_inject':
        o = np.argsort(on, kind='stable')
        lane = lane.copy()
        for a, b in zip(o[:-1], o[1:]):
            if on[b] > on[a] and rng.random() < dose:
                lane[b] = lane[a]
        return _renote(on, end, kind, lane, mode)
    if name == 'type_shuffle':  # types redrawn from the chart's own mix: share kept, order destroyed
        m = rng.random(n) < dose
        new = kind.copy()
        new[m] = rng.permutation(kind)[m]
        endn = end.copy()
        endn[(new == 1) & (kind != 1)] = on[(new == 1) & (kind != 1)]+250_000
        endn[(new != 1) & (kind == 1)] = on[(new != 1) & (kind == 1)]
        return _renote(on, endn, new, lane, mode)
    if name == 'type_collapse':  # non-tap notes become taps: the r5 Phira symptom
        m = (rng.random(n) < dose) & (kind != 0)
        new = kind.copy()
        new[m] = 0
        endn = end.copy()
        endn[m] = on[m]
        return _renote(on, endn, new, lane, mode)
    if name == 'drop':
        m = rng.random(n) >= dose
        return _renote(on[m], end[m], kind[m], lane[m], mode)
    if name == 'insert':
        k = int(dose*n)
        t = rng.integers(on.min(), on.max()+1, k)
        i = rng.integers(0, n, k)
        return _renote(np.r_[on, t], np.r_[end, t+dur[i]], np.r_[kind, kind[i]], np.r_[lane, lane[i]], mode)
    if name in ('bar_shuffle', 'loop'):
        cell = int(dose*1_000_000)
        cells = np.maximum(on//cell, 0)
        m = int(cells.max())+1
        if name == 'bar_shuffle':
            perm = rng.permutation(m)
        else:  # every cell replays the first 4 cells: collapse to a loop
            first = int(cells.min())
            perm = first+(np.arange(m)-first) % 4
        new_on, new_end, new_k, new_l = [], [], [], []
        for c in range(m):
            src = perm[c]
            g = cells == src
            if g.any():
                d = (c-src)*cell
                new_on.append(on[g]+d)
                new_end.append(end[g]+d)
                new_k.append(kind[g])
                new_l.append(lane[g])
        return _renote(np.concatenate(new_on), np.concatenate(new_end), np.concatenate(new_k), np.concatenate(new_l), mode)
    if name == 'burst':  # dose = share of the song turned into 4x density bursts
        k = max(int(dose*duration_us/2_000_000), 1)
        t = []
        for s in rng.integers(0, max(duration_us-2_000_000, 1), k):
            local = int(((on >= s) & (on < s+2_000_000)).sum())
            t.append(rng.integers(s, s+2_000_000, max(local*3, 1)))
        t = np.concatenate(t)
        return _renote(np.r_[on, t], np.r_[on+dur, t], np.r_[kind, np.zeros(len(t), np.int64)], np.r_[lane, rng.integers(0, 4, len(t))], mode)
    if name == 'group_jitter':  # every onset group moves together: chords stay chords, only micro timing changes
        groups, idx = np.unique(on, return_inverse=True)
        d = rng.integers(-int(dose*1000), int(dose*1000)+1, len(groups))
        return _renote(on+d[idx], on+d[idx]+dur, kind, lane, mode)
    if name == 'null_flat':
        # H null: note count, span, IOI multiset and chord sizes survive; rhythm order, lanes and all note types do not (everything is a tap).
        shuffled = corrupt(notes, 'null_shuffle', 1, mode, seed, duration_us)
        return [(a, a, 'tap', l) for a, _, _, l in shuffled]
    if name == 'null_shuffle':
        # H null: keeps note count, song span, the IOI multiset, the type mix and chord sizes; destroys every local pattern.
        groups = np.unique(on)
        idx = np.searchsorted(groups, on)
        new_times = np.empty(len(groups), np.int64)
        new_times[0] = groups[0]
        new_times[1:] = groups[0]+np.cumsum(rng.permutation(np.diff(groups)))
        map_group = rng.permutation(len(groups))
        new_on = new_times[map_group[idx]]
        p = rng.permutation(n)
        new_kind = kind[p]
        new_dur = dur[p]
        new_lane = rng.integers(0, 4, n) if mode == 'mania_4k' else lane
        return _renote(new_on, new_on+new_dur, new_kind, new_lane, mode)
    raise ValueError(name)


H_DOSES = dict(jitter=(10, 20, 30), group_jitter=(10, 20, 30), lane_shuffle=(.25, .5, 1.), jack_inject=(.1, .25, .5), type_shuffle=(.25, .5, 1.), type_collapse=(.25, .5, 1.),
               drop=(.1, .25, .5), insert=(.1, .25, .5), bar_shuffle=(16, 8, 4), loop=(16, 8, 4), burst=(.05, .1, .2), null_shuffle=(1,), null_flat=(1,))
H_MODE_SKIP = dict(phira={'lane_shuffle', 'jack_inject'}, mania_4k=set())


# ------------------------------------------------------------------ floor
D_CONSTANTS = dict(phira=dict(exact=.507, best=.455), mania_4k=dict(exact=.499))  # D_human_chart_data.md "用哪把尺子"
D_NOTE = dict(
    phira=dict(exact='same audio, different charter, density within 1.25x, near-duplicates (CER<.05) dropped; exact clock (D: .507, 42 pairs / 40 songs)',
               best='same song_group across audio files, onset cross-correlation aligned, peak_frac>=.5, peak_over_med>=3, different charter, density within 1.25x (D: .455, 875 pairs / 357 groups)'),
    mania_4k=dict(exact='same audio, different set AND different mapper, density within 1.25x, near-duplicates dropped; exact clock (D: .499, 480 pairs / 137 songs)',
                  best='same pairs as exact, each pair realigned by its 1 ms onset-difference mode (recomputed here from D pairs_labeled.json cer_al)'))


def ruler_pairs(P, mode, clock):
    if mode == 'phira':
        if clock == 'exact':
            return [p for p in P if p['mode'] == mode and p['kind'] == 'same_audio' and not p.get('dup') and p['m'] == 'diff_charter']
        return [p for p in P if p['mode'] == mode and p['kind'] == 'cross_audio' and p.get('peak_frac', 0) >= .5 and p.get('peak_over_med', 0) >= 3
                and p.get('cer_ab_al') is not None and min(p['cer_ab_al'], p['cer_ba_al']) >= .05 and p['m'] == 'diff_charter']
    return [p for p in P if p['mode'] == mode and p['kind'] == 'same_audio' and not p.get('dup') and p['m'] == 'other_set_diff_mapper']


def human_rho_table(P, mode, clock):
    Q = ruler_pairs(P, mode, clock)
    ka, kb = ('cer_ab', 'cer_ba') if clock == 'exact' else ('cer_ab_al', 'cer_ba_al')
    bins = [[] for _ in RHO_BINS]
    for p in Q:
        if p.get(ka) is None or p['na'] <= 0 or p['nb'] <= 0:
            continue
        bins[rho_bin(p['na']/p['nb'])].append(p[ka])
        bins[rho_bin(p['nb']/p['na'])].append(p[kb])
    return [float(np.mean(b)) if len(b) >= 10 else None for b in bins], [len(b) for b in bins]


def null_job(a):
    mode, t, sources, tape, seg_t, seg_s = a
    target = load_chart(tape, t['target'], mode)
    out = []
    for s, ss in zip(sources, seg_s):
        src = load_chart(tape, s['target'], mode)
        warped = warp_notes(src, ss, seg_t, t['duration_us'])
        if not warped:
            out.append(None)
            continue
        cur = [chart_edit_rate([(x+ms*1000, y+ms*1000, k, l) for x, y, k, l in warped], target, mode)['cer'] for ms in SCAN_MS]
        plain = chart_edit_rate(src, target, mode)['cer']
        out.append(dict(rho=len(warped)/len(target), exact=cur[SCAN_MS.index(0)], best=min(cur), plain_exact=plain))
    return out


def rules_job(a):
    mode, r, tape = a
    return rule_counts(load_chart(tape, r['target'], mode), mode, r['duration_us'])


def corrupt_features_job(a):
    mode, r, tape, doses, names = a
    notes = load_chart(tape, r['target'], mode)
    out = {}
    for name, dose in doses:
        c = corrupt(notes, name, dose, mode, r['chart_id'], r['duration_us'])
        o, e, k, l = arrays(c)
        f = features(o, e, k, l, mode, r['duration_us'], None) if len(c) >= 2 else None
        out[f'{name}:{dose}'] = None if f is None else [float(f['log_nps'])]+[float(tf(n, f[n])) if f.get(n) is not None else float('nan') for n in names]
    return out


def build_floor(args):
    d = Path(args.d_data)
    t0 = time.time()
    with open(d/'meta.json') as f:
        META = json.load(f)
    groups_elsewhere = {r['song_group'] for r in META if r['split'] != 'train'}
    train = [r for r in META if r['split'] == 'train' and r['song_group'] not in groups_elsewhere]
    with open(d/'human_feats.json') as f:
        HF = json.load(f)
    with open(d/'pairs_labeled.json') as f:
        P = json.load(f)
    assert not any(p['split'] == 'test' for p in P)
    with open(args.timing) as f:
        TT = json.load(f)['charts']
    out = dict(schema=SCHEMA, created_unix=int(time.time()), sources=dict(d_data=str(d), tape=args.tape, timing=args.timing), H=dict(reference=dict(file='human_reference.npz', names={}, rule=H_PROTOCOL['ref_split'])),
               W={}, rules={}, costs=DEFAULT_COSTS, scan_ms=list(SCAN_MS), rho_bins=list(RHO_BINS), notes=[
                   'Human rulers and wrong-song nulls use non-test charts only; test notes are never opened.',
                   'W = (null - CER)/(null - human) by ratio of means. rho = draft notes / reference notes; both null and human are looked up at the song rho bin, and in the central bin (.8-1.25) human is exactly the D ruler.'])
    arrays_out = {}
    admit = {}
    for mode in ('phira', 'mania_4k'):
        pick = {}
        for r in sorted((r for r in train if r['mode'] == mode and HF.get(r['chart_id'], {}).get('f')), key=lambda r: stable_int('ref', r['chart_id'])):
            pick.setdefault(r['song_group'], r)
        fold = defaultdict(list)
        for g, r in pick.items():
            fold[stable_int('fold', g) % 5].append(r)
        fit, ev = (fold[0]+fold[1])[:1500], (fold[2]+fold[3])[:1500]
        admit[mode] = fold[4][:400]
        names = core_features(mode)+GRID_FEATURES
        vec = lambda r: _vec(HF[r['chart_id']]['f'], names)
        arrays_out[f'{mode}_fit'] = np.array([vec(r) for r in fit], np.float32)
        arrays_out[f'{mode}_eval'] = np.array([vec(r) for r in ev], np.float32)
        arrays_out[f'{mode}_fit_lognps'] = np.array([HF[r['chart_id']]['f']['log_nps'] for r in fit], np.float32)
        arrays_out[f'{mode}_eval_lognps'] = np.array([HF[r['chart_id']]['f']['log_nps'] for r in ev], np.float32)
        out['H']['reference']['names'][mode] = names
        out['H'][mode] = dict(n_fit=len(fit), n_eval=len(ev), n_admit=len(admit[mode]))
        # feature rulers: median |difference| over D's different-mapper density-matched pairs
        if mode == 'phira':
            Q = [p for p in P if p['mode'] == mode and p['m'] == 'diff_charter' and ((p['kind'] == 'same_audio' and not p.get('dup')) or (p['kind'] == 'cross_audio' and p.get('peak_frac', 0) >= .5 and p.get('cer_ab_al') is not None and min(p['cer_ab_al'], p['cer_ba_al']) >= .05))]
        else:
            Q = [p for p in P if p['mode'] == mode and p['kind'] == 'same_audio' and not p.get('dup') and p['m'] in ('other_set_diff_mapper', 'same_set_gd')]
        Q = [p for p in Q if p['d'] == 'dens<=1.25x']
        rulers = {}
        for k in core_features(mode):
            x = np.array([abs(tf(k, HF[p['a']]['f'][k])-tf(k, HF[p['b']]['f'][k])) for p in Q if HF[p['a']]['f'].get(k) is not None and HF[p['b']]['f'].get(k) is not None], float)
            rulers[k] = float(np.nanmedian(x))
        out['H'][mode].update(rulers=rulers, rulers_pairs=len(Q))
        # W human table
        hu = {}
        table = {}
        for clock in ('exact', 'best'):
            means, counts = human_rho_table(P, mode, clock)
            Qc = [p for p in ruler_pairs(P, mode, clock) if p['d'] == 'dens<=1.25x']
            key = 'cer' if clock == 'exact' else 'cer_al'
            central = float(np.mean([p[key] for p in Qc if p.get(key) is not None]))
            hu[clock] = dict(central=central, n_pairs=len(Qc), songs=len({p['group'] for p in Qc}), d_constant=D_CONSTANTS[mode].get(clock), how=D_NOTE[mode][clock], bins_raw=means, bin_counts=counts)
            means[CENTRAL_BIN] = central
            table[clock] = means
        # fill sparse bins: Phira exact takes the shape of the aligned table
        other = {'exact': 'best', 'best': 'exact'}
        for clock in table:
            for i in range(len(RHO_BINS)):
                if table[clock][i] is None:
                    o = table[other[clock]][i]
                    table[clock][i] = (table[clock][CENTRAL_BIN]+o-table[other[clock]][CENTRAL_BIN]) if o is not None else None
            filled = np.array([np.nan if x is None else x for x in table[clock]], float)
            for i in range(len(filled)):
                if np.isnan(filled[i]):
                    j = np.flatnonzero(~np.isnan(filled))
                    filled[i] = filled[j[np.argmin(np.abs(j-i))]]
            table[clock] = filled.tolist()
        # min-ref floor by reference count K (D's min-other form, no mapper filter, near-duplicates dropped)
        by_audio = defaultdict(lambda: defaultdict(dict))
        for p in P:
            if p['mode'] == mode and p['kind'] == 'same_audio' and not p.get('dup'):
                by_audio[p['audio']][p['a']][p['b']] = (p['cer_ab'], p.get('cer_ab_al'))
                by_audio[p['audio']][p['b']][p['a']] = (p['cer_ba'], p.get('cer_ba_al'))
        minref = {}
        for ci, clock in enumerate(('exact', 'best')):
            vals = {}
            for K in range(2, 6):
                song_means = []
                for audio, charts in by_audio.items():
                    v = []
                    for c, others in charts.items():
                        if len(others) < K:
                            continue
                        chosen = sorted(others, key=lambda o: stable_int('minK', c, o))[:K]
                        x = [others[o][ci] for o in chosen if others[o][ci] is not None]
                        if len(x) == K:
                            v.append(min(x))
                    if v:
                        song_means.append(float(np.mean(v)))
                vals[K] = (float(np.mean(song_means)) if len(song_means) >= 8 else None, len(song_means))
            minref[clock] = vals
        # wrong-song null: 5 near-density (K ladder + table) + 3 far-density (table tails only) charts warped onto each target
        with_grid = [r for r in fold[4]+fold[3] if TT.get(r['chart_id'], {}).get('valid') and TT[r['chart_id']].get('segments')]
        rng = rng_for('nullpairs', mode)
        targets = [with_grid[i] for i in rng.choice(len(with_grid), min(args.null_targets, len(with_grid)), replace=False)]
        jobs = []
        for t in targets:
            cand = [r for r in with_grid if r['song_group'] != t['song_group'] and .8 <= r['duration_us']/t['duration_us'] <= 1.25 and .6 <= r['n_events']/max(t['n_events'], 1) <= 1.67]
            if len(cand) < 5:
                continue
            src = [cand[i] for i in rng.choice(len(cand), 5, replace=False)]
            far = [r for r in with_grid if r['song_group'] != t['song_group'] and .8 <= r['duration_us']/t['duration_us'] <= 1.25 and not .6 <= r['n_events']/max(t['n_events'], 1) <= 1.67]
            src += [far[i] for i in rng.choice(len(far), min(3, len(far)), replace=False)] if far else []
            jobs.append((mode, t, src, args.tape, TT[t['chart_id']]['segments'], [TT[s['chart_id']]['segments'] for s in src]))
        res = run_jobs(null_job, jobs, args.workers)
        print(mode, 'null done', round(time.time()-t0), flush=True)
        flat = [x for r in res for x in r if x]
        null_table = {}
        null_counts = {}
        for clock in ('exact', 'best'):
            bins = [[] for _ in RHO_BINS]
            for x in flat:
                bins[rho_bin(x['rho'])].append(x[clock])
            null_counts[clock] = [len(b) for b in bins]
            arr = np.array([np.mean(b) if len(b) >= 10 else np.nan for b in bins], float)
            arr[CENTRAL_BIN+1:] = np.maximum.accumulate(np.nan_to_num(arr[CENTRAL_BIN+1:], nan=-1))  # a sparse tail bin keeps at least the previous one
            arr[arr < 0] = np.nan
            for i in range(len(arr)):
                if np.isnan(arr[i]):
                    j = np.flatnonzero(~np.isnan(arr))
                    arr[i] = arr[j[np.argmin(np.abs(j-i))]]
            # at far density the null can sit under the human floor by sampling noise; keep the denominator positive
            null_table[clock] = np.maximum(arr, np.array(table[clock])+.02).tolist()
        null_k = {'exact': {}, 'best': {}}
        for clock in null_k:
            for K in range(1, 6):
                null_k[clock][K] = float(np.mean([min(x[clock] for x in r[:K] if x) for r in res if any(r[:K])]))  # r[:5] are near-density
        W = dict(rho_table={c: dict(null=null_table[c], human=table[c]) for c in ('exact', 'best')},
                 human_ruler=hu, null=dict(rule='5 wrong-song train charts warped beat-for-beat to the target chart grid (B null N3); table tails add 3 far-density charts per target; K ladder uses only the 5 within 1.67x; best = minimum over the same +-40 ms scan as the draft',
                                           n_targets=len(res), n_pairs=len(flat), counts=null_counts, plain_exact_mean=float(np.mean([x['plain_exact'] for x in flat])), d11_aligned_reference={'phira': .901, 'mania_4k': .916}[mode]),
                 minref={})
        for clock in ('exact', 'best'):
            hk = {'1': table[clock][CENTRAL_BIN]}
            nk = {'1': null_table[clock][CENTRAL_BIN]}
            counts = {'1': hu[clock]['n_pairs']}
            for K in range(2, 6):
                v, c = minref[clock][K]
                counts[str(K)] = c
                hk[str(K)] = v if v is not None else hk[str(K-1)]
                nk[str(K)] = null_k[clock][K]
            W['minref'][clock] = dict(human=hk, null=nk, human_songs=counts)
        out['W'][mode] = W
        # human rule rates on the admission sample
        rr = run_jobs(rules_job, [(mode, r, args.tape) for r in admit[mode][:args.rules_charts]], args.workers)
        notes_total = sum(r['notes'] for r in rr)
        out['rules'][mode] = dict(n_charts=len(rr), per_1k={k: 1000*sum(r[k] for r in rr)/max(notes_total, 1) for k in RATE_RULES[mode]},
                                  hard_rule_totals={k: int(sum(r[k] for r in rr)) for k in HARD_RULES[mode]})
        print(mode, 'rules done', round(time.time()-t0), out['rules'][mode], flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path.parent/'human_reference.npz', **arrays_out)
    with open(out_path.parent/'admit_charts.json', 'w') as f:
        json.dump({m: [dict(chart_id=r['chart_id'], target=r['target'], duration_us=r['duration_us'], mode=m, song_group=r['song_group']) for r in v] for m, v in admit.items()}, f)
    for mode in admit:
        out['H'][mode]['auc_null'] = 1.
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    # H null: needs the reference saved above
    floor = Floor(out_path)
    for mode in ('phira', 'mania_4k'):
        names = core_features(mode)
        rows = run_jobs(corrupt_features_job, [(mode, r, args.tape, [('null_flat', 1)], names) for r in admit[mode][:args.null_charts]], args.workers)
        X = np.array([r['null_flat:1'] for r in rows if r['null_flat:1'] is not None], float)
        ref = floor.ref(mode)
        idx = [ref.names.index(k) for k in names]
        Xf = np.full((len(X), len(ref.names)), np.nan)
        Xf[:, idx] = X[:, 1:]
        Z = ref.z(Xf, X[:, 0])[:, idx]
        solver = floor.solver(mode, names)
        rng = rng_for('h-null', mode)
        sub = lambda n: [loso_auc(Z[rng.choice(len(Z), n, replace=False)], solver)[0] for _ in range(30)]
        a50, a15 = sub(min(50, len(Z))), sub(min(15, len(Z)))
        out['H'][mode].update(auc_null=float(np.mean(a50)), auc_null_n15=float(np.mean(a15)), auc_null_sd_n15=float(np.std(a15)), auc_null_rule='null_flat: held-out human charts with IOIs permuted, note groups reassigned, mania lanes uniform and every note a tap; AUC at 50 songs under the fixed protocol', n_null_charts=len(Z))
        print(mode, 'H null', out['H'][mode]['auc_null'], out['H'][mode]['auc_null_n15'], flush=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('floor written', out_path, round(time.time()-t0), flush=True)


# ------------------------------------------------------------------ admission
def spearman_dose(u_by_dose):
    """Mean per-chart Spearman of dose rank against value, with a 95% chart bootstrap. u [charts, doses]."""
    x = np.arange(u_by_dose.shape[1])
    rho = np.array([spearman(x, row) for row in u_by_dose], float)
    rho = rho[~np.isnan(rho)]
    if not len(rho):
        return dict(mean=None, ci95=None)
    b = rho[np.random.default_rng(1).integers(0, len(rho), (1000, len(rho)))].mean(1)
    return dict(mean=float(rho.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def admission_h(floor, tape, admit, workers, n_charts=100):
    rep = {}
    for mode, charts in admit.items():
        names = core_features(mode)
        ref = floor.ref(mode)
        idx = [ref.names.index(k) for k in names]
        doses = [('control', 0)]+[(n, d) for n, ds in H_DOSES.items() if n not in H_MODE_SKIP[mode] for d in ds]+[('shift', 1000), ('lane_mirror', 1)]
        rows = run_jobs(corrupt_features_job, [(mode, r, tape, doses, names) for r in charts[:n_charts]], workers)
        solver = floor.solver(mode, names)
        def z_of(key):
            X = np.array([r[key] for r in rows if all(r[f'{a}:{b}'] is not None for a, b in doses)], float)
            Xf = np.full((len(X), len(ref.names)), np.nan)
            Xf[:, idx] = X[:, 1:]
            return ref.z(Xf, X[:, 0])[:, idx]
        Zc = z_of('control:0')
        _, uc = loso_auc(Zc, solver)
        out = dict(n_charts=len(Zc), control=dict(auc=float(uc.mean())), invariance=dict(
            shift_1s_max_abs_z=float(np.abs(z_of('shift:1000')-Zc).max()), lane_mirror_max_abs_z=float(np.abs(z_of('lane_mirror:1')-Zc).max())), corruptions={})
        rng = rng_for('admit-h', mode)
        for name, ds in H_DOSES.items():
            if name in H_MODE_SKIP[mode]:
                continue
            us = [uc]
            levels = []
            for dose in ds:
                Z = z_of(f'{name}:{dose}')
                _, u = loso_auc(Z, solver)
                us.append(u)
                s15 = [loso_auc(Z[rng.choice(len(Z), 15, replace=False)], solver)[0] for _ in range(30)]
                levels.append(dict(dose=dose, auc=float(u.mean()), auc_n15=float(np.mean(s15)), sd_n15=float(np.std(s15))))
            top = us[-1]-uc
            b = top[np.random.default_rng(2).integers(0, len(top), (1000, len(top)))].mean(1)
            out['corruptions'][name] = dict(levels=levels, spearman=spearman_dose(np.array(us).T), top_minus_control=dict(mean=float(top.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]))
            out['corruptions'][name]['responds'] = bool(out['corruptions'][name]['top_minus_control']['ci95'][0] > 0)
        rep[mode] = out
    return rep


W_DOSES = dict(shift=(15, 30, 60), jitter=(10, 20, 30), lane_shuffle=(.25, .5, 1.), type_shuffle=(.25, .5, 1.), drop=(.1, .25, .5), insert=(.1, .25, .5), bar_shuffle=(16, 8, 4), null_shuffle=(1,))


def admission_w_job(a):
    mode, pa, pb, tape, table = a
    draft = load_chart(tape, pa['target'], mode)
    ref = load_chart(tape, pb['target'], mode)
    out = {}
    jobs = [('control', 0)]+[(n, d) for n, ds in W_DOSES.items() for d in ds if not (mode == 'phira' and n == 'lane_shuffle')]
    for name, dose in jobs:
        c = corrupt(draft, name, dose, mode, pa['chart_id'], pa['duration_us'])
        cur = [chart_edit_rate([(x+ms*1000, y+ms*1000, k, l) for x, y, k, l in c], ref, mode)['cer'] for ms in (SCAN_MS if name in ('control', 'shift', 'jitter') else (0,))]
        rho = len(c)/len(ref)
        exact = cur[SCAN_MS.index(0)] if len(cur) > 1 else cur[0]
        n = table['exact']['null'][rho_bin(rho)]
        lb = count_lower_bound(rho)
        out[f'{name}:{dose}'] = dict(exact=exact, best=min(cur) if len(cur) > 1 else None, norm=(exact-lb)/max(n-lb, .1), rho=rho)
    return out


def admission_w(floor, tape, P, BY, workers, n_pairs=30):
    rep = {}
    for mode in ('phira', 'mania_4k'):
        Q = [p for p in ruler_pairs(P, mode, 'exact') if p['d'] == 'dens<=1.25x' and p['split'] == 'train']
        Q = sorted(Q, key=lambda p: stable_int('admitw', p['a'], p['b']))
        seen, use = set(), []
        for p in Q:
            if p['group'] not in seen:
                seen.add(p['group'])
                use.append(p)
        use = use[:n_pairs]
        rows = run_jobs(admission_w_job, [(mode, BY[p['a']], BY[p['b']], tape, floor.data['W'][mode]['rho_table']) for p in use], workers)
        out = dict(n_pairs=len(rows), corruptions={})
        ctl = np.array([r['control:0']['exact'] for r in rows])
        ctl_best = np.array([r['control:0']['best'] for r in rows])
        ctl_norm = np.array([r['control:0']['norm'] for r in rows])
        wt = floor.data['W'][mode]['rho_table']['exact']
        out['control'] = dict(exact=float(ctl.mean()), best=float(ctl_best.mean()), norm=float(ctl_norm.mean()), rho=float(np.mean([r['control:0']['rho'] for r in rows])),
                              W=(wt['null'][CENTRAL_BIN]-float(ctl.mean()))/(wt['null'][CENTRAL_BIN]-wt['human'][CENTRAL_BIN]))
        for name, ds in W_DOSES.items():
            if mode == 'phira' and name == 'lane_shuffle':
                continue
            lv = []
            exact_all = [ctl]
            norm_all = [ctl_norm]
            for dose in ds:
                ex = np.array([r[f'{name}:{dose}']['exact'] for r in rows])
                nm = np.array([r[f'{name}:{dose}']['norm'] for r in rows])
                bs = [r[f'{name}:{dose}']['best'] for r in rows]
                exact_all.append(ex)
                norm_all.append(nm)
                wt = floor.data['W'][mode]['rho_table']['exact']
                lv.append(dict(dose=dose, exact=float(ex.mean()), best=None if bs[0] is None else float(np.mean(bs)), norm=float(nm.mean()), rho=float(np.mean([r[f'{name}:{dose}']['rho'] for r in rows])),
                               W=(wt['null'][CENTRAL_BIN]-float(ex.mean()))/(wt['null'][CENTRAL_BIN]-wt['human'][CENTRAL_BIN])))
            sp = spearman_dose(np.array(exact_all).T)
            sn = spearman_dose(np.array(norm_all).T)
            out['corruptions'][name] = dict(levels=lv, spearman_exact=sp, spearman_norm=sn, responds_exact=bool(sp['ci95'] and sp['ci95'][0] > 0), responds_norm=bool(sn['ci95'] and sn['ci95'][0] > 0))
        rep[mode] = out
    return rep


def admission_m_job(a):
    r, segs, tape, mel_dir, other_sha, other_segs, other_target = a
    mode = r['mode']
    notes = load_chart(tape, r['target'], mode)
    cur = audio_curves(np.load(find_mel(mel_dir, r['audio']), mmap_mode='r', allow_pickle=False))
    oc = audio_curves(np.load(find_mel(mel_dir, other_sha), mmap_mode='r', allow_pickle=False))
    on = np.array([n[0] for n in notes], np.int64)
    dur = int(r['duration_us'])
    clock = BeatClock(segs)
    L = lambda t, star=None, s=segs: onset_lift(cur['star'] if star is None else star, t, s)['lift']
    wrong_star = np.resize(oc['star'], len(cur['star']))
    rng = rng_for('adm-m', r['chart_id'])
    rolls = [L(on, np.roll(cur['star'], x)) for x in shifted_rolls(r['chart_id'], dur)]
    wrong_chart = warp_notes(load_chart(tape, other_target, mode), other_segs, segs, dur)
    beats = clock.to_beats(on)
    half = np.round(beats*2)/2
    t_grid = clock.to_time(half)  # metronome-like: same chart, every onset snapped to the half-beat grid
    uniform = np.sort(rng.integers(0, dur, len(on)))
    row = dict(real=L(on), shifted_audio=float(np.mean(rolls)), wrong_song=L(on, wrong_star),
               a2=density_energy(cur['energy'], on, dur), a2_wrong_song=density_energy(np.resize(oc['energy'], len(cur['energy'])), on, dur),
               real_z=onset_lift(cur['star'], on, segs)['z'],
               chart_wrong_song_warped=L([n[0] for n in wrong_chart]) if wrong_chart else None,
               a2_chart_wrong_song=density_energy(cur['energy'], [n[0] for n in wrong_chart], dur) if wrong_chart else None,
               half_beat_snap=L(t_grid), uniform_random=L(uniform), n=len(on), mode=mode)
    for ms in (30, 60, 120):
        row[f'jitter_{ms}'] = L(on+rng.integers(-ms*1000, ms*1000+1, len(on)))
    for ms in (30, 60, 120, 250):
        row[f'shift_{ms}'] = L(on+ms*1000)
    for frac in (.25, .5):
        row[f'drop_{frac}'] = L(on[rng.random(len(on)) >= frac])
    return row


def admission_m(floor, tape, BY_ROWS, TT, mel_dir, workers, per_mode=120):
    by_audio = defaultdict(list)
    for r in BY_ROWS:
        if r['split'] != 'test' and find_mel(mel_dir, r['audio']) is not None and TT.get(r['chart_id'], {}).get('valid') and TT[r['chart_id']].get('segments'):
            by_audio[(r['mode'], r['audio'])].append(r)
    rep = {}
    for mode in ('phira', 'mania_4k'):
        keys = sorted((k for k in by_audio if k[0] == mode), key=lambda k: stable_int('m', k[1]))
        jobs = []
        for i, k in enumerate(keys[:per_mode]):
            r = min(by_audio[k], key=lambda x: stable_int('m', x['chart_id']))
            o = keys[(i+1) % len(keys)]
            q = min(by_audio[o], key=lambda x: stable_int('m', x['chart_id']))
            jobs.append((r, TT[r['chart_id']]['segments'], tape, mel_dir, o[1], TT[q['chart_id']]['segments'], q['target']))
        rows = run_jobs(admission_m_job, jobs, workers)
        vals = {k: np.array([np.nan if x.get(k) is None else x[k] for x in rows], float) for k in rows[0] if k not in ('mode',)}
        real = vals['real']
        a2_keys = ('a2', 'a2_wrong_song', 'a2_chart_wrong_song')
        tab = {k: dict(mean=float(np.nanmean(v)), minus_real=boot_mean(v-real, seed=5)) for k, v in vals.items() if k not in ('real', 'n', 'real_z')+a2_keys}
        a2 = dict(real=boot_mean(vals['a2'], seed=7), wrong_song_audio=boot_mean(vals['a2_wrong_song'], seed=8), wrong_song_chart=boot_mean(vals['a2_chart_wrong_song'], seed=9),
                  minus_wrong_song_audio=boot_mean(vals['a2']-vals['a2_wrong_song'], seed=10), minus_wrong_song_chart=boot_mean(vals['a2']-vals['a2_chart_wrong_song'], seed=11))
        rep[mode] = dict(n_songs=len(rows), real=boot_mean(real, seed=6), real_z=float(np.nanmean(vals['real_z'])), table=tab, a2=a2)
    return rep


def admission(args):
    floor = Floor(args.floor)
    with open(Path(args.floor).parent/'admit_charts.json') as f:
        admit = json.load(f)
    out = dict(floor=floor.sha256)
    parts = args.parts.split(',')
    d = Path(args.d_data)
    if 'H' in parts:
        out['H'] = admission_h(floor, args.tape, admit, args.workers, args.n_charts)
    if 'W' in parts or 'M' in parts:
        with open(d/'meta.json') as f:
            META = [r for r in json.load(f) if r['split'] != 'test']
        BY = {r['chart_id']: r for r in META}
    if 'W' in parts:
        with open(d/'pairs_labeled.json') as f:
            P = json.load(f)
        out['W'] = admission_w(floor, args.tape, P, BY, args.workers, args.n_pairs)
    if 'M' in parts:
        with open(args.timing) as f:
            TT = json.load(f)['charts']
        out['M'] = admission_m(floor, args.tape, META, TT, args.mel_dir, args.workers, args.n_mel)
    path = Path(args.out)
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=1))
    print('written', path)


# ------------------------------------------------------------------ CLI
def silence_from_readout(panel_path, drafts_path, seeds, readouts, drafts):
    """Silence numbers saved by the arm's own benchmark, only when the notes prove they describe these drafts."""
    if Path(panel_path).resolve().parent != Path(drafts_path).resolve().parent:
        return None
    name = Path(drafts_path).name
    key = 'consensus_8_jackfree' if 'jackfree' in name else 'consensus_8' if 'consensus' in name else 'single_sample' if seeds == [0] else None
    if key is None or key not in readouts:
        return None
    out = {}
    for r in readouts[key].get('draft_rows', []):
        state = drafts.get(r['song_id'])
        if state is not None and len(state['events']) == r.get('draft_notes') and r.get('silence_split'):
            out[r['song_id']] = r['silence_split'].get('true_silence_events_per_s')
    return out or None


def cmd_score(args):
    panel, readouts = load_panel(args.panel)
    songs = panel['songs']
    if args.modes:
        songs = [s for s in songs if s['mode'] in args.modes.split(',')]
    if args.max_songs_per_mode:
        n = defaultdict(int)
        keep = []
        for s in songs:
            n[s['mode']] += 1
            if n[s['mode']] <= args.max_songs_per_mode:
                keep.append(s)
        songs = keep
    if panel.get('split') == 'test':
        raise PermissionError('test panel refused')
    seeds = None if args.seeds is None else [int(x) for x in args.seeds.split(',')]
    draft_sets = []
    silence = None
    if seeds is None:
        d = load_drafts(args.drafts)
        silence = silence_from_readout(args.panel, args.drafts, None, readouts, d)
        songs = [s for s in songs if s['song_id'] in d]
        draft_sets.append({s['song_id']: d[s['song_id']] for s in songs})
    else:
        for sd in seeds:
            d = load_drafts(args.drafts, sd)
            if sd == seeds[0] and seeds == [0]:
                silence = silence_from_readout(args.panel, args.drafts, seeds, readouts, d)
            songs = [s for s in songs if s['song_id'] in d]
            draft_sets.append({s['song_id']: d[s['song_id']] for s in songs})
            del d
    floor = Floor(args.floor)
    timing = load_timing(args.timing)
    doc = score_panel(songs, draft_sets, tape=args.tape, timing=timing, floor=floor, features_dir=args.features_dir, workers=args.workers, label=args.label or Path(args.drafts).parent.name+'/'+Path(args.drafts).stem, cost=args.cost,
                      source=dict(drafts=str(args.drafts), panel=str(args.panel), seeds=seeds, panel_sha256=panel.get('panel_sha256'), tape=args.tape), silence_rows=silence)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc))
    table = text_table(doc)
    Path(args.out).with_suffix('.txt').write_text(table+'\n')
    print(table)


def compare(a_path, b_path, *, floor=None, cost_a=None, cost_b=None, samples=500):
    """compare(a.json, b.json): the paired report as a dict; also what the CLI prints via compare_text()."""
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    floor = Floor(floor or a['floor']['path'])
    return compare_docs(a, b, floor, cost_a=cost_a, cost_b=cost_b, samples=samples)


def cmd_compare(args):
    rep = compare(args.a, args.b, floor=args.floor, cost_a=args.cost_a, cost_b=args.cost_b, samples=args.samples)
    text = compare_text(rep)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
        Path(args.out).with_suffix('.txt').write_text(text+'\n')


def cmd_noise(args):
    doc = json.loads(Path(args.scored).read_text())
    floor = Floor(args.floor or doc['floor']['path'])
    other = json.loads(Path(args.other).read_text()) if args.other else None
    aa = [json.loads(Path(p).read_text()) for p in args.aa.split(',')] if args.aa else None
    rep = noise_report(doc, floor, [int(x) for x in args.sizes.split(',')], other=other, reps=args.reps, aa_docs=aa, aa_reps=args.aa_reps)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
    for mode, by in rep.items():
        for size, row in by.items():
            if size == 'aa_false_alarm':
                print(mode, 'A/A false alarm', json.dumps(row))
                continue
            print(mode, f'n={size}', ' '.join(f"{k} {row[k]['mean']:.3f}±{row[k]['halfwidth80']:.3f}" for k in ('H', 'AUC', 'H_gap', 'W_nearest_exact', 'W_nearest_best')))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--drafts', required=True)
    s.add_argument('--panel', required=True, help='workflow-benchmark.json holding the panel')
    s.add_argument('--seeds', help='for drafts-samples.json: comma list, scored separately and averaged per song')
    s.add_argument('--tape', default=DEFAULT_TAPE)
    s.add_argument('--timing', default=DEFAULT_TIMING)
    s.add_argument('--floor', default=DEFAULT_FLOOR)
    s.add_argument('--features-dir', help='folder with <audio_sha>/mel10_complete.npy (or features/<sha>/..., or <sha>.npy): enables M and the silence check')
    s.add_argument('--out', required=True)
    s.add_argument('--label')
    s.add_argument('--cost', type=float, help='cost per song or minute, any unit, same across arms')
    s.add_argument('--workers', type=int, default=3)
    s.add_argument('--max-songs-per-mode', type=int)
    s.add_argument('--modes')
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--floor')
    c.add_argument('--cost-a', type=float)
    c.add_argument('--cost-b', type=float)
    c.add_argument('--samples', type=int, default=500)
    c.add_argument('--out')
    c.set_defaults(fn=cmd_compare)
    n = sub.add_parser('noise')
    n.add_argument('scored')
    n.add_argument('--other')
    n.add_argument('--aa', help='a.json,b.json: same arm, different seeds')
    n.add_argument('--sizes', default='15,30,50')
    n.add_argument('--reps', type=int, default=300)
    n.add_argument('--aa-reps', type=int, default=60)
    n.add_argument('--floor')
    n.add_argument('--out')
    n.set_defaults(fn=cmd_noise)
    f = sub.add_parser('build-floor')
    f.add_argument('--d-data', default=str(REPO/'reports/research-20261104/data'))
    f.add_argument('--tape', default=DEFAULT_TAPE)
    f.add_argument('--timing', default=DEFAULT_TIMING)
    f.add_argument('--out', default=DEFAULT_FLOOR)
    f.add_argument('--null-targets', type=int, default=120)
    f.add_argument('--null-charts', type=int, default=200)
    f.add_argument('--rules-charts', type=int, default=200)
    f.add_argument('--workers', type=int, default=3)
    f.set_defaults(fn=build_floor)
    a = sub.add_parser('admission')
    a.add_argument('--floor', default=DEFAULT_FLOOR)
    a.add_argument('--d-data', default=str(REPO/'reports/research-20261104/data'))
    a.add_argument('--tape', default=DEFAULT_TAPE)
    a.add_argument('--timing', default=DEFAULT_TIMING)
    a.add_argument('--parts', default='H,W')
    a.add_argument('--mel-dir')
    a.add_argument('--n-charts', type=int, default=100)
    a.add_argument('--n-pairs', type=int, default=30)
    a.add_argument('--n-mel', type=int, default=120)
    a.add_argument('--out', required=True)
    a.add_argument('--workers', type=int, default=3)
    a.set_defaults(fn=admission)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()

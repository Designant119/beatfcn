"""Workflow-readiness benchmark v1: whole-song Chart Edit Rate (CER) of drafts vs human charts.

Estimates how much charting work starting from a draft saves compared with
charting from scratch (see design/WORKFLOW_BENCHMARK_v1.md and
beatfcn/train/workflow_metrics.py). Never schedules cloud work, never trains.

Draft sources (exactly one):
  --checkpoint CKPT --plan PLAN.json   model drafts via local_song_rollout.generate_complete_songs
  --local-smoke                        tiny RANDOM-INIT model on local MERT features (plumbing only)
  --human-as-draft                     another human chart of the same song (human-vs-human floor)
  --null-draft                         empty draft (CER must be 1) and time-shifted human drafts

  .venv/bin/python -m scripts.run_workflow_benchmark --human-as-draft --out-dir /tmp/wf-human
  .venv/bin/python -m scripts.run_workflow_benchmark --null-draft --out-dir /tmp/wf-null
  .venv/bin/python -m scripts.run_workflow_benchmark --local-smoke --max-songs 4 --out-dir /tmp/wf-smoke
  .venv/bin/python -m scripts.run_workflow_benchmark --plan PLAN.json --checkpoint step-X.pt \
      --device cuda --temperature 1 --seed 0 --out-dir reports/workflow-benchmark-v1/<run>

  --scheduler legacy|grouped|continuous, --flush-reserve-s, --no-score, --resume, --score-from: see
      design notes in beatfcn/model/local_song_rollout.py. Every model run writes <out>/samples.jsonl (one fsynced
      line per finished (seed, song)), <out>/progress.json (atomic) and <out>/run-meta.json. --max-wall-s is an
      ABSOLUTE process limit: decoding and scoring stop --flush-reserve-s before it and raw samples are kept.
      Score elsewhere, without a model: --score-from <decode out-dir> --out-dir <new dir> + the same readout flags.

  --best-of N (model drafts, sampling only): N whole-song samples per song (seeds seed..seed+N-1);
      readouts single_sample (capability) / best_of_N_reranked (reference-free reranker, per song) /
      oracle_best_of_N (ORACLE upper bound: picked by CER vs human). --export-dir writes the selected
      (never the oracle) draft + audio; --pilot-segments K cuts a pilot kit (design/PILOT_KIT_v1.md).

Opt-in decoding/conditioning switches (model drafts; all default off, outputs unchanged without them):
  --history-reference      each core sees the previous 8 s of the model's OWN draft as a model_history
                           VisibleReference (matches history_context training); recorded in decoding.
  --onset-fusion W         D3 shallow fusion: head-time logits += W*log(p_onset+1e-4) from the model's onset head.
  --consensus N            D6: N whole-song samples -> one draft by onset clustering (30 ms) + majority vote;
                           readout consensus_N next to single_sample / best_of_N_reranked / oracle.
  --density-request reference [--guidance W]
                           D4: per core the 2 s density curve of the song's PRIMARY human reference (the chart
                           whose stratum the panel is stratified by) is requested -- labelled
                           charter_intent_density, NEVER unconditioned capability; W != 1 adds CFG against the
                           rollout's own unspecified conditions (W = 1 is plain conditioning).
  --charter-timing TABLE   model with config.beat_features: every core gets beat features from the charter
                           timing (BPM/offset/meter) of the song's PRIMARY (scored) human reference, from a
                           scripts/build_chart_timing.py table (absent state when that chart has no valid
                           timing). Labelled charter_timing_input -- a workflow input, NEVER unconditioned
                           capability; run without it for the capability number.

Readouts (never mixed): raw CER; snapped CER (draft snapped to the REFERENCE
chart's own BPM/offset grid -- assumes charter-supplied timing); per-reference
CER labelled by the reference's difficulty stratum; min-over-references CER
(less pessimistic, like multi-reference HTER). Bootstrap 80%/95% CIs are over
songs (cluster bootstrap: a song's references move together).
"""
import argparse
from collections import defaultdict
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

PROCESS_START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.events.canonical import TYPES  # noqa: E402
from beatfcn.model.rerank import DEFAULT_WEIGHTS as RERANK_WEIGHTS  # noqa: E402
from beatfcn.train.workflow_metrics import (CHARTER_DIVISORS, CHARTER_SNAP_MAX_US, DEFAULT_COSTS,  # noqa: E402
    DEFAULT_TOLERANCES, ON_GRID_US, PROFILE_METRICS, SNAP_DIVISORS, SNAP_MAX_US, bootstrap_mean, chart_edit_rate,
    charter_timing_snap, compare_patterns, cost_sensitivity, density_profile, grid_alignment, grid_comparison,
    grid_from_timing, playability, snap_notes, snap_times, stable_rank)

TAPE = 'inputs/native-v2-d94-580/tape.json'
RAW_ROOT = 'inputs/d66_local_cohort/source/raw'
PANEL_SEED = 20261102
SHIFTS_MS = (10, 50, 150, 250, 1000)
GRID_FIT_MIN = .9
NEAR_DUPLICATE_CER = .05


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def coarse_stratum(mode, stratum):
    if stratum and stratum.startswith('density/'):
        return stratum.split('/', 1)[1]
    if mode == 'mania_4k':
        return stratum
    level = stratum.split('/')[-1]
    return {'0-4': 'low', '4-8': 'low', '8-11': 'low', '11-13': 'medium', '13-15': 'medium',
            '15-16': 'high', '16+': 'high'}.get(level, 'unknown')


# ------------------------------------------------------------------ data
def load_tape(root, tape):
    manifest = json.loads((root/tape).read_text())
    if not manifest['complete'] or manifest['failures']:
        raise ValueError('Request tape is incomplete or failed')
    assign_density_strata(manifest)
    return manifest


def assign_density_strata(manifest):
    """Full-database receipts carry no difficulty label: derive an objective
    proxy (heads/second tertiles of the train split, per mode) in memory only.
    Labelled 'density/<low|medium|high>' so it is never confused with charter
    difficulty strata; charts that already have a stratum are left untouched."""
    import numpy as np
    def rate(c):
        return sum(r['event_count'] for r in c['requests']) / max(c['duration_us'] / 1e6, 1e-6)
    missing = [c for c in manifest['charts'] if not c.get('stratum')]
    if not missing:
        return
    edges = {}
    for mode in {c['mode'] for c in missing}:
        train = [rate(c) for c in manifest['charts'] if c['mode'] == mode and c['split'] == 'train']
        edges[mode] = tuple(np.quantile(train, [1/3, 2/3])) if train else (0., 0.)
    for c in missing:
        lo, hi = edges[c['mode']]
        value = rate(c)
        c['stratum'] = 'density/' + ('low' if value < lo else 'medium' if value < hi else 'high')
    manifest['derived_strata'] = dict(rule='train-split heads/s tertiles per mode',
        edges={m: [round(float(x), 4) for x in v] for m, v in edges.items()})


def song_groups(charts, splits):
    groups = defaultdict(list)
    for c in charts:
        if c['split'] in splits:
            groups[(c['audio_sha256'], c['mode'])].append(c)
    for key, refs in groups.items():
        refs.sort(key=lambda c: c['chart_id'])
        if len({c['duration_us'] for c in refs}) != 1 or len({c['split'] for c in refs}) != 1:
            raise ValueError(f'Inconsistent duration/split within song {key}')
    return groups


def freeze_panel(manifest, tape_sha, *, max_songs=60, seed=PANEL_SEED, split='val'):
    """Deterministic song list: per mode, round-robin over coarse strata in hash order."""
    groups = song_groups(manifest['charts'], {split})
    by_mode = defaultdict(lambda: defaultdict(list))
    for (audio, mode), refs in groups.items():
        primary = min(refs, key=lambda c: stable_rank(seed, 'primary', c['chart_id']))
        by_mode[mode][coarse_stratum(mode, primary['stratum'])].append((audio, mode))
    modes = sorted(by_mode)
    available = {m: sum(map(len, by_mode[m].values())) for m in modes}
    quota = {m: 0 for m in modes}
    remaining = min(max_songs, sum(available.values()))
    while remaining:
        for m in modes:
            if remaining and quota[m] < available[m]:
                quota[m] += 1
                remaining -= 1
    chosen = []
    for m in modes:
        queues = {s: sorted(v, key=lambda k: stable_rank(seed, 'song', k[0], k[1])) for s, v in sorted(by_mode[m].items())}
        picked = []
        while len(picked) < quota[m]:
            for s in sorted(queues):
                if queues[s] and len(picked) < quota[m]:
                    picked.append(queues[s].pop(0))
        chosen += picked
    songs = []
    for audio, mode in chosen:
        refs = groups[(audio, mode)]
        songs.append(dict(song_id=f'{mode}:{audio[:16]}', audio_sha256=audio, mode=mode,
            duration_us=int(refs[0]['duration_us']), component=refs[0]['component'],
            references=[dict(chart_id=c['chart_id'], stratum=c['stratum'], coarse_stratum=coarse_stratum(mode, c['stratum']),
                             target_path=c['target_path'], target_sha256=c['target_sha256']) for c in refs]))
    panel = dict(schema='workflow-benchmark-v1-panel', tape_sha256=tape_sha, split=split, seed=seed,
                 max_songs=max_songs, selection='per mode round-robin over coarse strata (of a hash-chosen primary '
                 'chart), hash-ordered; every human chart of a chosen song is a reference',
                 available_songs=available, songs=songs)
    panel['panel_sha256'] = sha_json(panel)
    return panel


def _chart_heads(chart):
    return sum(int(r['event_count']) for r in chart['requests'])


def freeze_spread_panel(manifest, tape_sha, *, max_songs=40, seed=PANEL_SEED, split='val', mode='mania_4k'):
    """Difficulty-spread panel: songs of ``split``/``mode`` with >= 2 distinct charts. Per song the TARGET is a
    hash-chosen chart that has a strictly denser chart; the SIBLING (the charter's own chart, an input) is the
    nearest denser chart. Only the target is a scoring reference. Round-robin over the target's coarse stratum."""
    groups = song_groups(manifest['charts'], {split})
    by_stratum = defaultdict(list)
    for (audio, m), refs in groups.items():
        if m != mode:
            continue
        distinct = {c['canonical_sha256']: c for c in refs if c.get('canonical_sha256')}.values() \
            if all(c.get('canonical_sha256') for c in refs) else refs
        distinct = sorted(distinct, key=lambda c: c['chart_id'])
        heads = {c['chart_id']: _chart_heads(c) for c in distinct}
        eligible = [c for c in distinct if any(heads[o['chart_id']] > heads[c['chart_id']] for o in distinct)]
        if not eligible:
            continue
        target = min(eligible, key=lambda c: stable_rank(seed, 'spread-target', c['chart_id']))
        sibling = min((o for o in distinct if heads[o['chart_id']] > heads[target['chart_id']]),
                      key=lambda o: (heads[o['chart_id']], o['chart_id']))
        by_stratum[coarse_stratum(m, target['stratum'])].append((audio, target, sibling, heads))
    queues = {k: sorted(v, key=lambda e: stable_rank(seed, 'spread-song', e[0])) for k, v in sorted(by_stratum.items())}
    picked = []
    while len(picked) < max_songs and any(queues.values()):
        for k in sorted(queues):
            if queues[k] and len(picked) < max_songs:
                picked.append(queues[k].pop(0))
    entry = lambda c: dict(chart_id=c['chart_id'], stratum=c['stratum'], coarse_stratum=coarse_stratum(mode, c['stratum']),
                           target_path=c['target_path'], target_sha256=c['target_sha256'])
    songs = [dict(song_id=f'{mode}:{audio[:16]}', audio_sha256=audio, mode=mode, duration_us=int(t['duration_us']),
                  component=t['component'], references=[entry(t)], sibling=dict(entry(s), heads=heads[s['chart_id']]),
                  target_heads=heads[t['chart_id']]) for audio, t, s, heads in picked]
    panel = dict(schema='workflow-benchmark-v1-spread-panel', tape_sha256=tape_sha, split=split, seed=seed,
                 max_songs=max_songs, mode=mode, available_songs={k: len(v) for k, v in by_stratum.items()},
                 selection='songs with >= 2 distinct charts; target = hash-chosen chart with a denser sibling; sibling = '
                           'nearest denser chart (input, never scored); round-robin over the target coarse stratum',
                 songs=songs)
    panel['panel_sha256'] = sha_json(panel)
    return panel


def thin_to_count(notes, n):
    """Naive downmap baseline: the sibling thinned to ``n`` notes (chord members dropped highest lane first,
    then onset groups decimated evenly). Deterministic; no audio, no model."""
    groups = defaultdict(list)
    for note in sorted(notes):
        groups[note[0]].append(note)
    groups = [sorted(g, key=lambda x: x[3]) for _, g in sorted(groups.items())]
    total = sum(map(len, groups))
    for g in sorted(groups, key=len, reverse=True):
        while len(g) > 1 and total > n:
            g.pop()
            total -= 1
    if total > n:
        keep = set(np.round(np.linspace(0, len(groups) - 1, max(n, 1))).astype(int).tolist()) if n else set()
        groups = [g for i, g in enumerate(groups) if i in keep]
    return [x for g in groups for x in g]


def spread_baselines(root, panel, *, samples, seed):
    """Human-only baselines of a spread panel vs each target: the sibling as is, and thinned to the target count."""
    rows = []
    for song in panel['songs']:
        target = song['references'][0]
        t_notes = reference_notes(root, target, song['mode'])
        s_notes = reference_notes(root, song['sibling'], song['mode'])
        raw = chart_edit_rate(s_notes, t_notes, song['mode'])
        thin = chart_edit_rate(thin_to_count(s_notes, len(t_notes)), t_notes, song['mode'])
        cost = raw['op_cost']
        rows.append(dict(song_id=song['song_id'], coarse_stratum=target['coarse_stratum'], sibling=song['sibling']['chart_id'],
                         target=target['chart_id'], sibling_cer=raw['cer'], thinned_cer=thin['cer'],
                         oracle_thinned_cer=(sum(cost.values()) - cost['delete']) / max(1, raw['reference_notes']),
                         notes_ratio=len(s_notes) / max(1, len(t_notes))))
    out = dict(label='human-only spread baselines (no model): sibling chart as the draft for the target',
               rows=rows, by_stratum={})
    for key, metric in (('sibling_cer', 'sibling_cer'), ('thinned_cer', 'thinned_cer'), ('oracle_thinned_cer', 'oracle_thinned_cer')):
        out[key] = _agg(rows, lambda r, m=metric: r[m], samples=samples, seed=seed)
        for stratum in sorted({r['coarse_stratum'] for r in rows}):
            out['by_stratum'].setdefault(stratum, {})[key] = _agg([r for r in rows if r['coarse_stratum'] == stratum],
                                                                  lambda r, m=metric: r[m], samples=samples, seed=seed)
    return out


def reference_notes(root, ref, mode):
    path = root/ref['target_path']
    if sha_file(path) != ref['target_sha256']:
        raise ValueError(f'Human target changed: {path}')
    with np.load(path, allow_pickle=False) as z:
        return [(int(a), int(b), TYPES[int(k)], int(l) if mode == 'mania_4k' else -1)
                for a, b, k, l in zip(z['onset_us'], z['end_us'], z['kind'], z['lane'])]


def raw_chart_path(raw_root, chart_id):
    source, number = chart_id.split(':', 1)
    return raw_root/('osu/charts' if source == 'osu' else 'phira/zips')/(number+('.osu' if source == 'osu' else '.zip'))


def _info_scan(text):
    """Tolerant top-level ``key: value`` scan of a Phira info.yml (images without PyYAML)."""
    import re
    out = {}
    for line in text.splitlines():
        m = re.match(r'^([A-Za-z_][\w]*)\s*:\s*(.*?)\s*$', line)
        if m:
            value = m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
                value = value[1:-1]
            out[m.group(1)] = value
    if 'offset' in out:
        try:
            out['offset'] = float(out['offset'])
        except ValueError:
            out['offset'] = 0.
    return out


def load_raw_chart(path, chart_id):
    """Parsed raw human chart. Phira packages fall back to a yaml-free info.yml scan when PyYAML is
    missing (the cloud image has none; it made every Phira grid unavailable before)."""
    if chart_id.startswith('osu:'):
        from beatfcn.events.osu_mania import parse_osu
        return parse_osu(path.read_bytes(), path.name)
    from beatfcn.events.phira import load_package
    try:
        import yaml  # noqa: F401
        return load_package(path)
    except ImportError:
        pass
    import zipfile
    from beatfcn.events.phira import UnsupportedFormat, parse_phira
    with zipfile.ZipFile(path) as z:
        names = {n.lower().rsplit('/', 1)[-1]: n for n in z.namelist()}
        info_name = names.get('info.yml') or names.get('info.yaml')
        info = _info_scan(z.read(info_name).decode('utf-8-sig')) if info_name else {}
        if str(info.get('format') or '').lower() in ('pec', 'pbc'):
            raise UnsupportedFormat(f"info.yml format={info['format']}")
        name = info.get('chart')
        member = names.get(str(name).lower().rsplit('/', 1)[-1]) if name else None
        if member is None:
            raise UnsupportedFormat(f'chart file {name!r} not in package')
        return parse_phira(z.read(member), source_name=str(name), info_offset_sec=info.get('offset', 0))


class ClockShifts:
    """chart_id -> the constant model-clock shift (us) baked into its tape target, from the
    full-database target receipts (``<receipts_root>/targets/<sha256(chart_id:canonical_sha256)>/
    receipt.json``, ``statistics.model_shift_us`` = offset_decision shift_ms + codec c_ms).

    Full-database targets are the raw chart's events + this shift (beatfcn.train.full_database.
    event_arrays), so the raw chart's BPM/offset must be moved by the same amount. Lookups are lazy
    (only charts whose raw clock disagrees at shift 0) and verified (receipt chart_id/canonical)."""

    def __init__(self, receipts_root=None, canonical=None):
        self.root = Path(receipts_root) if receipts_root else None
        self.canonical = dict(canonical or {})

    def __call__(self, chart_id):
        if self.root is None or chart_id not in self.canonical:
            return None, 'no receipt root or canonical sha for chart'
        canonical = self.canonical[chart_id]
        key = hashlib.sha256(f'{chart_id}:{canonical}'.encode()).hexdigest()
        path = self.root/'targets'/key/'receipt.json'
        if not path.is_file():
            return None, f'receipt missing: {path}'
        receipt = json.loads(path.read_text())
        if receipt.get('chart_id') != chart_id or receipt.get('canonical_sha256') != canonical:
            return None, 'receipt identity mismatch'
        shift = (receipt.get('statistics') or {}).get('model_shift_us')
        return (None, 'receipt has no model_shift_us') if shift is None else (int(shift), str(path))


def default_receipts_root(root, tape):
    """Full-database layout: <db>/native-v2-tape/tape.json next to <db>/targets/targets/<key>/receipt.json."""
    candidate = (Path(root)/tape).parent.parent/'targets'
    return candidate if (candidate/'targets').is_dir() else None


def estimate_clock_shift(ref_onsets_us, chart_onsets_us, *, max_us=200_000):
    """Most common (target - nearest raw onset) difference in whole ms within +-max_us, or None."""
    a = np.sort(np.asarray(chart_onsets_us, dtype=np.int64))
    r = np.asarray(ref_onsets_us, dtype=np.int64)
    if not len(a) or not len(r):
        return None
    i = np.clip(np.searchsorted(a, r), 0, len(a)-1)
    j = np.maximum(i-1, 0)
    d = np.where(np.abs(r-a[i]) <= np.abs(r-a[j]), r-a[i], r-a[j])
    d = d[np.abs(d) <= max_us]
    if not len(d):
        return None
    values, counts = np.unique(np.round(d/1000).astype(np.int64), return_counts=True)
    return int(values[np.argmax(counts)])*1000


def resolve_clock_shift(chart, chart_id, ref_onsets, *, clock_shifts=None):
    """(shift_us, source, alignment, tried) mapping the raw chart clock onto the tape target clock.

    Tries 0, then the target receipt's model_shift_us (``clock_shifts``), then the
    modal ms difference (estimated); the first with >= 95% target onsets within
    1 ms of a raw onset wins. ``alignment`` < .95 means no shift was accepted
    (shift 0 / source 'none' are then returned)."""
    raw_onsets = np.asarray([e.onset_us for e in chart.events], dtype=np.int64)
    shift, source = 0, 'none'
    alignment = grid_alignment(ref_onsets, raw_onsets)
    tried = {0: alignment}
    if alignment < .95:
        candidates = []
        if clock_shifts is not None:
            got, why = clock_shifts(chart_id)
            if got is not None:
                candidates.append((got, 'receipt'))
        estimated = estimate_clock_shift(ref_onsets, raw_onsets)
        if estimated is not None:
            candidates.append((estimated, 'estimated'))
        for value, label in candidates:
            if value not in tried:
                tried[value] = grid_alignment(ref_onsets, raw_onsets+value)
            if tried[value] >= .95:
                shift, source, alignment = value, label, tried[value]
                break
    return shift, source, alignment, tried


def reference_grid(raw_root, chart_id, notes, *, clock_shifts=None):
    """Charter timing of the human reference (BPM/offset) on the audio clock, or a reason it is unavailable.

    The raw chart is on its native clock; the tape target may carry a constant model-clock shift
    (full database: offset decision + codec constant, e.g. +14 ms for osu). Shift resolution, each
    accepted only if >= 95% of target onsets then sit within 1 ms of a raw onset: 0 (local tape),
    the target receipt (``clock_shifts``), else the modal ms difference (estimated). The grid is
    moved by the accepted shift. Recorded as ``clock_shift_us`` / ``clock_shift_source``."""
    path = raw_chart_path(raw_root, chart_id)
    if not path.is_file():
        return None, dict(available=False, reason=f'raw chart not local: {path}')
    try:
        chart = load_raw_chart(path, chart_id)
    except Exception as exc:  # noqa: BLE001 - recorded, readout skipped
        return None, dict(available=False, reason=f'{type(exc).__name__}: {exc}')
    if chart.timing is None:
        return None, dict(available=False, reason='chart has no single timing map (e.g. per-line BPMs)')
    ref_onsets = [n[0] for n in notes]
    shift, source, alignment, tried = resolve_clock_shift(chart, chart_id, ref_onsets, clock_shifts=clock_shifts)
    if alignment < .95:
        return None, dict(available=False, reason=f'raw chart clock disagrees with target ({alignment:.3f} within 1 ms '
                          f'at every tried shift: {sorted(tried)} us)', shifts_tried_us=sorted(tried))
    from beatfcn.train.workflow_metrics import UnusableTiming
    try:
        native_grid = grid_from_timing(chart.timing)
    except UnusableTiming as exc:
        return None, dict(available=False, reason=f'gimmick timing: {exc}')
    grid = [(start+shift, beat) for start, beat in native_grid]
    onsets = np.array(ref_onsets, dtype=np.int64)
    snapped, _ = snap_times(onsets, grid)
    fit = float(np.mean(np.abs(snapped-onsets) <= 2000)) if len(onsets) else 0.
    info = dict(segments=len(grid), first_bpm=round(60e6/grid[0][1], 4), clock_alignment_1ms=alignment,
                clock_shift_us=int(shift), clock_shift_source=source, self_grid_fit_2ms=fit,
                source_format=chart.source_format)
    if fit < GRID_FIT_MIN:
        return None, dict(info, available=False, reason=f'reference notes do not sit on their own BPM grid '
                          f'({fit:.3f} within 2 ms < {GRID_FIT_MIN})')
    return grid, dict(info, available=True)


def timing_coverage(timing):
    """Charter-timing availability counts per source/mode and shift source, plus unavailability reasons."""
    import re
    out = dict(charts=len(timing), available=sum(bool(v['available']) for v in timing.values()), by_source={},
               shift_source={}, reasons={})
    for chart_id, v in timing.items():
        src = chart_id.split(':', 1)[0]
        b = out['by_source'].setdefault(src, dict(charts=0, available=0))
        b['charts'] += 1
        b['available'] += bool(v['available'])
        if v.get('clock_shift_source'):
            key = f"{v['clock_shift_source']}:{v.get('clock_shift_us')}"
            out['shift_source'][key] = out['shift_source'].get(key, 0)+1
        if not v['available']:
            reason = re.sub(r':\s*/\S*$', '', str(v.get('reason', '')).split(' (')[0].strip())
            out['reasons'][reason] = out['reasons'].get(reason, 0)+1
    out['available_fraction'] = out['available']/out['charts'] if out['charts'] else None
    return out


# ------------------------------------------------------------ evaluation
def evaluate(draft, draft_info, song, refs, *, grids, per_op, split, energy=None):
    """``energy`` (optional, model drafts): requirement_diagnostics.song_energy of the song's complete mel;
    adds the whole-song silence split (draft notes inside regions where that reference has no note for
    >= 4 s, split into true-silence vs uncharted-music by local energy) to every reference row."""
    from beatfcn.train import requirement_diagnostics as D
    mode, duration = song['mode'], song['duration_us']
    rows = []
    for ref in refs:
        notes = ref['notes']
        cer = chart_edit_rate(draft, notes, mode)
        row = dict(song_id=song['song_id'], split=split, mode=mode, draft=draft_info['draft_id'],
                   draft_stratum=draft_info.get('stratum'), draft_coarse=draft_info.get('coarse_stratum'),
                   reference=ref['chart_id'], stratum=ref['stratum'], coarse_stratum=ref['coarse_stratum'],
                   primary_reference=bool(song.get('references'))
                   and ref['chart_id'] == primary_reference(song)['chart_id'],
                   cer=cer['cer'], savings=cer['savings'], ops=cer['ops'], op_cost=cer['op_cost'],
                   draft_notes=cer['draft_notes'], reference_notes=cer['reference_notes'],
                   sensitivity=cost_sensitivity(draft, notes, mode, per_op=per_op),
                   density=density_profile(draft, notes, duration))
        grid = grids.get(ref['chart_id'])
        if grid is not None:
            snapped, stats = snap_notes(draft, grid)
            s = chart_edit_rate(snapped, notes, mode)
            row.update(snapped_cer=s['cer'], snapped_ops=s['ops'], snap_stats=stats)
            # Charter timing (the reference's own BPM/offset): grid metrics + charter_timing_snap readout.
            row['grid'] = grid_comparison(draft, notes, grid)
            row['charter_timing_snap'] = charter_timing_snap(draft, notes, mode, grid)
        else:
            row.update(snapped_cer=None)
        row['pattern'] = compare_patterns(draft, notes, mode, grid=grid, duration_us=duration)
        if energy is not None:
            row['silence_split'] = D.uncharted_split([n[0] for n in draft], D.uncharted_regions(notes, duration), energy)
        rows.append(row)
    best = min(rows, key=lambda r: r['cer'])
    snapped = [r['snapped_cer'] for r in rows if r['snapped_cer'] is not None]
    charter = [r['charter_timing_snap']['cer'] for r in rows if r.get('charter_timing_snap')]
    draft_row = dict(song_id=song['song_id'], split=split, mode=mode, draft=draft_info['draft_id'],
                     draft_stratum=draft_info.get('stratum'), complete=draft_info.get('complete', True),
                     failure=draft_info.get('failure'), draft_notes=len(draft),
                     min_ref_cer=best['cer'], min_ref=best['reference'], min_ref_stratum=best['stratum'],
                     min_ref_snapped_cer=min(snapped) if snapped else None,
                     min_ref_charter_snap_cer=min(charter) if charter else None,
                     playability=playability(draft, mode))
    if energy is not None:
        draft_row['silence_split'] = best['silence_split']  # vs the min-CER reference
    if draft_info.get('intent_reference') is not None:
        intent = next((r for r in rows if r['reference'] == draft_info['intent_reference']), None)
        draft_row.update(intent_reference=draft_info['intent_reference'],
                         intent_reference_cer=intent['cer'] if intent else None)
    return rows, draft_row


def _agg(rows, value, *, samples, seed):
    by_song = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            by_song[r['song_id']].append(v)
    return bootstrap_mean(list(by_song.values()), samples=samples, seed=seed)


def aggregate(rows, draft_rows, *, samples, seed, group_keys=(('split', 'mode', 'stratum'), ('split', 'mode', 'coarse_stratum'),
                                                         ('split', 'mode'))):
    metrics = dict(cer=lambda r: r['cer'], snapped_cer=lambda r: r['snapped_cer'],
                   cer_edits_x0_5=lambda r: r['sensitivity']['edits_x0.5'],
                   cer_edits_x2=lambda r: r['sensitivity']['edits_x2'],
                   density_pearson_1s=lambda r: r['density']['pearson'],
                   draft_over_reference_notes=lambda r: r['density']['count_ratio'])
    silence = any('silence_split' in r for r in rows)
    if silence:
        metrics.update(true_silence_events_per_s=lambda r: (r.get('silence_split') or {}).get('true_silence_events_per_s'),
                       uncharted_music_events_per_s=lambda r: (r.get('silence_split') or {}).get(
                           'uncharted_music_events_per_s'))
    out = {}
    for keys in group_keys:
        groups = defaultdict(list)
        for r in rows:
            groups['/'.join(str(r[k]) for k in keys)].append(r)
        out['by_'+'+'.join(keys)] = {g: dict(rows=len(v), songs=len({r['song_id'] for r in v}),
            **{m: _agg(v, f, samples=samples, seed=seed) for m, f in metrics.items()})
            for g, v in sorted(groups.items())}
    # Difficulty match (W5 note ratio) against the chart the draft was made FOR: one draft per song cannot
    # match every difficulty of a multi-chart song, so the all-references ratio mixes tiers. The primary
    # reference is the hash-chosen chart that sets the song stratum (and, for charter-intent readouts,
    # the requested tier/curve).
    primary_groups = defaultdict(list)
    for r in rows:
        if r.get('primary_reference'):
            primary_groups[f"{r['split']}/{r['mode']}/{r['coarse_stratum']}"].append(r)
    if primary_groups:
        out['primary_target_by_split+mode+coarse_stratum'] = {g: dict(
            rows=len(v), songs=len({r['song_id'] for r in v}),
            draft_over_reference_notes=_agg(v, metrics['draft_over_reference_notes'], samples=samples, seed=seed),
            within_0_7_1_4=float(np.mean([.7 <= r['density']['count_ratio'] <= 1.4 for r in v])),
            density_pearson_1s=_agg(v, metrics['density_pearson_1s'], samples=samples, seed=seed),
            cer=_agg(v, metrics['cer'], samples=samples, seed=seed)) for g, v in sorted(primary_groups.items())}
    per_draft = defaultdict(list)
    for d in draft_rows:
        per_draft[f"{d['split']}/{d['mode']}"].append(d)
    rate_keys = dict(mania_4k=('same_lane_overlap_rate', 'hold_covers_head_rate', 'jack_lt_40ms_rate', 'any_violation_rate',
                               'jack_lt_100ms_rate'),
                     phira=('short_gap_lt_40ms_rate', 'short_gap_nondrag_rate', 'stack_gt4_rate', 'zero_length_hold_rate'))
    out['min_over_references'] = {}
    for g, v in sorted(per_draft.items()):
        mode = v[0]['mode']
        out['min_over_references'][g] = dict(drafts=len(v), songs=len({d['song_id'] for d in v}),
            complete_fraction=float(np.mean([d['complete'] for d in v])),
            min_ref_cer=_agg(v, lambda d: d['min_ref_cer'], samples=samples, seed=seed),
            min_ref_snapped_cer=_agg(v, lambda d: d['min_ref_snapped_cer'], samples=samples, seed=seed),
            playability={k: _agg(v, lambda d, k=k: d['playability'].get(k), samples=samples, seed=seed)
                         for k in rate_keys[mode]})
        if silence:
            splits = [d.get('silence_split') for d in v if d.get('silence_split')]
            seconds = sum(x['uncharted_s'] for x in splits)
            out['min_over_references'][g]['silence_split'] = dict(
                rule='draft notes in regions where the min-CER reference has no note for >= 4 s (100 ms margins); '
                     'true_silence = max mean-log-mel within +-50 ms < song p2 + 1.0, else uncharted_music',
                true_silence_events_per_s=_agg(v, lambda d: (d.get('silence_split') or {}).get(
                    'true_silence_events_per_s'), samples=samples, seed=seed),
                uncharted_music_events_per_s=_agg(v, lambda d: (d.get('silence_split') or {}).get(
                    'uncharted_music_events_per_s'), samples=samples, seed=seed),
                pooled_uncharted_s=seconds,
                pooled_true_silence_events_per_s=sum(x['true_silence_notes'] for x in splits)/seconds if seconds else None,
                pooled_uncharted_music_events_per_s=sum(x['uncharted_music_notes'] for x in splits)/seconds
                if seconds else None)
    if any('pattern' in r for r in rows):
        out['human_likeness_profile'] = profile_aggregate(rows, samples=samples, seed=seed)
    if any('grid' in r for r in rows):
        out['charter_timing'] = charter_aggregate(rows, draft_rows, samples=samples, seed=seed)
    return out


PROFILE_FIELDS = ('draft', 'reference', 'abs_diff', 'js_nats')
CHARTER_RULE = (f"on-grid: distinct onset within {ON_GRID_US//1000} ms of a "
                f"{', '.join(f'1/{k}' for k in CHARTER_DIVISORS)} point of the reference's charter timing; "
                f"charter_timing_snap: heads/tails to the nearest such point within {CHARTER_SNAP_MAX_US//1000} ms "
                "(farther notes kept), then CER and F1@50 ms rescored")


def profile_aggregate(rows, *, samples, seed):
    """Per split/mode and metric: mean over songs (cluster bootstrap) of the draft value, the reference
    value, |draft - reference| and JS, over every (draft, reference) row that defines it."""
    groups = defaultdict(list)
    for r in rows:
        if 'pattern' in r:
            groups[f"{r['split']}/{r['mode']}"].append(r)
    out = {}
    for g, v in sorted(groups.items()):
        mode = v[0]['mode']
        metrics = {}
        for name, _, dist, meaning in PROFILE_METRICS[mode]:
            have = [r for r in v if name in r['pattern']['metrics']]
            if not have:
                continue
            entry = dict(meaning=meaning, rows=len(have))
            for field in PROFILE_FIELDS:
                if field == 'js_nats' and dist is None:
                    continue
                entry[field] = _agg(have, lambda r, f=field: r['pattern']['metrics'][name].get(f),
                                    samples=samples, seed=seed)
            metrics[name] = entry
        out[g] = dict(rows=len(v), songs=len({r['song_id'] for r in v}),
                      rows_with_charter_timing=sum(bool(r['pattern'].get('charter_timing')) for r in v), metrics=metrics)
    return out


def charter_aggregate(rows, draft_rows, *, samples, seed):
    """Grid metrics and the charter_timing_snap readout per split/mode, on rows whose reference has
    charter timing (raw CER / F1 of the same rows reported next to it for a like-for-like delta)."""
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['split']}/{r['mode']}"].append(r)
    by_draft = defaultdict(list)
    for d in draft_rows:
        by_draft[f"{d['split']}/{d['mode']}"].append(d)
    out = {}
    for g, v in sorted(groups.items()):
        have = [r for r in v if r.get('grid')]
        if not have:
            continue
        a = lambda f: _agg(have, f, samples=samples, seed=seed)  # noqa: E731
        out[g] = dict(label='charter_timing_snap: draft snapped to the REFERENCE chart\'s own BPM/offset (charter '
                            'timing, a legitimate workflow input) -- never a capability number',
            rows=len(v), rows_with_charter_timing=len(have), songs_with_charter_timing=len({r['song_id'] for r in have}),
            draft_on_grid_fraction=a(lambda r: r['grid']['draft_on_grid_fraction']),
            reference_on_grid_fraction=a(lambda r: r['grid']['reference_on_grid_fraction']),
            chance_on_grid_fraction=a(lambda r: r['grid'].get('chance_on_grid_fraction')),
            on_grid_abs_diff=a(lambda r: r['grid']['on_grid_abs_diff']),
            subdivision_js_nats=a(lambda r: r['grid']['subdivision_js_nats']),
            raw_cer=a(lambda r: r['cer']), charter_snap_cer=a(lambda r: r['charter_timing_snap']['cer']),
            raw_note_f1_50ms=a(lambda r: r['charter_timing_snap']['raw_note_f1_50ms']),
            charter_snap_note_f1_50ms=a(lambda r: r['charter_timing_snap']['note_f1_50ms']),
            raw_onset_f1_50ms=a(lambda r: r['charter_timing_snap']['raw_onset_f1_50ms']),
            charter_snap_onset_f1_50ms=a(lambda r: r['charter_timing_snap']['onset_f1_50ms']),
            min_ref_charter_snap_cer=_agg(by_draft.get(g, []), lambda d: d.get('min_ref_charter_snap_cer'),
                                          samples=samples, seed=seed),
            rule=CHARTER_RULE)
    return out


def human_playability_baseline(root, manifest):
    """Reference-free validity rates of every human chart on the tape, per split and mode."""
    acc = defaultdict(lambda: defaultdict(list))
    for c in manifest['charts']:
        notes = reference_notes(root, dict(target_path=c['target_path'], target_sha256=c['target_sha256']), c['mode'])
        p = playability(notes, c['mode'])
        acc[f"{c['split']}/{c['mode']}"]['charts'].append(p)
    out = {}
    for g, v in sorted(acc.items()):
        charts = v['charts']
        mode = g.split('/')[1]
        if mode == 'mania_4k':
            counts = ('same_lane_overlap', 'hold_covers_head', 'jack_lt_40ms')
            denom = sum(p['notes'] for p in charts)
            pooled = {k+'_rate': sum(p[k] for p in charts)/denom for k in counts}
        else:
            gaps = sum(max(p['onset_groups']-1, 0) for p in charts)
            pooled = dict(short_gap_lt_40ms_rate=sum(p['short_gap_lt_40ms'] for p in charts)/gaps,
                          short_gap_nondrag_rate=sum(p['short_gap_nondrag'] for p in charts)/gaps,
                          stack_gt4_rate=sum(p['stack_gt4'] for p in charts)/sum(p['onset_groups'] for p in charts),
                          zero_length_hold_rate=sum(p['zero_length_hold'] for p in charts)/max(1, sum(p['holds'] for p in charts)))
        keys = list(pooled)
        out[g] = dict(charts=len(charts), pooled=pooled,
                      chart_mean={k: float(np.mean([p[k] for p in charts if p[k] is not None])) for k in keys},
                      charts_with_any={k: float(np.mean([bool(p[k]) for p in charts if p[k] is not None])) for k in keys})
    return out


# ---------------------------------------------------------------- drafts
def near_duplicate(a, b, mode):
    return min(chart_edit_rate(a, b, mode)['cer'], chart_edit_rate(b, a, mode)['cer']) < NEAR_DUPLICATE_CER


def human_drafts(root, manifest, raw_root, splits, per_op, keep_duplicates=False, clock_shifts=None):
    """Every human chart as a draft against the other charts of its song.

    Near-identical re-uploads (CER < NEAR_DUPLICATE_CER either way) would fake a
    zero floor, so each later twin is dropped (listed in the report) unless keep_duplicates.
    """
    rows, draft_rows, grids, timing, duplicates, used = [], [], {}, {}, [], 0
    groups = {k: v for k, v in song_groups(manifest['charts'], splits).items() if len(v) > 1}
    for (audio, mode), charts in sorted(groups.items()):
        song = dict(song_id=f'{mode}:{audio[:16]}', mode=mode, duration_us=int(charts[0]['duration_us']))
        refs = []
        for c in charts:
            notes = reference_notes(root, c, mode)
            if not keep_duplicates:
                twin = next((b['chart_id'] for b in refs if near_duplicate(notes, b['notes'], mode)), None)
                if twin is not None:
                    duplicates.append(dict(chart_id=c['chart_id'], duplicate_of=twin, song_id=song['song_id']))
                    continue
            grid, info = reference_grid(raw_root, c['chart_id'], notes, clock_shifts=clock_shifts)
            grids[c['chart_id']], timing[c['chart_id']] = grid, info
            refs.append(dict(chart_id=c['chart_id'], stratum=c['stratum'],
                             coarse_stratum=coarse_stratum(mode, c['stratum']), notes=notes))
        if len(refs) < 2:
            continue
        used += 1
        for a in refs:
            others = [b for b in refs if b is not a]
            r, d = evaluate(a['notes'], dict(draft_id=a['chart_id'], stratum=a['stratum'],
                                             coarse_stratum=a['coarse_stratum']),
                            song, others, grids=grids, per_op=per_op, split=charts[0]['split'])
            rows += r
            draft_rows.append(d)
    return rows, draft_rows, timing, used, duplicates


def panel_references(root, panel, raw_root, clock_shifts=None):
    refs, grids, timing = {}, {}, {}
    for song in panel['songs']:
        items = []
        for ref in song['references']:
            notes = reference_notes(root, ref, song['mode'])
            grid, info = reference_grid(raw_root, ref['chart_id'], notes, clock_shifts=clock_shifts)
            grids[ref['chart_id']], timing[ref['chart_id']] = grid, info
            items.append(dict(ref, notes=notes))
        refs[song['song_id']] = items
    return refs, grids, timing


HUMAN_FLOOR_LABEL = ('Human-vs-human floor on the SAME panel: every reference of a multi-chart panel song as the '
                     'draft against the song\'s other references (near-identical re-uploads dropped). Shown next to '
                     'the model in the human-likeness profile / charter-timing tables; not a model result.')


def panel_human_floor(panel, refs, grids, *, samples, seed):
    """Pattern / grid / charter_timing_snap floor of the panel's own human charts (no model involved)."""
    rows, draft_rows, dropped, used = [], [], [], 0
    for song in panel['songs']:
        mode, kept = song['mode'], []
        for ref in refs[song['song_id']]:
            twin = next((b['chart_id'] for b in kept if near_duplicate(ref['notes'], b['notes'], mode)), None)
            if twin is not None:
                dropped.append(dict(chart_id=ref['chart_id'], duplicate_of=twin, song_id=song['song_id']))
            else:
                kept.append(ref)
        if len(kept) < 2:
            continue
        used += 1
        for a in kept:
            r, d = evaluate(a['notes'], dict(draft_id=a['chart_id'], stratum=a['stratum'],
                                             coarse_stratum=a['coarse_stratum']),
                            song, [b for b in kept if b is not a], grids=grids, per_op=False, split=panel['split'])
            rows += r
            draft_rows.append(d)
    agg = aggregate(rows, draft_rows, samples=samples, seed=seed) if rows else {}
    return dict(label=HUMAN_FLOOR_LABEL, songs=used, pairs=len(rows), dropped_near_duplicates=dropped,
                aggregate={k: agg[k] for k in ('human_likeness_profile', 'charter_timing', 'by_split+mode') if k in agg})


def null_drafts(root, panel, raw_root, per_op, clock_shifts=None):
    refs, grids, timing = panel_references(root, panel, raw_root, clock_shifts=clock_shifts)
    rows, draft_rows = [], []
    for song in panel['songs']:
        r, d = evaluate([], dict(draft_id='empty'), song, refs[song['song_id']], grids=grids, per_op=per_op, split=panel['split'])
        rows += r
        draft_rows.append(d)
        for ref in refs[song['song_id']]:
            for shift in SHIFTS_MS:
                draft = [(a+shift*1000, b+shift*1000, t, l) for a, b, t, l in ref['notes']]
                r, d = evaluate(draft, dict(draft_id=f'self+{shift}ms', stratum=ref['stratum']), song, [ref],
                                grids=grids, per_op=per_op, split=panel['split'])
                rows += r
                draft_rows.append(d)
    return rows, draft_rows, timing


def make_generator(temperature, top_p, seed, field_temperature=None, decode_options=None, *, onset_fusion=None,
                   guidance=None, stats=None, on_step=None):
    """Wrapper around full_event_generate.generate_batch with per-row seeds (batch-composition independent).

    ``row_seeds`` (passed by local_song_rollout.generate_complete_song_samples) gives each row its own
    sample seed; the per-row generator seed is the same function of (seed, mode, duration, core start).
    ``decode_options`` (e.g. fast=False / cuda_graph=False) are forwarded only when set; so are
    ``onset_fusion`` (weight -> generate_batch onset_fusion=dict(weight=..)) and ``guidance`` (CFG
    weight, applied only to batches for which the rollout passes ``unconditional_inputs``). ``stats`` (dict)
    accumulates the decode counters over all batches. ``generator.continuous`` / ``generator.row_generators`` are
    the same decoding settings and the same seed function for scheduler='continuous'."""
    import inspect
    import torch
    options = dict(decode_options or {})
    if onset_fusion:
        options['onset_fusion'] = dict(weight=float(onset_fusion))

    def row_generators(seeds, requests, device):
        gen_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
        return [torch.Generator(device=gen_device).manual_seed(int(stable_rank(
            s, r.mode, r.duration_us, r.start_us)[:15], 16)) for s, r in zip(seeds, requests)]

    def generator(model, inputs, requests, max_tokens, *, deadline=None, row_seeds=None, unconditional_inputs=None,
                  memory_index=None):
        from beatfcn.model.full_event_generate import generate_batch, merge_stats
        extra = (dict(guidance=float(guidance), unconditional_inputs=unconditional_inputs)
                 if guidance is not None and unconditional_inputs is not None else {})
        if memory_index is not None:
            extra['memory_index'] = memory_index
        if on_step is not None:
            extra['on_step'] = on_step
        batch_stats = {}
        if stats is not None:
            extra['stats'] = batch_stats
        if temperature is None:
            out = generate_batch(model, inputs, requests, max_tokens, deadline=deadline, **options, **extra)
        else:
            if 'temperature' not in inspect.signature(generate_batch).parameters:
                raise RuntimeError('generate_batch has no sampling support; use --temperature none (greedy)')
            seeds = row_seeds if row_seeds is not None else [seed] * len(requests)
            gens = row_generators(seeds, requests, inputs['local'].device)
            if field_temperature:
                extra['field_temperature'] = field_temperature
            out = generate_batch(model, inputs, requests, max_tokens, deadline=deadline,
                                 temperature=float(temperature), top_p=top_p, generator=gens, **options, **extra)
        if stats is not None:
            merge_stats(stats, batch_stats)
        return out
    generator.row_generators = row_generators
    if options.get('fast', True):
        generator.continuous = dict({k: v for k, v in options.items() if k != 'fast'},
                                    temperature=None if temperature is None else float(temperature), top_p=top_p,
                                    field_temperature=field_temperature, guidance=guidance, stats=stats,
                                    on_step=on_step)
    return generator


def decode_options(args):
    """generate_batch speed switches from the CLI (empty = module defaults: fast path, CUDA graphs on CUDA)."""
    options = {}
    if getattr(args, 'reference_decode', False):
        options['fast'] = False
    if getattr(args, 'no_cuda_graph', False):
        options['cuda_graph'] = False
    if getattr(args, 'autocast', None):
        options['autocast'] = args.autocast
    if getattr(args, 'compile_decode', False):
        options['compile_decode'] = True
    if getattr(args, 'kind_bias', None):
        options['kind_bias'] = args.kind_bias
    return options


def onset_probe(model):
    """Per-core onset-head probabilities for the reference-free reranker, or None without an onset head.

    Called through ``generate_complete_songs(on_core_inputs=...)`` with the exact
    device inputs of each batch (no duplicated input construction). Channel 0 of
    ``memory['onset_logits']`` is "any onset" (sigmoid/BCE head); times are
    seconds relative to the core start. Frames outside [0, core length) or
    invalid are dropped. Inputs are audio/request only -- never a target.
    """
    config = getattr(model, 'c', None)
    if not getattr(config, 'onset_head', False):
        return None
    import torch

    @torch.no_grad()
    def probe(model, inputs, requests, states):
        memory = model.encode(inputs)
        logits = memory['onset_logits'].float()
        if logits.ndim == 3:
            logits = logits[..., 0]
        probs = torch.sigmoid(logits).cpu().numpy()
        times = memory['onset_times'].float().cpu().numpy()
        valid = memory['onset_valid'].bool().cpu().numpy()
        out = []
        for row, request in enumerate(requests):
            span_s = (request.end_us-request.start_us)/1e6
            keep = valid[row] & (times[row] >= 0) & (times[row] < span_s)
            out.append(dict(onset_probs=probs[row][keep].astype(float).tolist(),
                            onset_times_s=times[row][keep].astype(float).tolist()))
        return out
    return probe


def parse_temperature(value):
    return None if value in (None, 'none', 'greedy') else float(value)


def rollout_kwargs(args, probe=None, condition_provider=None):
    """Opt-in rollout switches; empty unless enabled (historical calls stay byte-identical)."""
    kwargs = dict(on_core_inputs=probe) if probe is not None else {}
    if condition_provider is not None:
        kwargs['condition_provider'] = condition_provider
    if getattr(args, 'timing_provider', None) is not None:
        kwargs['timing_provider'] = args.timing_provider
    if getattr(args, 'alternative_provider', None) is not None:
        kwargs['alternative_provider'] = args.alternative_provider
    return kwargs


def charter_timing_provider(table, panel):
    """local_song_rollout timing_provider: the charter timing of each song's PRIMARY (scored) reference
    (table segments are already on the model clock); None (absent state) without valid timing."""
    chart_of = {song['song_id']: primary_reference(song)['chart_id'] for song in panel['songs']}

    def provider(state, request):
        return table.segments(chart_of[state['chart_id']])
    coverage = dict(songs=len(chart_of), with_timing=sum(table.segments(c) is not None for c in chart_of.values()),
                    by_status=dict(sorted({st: sum(table.status(c) == st for c in chart_of.values())
                                           for st in {table.status(c) for c in chart_of.values()}}.items())))
    return provider, chart_of, coverage


def draft_generator(args, temperature, seed):
    progress = getattr(args, 'progress', None)
    return make_generator(temperature, args.top_p, seed, getattr(args, 'field_temperature', None),
                          decode_options(args), onset_fusion=getattr(args, 'onset_fusion', None),
                          guidance=cfg_weight(args), stats=getattr(args, 'decode_stats', None),
                          on_step=None if progress is None else progress.tick)


def cfg_weight(args):
    """CFG weight passed to generate_batch: None without --density-request or for W == 1 (plain conditioning)."""
    weight = getattr(args, 'guidance', None)
    if weight is None or not (getattr(args, 'density_request', None) or getattr(args, 'control_band', None)) \
            or float(weight) == 1.:
        return None
    return float(weight)


def remaining_wall_s(args):
    """Seconds of decoding left. One absolute deadline for the process (--max-wall-s minus --flush-reserve-s), so
    the time to save always exists; without ``process_start`` (library callers) --max-wall-s is per call."""
    start = getattr(args, 'process_start', None)
    if start is None:
        return args.max_wall_s
    return max(0., start + args.max_wall_s - args.flush_reserve_s - time.monotonic())


def finished_pair(state):
    """A pair worth keeping on --resume. Wall-limit failures are retried; token-cap / legality ones would repeat."""
    return bool(state['complete']) or state['failure'] in ('core_token_limit', 'independent_legality')


def load_samples(path, *, repair=False):
    """{(seed, chart_id): state} of a samples.jsonl (last line of a pair wins). A torn last line (killed process) is
    ignored; ``repair`` also cuts it off so that appending stays valid."""
    data = Path(path).read_bytes()
    keep = data.rfind(b'\n') + 1
    if repair and keep != len(data):
        with open(path, 'r+b') as handle:
            handle.truncate(keep)
    out = {}
    for line in data[:keep].splitlines():
        record = json.loads(line)
        out[record['seed'], record['state']['chart_id']] = record['state']
    return out


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, default=_json_default) + '\n')
    os.replace(tmp, path)


class ProgressWriter:
    """Incremental results of one run, so power-off or a kill loses at most the rows in flight.

    ``samples.jsonl``: one fsynced line per finished (seed, song), also the partial pairs of a wall limit.
    ``progress.json``: rewritten atomically at least every 30 s with phase (decode, score, write, done), cores,
    tokens, rate and eta. In decode only the decoding thread writes it, so a stale file means a stalled decode; other
    phases (chart reading, scoring) have a heartbeat thread."""

    def __init__(self, out, songs, seeds, *, completed=None, interval_s=20.):
        self.dir, self.interval_s = Path(out), interval_s
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.total = sum(math.ceil(song['duration_us']/8_000_000) for song in songs) * len(seeds)
        self.pairs_total = len(songs) * len(seeds)
        self.written = {key for key, state in (completed or {}).items() if finished_pair(state)}
        self.seen = {key: len(state['cores']) for key, state in (completed or {}).items()}
        states = (completed or {}).values()
        self.base_cores = self.cores = sum(len(state['cores']) for state in states)
        self.base_tokens = self.tokens = sum(len(core['tokens']) for state in states for core in state['cores'])
        self.pairs = len(self.written)
        self.phase_name, self.done, self.phase_total = 'decode', 0, 0
        self.started = self.phase_started = self.last = time.monotonic()
        self.samples = None

    def __call__(self, state, seed):
        key = seed, state['chart_id']
        new = state['cores'][self.seen.get(key, 0):]
        self.seen[key] = len(state['cores'])
        self.cores += len(new)
        self.tokens += sum(len(core['tokens']) for core in new)
        if (state['complete'] or state['failure'] is not None) and key not in self.written:
            self.written.add(key)
            self.pairs += finished_pair(state)
            if self.samples is None:
                self.samples = open(self.dir/'samples.jsonl', 'a')
            self.samples.write(json.dumps(dict(seed=seed, state=state), default=_json_default) + '\n')
            self.samples.flush()
            os.fsync(self.samples.fileno())
        self.tick()

    def snapshot(self):
        elapsed = time.monotonic() - self.phase_started
        if self.phase_name == 'decode':
            done, total, base = self.cores, self.total, self.base_cores
        else:
            done, total, base = self.done, self.phase_total, 0
        rate = (done - base) / elapsed if elapsed > 0 else 0.
        return dict(phase=self.phase_name, done=done, total=total, cores_done=self.cores, cores_total=self.total,
                    pairs_done=self.pairs, pairs_total=self.pairs_total, tokens=self.tokens,
                    tok_s=(self.tokens - self.base_tokens) / max(time.monotonic() - self.started, 1e-9),
                    eta_s=(total - done) / rate if rate > 0 else None, elapsed_s=time.monotonic() - self.started,
                    at=time.time())

    def write(self):
        with self.lock:
            self.last = time.monotonic()
            snapshot = self.snapshot()
            write_json_atomic(self.dir/'progress.json', snapshot)
        print('progress ' + json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in snapshot.items()
                                        if k != 'at'}), flush=True)

    def tick(self):
        if time.monotonic() - self.last >= self.interval_s:
            self.write()

    def _heartbeat(self):
        while not self.stop.wait(self.interval_s):
            self.write()

    def phase(self, name, *, total=0):
        first_other = self.phase_name == 'decode' and name != 'decode'
        self.phase_name, self.done, self.phase_total, self.phase_started = name, 0, total, time.monotonic()
        self.write()
        if first_other:
            threading.Thread(target=self._heartbeat, daemon=True).start()

    def advance(self):
        self.done += 1
        self.tick()

    def close(self):
        self.stop.set()
        if self.samples is not None:
            self.samples.close()
            self.samples = None
        self.phase_name = 'done'
        self.write()


def model_drafts(args, panel, model, features, *, intensity_dim, cell_us, seed=None, probe=None,
                 condition_provider=None, progress=None, completed=None):
    from beatfcn.model.local_song_rollout import generate_complete_songs
    seed = args.seed if seed is None else seed
    songs = [dict(chart_id=s['song_id'], audio_sha256=s['audio_sha256'], mode=s['mode'], duration_us=s['duration_us'])
             for s in panel['songs']]
    provider = None
    if intensity_dim:
        from beatfcn.train.density_curve import grid_density_curve
        provider = lambda audio, start, end: grid_density_curve([], start, end, cell_us=cell_us, scale=16.)  # noqa: E731
    temperature = parse_temperature(args.temperature)
    kwargs = rollout_kwargs(args, probe, condition_provider)
    if progress is not None:
        kwargs['on_progress'] = lambda state: progress(state, seed)
    if completed:
        kwargs['completed'] = {chart: state for (sample, chart), state in completed.items() if sample == seed}
    result = generate_complete_songs(model, features, songs, device=args.device, batch_size=args.batch_size,
        token_cap=args.token_cap, max_wall_s=remaining_wall_s(args),
        history_reference=bool(getattr(args, 'history_reference', False)),
        generator=draft_generator(args, temperature, seed), intensity_provider=provider, **kwargs)
    return result, draft_decoding(args, temperature, seed)


def draft_decoding(args, temperature, seed):
    out = dict(rule='greedy' if temperature is None else 'ancestral', temperature=temperature, top_p=args.top_p,
               field_temperature=getattr(args, 'field_temperature', None),
               seed=seed, rng='per-row torch.Generator seeded from (seed, mode, duration, core start)',
               token_cap_per_core=args.token_cap, conditioning='unconditional controls + unspecified density curve')
    # Opt-in switches are recorded only when enabled (default runs keep the historical metadata).
    if getattr(args, 'autocast', None):
        out['autocast'] = args.autocast
    if getattr(args, 'history_reference', False):
        out['history_reference'] = dict(enabled=True, role='model_history', window_s=8,
            rule="each core after the first sees the model's OWN previous 8 s of draft as a VisibleReference")
    if getattr(args, 'onset_fusion', None):
        out['onset_fusion'] = dict(weight=float(args.onset_fusion), window_ms=20, floor=1e-4,
                                   fields=['head_block', 'head_cell', 'head_ms'])
    if getattr(args, 'charter_timing', None):
        out['charter_timing_input'] = dict(getattr(args, 'charter_timing_identity', None) or {},
            label='charter_timing_input', reference='primary reference (freeze_panel hash-chosen chart)',
            rule='per core: beat features of the primary reference chart timing; absent state without valid timing')
    if getattr(args, 'consensus_min_jack_ms', None) is not None:
        out['consensus_min_jack_ms'] = float(args.consensus_min_jack_ms)
    if getattr(args, 'alternative_reference', False):
        out['alternative_reference'] = dict(label='charter_alternative_chart', role='human_alternative', pad_s=2.0,
                                            rule="per core the song's sibling chart (the charter's own denser chart)")
    if getattr(args, 'control_band', None):
        out['style_request'] = dict(label='charter_intent_style', bands=list(args.control_band),
                                    guidance=None if getattr(args, 'guidance', None) is None else float(args.guidance),
                                    cfg='off (plain conditioning)' if cfg_weight(args) is None else
                                        'logits_u + w*(logits_c - logits_u); unconditional = unspecified curve + controls')
    if getattr(args, 'density_request', None):
        weight = getattr(args, 'guidance', None)
        out['conditioning'] = ('charter_intent_density: per core the 2 s achieved-density curve of the song\'s '
                               'primary human reference (target-derived stand-in for the charter\'s intent; never '
                               'an unconditioned-capability number) + unconditional controls')
        out['density_request'] = dict(source=args.density_request, label='charter_intent_density',
            reference='primary reference (freeze_panel hash-chosen chart that sets the song stratum)',
            curve_cell_us=2_000_000, render_cell_us='plan counterfactual_cell_us',
            guidance=None if weight is None else float(weight),
            cfg='off (plain conditioning)' if cfg_weight(args) is None else
                'logits_u + w*(logits_c - logits_u); unconditional = unspecified curve + controls')
    return out


def model_drafts_parallel(args, panel, model, features, *, intensity_dim, cell_us, seeds, probe=None,
                          condition_provider=None, progress=None, completed=None):
    """All sample seeds in one rollout (rows = seeds x songs); per-row generators make each sample identical
    to a separate model_drafts call with that seed. The wall limit is the absolute process deadline."""
    from beatfcn.model.local_song_rollout import generate_complete_song_samples
    songs = [dict(chart_id=s['song_id'], audio_sha256=s['audio_sha256'], mode=s['mode'], duration_us=s['duration_us'])
             for s in panel['songs']]
    provider = None
    if intensity_dim:
        from beatfcn.train.density_curve import grid_density_curve
        provider = lambda audio, start, end: grid_density_curve([], start, end, cell_us=cell_us, scale=16.)  # noqa: E731
    temperature = parse_temperature(args.temperature)
    kwargs = rollout_kwargs(args, probe, condition_provider)
    results = generate_complete_song_samples(model, features, songs, seeds, device=args.device,
        batch_size=args.batch_size, token_cap=args.token_cap, max_wall_s=remaining_wall_s(args),
        history_reference=bool(getattr(args, 'history_reference', False)), intensity_provider=provider, **kwargs,
        generator=draft_generator(args, temperature, seeds[0]), on_progress=progress, completed=completed,
        scheduler=getattr(args, 'scheduler', 'legacy'))
    return [(result, draft_decoding(args, temperature, seed)) for seed, result in zip(seeds, results)]


def sample_pool(args, panel, model, features, *, intensity_dim, cell_us, best_of, condition_provider=None,
                rerank=None):
    """N complete whole-song samples per song (seeds seed..seed+N-1, same temperature/top_p).

    Sample 0 (seed ``--seed``) is the single-sample capability readout. The onset
    probe is attached only when the reranker will run (``rerank``, default N > 1)."""
    if best_of < 1:
        raise ValueError('--best-of must be >= 1')
    if best_of > 1 and parse_temperature(args.temperature) is None:
        raise ValueError('--best-of > 1 needs sampling (--temperature > 0); greedy samples are identical')
    probe = onset_probe(model) if (best_of > 1 if rerank is None else rerank) else None
    samples = []
    resume = dict(progress=getattr(args, 'progress', None), completed=getattr(args, 'completed', None))
    if (getattr(args, 'parallel_samples', False) and best_of > 1) or getattr(args, 'scheduler', 'legacy') != 'legacy':
        seeds = [args.seed+k for k in range(best_of)]
        for seed, (result, decoding) in zip(seeds, model_drafts_parallel(
                args, panel, model, features, intensity_dim=intensity_dim, cell_us=cell_us, seeds=seeds, probe=probe,
                condition_provider=condition_provider, **resume)):
            samples.append(dict(seed=seed, result=result, decoding=decoding,
                                by_id={st['chart_id']: st for st in result['songs']}))
        return samples, probe is not None
    for k in range(best_of):
        result, decoding = model_drafts(args, panel, model, features, intensity_dim=intensity_dim, cell_us=cell_us,
                                        seed=args.seed+k, probe=probe, condition_provider=condition_provider, **resume)
        samples.append(dict(seed=args.seed+k, result=result, decoding=decoding,
                            by_id={st['chart_id']: st for st in result['songs']}))
    return samples, probe is not None


def draft_tuples(song, state):
    return [(int(e['onset_us']), int(e['end_us']), e['type'], int(e['lane']) if song['mode'] == 'mania_4k' else -1)
            for e in state['events']]


def draft_record(song, state, seed):
    return dict(mode=song['mode'], audio_sha256=song['audio_sha256'], seed=seed, complete=state['complete'],
                failure=state['failure'], completed_until_us=state['next_start_us'], duration_us=song['duration_us'],
                legality=state.get('legality', []), events=state['events'],
                core_tokens=[len(c['tokens']) for c in state['cores']])


def rerank_selection(pool, songs, *, weights=None):
    """Reference-free per-song pick over the sample pool.

    ``pool``: list over samples of {song_id: rollout state}; ``songs``: panel song
    descriptors (only ``song_id`` is read). Never touches references/targets --
    the unit tests make every target accessor raise while this runs."""
    from beatfcn.model.rerank import select_best_song
    out = {}
    for song in songs:
        sid = song['song_id']
        best, scores, parts = select_best_song([sample[sid] for sample in pool], weights=weights)
        out[sid] = dict(index=best, scores=scores, parts=parts)
    return out


def oracle_selection(evaluated, songs):
    """ORACLE: per song the sample with the lowest min-over-references CER vs the HUMAN charts (ties: lowest seed)."""
    out = {}
    for song in songs:
        sid = song['song_id']
        cers = [ev[sid][1]['min_ref_cer'] for ev in evaluated]
        out[sid] = dict(index=int(np.argmin(cers)), min_ref_cers=cers)
    return out


# ------------------------------------------------------------ D6 consensus
CONSENSUS_WINDOW_US = 30_000


def _lower_median(values):
    values = sorted(values)
    return values[(len(values)-1)//2]


def _onset_groups(notes, mode):
    """One sample's notes -> [(onset, pattern, tails)] per distinct onset.

    mania: pattern = ((lane, type), ...) sorted by lane, tails = {lane: end};
    phira: pattern = sorted type multiset, tails = sorted hold ends."""
    by_onset = defaultdict(list)
    for n in notes:
        by_onset[int(n[0])].append(n)
    out = []
    for onset in sorted(by_onset):
        members = by_onset[onset]
        if mode == 'mania_4k':
            pattern = tuple(sorted((int(n[3]), n[2]) for n in members))
            tails = {int(n[3]): int(n[1]) for n in members if n[2] == 'hold'}
        else:
            pattern = tuple(sorted(n[2] for n in members))
            tails = sorted(int(n[1]) for n in members if n[2] == 'hold')
        out.append((onset, pattern, tails))
    return out


UNPLAYABLE_REPEAT_US = 40_000


def _without_weak_fragments(clusters, need, half):
    """Drop a kept onset that sits alone in a squeezed gap (a gap under .6 of the local median and under 200 ms, with
    neighbouring gaps that are not short) when fewer than half of the samples wrote it. Fast runs stay, and so do squeezes
    most samples agree on; a lone squeeze few samples wrote reads as noise (2026-11-05 owner playtest)."""
    kept = [c for c in clusters if len(c) >= need]
    if len(kept) < 20:
        return clusters
    t = [sorted(m[0] for m in c)[(len(c)-1)//2] for c in kept]
    gaps = [b-a for a, b in zip(t, t[1:])]
    med = [sorted(gaps[max(0, i-8):i+9])[len(gaps[max(0, i-8):i+9])//2] for i in range(len(gaps))]
    short = [g < .6*m and g < 200_000 for g, m in zip(gaps, med)]
    drop = set()
    for i, s in enumerate(short):
        lone = s and not (i > 0 and short[i-1]) and not (i+1 < len(short) and short[i+1])
        if lone:
            a, b = kept[i], kept[i+1]
            weaker = i if len(a) <= len(b) else i+1
            if len(kept[weaker]) < half:
                drop.add(weaker)
    return [c for i, c in enumerate(kept) if i not in drop]


def consensus_draft(samples, mode, duration_us, *, window_us=CONSENSUS_WINDOW_US, min_jack_us=None,
                    pattern_source='majority', min_votes=None, onset_source='clusters', fragment_guard=False):
    """D6 MBR-style consensus of N whole-song drafts (lists of (onset, end, type, lane) tuples).

    1. Every sample's distinct onsets (mania: lane-agnostic onset = one chord) are pooled
       and swept in time order (ties: lower sample index); a cluster starts at its first
       onset and takes later onsets within ``window_us`` of that first onset, at most one
       per sample (a second onset of the same sample starts the next cluster).
    2. Clusters with members from >= ``min_votes`` (default ceil(N/2)) samples are kept; onset = lower median of
       the member onsets (always a sampled ms). A kept cluster whose onset is not after
       the previous kept onset is skipped (cannot happen unless windows touch).
    3. Pattern (mania: the lane->type chord; phira: the type multiset) = majority among
       members; ties -> the pattern of the earliest sample. Hold tail = lower median of the
       tails of the members with that pattern (per lane / per sorted hold rank), clamped
       to the song; a hold whose tail is not after the onset becomes a tap.
    4. mania: a hold that would contain a later same-lane head is dropped (its head kept
       as a tap), so the draft has no same-lane overlap / head-inside-hold.
    5. ``min_jack_us`` (mania, default None = off): per-cluster majority ignores the neighbours,
       so samples that alternate keys with opposite parity vote same-key repeats no sample
       made (consensus 8 had 2.6x the single-sample rate of same-key repeats < 100 ms). With it
       set, the pattern is the most-voted member pattern that presses no key pressed less than
       ``min_jack_us`` earlier; if every candidate does, the majority is kept (``jack_relaxed``).
    6. ``pattern_source='anchor'``: the onsets are still the majority clusters, but the PATTERN of a kept cluster
       is the anchor sample's (the medoid: most kept-cluster members sharing its patterns) whenever the
       anchor has a member there; majority otherwise. Per-cluster majority stitches lanes from different samples
       (same-key repeats no sample made, rolls broken) and splits hold-vs-tap votes (LNs lost); one sample's
       patterns keep one coherent hand flow. ``min_jack_us`` still applies (the anchor pattern is ranked first).
    7. ``onset_source='anchor'`` (needs pattern_source='anchor'): only clusters holding the anchor sample survive and they
       keep its own onset, so the rhythm is one sample's and the vote only removes its unsupported notes. Pooling onsets
       of different samples stitches different rhythms into bursts no sample wrote.
    Returns (notes, info). Never reads a human chart.
    """
    n = len(samples)
    if n < 1:
        raise ValueError('consensus needs >= 1 sample')
    need = math.ceil(n/2) if min_votes is None else int(min_votes)
    items = sorted((onset, k, pattern, tails) for k, notes in enumerate(samples)
                   for onset, pattern, tails in _onset_groups(notes, mode))
    clusters, current = [], None
    for item in items:
        if current is not None and item[0]-current[0][0] <= window_us and item[1] not in {m[1] for m in current}:
            current.append(item)
        else:
            current = [item]
            clusters.append(current)
    if pattern_source not in ('majority', 'anchor'):
        raise ValueError(f'pattern_source {pattern_source!r}')
    if onset_source == 'anchor' and pattern_source != 'anchor':
        raise ValueError('onset_source anchor needs pattern_source anchor')
    anchor, from_anchor = None, 0
    if pattern_source == 'anchor':
        # medoid: per kept cluster a sample scores the number of members sharing its pattern (presence x agreement)
        score = defaultdict(int)
        for members in clusters:
            if len(members) >= need:
                votes = defaultdict(int)
                for m in members:
                    votes[m[2]] += 1
                for m in members:
                    score[m[1]] += votes[m[2]]
        anchor = min(score, key=lambda k: (-score[k], k)) if score else None
    if fragment_guard:
        clusters = _without_weak_fragments(clusters, need, math.ceil(n/2))
    notes, kept, skipped, last_onset, dropped, short = [], 0, 0, None, 0, 0
    lane_hold = {}  # lane -> index in notes of the active hold
    last_press, rechosen, relaxed, relaned, dropped_jack = {}, 0, 0, 0, 0
    for members in clusters:
        if len(members) < need:
            continue
        if onset_source == 'anchor':
            own = [m for m in members if m[1] == anchor]
            if not own:
                continue
            onset = own[0][0]
        else:
            onset = _lower_median([m[0] for m in members])
        if last_onset is not None and onset <= last_onset or not 0 <= onset < duration_us:
            skipped += 1
            continue
        counts = defaultdict(int)
        for m in members:
            counts[m[2]] += 1
        best = max(counts.values())
        pattern = min((m for m in members if counts[m[2]] == best), key=lambda m: m[1])[2]
        anchor_pattern = next((m[2] for m in members if m[1] == anchor), None) if anchor is not None else None
        if anchor_pattern is not None:
            pattern, from_anchor = anchor_pattern, from_anchor + 1
        tail_source = None  # mania re-laned keys: lane -> the voted lane whose tails it carries
        if mode == 'mania_4k' and min_jack_us:
            first_vote = {}
            for m in members:
                first_vote.setdefault(m[2], m[1])
            ranked = sorted(counts, key=lambda pat: (pat != anchor_pattern, -counts[pat], first_vote[pat]))
            ok = [pat for pat in ranked if all(onset - last_press.get(lane, -10**12) >= min_jack_us for lane, _ in pat)]
            if ok and ok[0] != pattern:
                pattern, rechosen = ok[0], rechosen + 1
            elif not ok:
                # No voted pattern avoids the repeat: move each offending key to the free key pressed longest
                # ago (not in the chord, not held at this onset, not itself a repeat); if none, drop that
                # member of a multi-key chord; a lone unmovable note is kept (counted as relaxed).
                busy = {lane for lane, _ in pattern}
                held = {lane for lane, i in lane_hold.items() if notes[i][1] > onset}
                fixed, tail_source, voted = [], {}, pattern
                for lane, kind in sorted(pattern):
                    if onset - last_press.get(lane, -10**12) >= min_jack_us:
                        fixed.append((lane, kind))
                        tail_source[lane] = lane
                        continue
                    free = [l for l in range(4) if l not in busy and l not in held
                            and onset - last_press.get(l, -10**12) >= min_jack_us]
                    if free:
                        new = min(free, key=lambda l: (last_press.get(l, -10**12), l))
                        busy.discard(lane)
                        busy.add(new)
                        fixed.append((new, kind))
                        tail_source[new] = lane
                        relaned += 1
                    elif len(pattern) > 1:
                        busy.discard(lane)
                        dropped_jack += 1
                    else:
                        fixed.append((lane, kind))
                        tail_source[lane] = lane
                        relaxed += 1
                pattern = tuple(sorted(fixed))
        voters = [m for m in members if m[2] == (voted if tail_source is not None else pattern)]
        if mode == 'mania_4k':
            group = []
            for lane, kind in pattern:
                end = onset
                if kind == 'hold':
                    src = lane if tail_source is None else tail_source[lane]
                    end = min(_lower_median([m[3][src] for m in voters]), int(duration_us))
                    if end <= onset:
                        kind, end, short = 'tap', onset, short+1
                group.append((onset, end, kind, lane))
            for lane in {g[3] for g in group}:
                held = lane_hold.pop(lane, None)
                if held is not None and notes[held][1] > onset:  # earlier hold would contain this head
                    a, _, _, l = notes[held]
                    notes[held] = (a, a, 'tap', l)
                    dropped += 1
            for g in group:
                if g[2] == 'hold':
                    lane_hold[g[3]] = len(notes)
                notes.append(g)
                last_press[g[3]] = onset
        else:
            rank = 0
            for kind in pattern:
                end = onset
                if kind == 'hold':
                    end = min(_lower_median([m[3][rank] for m in voters]), int(duration_us))
                    rank += 1
                    if end <= onset:
                        kind, end, short = 'tap', onset, short+1
                notes.append((onset, end, kind, -1))
        kept += 1
        last_onset = onset
    notes.sort(key=lambda x: (x[0], x[3], x[2]))
    unplayable = 0
    if mode == 'mania_4k' and min_jack_us:
        # Humans never press one key twice within UNPLAYABLE_REPEAT_US; a lone note step 5 could not move is dropped.
        playable, pressed = [], {}
        for note in notes:
            if note[0] - pressed.get(note[3], -10**12) < UNPLAYABLE_REPEAT_US:
                unplayable += 1
                continue
            playable.append(note)
            pressed[note[3]] = note[0]
        notes = playable
    info = dict(samples=n, min_votes=need, window_us=window_us, clusters=len(clusters), kept_clusters=kept,
                skipped_clusters=skipped, notes=len(notes), holds_dropped_for_overlap=dropped,
                holds_too_short=short, unplayable_repeats_dropped=unplayable)
    info.update(pattern_source=pattern_source)
    if anchor is not None:
        info.update(anchor_sample=int(anchor), patterns_from_anchor=from_anchor)
    if min_jack_us:
        info.update(min_jack_us=int(min_jack_us), jack_rechosen=rechosen, jack_relaned=relaned,
                    jack_dropped=dropped_jack, jack_relaxed=relaxed)
    return notes, info


def consensus_legality(notes, mode, duration_us):
    """local_song_rollout.independent_legality of a consensus draft (must be empty)."""
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.model.local_song_rollout import independent_legality
    return independent_legality([RhythmEvent(a, b, t, l) for a, b, t, l in notes], mode, duration_us)


def consensus_evaluated(samples, songs, refs, grids, *, per_op, split, energies=None, intent=None, min_jack_us=None,
                        hold_release_us=None, pattern_source='majority', min_votes=None, on_song=None):
    """Consensus draft per song from ALL ``samples`` (reference-free construction), then scored like a sample."""
    out, drafts = {}, {}
    for song in songs:
        sid = song['song_id']
        states = [smp['by_id'][sid] for smp in samples]
        notes, info = consensus_draft([draft_tuples(song, st) for st in states], song['mode'], song['duration_us'],
                                      min_jack_us=min_jack_us, pattern_source=pattern_source, min_votes=min_votes)
        if hold_release_us:
            from beatfcn.train.workflow_metrics import repair_hold_release
            notes, info['hold_release_repair'] = repair_hold_release(notes, song['mode'], min_release_us=hold_release_us)
        info['legality'] = consensus_legality(notes, song['mode'], song['duration_us'])
        complete = all(st['complete'] for st in states)
        draft_info = dict(draft_id=f'consensus_{len(samples)}', complete=complete,
                          failure=None if complete else 'incomplete_samples')
        if intent is not None:
            draft_info.update(stratum=intent[sid]['stratum'], intent_reference=intent[sid]['chart_id'])
        r, d = evaluate(notes, draft_info, song, refs[sid], grids=grids, per_op=per_op, split=split,
                        energy=(energies or {}).get(sid))
        d['seeds'] = [smp['seed'] for smp in samples]
        out[sid] = (r, d)
        drafts[sid] = dict(mode=song['mode'], audio_sha256=song['audio_sha256'], seeds=d['seeds'], complete=complete,
                           duration_us=song['duration_us'], info=info,
                           events=[dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in notes])
        if on_song is not None:
            on_song()
    return out, drafts


# ------------------------------------------------------- D4 charter intent
def intent_references(root, panel):
    """Per song the reference whose density is requested by --density-request reference:
    the PRIMARY reference (hash-chosen with PANEL_SEED, the chart whose stratum placed the
    song in the frozen panel), so the draft's matched stratum is that reference's stratum."""
    out = {}
    for song in panel['songs']:
        ref = primary_reference(song)
        out[song['song_id']] = dict(ref, notes=reference_notes(root, ref, song['mode']))
    return out


def chart_style_bands(mode, notes, duration_us):
    """A chart's own style as table bands: a stand-in for what its charter would ask for. Hold, flick, drag and chord shares of heads."""
    from beatfcn.events import controls as C
    heads = max(len(notes), 1)
    onsets = {}
    for n in notes:
        onsets[n[0]] = onsets.get(n[0], 0)+1
    values = dict(hold_head_ratio=sum(n[2] == 'hold' for n in notes)/heads, chord_ratio=sum(c >= 2 for c in onsets.values())/max(len(onsets), 1))
    if mode == 'phira':
        values.update(flick_ratio=sum(n[2] == 'flick' for n in notes)/heads, drag_ratio=sum(n[2] == 'drag' for n in notes)/heads)
    out = {}
    for field, v in values.items():
        edges = C.BAND_EDGES[field]
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi or (hi == edges[-1] and v == hi):
                out[field] = [lo, hi]
                break
    return out


def density_condition_provider(intent, *, cell_us, curve_cell_us=2_000_000, bands=None, reference_style=False, self_curve=None):
    """local_song_rollout condition_provider: per core the reference's achieved 2 s density
    curve (counterfactual_exposure.condition_inputs, the training renderer), labelled
    charter_intent_density. Controls stay unconditional unless style ``bands`` are given."""
    from beatfcn.events import controls as C
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.train.counterfactual_density import achieved_density_segments
    from beatfcn.train.counterfactual_exposure import condition_inputs
    events = {sid: [RhythmEvent(int(a), int(b), t, int(l)) for a, b, t, l in ref['notes']]
              for sid, ref in intent.items()}

    def provider(state, request):
        sid = state['chart_id']
        if self_curve is not None:  # densities the model itself proposed in an earlier pass; no human chart involved
            segments = [dict(start_us=max(a, request.start_us), end_us=min(b, request.end_us), density=d)
                        for a, b, d in self_curve[sid] if a < request.end_us and b > request.start_us]
        elif curve_cell_us > request.end_us-request.start_us:  # coarser than one core: the density of the whole song section
            lo = request.start_us//curve_cell_us*curve_cell_us
            count = sum(lo <= e.onset_us < lo+curve_cell_us for e in events[sid])
            span = min(lo+curve_cell_us, state['duration_us'])-lo
            segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=count/(span/1e6))]
        else:
            segments = achieved_density_segments(events[sid], request.start_us, request.end_us, cell_us=curve_cell_us)
        mode_bands = {k: v for k, v in (bands or {}).items() if k in C.MODE_FIELDS[state['mode']]}
        if reference_style:
            mode_bands = chart_style_bands(state['mode'], intent[sid]['notes'], state['duration_us'])
        update = condition_inputs(state['mode'], request, bands=mode_bands, segments=segments,
                                  render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_intent_density', reference=intent[sid]['chart_id'],
                                               segments=segments, bands=mode_bands))
    return provider


def difficulty_presets(manifest):
    """Median heads/s of TRAIN charts per (mode, coarse density stratum): the constant density a
    charter implicitly asks for by choosing a difficulty tier. Uses only train charts."""
    import statistics
    rates = {}
    for c in manifest['charts']:
        if c['split'] != 'train':
            continue
        rate = sum(r['event_count'] for r in c['requests']) / max(c['duration_us'] / 1e6, 1e-6)
        rates.setdefault((c['mode'], coarse_stratum(c['mode'], c.get('stratum'))), []).append(rate)
    return {key: statistics.median(v) for key, v in rates.items() if v}


def preset_condition_provider(panel, presets, *, cell_us):
    """condition_provider: every core requests the constant train-median density of the song's
    primary reference difficulty tier (charter_difficulty_preset). Reads only the tier label."""
    from beatfcn.train.counterfactual_exposure import condition_inputs
    tier = {}
    for song in panel['songs']:
        ref = primary_reference(song)
        tier[song['song_id']] = (song['mode'], coarse_stratum(song['mode'], ref.get('stratum')))

    def provider(state, request):
        key = tier[state['chart_id']]
        density = float(presets[key])
        segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=density)]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_difficulty_preset', tier=key[1], density=density))
    return provider


def level_preset_provider(panel, chart_levels, *, cell_us):
    """condition_provider (difficulty_level_preset): the primary reference's Phira catalog level / approximate
    mania star -> the bundled preset table's heads/s (beatfcn.events.difficulty_presets), constant per core. Reads
    only the level label -- what a user picking "Lv.14" / "3.5*" would give."""
    from beatfcn.events.difficulty_presets import DifficultyPresets
    from beatfcn.train.counterfactual_exposure import condition_inputs
    table = DifficultyPresets.bundled()
    plan = {}
    for song in panel['songs']:
        ref = primary_reference(song)
        lv = chart_levels.get(ref['chart_id'])
        if lv is None:
            plan[song['song_id']] = None
            continue
        value = lv['phira_level'] if song['mode'] == 'phira' else lv['mania_star']
        value = min(max(value, 6.0), 18.0) if song['mode'] == 'phira' else min(max(value, 1.0), 7.0)
        plan[song['song_id']] = (value, table.heads_per_s(song['mode'], value))

    def provider(state, request):
        entry = plan[state['chart_id']]
        if entry is None:
            return None
        segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=float(entry[1]))]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='difficulty_level_preset', level=entry[0], density=entry[1]))
    return provider, plan


def shaped_preset_provider(panel, presets, shapes, *, cell_us, curve_cell_us=2_000_000):
    """condition_provider (charter_difficulty_preset_shaped): the tier's train-median density as the LEVEL, the
    first-pass unconditioned draft's 2 s density curve as the SHAPE (curve x tier / draft mean). Reads only the
    tier label and the model's own first pass; falls back to the constant preset when the first pass is empty."""
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.train.counterfactual_density import achieved_density_segments
    from beatfcn.train.counterfactual_exposure import condition_inputs
    plan = {}
    for song in panel['songs']:
        key = (song['mode'], coarse_stratum(song['mode'], primary_reference(song).get('stratum')))
        events = [RhythmEvent(int(e['onset_us']), int(e['end_us']), e['type'], int(e['lane']))
                  for e in shapes.get(song['song_id'], ())]
        mean = len({e.onset_us for e in events}) / max(song['duration_us'] / 1e6, 1e-6)
        plan[song['song_id']] = (key, events, float(presets[key]) / mean if mean > 0 else None)

    def provider(state, request):
        key, events, factor = plan[state['chart_id']]
        level = float(presets[key])
        if factor is None:
            segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=level)]
        else:
            segments = [dict(seg, density=seg['density'] * factor) for seg in
                        achieved_density_segments(events, request.start_us, request.end_us, cell_us=curve_cell_us)]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_difficulty_preset_shaped', tier=key[1], level=level,
                                               shape_factor=factor))
    return provider


def parse_control_bands(specs):
    """--control-band FIELD=LO,HI (repeatable) -> {field: [lo, hi]}; bands must be table bands of
    beatfcn.events.controls (the training condition renderer only knows those)."""
    from beatfcn.events import controls as C
    out = {}
    for spec in specs or ():
        field, _, rng = spec.partition('=')
        try:
            band = [float(x) for x in rng.split(',')]
        except ValueError:
            band = None
        if field not in C.BAND_EDGES or not band or not C.is_table_band(field, band):
            raise SystemExit(f'--control-band {spec!r}: need FIELD=LO,HI with a table band of {field!r} '
                             f'(edges {C.BAND_EDGES.get(field)})')
        out[field] = band
    return out


def style_condition_provider(bands):
    """local_song_rollout condition_provider: every core requests the charter's style bands
    (charter_intent_style; density curve stays unspecified). Bands a mode has no field for are
    left unconditional for that mode."""
    from beatfcn.events import controls as C
    from beatfcn.train.counterfactual_exposure import condition_inputs

    def provider(state, request):
        mode_bands = {k: v for k, v in bands.items() if k in C.MODE_FIELDS[state['mode']]}
        if not mode_bands:
            return None
        update = condition_inputs(state['mode'], request, bands=mode_bands, segments=(), scale=16.)
        return dict(inputs=update, record=dict(label='charter_intent_style', bands=mode_bands))
    return provider


def style_response(drafts, panel, bands):
    """Achieved whole-song control values of each selected draft vs the requested bands."""
    from beatfcn.events import controls as C
    from beatfcn.events.canonical import Event, LaneLoc, LineLoc
    songs = {s['song_id']: s for s in panel['songs']}
    per_song, summary = {}, {}
    for song_id, d in drafts.items():
        mode, duration = songs[song_id]['mode'], int(songs[song_id]['duration_us'])
        events = [Event(str(i), int(e['onset_us']), int(e['end_us']), e['type'],
                        (LaneLoc(int(e['lane'])) if int(e['lane']) >= 0 else LineLoc('rhythm', 'above', 0.))
                        if mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
                  for i, e in enumerate(d['events'])]
        achieved = C.raw_controls(mode, events, [(0, duration)])
        mode_bands = {k: v for k, v in bands.items() if k in C.MODE_FIELDS[mode]}
        per_song[song_id] = dict(mode=mode, requested=mode_bands, achieved=achieved)
        for field, band in mode_bands.items():
            value = achieved.get(field)
            entry = summary.setdefault(f'{mode}/{field}', dict(band=band, values=[]))
            entry['values'].append(value)
    for entry in summary.values():
        vals = [v for v in entry.pop('values') if v is not None]
        lo, hi = entry['band']
        entry.update(songs=len(vals), mean=float(np.mean(vals)) if vals else None,
                     median=float(np.median(vals)) if vals else None,
                     in_band=float(np.mean([lo <= v <= hi for v in vals])) if vals else None)
    return dict(label='charter_intent_style', bands=bands, summary=summary, songs=per_song)


def song_energies(features, panel):
    """requirement_diagnostics.song_energy per panel song from ``features.mel_frames`` (None if unavailable)."""
    from beatfcn.train import requirement_diagnostics as D
    mel_frames = getattr(features, 'mel_frames', None)
    if mel_frames is None:
        return None
    out = {}
    for song in panel['songs']:
        try:
            out[song['song_id']] = D.song_energy(mel_frames(song['audio_sha256']))
        except (KeyError, OSError, ValueError, AttributeError):
            out[song['song_id']] = None
    return out


READOUT_LABELS = dict(
    single_sample='MODEL CAPABILITY: one sample per song (seed --seed); the number to quote for the model.',
    best_of_N_reranked='DECODING readout: N samples per song, one chosen per song by the reference-free reranker '
                       '(beatfcn.model.rerank.select_best_song; never reads human charts). A legitimate product '
                       'decoding policy, reported separately from model capability.',
    oracle_best_of_N='ORACLE upper bound only: per song the sample with the lowest min-over-references CER vs the '
                     'HUMAN charts (selected with the answer). NEVER a capability or product claim.',
    consensus_N='DECODING readout (D6): one draft per song built from N samples by onset clustering (30 ms) + '
                'majority vote (>= ceil(N/2) samples; type/lane pattern by majority); reference-free, a legitimate '
                'product decoding policy, reported separately from model capability.')


def build_readouts(samples, evaluated, songs, *, rerank=None, oracle=None, samples_n=2000, seed=0, consensus=None,
                   consensus_jack=None):
    """Same CER/playability/density/bootstrap machinery for every readout; picks are sample indices per song.
    ``consensus``: optional {song_id: (rows, draft_row)} of the consensus draft (readout consensus_<N>)."""
    picks = dict(single_sample={s['song_id']: 0 for s in songs})
    if rerank is not None:
        picks['best_of_N_reranked'] = {sid: v['index'] for sid, v in rerank.items()}
    if oracle is not None:
        picks['oracle_best_of_N'] = {sid: v['index'] for sid, v in oracle.items()}
    readouts = {}
    for name, pick in picks.items():
        rows, draft_rows = [], []
        for song in songs:
            r, d = evaluated[pick[song['song_id']]][song['song_id']]
            rows += r
            draft_rows.append(d)
        readouts[name] = dict(label=READOUT_LABELS[name], indices=dict(pick),
                              seeds_used={sid: samples[i]['seed'] for sid, i in pick.items()},
                              complete_fraction=float(np.mean([d['complete'] for d in draft_rows])),
                              aggregate=aggregate(rows, draft_rows, samples=samples_n, seed=seed),
                              rows=rows, draft_rows=draft_rows)
    if consensus is not None:
        rows, draft_rows = [], []
        for song in songs:
            r, d = consensus[song['song_id']]
            rows += r
            draft_rows.append(d)
        n = len(draft_rows[0]['seeds']) if draft_rows else 0
        readouts[f'consensus_{n}'] = dict(label=READOUT_LABELS['consensus_N'].replace('N samples', f'{n} samples'),
            indices=None, seeds_used={d['song_id']: d['seeds'] for d in draft_rows},
            complete_fraction=float(np.mean([d['complete'] for d in draft_rows])),
            aggregate=aggregate(rows, draft_rows, samples=samples_n, seed=seed), rows=rows, draft_rows=draft_rows)
    if consensus_jack is not None:
        rows, draft_rows = [], []
        for song in songs:
            r, d = consensus_jack[song['song_id']]
            rows += r
            draft_rows.append(d)
        n = len(draft_rows[0]['seeds']) if draft_rows else 0
        readouts[f'consensus_{n}_jackfree'] = dict(
            label=READOUT_LABELS['consensus_N'].replace('N samples', f'{n} samples') +
                  ' + no key re-pressed within --consensus-min-jack-ms (same samples as consensus_N)',
            indices=None, seeds_used={d['song_id']: d['seeds'] for d in draft_rows},
            complete_fraction=float(np.mean([d['complete'] for d in draft_rows])),
            aggregate=aggregate(rows, draft_rows, samples=samples_n, seed=seed), rows=rows, draft_rows=draft_rows)
    if rerank is not None and oracle is not None:
        same = [rerank[s['song_id']]['index'] == oracle[s['song_id']]['index'] for s in songs]
        readouts['best_of_N_reranked']['agreement_with_oracle'] = float(np.mean(same))
    return readouts


def local_smoke_model(args, panel):
    import torch
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from scripts.run_benchmark_v2 import local_mert_store
    store = local_mert_store()
    keep = []
    for s in panel['songs']:
        if store.available(s['audio_sha256']):
            store.ensure(s['audio_sha256'])
            if abs(store.records[s['audio_sha256']]['duration_us']-s['duration_us']) > 1:
                raise ValueError('Tape and local feature clocks disagree')
            store.records[s['audio_sha256']]['duration_us'] = s['duration_us']
            keep.append(s)
    torch.manual_seed(0)
    config = FullEventConfig(d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2,
        tap_dim=1024, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=128, intensity_dim=3,
        onset_head=True, **(dict(beat_features=True) if getattr(args, 'charter_timing', None) else {}))
    model = FullEventModel(config).to(args.device).eval()
    return model, store, keep, dict(path=None, sha256=None, note='RANDOM-INIT smoke model; plumbing only'), 3, 100_000


def checkpoint_model(args):
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from beatfcn.train.training_panel import build_training_panel
    plan = json.loads(args.plan.read_text())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = FullEventConfig(**saved['config'])
    model = FullEventModel(config, EventVocabulary(**saved.get('vocabulary', {}))).to(args.device)
    weights = getattr(args, 'weights', 'raw')
    if weights == 'ema':
        if not saved.get('ema'):
            raise SystemExit('Checkpoint has no EMA state; use --weights raw')
        model.load_state_dict(saved['ema']['model'], strict=True)
    else:
        model.load_state_dict(saved['model'], strict=True)
    model.eval()
    del saved
    panel = build_training_panel(args.root, plan)
    identity = dict(path=str(args.checkpoint), sha256=sha_file(args.checkpoint), plan=str(args.plan),
                    plan_sha256=sha_file(args.plan))
    return model, panel.features, identity, config.intensity_dim, int(plan.get('counterfactual_cell_us', 100_000))


# ---------------------------------------------------------------- report
def _fmt(a, key='mean'):
    if not a or a.get('mean') is None:
        return 'n/a'
    s = f"{a['mean']:.3f}"
    if a.get('ci80'):
        s += f" [{a['ci80'][0]:.3f}, {a['ci80'][1]:.3f}]"
    if a.get('ci95'):
        s += f" ({a['ci95'][0]:.3f}, {a['ci95'][1]:.3f})"
    return s


def charter_table(charter, level='##', floor=None):
    lines = [f'{level} Charter timing: grid metrics + charter_timing_snap (reference BPM/offset = charter timing; '
             'NEVER capability)', '',
             '| group | rows with timing | draft on-grid | reference on-grid | chance on-grid | subdivision JS | '
             'CER raw (same rows) | CER charter_timing_snap | min-ref charter_timing_snap CER | note F1@50 raw -> snap |',
             '|---|---|---|---|---|---|---|---|---|---|']
    tables = [('', charter)] + ([(' (human-vs-human floor, same panel)', floor)] if floor else [])
    for tag, block in tables:
        for g, v in block.items():
            f1 = f"{_fmt(v['raw_note_f1_50ms'])} -> {_fmt(v['charter_snap_note_f1_50ms'])}"
            lines.append(f"| {g}{tag} | {v['rows_with_charter_timing']}/{v['rows']} | {_fmt(v['draft_on_grid_fraction'])} | "
                         f"{_fmt(v['reference_on_grid_fraction'])} | {_short(v.get('chance_on_grid_fraction'))} | "
                         f"{_fmt(v['subdivision_js_nats'])} | {_fmt(v['raw_cer'])} | "
                         f"{_fmt(v['charter_snap_cer'])} | {_fmt(v['min_ref_charter_snap_cer'])} | {f1} |")
    return lines+['', f'Rule: {CHARTER_RULE}. Chance on-grid = fraction of a beat within 5 ms of such a point '
                  '(uniformly random onsets); dense 1/12+1/16 unions make it large at fast tempos.', '']


def _short(a):
    return 'n/a' if not a or a.get('mean') is None else f"{a['mean']:.3f}"


def profile_table(profile, level='##', floor=None):
    """Markdown 'human-likeness profile': per mode and metric, draft vs human reference (+ panel floor)."""
    lines = [f'{level} Human-likeness profile (pattern level; draft vs every human reference of its song)', '',
             'Mean over songs [80% CI]. |d| = |draft - human| per (draft, reference) row; JS in nats (0 = same '
             'distribution, 0.693 = disjoint or one side empty).' + (' Floor = human-vs-human on the same panel '
             '(multi-chart songs, near-duplicates dropped): a model at the floor is as human-like as another charter.'
             if floor else ''), '',
             '| group | metric | draft | human | abs diff | JS | ' + ('floor abs diff | floor JS |' if floor else ''),
             '|---|---|---|---|---|---|' + ('---|---|' if floor else '')]
    for g, v in profile.items():
        fl = (floor or {}).get(g, {}).get('metrics', {})
        for name, m in v['metrics'].items():
            row = (f"| {g} | {name} | {_short(m['draft'])} | {_short(m['reference'])} | {_fmt(m['abs_diff'])} | "
                   f"{_fmt(m['js_nats']) if 'js_nats' in m else '-'} |")
            if floor:
                f = fl.get(name, {})
                row += f" {_short(f.get('abs_diff'))} | {_short(f.get('js_nats')) if 'js_nats' in m else '-'} |"
            lines.append(row)
        lines.append(f"| {g} | rows (with charter timing) | {v['rows']} ({v['rows_with_charter_timing']}) | | | |"
                     + (' | |' if floor else ''))
    lines += ['']
    metrics = {name: meaning for mode in PROFILE_METRICS.values() for name, _, _, meaning in mode}
    lines += [f'- `{k}`: {m}' for k, m in metrics.items() if any(k in v['metrics'] for v in profile.values())]
    return lines+['']


def aggregate_tables(agg, level='##', floor=None):
    """``floor``: optional human-vs-human aggregate (same panel) shown next to the profile/charter tables."""
    lines = []
    for title, key in (('Per reference stratum', 'by_split+mode+stratum'),
                       ('Per coarse stratum', 'by_split+mode+coarse_stratum'), ('Per mode (all references)', 'by_split+mode')):
        groups = agg.get(key, {})
        if not groups:
            continue
        lines += [f'{level} {title}', '', '| group | songs | rows | CER raw | CER snapped | CER edits x0.5 | CER edits x2 | notes draft/ref |',
                  '|---|---|---|---|---|---|---|---|']
        for g, v in groups.items():
            lines.append(f"| {g} | {v['songs']} | {v['rows']} | {_fmt(v['cer'])} | {_fmt(v['snapped_cer'])} | "
                         f"{_fmt(v['cer_edits_x0_5'])} | {_fmt(v['cer_edits_x2'])} | {_fmt(v['draft_over_reference_notes'])} |")
        lines.append('')
    mins = agg.get('min_over_references', {})
    if mins:
        lines += [f'{level} Min over references (per draft) + playability', '',
                  '| group | drafts | complete | min-ref CER | min-ref snapped CER | playability (mean rate over songs) |',
                  '|---|---|---|---|---|---|']
        for g, v in mins.items():
            play = '; '.join(f"{k.replace('_rate', '')} {a['mean']:.4f}" for k, a in v['playability'].items()
                             if a.get('mean') is not None)
            lines.append(f"| {g} | {v['drafts']} | {v['complete_fraction']:.2f} | {_fmt(v['min_ref_cer'])} | "
                         f"{_fmt(v['min_ref_snapped_cer'])} | {play} |")
        lines.append('')
        split = {g: v['silence_split'] for g, v in mins.items() if v.get('silence_split')}
        if split:
            lines += [f'{level} Silence split (draft notes where the min-CER reference is empty >= 4 s)', '',
                      '| group | uncharted s (pooled) | true-silence notes/s | uncharted-music notes/s | '
                      'pooled true-silence / music notes/s |', '|---|---|---|---|---|']
            for g, v in split.items():
                pooled = '/'.join('n/a' if x is None else f'{x:.3f}' for x in (
                    v['pooled_true_silence_events_per_s'], v['pooled_uncharted_music_events_per_s']))
                lines.append(f"| {g} | {v['pooled_uncharted_s']:.1f} | {_fmt(v['true_silence_events_per_s'])} | "
                             f"{_fmt(v['uncharted_music_events_per_s'])} | {pooled} |")
            lines.append('')
    if agg.get('charter_timing'):
        lines += charter_table(agg['charter_timing'], level, (floor or {}).get('charter_timing'))
    if agg.get('human_likeness_profile'):
        lines += profile_table(agg['human_likeness_profile'], level, (floor or {}).get('human_likeness_profile'))
    return lines


def readout_comparison(readouts):
    lines = ['## Readouts (never mixed: capability vs decoding vs ORACLE)', '',
             '| readout | group | drafts | complete | min-ref CER | CER raw (all refs) |', '|---|---|---|---|---|---|']
    for name, r in readouts.items():
        agg = r['aggregate']
        for g, v in agg.get('min_over_references', {}).items():
            raw = agg.get('by_split+mode', {}).get(g, {}).get('cer')
            tag = 'ORACLE ' if name.startswith('oracle') else ''
            lines.append(f"| {tag}{name} | {g} | {v['drafts']} | {v['complete_fraction']:.2f} | "
                         f"{_fmt(v['min_ref_cer'])} | {_fmt(raw)} |")
    lines.append('')
    for name, r in readouts.items():
        lines.append(f"- **{name}**: {r['label']}" + (f" Reranker pick == oracle pick on "
                     f"{r['agreement_with_oracle']:.2f} of songs." if 'agreement_with_oracle' in r else ''))
    return lines+['']


def markdown(report):
    lines = [f"# Workflow benchmark v1 -- {report['draft_source']}", '',
             f"Panel SHA `{report.get('panel', {}).get('panel_sha256', 'n/a')}`; tape SHA `{report['tape_sha256']}`.",
             'Values: mean over songs [80% CI] (95% CI), cluster bootstrap over songs. CER 0 = draft equals the',
             'reference; 1 = no better than charting from scratch; savings = 1 - CER. Phira is rhythm-only.', '']
    if report.get('notes'):
        lines += ['> '+n for n in report['notes']]+['']
    readouts = report.get('readouts') or {}
    floor = (report.get('human_floor_panel') or {}).get('aggregate')
    if len(readouts) > 1:
        lines += readout_comparison(readouts)
        for name, r in readouts.items():
            tag = 'ORACLE -- ' if name.startswith('oracle') else ''
            lines += [f'## {tag}{name}', '', r['label'], '']
            lines += aggregate_tables(r['aggregate'], level='###', floor=floor)
    else:
        lines += aggregate_tables(report['aggregate'], floor=floor)
    if report.get('extra_tables'):
        lines += report['extra_tables']
    if report.get('exports'):
        ex = report['exports']
        ok = sum(1 for v in ex['songs'].values() if v.get('chart'))
        lines += ['## Exports', '', f"Readout exported: `{ex['readout']}` -> `{ex['dir']}`; {ok}/{len(ex['songs'])} charts written.", '']
        for sid, v in ex['songs'].items():
            lines.append(f"- {sid}: chart {'ok' if v.get('chart') else 'FAILED'}; audio "
                         f"{'ok (' + v['audio']['source'] + ')' if v.get('audio', {}).get('ok') else 'MISSING: ' + str(v.get('audio', {}).get('reason'))}"
                         + (f"; {v['failed']}" if v.get('failed') else ''))
        if ex.get('pilot'):
            pl = ex['pilot']
            lines += ['', f"Pilot kit: {len(pl['segments'])} segments in {len(pl['pairs'])} pairs + "
                      f"{'1 practice segment' if pl.get('practice') else 'no practice segment'} -> `{pl['dir']}`"]
            for seg in pl['segments']:
                lines.append(f"  - {seg['segment_id']} pair {seg['pair']} {seg['mode']} {seg['coarse_stratum']} "
                             f"{'(pair stratum-matched) ' if seg['pair_stratum_matched'] else '(pair NOT stratum-matched) ' if seg['pair'] is not None else ''}"
                             f"{seg['song_id']} {seg['start_mmss']}-{seg['end_mmss']} timing={seg['timing']} "
                             f"audio={'ok' if seg['audio_ok'] else 'MISSING'}")
        lines.append('')
    lines += ['## Human playability baseline (pooled rate over all tape charts)', '']
    for g, v in report['human_playability'].items():
        lines.append(f"- {g} ({v['charts']} charts): " + ', '.join(f'{k} {x:.4f}' for k, x in v['pooled'].items()))
    unavailable = [k for k, v in report.get('timing', {}).items() if not v['available']]
    lines += ['', f"Timing grids: {sum(v['available'] for v in report.get('timing', {}).values())} available, "
              f"{len(unavailable)} unavailable" + (f" ({', '.join(unavailable[:8])})" if unavailable else ''), '']
    cov = report.get('timing_coverage')
    if cov:
        lines += [f"Charter-timing coverage: {cov['available']}/{cov['charts']} "
                  f"({0 if cov['available_fraction'] is None else cov['available_fraction']:.3f}); by source "
                  + ', '.join(f"{k} {b['available']}/{b['charts']}" for k, b in sorted(cov['by_source'].items()))
                  + '; clock shift ' + (', '.join(f'{k} x{n}' for k, n in sorted(cov['shift_source'].items())) or 'n/a')
                  + ('; unavailable: ' + ', '.join(f'{k} x{n}' for k, n in sorted(cov['reasons'].items()))
                     if cov['reasons'] else ''), '']
    return '\n'.join(lines)+'\n'


def shift_table(rows):
    lines = ['## Shifted-human sanity (draft = reference itself shifted later)', '',
             '| mode | shift | CER raw mean | CER snapped mean | rows |', '|---|---|---|---|---|']
    groups = defaultdict(list)
    for r in rows:
        if r['draft'].startswith('self+'):
            groups[(r['mode'], int(r['draft'][5:-2]))].append(r)
    for (mode, shift), v in sorted(groups.items()):
        snapped = [r['snapped_cer'] for r in v if r['snapped_cer'] is not None]
        lines.append(f"| {mode} | {shift} ms | {np.mean([r['cer'] for r in v]):.4f} | "
                     f"{np.mean(snapped) if snapped else float('nan'):.4f} | {len(v)} |")
    return lines+['']


def same_stratum_table(rows, samples, seed):
    lines = ['## Human-vs-human, same stratum only (the closest analogue of "another charter, same difficulty")', '',
             '| split/mode/stratum | songs | pairs | CER raw | CER snapped |', '|---|---|---|---|---|']
    out = {}
    for key in ('stratum', 'coarse_stratum'):
        draft_key = 'draft_stratum' if key == 'stratum' else 'draft_coarse'
        groups = defaultdict(list)
        for r in rows:
            if r.get(draft_key) is not None and r[draft_key] == r[key]:
                groups[f"{r['split']}/{r['mode']}/{key}={r[key]}"].append(r)
        for g, v in sorted(groups.items()):
            a = _agg(v, lambda r: r['cer'], samples=samples, seed=seed)
            b = _agg(v, lambda r: r['snapped_cer'], samples=samples, seed=seed)
            out[g] = dict(songs=a['n_units'], pairs=len(v), cer=a, snapped_cer=b)
            lines.append(f"| {g} | {a['n_units']} | {len(v)} | {_fmt(a)} | {_fmt(b)} |")
    return out, lines+['']


# ------------------------------------------------------------------- main
FIELD_TEMPERATURE_KEYS = ('group', 'head_block', 'head_cell', 'head_ms', 'mania', 'member', 'kind', 'tail_block',
                          'tail_cell', 'tail_ms')


class ScoringDeadline(Exception):
    """The process deadline passed during scoring: stop, the raw samples are already on disk."""


def scoring_tick(a):
    if getattr(a, 'progress', None) is not None:
        a.progress.advance()
    if a.score_from is None and remaining_wall_s(a) <= 0:
        if getattr(a, 'progress', None) is not None:
            a.progress.close()
        raise ScoringDeadline


def scored_samples(a):
    """--score-from: samples + metadata of a decode run, no model. Keeps songs whose samples are all complete."""
    meta = json.loads((a.score_from/'run-meta.json').read_text())
    panel = meta['panel']
    states = load_samples(a.score_from/'samples.jsonl')
    pool_n = max(a.best_of, a.consensus or 1)
    seeds = [a.seed+k for k in range(pool_n)]
    keep = [song for song in panel['songs'] if all(
        states.get((seed, song['song_id']), dict(complete=False))['complete'] for seed in seeds)]
    if not keep:
        raise SystemExit(f'{a.score_from}/samples.jsonl has no song with all of seeds {seeds} complete')
    samples = []
    for seed in seeds:
        songs = [states[seed, song['song_id']] for song in keep]
        result = dict(complete=True, songs=songs, wall_s=meta.get('generation_wall_s'))
        samples.append(dict(seed=seed, result=result, decoding=dict(meta['decoding'], seed=seed),
                            by_id={state['chart_id']: state for state in songs}))
    return meta, dict(panel, songs=keep), samples


def write_raw_samples(out, samples, songs):
    write_json_atomic(out/'drafts-samples.json', {
        smp['seed']: {song['song_id']: draft_record(song, smp['by_id'][song['song_id']], smp['seed']) for song in songs}
        for smp in samples})


def main():
    try:
        run_main()
    except ScoringDeadline:
        print('process deadline passed during scoring: scoring skipped, raw samples kept in samples.jsonl', flush=True)


def run_main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', type=Path)
    src.add_argument('--score-from', type=Path, help='score the samples of a finished or partial decode out-dir without '
                     'loading a model; songs with an incomplete sample are left out')
    src.add_argument('--local-smoke', action='store_true')
    src.add_argument('--human-as-draft', action='store_true')
    src.add_argument('--null-draft', action='store_true')
    p.add_argument('--plan', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--tape', default=TAPE)
    p.add_argument('--raw-root', default=RAW_ROOT)
    p.add_argument('--receipts-root', type=Path, help='full-database target receipts root (<root>/targets/<key>/'
                   'receipt.json) giving each chart\'s model-clock shift for charter timing; default: '
                   '<tape dir>/../targets when present. Without receipts the shift is estimated and verified.')
    p.add_argument('--max-songs', type=int, default=60)
    p.add_argument('--spread-panel', action='store_true', help='difficulty-spread panel (mania songs with a denser '
                   'sibling chart; only the target chart is scored); reports sibling raw / thinned baselines')
    p.add_argument('--alternative-reference', action='store_true', help='with --spread-panel: every core sees the '
                   "song's sibling (the charter's own denser chart) as a human_alternative reference "
                   '(charter_alternative_chart readout; never unconditioned capability)')
    p.add_argument('--panel-split', choices=('train', 'val', 'test'), default='val',
                   help='split the whole-song panel is frozen from; test = the held-out claim panel '
                        '(no decoding/threshold choice may be made on it)')
    p.add_argument('--human-splits', default='val,train', help='splits used by --human-as-draft')
    p.add_argument('--device', default='cpu')
    p.add_argument('--weights', choices=('raw', 'ema'), default='raw', help='checkpoint weights to evaluate')
    p.add_argument('--human-baseline', type=Path, help='reuse human_playability from a previous workflow-benchmark.json on the same tape')
    p.add_argument('--temperature', default='1', help="sampling temperature, or 'none' for greedy")
    p.add_argument('--field-temperature', type=json.loads, default=None, help='JSON per-grammar-field temperatures, e.g. {"kind": 0.3, "mania": 0.5}')
    p.add_argument('--top-p', type=float)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=4, help='rows (cores) per generation batch; the fast '
                   'decoder makes 32-64 cheap')
    p.add_argument('--parallel-samples', action='store_true', help='--best-of N: decode all N samples in one '
                   'rollout (rows = N x songs; identical samples, ~N x fewer sequential batches)')
    p.add_argument('--reference-decode', action='store_true', help='historical generate_batch loop (fast=False)')
    p.add_argument('--no-cuda-graph', action='store_true', help='fast decoder without CUDA-graph replay')
    p.add_argument('--token-cap', type=int, default=4096, help='token cap per 8 s core')
    p.add_argument('--max-wall-s', type=float, default=3600., help='ABSOLUTE wall limit of this process, for any '
                   'number of samples; decoding and scoring stop --flush-reserve-s before it')
    p.add_argument('--flush-reserve-s', type=float, default=300., help='time kept before --max-wall-s to save results')
    p.add_argument('--scheduler', choices=('legacy', 'grouped', 'continuous'), default='legacy',
                   help='legacy: batches seed by seed; grouped: seeds of a core together, sorted by previous-core '
                        'tokens, encoder shared in cold mode; continuous: seats refilled per step + the same sharing. '
                        'grouped/continuous always decode all samples in one rollout and finish songs in order')
    p.add_argument('--no-score', action='store_true', help='write drafts only (samples.jsonl, drafts-samples.json); '
                   'score later with --score-from')
    p.add_argument('--resume', action='store_true', help='continue in the same --out-dir: pairs in samples.jsonl that '
                   'finished are kept, wall-limit failures are decoded again (same draws)')
    p.add_argument('--autocast', choices=('bf16',), help='decode under bf16 autocast (default fp32; not bit-identical)')
    p.add_argument('--compile-decode', action='store_true', help='torch.compile the decode step (replaces CUDA graphs)')
    p.add_argument('--consensus-votes', type=int, help='samples an onset cluster needs in the consensus readouts '
                   '(default ceil(N/2))')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--per-op-sensitivity', action='store_true')
    p.add_argument('--keep-duplicates', action='store_true', help='human-as-draft: keep near-identical re-uploads')
    p.add_argument('--best-of', type=int, default=1, help='N complete samples per song (seeds seed..seed+N-1); '
                   'reports single_sample / best_of_N_reranked / ORACLE oracle_best_of_N. 1 = off')
    p.add_argument('--export-readout', choices=('auto', 'single_sample', 'best_of_N_reranked', 'consensus',
                                                 'consensus_jackfree'), default='auto',
                   help='which readout --export-dir / the pilot kit export (auto = historical: reranked with '
                        '--best-of, else single_sample)')
    p.add_argument('--export-dir', type=Path, help='write the selected draft per song as an editor-openable chart + '
                   'audio (mania .osu/.osz, Phira preview zip); audio from --audio-dir/<sha>.* or --raw-root')
    p.add_argument('--audio-dir', type=Path)
    p.add_argument('--pilot-segments', type=int, nargs='?', const=6, default=None,
                   help='also cut K (default 6, even) 30-45 s pilot segments into <export-dir>/pilot')
    p.add_argument('--pilot-seed', type=int, default=PANEL_SEED)
    p.add_argument('--history-reference', action='store_true', help="rollout: each core sees the model's own "
                   'previous 8 s as a model_history reference (history_context-trained models)')
    p.add_argument('--onset-fusion', type=float, help='D3 onset shallow fusion weight (e.g. 0.5): head-time logits '
                   '+= W*log(p_onset+1e-4) from the model onset head')
    p.add_argument('--consensus', type=int, help='D6: N >= 2 samples -> one consensus draft per song (readout '
                   'consensus_N); shares the sample pool with --best-of (pool = max of both)')
    p.add_argument('--density-request', choices=('reference', 'preset', 'preset_shaped', 'level_preset'), help='D4: request the primary human reference\'s '
                   '2 s density curve per core (charter_intent_density, never a capability number)')
    p.add_argument('--only-audio', type=Path, help='JSON list of audio sha256: keep only these songs of the frozen '
                   'panel (same selection, a subset; e.g. songs whose features are local)')
    p.add_argument('--chart-levels', type=Path, help='level_preset: {chart_id: {phira_level | mania_star}} (reports/presets/chart-levels-valtest.json)')
    p.add_argument('--consensus-pattern', choices=('majority', 'anchor'), default='majority',
                   help="pattern of the jack-free consensus readout: per-cluster majority, or the anchor (most central) "
                        "sample's pattern (one coherent hand flow; majority where the anchor has no member)")
    p.add_argument('--save-samples', action='store_true', help='write all consensus samples (drafts-samples.json) '
                   'so later consensus variants are offline')
    p.add_argument('--consensus-min-jack-ms', type=float, help='mania consensus: pick the most-voted pattern that '
                   'presses no key pressed < MS earlier (same-key repeats the per-cluster vote would stitch together)')
    p.add_argument('--hold-release-ms', type=float, help='jack-free consensus readout: shorten a mania hold so it '
                   'releases >= MS before the next press of the same key (else -> tap); playability repair (2026-11-02)')
    p.add_argument('--self-timing', type=Path, help='grid head (scripts/train_grid_head.py): estimate each song\'s beat grid '
                   'from audio and feed it as the timing input (audio-only capability readout; exclusive with --charter-timing)')
    p.add_argument('--control-band', action='append', metavar='FIELD=LO,HI',
                   help='charter style request per core (charter_intent_style), e.g. hold_occupancy_ratio=0.6,0.8; '
                        'repeatable; table bands of beatfcn.events.controls; exclusive with --density-request')
    p.add_argument('--self-curve', type=Path, help='with --density-request reference: JSON {song_id: [[start_us, end_us, heads/s], ...]} from an earlier pass of the model')
    p.add_argument('--kind-bias', type=json.loads, help='continuous scheduler: log-probability added per Phira note type, e.g. {"drag": 1.0}')
    p.add_argument('--density-curve-cell-s', type=float, default=2., help='seconds per point of the requested density curve')
    p.add_argument('--style-from-reference', action='store_true', help='with --density-request reference: also request the primary chart own style bands per song (simulated charter)')
    p.add_argument('--guidance', type=float, help='D4 CFG weight with --density-request / --control-band (1 = plain conditioning)')
    p.add_argument('--charter-timing', type=Path, help='chart timing table (scripts/build_chart_timing.py): model '
                   'with beat_features gets the primary reference\'s charter timing (readout charter_timing_input)')
    a = p.parse_args()
    if a.checkpoint and not a.plan:
        p.error('--checkpoint needs --plan')
    if a.best_of < 1:
        p.error('--best-of must be >= 1')
    if a.best_of > 1 and not (a.checkpoint or a.local_smoke or a.score_from):
        p.error('--best-of needs model drafts (--checkpoint or --local-smoke)')
    model_only = [f for f, on in (('--history-reference', a.history_reference), ('--onset-fusion', a.onset_fusion),
                  ('--consensus', a.consensus), ('--density-request', a.density_request), ('--guidance', a.guidance),
                  ('--control-band', a.control_band),
                  ('--alternative-reference', a.alternative_reference),
                  ('--charter-timing', a.charter_timing))
                  if on not in (None, False)]
    if model_only and not (a.checkpoint or a.local_smoke or a.score_from):
        p.error(f"{', '.join(model_only)} need model drafts (--checkpoint or --local-smoke)")
    decode_only = [f for f, on in (('--charter-timing', a.charter_timing), ('--self-timing', a.self_timing),
                                   ('--onset-fusion', a.onset_fusion), ('--control-band', a.control_band),
                                   ('--alternative-reference', a.alternative_reference), ('--no-score', a.no_score),
                                   ('--resume', a.resume), ('--history-reference', a.history_reference),
                                   ('--guidance', a.guidance), ('--autocast', a.autocast),
                                   ('--compile-decode', a.compile_decode)) if on not in (None, False)]
    if a.score_from and decode_only:
        p.error(f"--score-from reads saved samples; {', '.join(decode_only)} belong to the decode run")
    if a.score_from and (a.export_dir or a.density_request not in (None, 'reference', 'preset', 'level_preset')):
        p.error('--score-from does not support exports or this --density-request')
    if (a.no_score or a.resume or a.scheduler != 'legacy') and not (a.checkpoint or a.local_smoke):
        p.error('--no-score, --resume and --scheduler need model drafts (--checkpoint or --local-smoke)')
    if a.scheduler == 'continuous' and a.reference_decode:
        p.error('--scheduler continuous is built on the fast decoder; drop --reference-decode')
    if a.consensus_votes is not None and not (a.consensus and 1 <= a.consensus_votes <= a.consensus):
        p.error('--consensus-votes needs --consensus N and 1 <= votes <= N')
    if a.field_temperature is not None and (set(a.field_temperature) - set(FIELD_TEMPERATURE_KEYS)
                                            or not all(v > 0 for v in a.field_temperature.values())):
        p.error(f'--field-temperature needs positive values for fields in {FIELD_TEMPERATURE_KEYS}')
    if a.flush_reserve_s < 0:
        p.error('--flush-reserve-s must be >= 0')
    if a.consensus is not None and (a.consensus < 2 or parse_temperature(a.temperature) is None):
        p.error('--consensus needs N >= 2 and sampling (--temperature > 0)')
    if a.onset_fusion is not None and not a.onset_fusion > 0:
        p.error('--onset-fusion needs a positive weight')
    if a.alternative_reference and not a.spread_panel:
        p.error('--alternative-reference needs --spread-panel (the sibling is chosen there)')
    if a.control_band and a.density_request not in (None, 'reference'):
        p.error('--control-band combines only with --density-request reference')
    if a.guidance is not None and (not (a.density_request or a.control_band) or not a.guidance > 0):
        p.error('--guidance needs --density-request or --control-band and a positive weight')
    if a.pilot_segments is not None and (not a.export_dir or a.pilot_segments < 2 or a.pilot_segments % 2):
        p.error('--pilot-segments needs --export-dir and an even K >= 2 (segments are paired)')
    if a.export_dir and (a.export_dir.exists() and any(a.export_dir.iterdir())):
        p.error(f'--export-dir {a.export_dir} is not empty; exports are never overwritten')
    export_drafts_in, export_readout, refs = {}, None, {}
    root, raw_root = a.root, a.root/a.raw_root
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    target = out/'workflow-benchmark.json'
    if target.exists():
        raise FileExistsError(f'{target} exists; runs are never overwritten')
    if (out/'samples.jsonl').exists() and not (a.resume or a.score_from):
        raise FileExistsError(f'{out}/samples.jsonl exists; pass --resume or use a new --out-dir')
    a.process_start = PROCESS_START
    started = time.monotonic()
    manifest = load_tape(root, a.tape)
    tape_sha = sha_file(root/a.tape)
    receipts_root = a.receipts_root or default_receipts_root(root, a.tape)
    clock_shifts = ClockShifts(receipts_root, {c['chart_id']: c.get('canonical_sha256') for c in manifest['charts']
                                               if c.get('canonical_sha256')})
    a.clock_shifts = clock_shifts
    panel = (freeze_spread_panel(manifest, tape_sha, max_songs=a.max_songs, split=a.panel_split) if a.spread_panel
             else freeze_panel(manifest, tape_sha, max_songs=a.max_songs, split=a.panel_split))
    if getattr(a, 'only_audio', None):
        keep = set(json.loads(a.only_audio.read_text()))
        panel = dict(panel, songs=[s for s in panel['songs'] if s['audio_sha256'] in keep],
                     only_audio=dict(path=str(a.only_audio), kept=len(keep)))
    report = dict(schema='workflow-benchmark-v1', tape=a.tape, tape_sha256=tape_sha, costs=DEFAULT_COSTS,
                  tolerances=DEFAULT_TOLERANCES, snap=dict(divisors=SNAP_DIVISORS, max_snap_us=SNAP_MAX_US,
                  grid='reference chart own BPM/offset (assumes charter-supplied timing)'), notes=[],
                  charter_timing=dict(divisors=CHARTER_DIVISORS, on_grid_us=ON_GRID_US, max_snap_us=CHARTER_SNAP_MAX_US,
                                      rule=CHARTER_RULE, receipts_root=None if receipts_root is None else str(receipts_root),
                                      clock='raw chart grid moved by the target\'s model-clock shift (0 / receipt / '
                                            'estimated; accepted only at >= 95% onsets within 1 ms)'))
    drafts_out = {}
    if a.human_as_draft:
        splits = set(a.human_splits.split(','))
        rows, draft_rows, timing, n, duplicates = human_drafts(root, manifest, raw_root, splits, a.per_op_sensitivity,
                                                               keep_duplicates=a.keep_duplicates,
                                                               clock_shifts=clock_shifts)
        report.update(draft_source='human-as-draft', songs=n, splits=sorted(splits), timing=timing,
                      dropped_near_duplicates=duplicates)
        report['notes'].append(f'{len(duplicates)} near-duplicate re-uploads (CER < {NEAR_DUPLICATE_CER} vs a kept chart '
                               'of the same song) dropped' + (' -- disabled by --keep-duplicates' if a.keep_duplicates else '') + '.')
        report['notes'].append('Draft = another human chart of the same song (different difficulty/charter); '
                               'this is a realistic human-vs-human CER floor, not a model result.')
        extra, lines = same_stratum_table(rows, a.bootstrap, a.seed)
        report['same_stratum'] = extra
        report['extra_tables'] = lines
    elif a.null_draft:
        rows, draft_rows, timing = null_drafts(root, panel, raw_root, a.per_op_sensitivity, clock_shifts=clock_shifts)
        report.update(draft_source='null-draft', panel=panel, timing=timing)
        empty = [r['cer'] for r in rows if r['draft'] == 'empty']
        report['empty_draft_cer_all_one'] = bool(empty) and all(x == 1. for x in empty)
        report['notes'].append(f"Empty draft CER == 1 on all {len(empty)} references: {report['empty_draft_cer_all_one']}")
        report['extra_tables'] = shift_table(rows)
        rows_main = [r for r in rows if r['draft'] == 'empty']
        report['aggregate_shifted'] = aggregate([r for r in rows if r['draft'] != 'empty'],
            [d for d in draft_rows if d['draft'] != 'empty'], samples=a.bootstrap, seed=a.seed)
        rows, draft_rows = rows_main, [d for d in draft_rows if d['draft'] == 'empty']
    else:
        meta = None
        if a.score_from:
            meta, panel, loaded = scored_samples(a)
            model, features, identity, intensity_dim, cell_us = None, None, meta['checkpoint'], None, None
            report['notes'].append(f'Scored from {a.score_from} without a model; {len(panel["songs"])} songs had all '
                                   'samples complete.')
        elif a.local_smoke:
            model, features, songs, identity, intensity_dim, cell_us = local_smoke_model(a, panel)
            panel = dict(panel, songs=songs)
            panel['panel_sha256_smoke_subset'] = sha_json(songs)
            report['notes'].append('RANDOM-INIT smoke model: proves plumbing only, says nothing about any checkpoint.')
        else:
            model, features, identity, intensity_dim, cell_us = checkpoint_model(a)
            for s in panel['songs']:
                got = features.records[s['audio_sha256']]['duration_us']
                if abs(got-s['duration_us']) > 1:
                    raise ValueError(f"Feature clock {got} != tape {s['duration_us']} for {s['song_id']}")
                s['duration_us'] = got
        intent, condition_provider = None, None
        if a.charter_timing:
            if not getattr(model.c, 'beat_features', False):
                raise SystemExit('--charter-timing needs a model with config.beat_features')
            from beatfcn.train.charter_timing import ChartTimingTable
            table = ChartTimingTable.load(a.charter_timing)
            a.timing_provider, _, timing_input = charter_timing_provider(table, panel)
            a.charter_timing_identity = dict(table.identity(), panel_coverage=timing_input)
            report['charter_timing_input'] = a.charter_timing_identity
            report['notes'].append(f"charter_timing_input: every core got the beat grid of the song's primary human "
                                   f"reference ({timing_input['with_timing']}/{timing_input['songs']} songs have valid "
                                   'timing; others the absent state). Workflow input, NOT unconditioned capability.')
        if getattr(a, 'self_timing', None):
            if a.charter_timing:
                raise SystemExit('--self-timing and --charter-timing are exclusive')
            if not getattr(model.c, 'beat_features', False):
                raise SystemExit('--self-timing needs a model with config.beat_features')
            from beatfcn.train.self_timing import load_head, self_timing_provider
            head = load_head(a.self_timing, a.device)
            a.timing_provider, grids, diags = self_timing_provider(model, head, features, panel['songs'], a.device)
            report['self_timing'] = dict(head=str(a.self_timing), songs=len(grids), with_grid=sum(g is not None for g in grids.values()),
                                         per_song=diags, grids={k: (None if g is None else g.tolist()) for k, g in grids.items()})
            report['notes'].append('self_timing: the beat grid of every song was estimated FROM AUDIO by the frozen encoder + '
                                   'grid head and fed as the timing input; no charter data -> a capability readout.')
        if a.score_from:
            intent = intent_references(root, panel) if a.density_request else None
        elif a.density_request == 'preset_shaped':
            if not intensity_dim:
                raise SystemExit('--density-request needs a model with a density curve input (intensity_dim > 0)')
            presets = difficulty_presets(manifest)
            intent = intent_references(root, panel)
            first, _ = model_drafts(a, panel, model, features, intensity_dim=intensity_dim, cell_us=cell_us,
                                    seed=a.seed + 7919)
            shapes = {st['chart_id']: st['events'] for st in first['songs']}
            condition_provider = shaped_preset_provider(panel, presets, shapes, cell_us=cell_us)
            report['difficulty_presets'] = {f'{m}/{t}': round(v, 3) for (m, t), v in sorted(presets.items())}
            report['notes'].append('charter_difficulty_preset_shaped: level = the tier train-median density, shape = the '
                                   "2 s density curve of the model's own unconditioned first pass (seed +7919). A charter "
                                   'input (tier label only); NOT unconditioned capability.')
        elif a.density_request == 'level_preset':
            if not intensity_dim:
                raise SystemExit('--density-request needs a model with a density curve input (intensity_dim > 0)')
            if not getattr(a, 'chart_levels', None):
                raise SystemExit('--density-request level_preset needs --chart-levels')
            intent = intent_references(root, panel)
            condition_provider, level_plan = level_preset_provider(panel, json.loads(a.chart_levels.read_text()),
                                                                   cell_us=cell_us)
            report['level_presets'] = {k: (None if v is None else [round(v[0], 3), round(v[1], 3)])
                                       for k, v in level_plan.items()}
            report['notes'].append('difficulty_level_preset: every core requested the bundled preset density of the '
                                   "primary reference's Phira level / approximate mania star (a level label only).")
        elif a.density_request == 'preset':
            if not intensity_dim:
                raise SystemExit('--density-request needs a model with a density curve input (intensity_dim > 0)')
            presets = difficulty_presets(manifest)
            # Same draft labelling as the reference intent path (stratum + intent_reference_cer of the
            # primary reference), but the REQUEST is only the tier's train-median constant density.
            intent = intent_references(root, panel)
            condition_provider = preset_condition_provider(panel, presets, cell_us=cell_us)
            report['difficulty_presets'] = {f'{m}/{t}': round(v, 3) for (m, t), v in sorted(presets.items())}
            report['notes'].append('charter_difficulty_preset: every core requested the constant train-median density '
                                   "of the primary reference's difficulty tier (only the tier label is used). "
                                   'A realistic charter input; NOT unconditioned capability.')
        elif a.density_request:
            if not intensity_dim:
                raise SystemExit('--density-request needs a model with a density curve input (intensity_dim > 0)')
            # Charter-intent channel: the requested curve is read from the primary human reference
            # BEFORE generation (labelled charter_intent_density; never a capability number).
            intent = intent_references(root, panel)
            condition_provider = density_condition_provider(intent, cell_us=cell_us, bands=parse_control_bands(a.control_band) if a.control_band else None,
                                                            reference_style=a.style_from_reference,
                                                            curve_cell_us=int(a.density_curve_cell_s*1e6),
                                                            self_curve=json.loads(a.self_curve.read_text()) if a.self_curve else None)
            report['notes'].append('charter_intent_density: every core requested the 2 s density curve of the '
                                   "song's primary human reference (target-derived stand-in for charter intent); "
                                   'CER vs that reference is intent_reference_cer. NOT unconditioned capability.')
        if a.alternative_reference:
            from beatfcn.train.alternative_context import chart_arrays_from_notes, sibling_provider
            siblings = {s['song_id']: (s['sibling']['chart_id'],
                                       chart_arrays_from_notes(reference_notes(root, s['sibling'], s['mode'])))
                        for s in panel['songs']}
            a.alternative_provider = sibling_provider(siblings)
            report['notes'].append("charter_alternative_chart: every core saw the song's sibling chart (the charter's own "
                                   'denser chart, [start-2 s, end+2 s)) as human_alternative; scored only against the '
                                   'target chart. A charter input; NOT unconditioned capability.')
        if a.control_band:
            a.style_bands = parse_control_bands(a.control_band)
            if not a.density_request:
                condition_provider = style_condition_provider(a.style_bands)
            report['notes'].append(f"charter_intent_style: every core requested the style bands {a.style_bands} "
                                   '(density curve unspecified). A charter input; NOT unconditioned capability.')
        pool_n = max(a.best_of, a.consensus or 1)
        if a.score_from:
            samples, onset_term = loaded, meta['onset_term']
            a.decode_stats = meta.get('decode_stats')
            a.progress = ProgressWriter(out, panel['songs'], [smp['seed'] for smp in samples])
        else:
            samples, onset_term = decode_samples(a, out, panel, model, features, identity, intensity_dim, cell_us,
                                                 condition_provider, pool_n)
            if a.no_score or remaining_wall_s(a) <= 0:
                reason = 'no-score' if a.no_score else 'process deadline passed'
                a.progress.phase('write')
                write_raw_samples(out, samples, panel['songs'])
                a.progress.close()
                print(f'{reason}: drafts written, scoring skipped ({out}/samples.jsonl, drafts-samples.json)', flush=True)
                return
        scored_runs = len(samples[:max(1, a.best_of)]) + (1 if a.consensus else 0) \
            + (1 if a.consensus and a.consensus_min_jack_ms is not None else 0)
        a.progress.phase('score', total=scored_runs * len(panel['songs']))
        # Reference-free selection runs BEFORE any human chart is loaded (except the explicit
        # charter-intent density curve of --density-request).
        rerank = (rerank_selection([smp['by_id'] for smp in samples[:a.best_of]], panel['songs'])
                  if a.best_of > 1 else None)
        energies = song_energies(features, panel)
        refs, grids, timing = panel_references(root, panel, raw_root, clock_shifts=clock_shifts)
        evaluated = []
        # Only samples a readout can select are scored: index 0 (single_sample) and, with --best-of N, the first
        # N (reranked / oracle). Consensus-only pools (best_of 1) skip scoring the other N-1 samples (they enter
        # only through the consensus draft, scored below) -- a pure cost cut, no readout changes.
        for smp in samples[:max(1, a.best_of)]:
            ev = {}
            for song in panel['songs']:
                state = smp['by_id'][song['song_id']]
                info = dict(draft_id='model', complete=state['complete'], failure=state['failure'])
                if intent is not None:
                    info.update(stratum=intent[song['song_id']]['stratum'],
                                intent_reference=intent[song['song_id']]['chart_id'])
                r, d = evaluate(draft_tuples(song, state), info, song, refs[song['song_id']], grids=grids,
                                per_op=a.per_op_sensitivity, split=panel['split'],
                                energy=(energies or {}).get(song['song_id']))
                d['seed'] = smp['seed']
                ev[song['song_id']] = (r, d)
                scoring_tick(a)
            evaluated.append(ev)
        oracle = oracle_selection(evaluated[:a.best_of], panel['songs']) if a.best_of > 1 else None
        consensus, consensus_drafts, consensus_jack, jack_drafts = None, None, None, None
        if a.consensus:
            on_song = lambda: scoring_tick(a)  # noqa: E731
            consensus, consensus_drafts = consensus_evaluated(samples[:a.consensus], panel['songs'], refs, grids,
                per_op=a.per_op_sensitivity, split=panel['split'], energies=energies, intent=intent,
                min_votes=a.consensus_votes, on_song=on_song)
            if a.consensus_min_jack_ms is not None:
                # Same samples, jack-aware pattern choice: a paired readout at no generation cost.
                consensus_jack, jack_drafts = consensus_evaluated(samples[:a.consensus], panel['songs'], refs, grids,
                    per_op=a.per_op_sensitivity, split=panel['split'], energies=energies, intent=intent,
                    min_jack_us=int(round(a.consensus_min_jack_ms * 1000)),
                    hold_release_us=int(round(a.hold_release_ms * 1000)) if a.hold_release_ms else None,
                    pattern_source=getattr(a, 'consensus_pattern', 'majority'),
                    min_votes=a.consensus_votes, on_song=on_song)
                (out/f'drafts-consensus-{a.consensus}-jackfree.json').write_text(json.dumps(jack_drafts)+'\n')
            if getattr(a, 'save_samples', False):
                (out/'drafts-samples.json').write_text(json.dumps({
                    smp['seed']: {song['song_id']: draft_record(song, smp['by_id'][song['song_id']], smp['seed'])
                                  for song in panel['songs']} for smp in samples[:a.consensus]}) + '\n')
            (out/f'drafts-consensus-{a.consensus}.json').write_text(json.dumps(consensus_drafts)+'\n')
            votes = 'ceil(N/2)' if a.consensus_votes is None else a.consensus_votes
            report['notes'].append(f'consensus_{a.consensus}: one draft per song from samples seeds {a.seed}..'
                                   f'{a.seed+a.consensus-1} (30 ms onset clusters kept in >= {votes} samples, '
                                   'majority type/lane pattern); reference-free decoding readout.')
        readouts = build_readouts(samples, evaluated, panel['songs'], rerank=rerank, oracle=oracle,
                                  samples_n=a.bootstrap, seed=a.seed, consensus=consensus,
                                  consensus_jack=consensus_jack)
        rows, draft_rows = readouts['single_sample']['rows'], readouts['single_sample']['draft_rows']
        drafts_out = {song['song_id']: draft_record(song, samples[0]['by_id'][song['song_id']], samples[0]['seed'])
                      for song in panel['songs']}
        decoding = dict(samples[0]['decoding'], best_of=a.best_of,
                        seeds=[smp['seed'] for smp in samples],
                        generation_loop='reference' if a.reference_decode else
                        'fast' + ('' if a.no_cuda_graph else ' (+CUDA graph on cuda)'),
                        parallel_samples=bool(a.parallel_samples and pool_n > 1), batch_size=a.batch_size,
                        scheduler=meta['scheduler'] if meta else a.scheduler)
        if a.consensus:
            decoding['consensus'] = dict(samples=a.consensus, window_ms=CONSENSUS_WINDOW_US//1000,
                min_votes=a.consensus_votes or -(-a.consensus//2), onset='lower median',
                pattern='majority (ties: earliest sample)',
                hold_tail='lower median of the voting members', overlap='mania: holds containing a later '
                'same-lane head dropped (head kept as tap)',
                legal_songs=sum(not v['info']['legality'] for v in consensus_drafts.values()))
        if energies is not None:
            decoding['silence_split'] = ('whole song: draft notes in regions where a reference has no note for '
                                         '>= 4 s, true_silence vs uncharted_music by local mel energy')
        result = samples[0]['result']
        if a.best_of > 1:
            decoding['reranker'] = dict(selector='beatfcn.model.rerank.select_best_song',
                granularity='per song (sum of per-core scores); per-core splicing would break hold carries across cores',
                onset_term='on (model onset head, channel 0)' if onset_term else
                           'OFF: model has no onset_head -> no onset-alignment term, no expected-count density term',
                density_request=('charter_intent_density curve -> density scored against the requested curve'
                                 if intent is not None else
                                 'unspecified curve -> density scored against onset-head expected count'
                                 if onset_term else 'unscored'), weights=RERANK_WEIGHTS)
            report['notes'].append(f'best-of-{a.best_of}: seeds {a.seed}..{a.seed+a.best_of-1}; reranker onset term '
                                   + ('ON' if onset_term else 'OFF (no onset head in this model)') + '. '
                                   'single_sample = capability; best_of_N_reranked = decoding readout (reference-free); '
                                   'oracle_best_of_N = ORACLE upper bound (selected by CER vs human), never a capability claim.')
            pool = {}
            for song in panel['songs']:
                sid = song['song_id']
                pool[sid] = dict(samples=[draft_record(song, smp['by_id'][sid], smp['seed'])
                                          for smp in samples[:a.best_of]],
                                 reranked_index=rerank[sid]['index'], rerank_scores=rerank[sid]['scores'],
                                 rerank_parts=rerank[sid]['parts'], oracle_index=oracle[sid]['index'],
                                 oracle_min_ref_cers=oracle[sid]['min_ref_cers'])
            (out/f'drafts-best-of-{a.best_of}.json').write_text(json.dumps(pool)+'\n')
            report['selection'] = {sid: dict(reranked_seed=samples[v['reranked_index']]['seed'],
                                             oracle_seed=samples[v['oracle_index']]['seed'],
                                             rerank_scores=v['rerank_scores'],
                                             oracle_min_ref_cers=v['oracle_min_ref_cers']) for sid, v in pool.items()}
        report['readouts'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('rows', 'draft_rows')}
                              for k, v in readouts.items()}
        report['readout_rows'] = {k: dict(rows=v['rows'], draft_rows=v['draft_rows']) for k, v in readouts.items()
                                  if k != 'single_sample'}
        export_readout = a.export_readout if a.export_readout != 'auto' else (
            'best_of_N_reranked' if a.best_of > 1 else 'single_sample')
        if export_readout in ('consensus', 'consensus_jackfree'):
            source = consensus_drafts if export_readout == 'consensus' else jack_drafts
            if source is None:
                raise SystemExit(f'--export-readout {export_readout} needs --consensus'
                                 + (' and --consensus-min-jack-ms' if export_readout == 'consensus_jackfree' else ''))
            export_drafts_in = {sid: dict(d, seed='consensus:' + ','.join(map(str, d['seeds'])),
                                          failure=None if d['complete'] else 'incomplete_samples')
                                for sid, d in source.items()}
        else:
            picked = readouts[export_readout]['indices']
            export_drafts_in = {song['song_id']: draft_record(song, samples[picked[song['song_id']]]['by_id'][song['song_id']],
                                                              samples[picked[song['song_id']]]['seed'])
                                for song in panel['songs']}
        report.update(draft_source=meta['draft_source'] if meta else 'local-smoke' if a.local_smoke else 'checkpoint',
                      checkpoint=identity, decode_stats=a.decode_stats,
                      decoding=decoding, panel=panel, timing=timing, generation_wall_s=result['wall_s'],
                      generation_wall_s_all_samples=sum(smp['result']['wall_s'] or 0. for smp in samples),
                      complete_fraction=float(np.mean([st['complete'] for st in result['songs']])))
        report['notes'].append('Incomplete songs are scored on what was generated (missing region = insertions); '
                               f"complete fraction {report['complete_fraction']:.2f} (single sample).")
        if intent is None:
            report['notes'].append('Model has no difficulty request here: one draft per song, scored against every '
                                   'reference and labelled by the reference stratum.')
        (out/'drafts.json').write_text(json.dumps(drafts_out)+'\n')
        report['human_floor_panel'] = panel_human_floor(panel, refs, grids, samples=a.bootstrap, seed=a.seed)
    if getattr(a, 'progress', None) is not None:
        a.progress.phase('write')
    report['timing_coverage'] = timing_coverage(report.get('timing', {}))
    report['aggregate'] = aggregate(rows, draft_rows, samples=a.bootstrap, seed=a.seed)
    if getattr(a, 'style_bands', None) and drafts_out:
        report['style_response'] = style_response(drafts_out, panel, a.style_bands)
    if a.spread_panel:
        report['spread_baselines'] = spread_baselines(root, panel, samples=a.bootstrap, seed=a.seed)
    if a.human_baseline:
        # Precomputed over the same tape (e.g. the --human-as-draft run): reading
        # every chart again from the network volume costs ~30 min of GPU idle.
        report['human_playability'] = json.loads(Path(a.human_baseline).read_text())['human_playability']
        report['human_playability_source'] = str(a.human_baseline)
    else:
        report['human_playability'] = human_playability_baseline(root, manifest)
    # Persist the measured results before optional exports can fail.
    target.write_text(json.dumps(dict(report, rows=rows, draft_rows=draft_rows), indent=1, default=str)+'\n')
    if a.export_dir and drafts_out:
        try:
            report['exports'] = export_drafts(a, export_drafts_in, panel, raw_root, readout=export_readout, refs=refs)
            if a.pilot_segments:
                report['exports']['pilot'] = write_pilot_kit(a, export_drafts_in, panel, refs, raw_root,
                                                             readout=export_readout)
            (a.export_dir/'export-manifest.json').write_text(json.dumps(report['exports'], indent=1, default=str)+'\n')
        except Exception as exc:  # exports are a convenience, never the measurement
            report['exports_error'] = f'{type(exc).__name__}: {exc}'
    report['rows'] = rows
    report['draft_rows'] = draft_rows
    report['wall_s'] = time.monotonic()-started
    target.write_text(json.dumps(report, indent=1, default=str)+'\n')
    md = markdown(report)
    (out/'summary.md').write_text(md)
    print(md)
    print(f'wrote {target} ({sha_file(target)})')
    if getattr(a, 'progress', None) is not None:
        a.progress.close()


def decode_samples(a, out, panel, model, features, identity, intensity_dim, cell_us, condition_provider, pool_n):
    """Decode the sample pool with incremental results; a --resume keeps what is already finished."""
    a.decode_stats = {}
    seeds = [a.seed+k for k in range(pool_n)]
    sample_decoding = json.loads(json.dumps(draft_decoding(a, parse_temperature(a.temperature), a.seed),
                                            default=_json_default))
    meta = dict(schema='workflow-samples-v1', panel=panel, checkpoint=identity, decoding=sample_decoding, seeds=seeds,
                scheduler=a.scheduler, batch_size=a.batch_size,
                draft_source='local-smoke' if a.local_smoke else 'checkpoint')
    a.completed = {}
    if a.resume and (out/'samples.jsonl').exists():
        previous = json.loads((out/'run-meta.json').read_text())
        if previous['decoding'] != sample_decoding or previous['checkpoint'].get('sha256') != identity.get('sha256'):
            raise SystemExit(f'--resume: decoding or checkpoint differs from the run in {out}; samples would not compare')
        from beatfcn.train.rollout_resume import resume_deadline_prefix
        for key, state in load_samples(out/'samples.jsonl', repair=True).items():
            if finished_pair(state):
                a.completed[key] = state
            else:
                prefix = resume_deadline_prefix(state)
                if prefix is not None:
                    a.completed[key] = prefix
        finished = sum(finished_pair(state) for state in a.completed.values())
        print(f'resume: {finished} finished pairs; {len(a.completed)-finished} core prefixes kept', flush=True)
    write_json_atomic(out/'run-meta.json', meta)
    a.progress = ProgressWriter(out, panel['songs'], seeds, completed=a.completed)
    a.progress.write()
    samples, onset_term = sample_pool(a, panel, model, features, intensity_dim=intensity_dim, cell_us=cell_us,
                                      best_of=pool_n, condition_provider=condition_provider, rerank=a.best_of > 1)
    states = [state for smp in samples for state in smp['result']['songs']]
    write_json_atomic(out/'run-meta.json', dict(meta, onset_term=onset_term, decode_stats=a.decode_stats,
        generation_wall_s=samples[0]['result']['wall_s'],
        complete_fraction=float(np.mean([state['complete'] for state in states]))))
    return samples, onset_term


# ---------------------------------------------------------------- export
PILOT_SEGMENT_US = 40_000_000
PILOT_MIN_SEGMENT_US = 30_000_000
PILOT_ORDERS = ('P1', 'P2')


def _mmss(us):
    total = int(us)//1000
    return f'{total//60000}:{total % 60000/1000:06.3f}'


def resolve_audio(raw_root, song, audio_dir=None):
    """The song's audio as dict(ok, source, ext, path|data), or dict(ok=False, reason).

    Layout (inputs/d66_local_cohort/source/raw): osu audio is ``osu/audio/<audio_sha256>.<ext>``
    (the .osu files say ``AudioFilename: audio.mp3``); Phira audio lives inside each
    ``phira/zips/<id>.zip`` as the member named by ``info.yml: music``. Every hit is
    SHA-256 checked against the tape's ``audio_sha256``.
    """
    import zipfile
    import yaml
    sha = song['audio_sha256']
    tried = []
    if audio_dir:
        for path in sorted(glob.glob(str(Path(audio_dir)/f'{sha}.*'))):
            if sha_file(path) == sha:
                return dict(ok=True, source=f'audio-dir:{Path(path).name}', ext=Path(path).suffix, path=Path(path))
        tried.append(f'{audio_dir}/{sha}.*')
    if song['mode'] == 'mania_4k':
        for path in sorted(glob.glob(str(raw_root/'osu/audio'/f'{sha}.*'))):
            if sha_file(path) == sha:
                return dict(ok=True, source=f'raw osu/audio/{Path(path).name}', ext=Path(path).suffix, path=Path(path))
        tried.append(f'{raw_root}/osu/audio/{sha}.*')
    else:
        for ref in song['references']:
            zp = raw_chart_path(raw_root, ref['chart_id'])
            if not zp.is_file():
                tried.append(str(zp))
                continue
            with zipfile.ZipFile(zp) as z:
                names = z.namelist()
                try:
                    music = (yaml.safe_load(z.read('info.yml')) or {}).get('music') if 'info.yml' in names else None
                except yaml.YAMLError:
                    music = None
                order = ([music] if music in names else []) + [n for n in names if n != music
                         and Path(n).suffix.lower() in ('.mp3', '.ogg', '.wav', '.flac', '.m4a')]
                for name in order:
                    data = z.read(name)
                    if hashlib.sha256(data).hexdigest() == sha:
                        return dict(ok=True, source=f'raw phira/zips/{zp.name}:{name}', ext=Path(name).suffix, data=data)
            tried.append(f'{zp} (no member with the tape audio SHA)')
    return dict(ok=False, reason='audio not found or SHA mismatch; tried ' + '; '.join(tried))


def write_audio(audio, dest):
    """Copy resolved audio to ``dest`` (umask-respecting default permissions)."""
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    if 'path' in audio:
        shutil.copyfile(audio['path'], dest)
    else:
        dest.write_bytes(audio['data'])
    return dest


def song_metadata(raw_root, song):
    """(title, artist) from the first local raw reference chart; falls back to the song id."""
    import re
    import zipfile
    try:
        import yaml
        yaml_error = yaml.YAMLError
    except ImportError:  # cloud image has no PyYAML: tolerant key scan instead
        yaml, yaml_error = None, ValueError
    for ref in song['references']:
        path = raw_chart_path(raw_root, ref['chart_id'])
        if not path.is_file():
            continue
        try:
            if song['mode'] == 'mania_4k':
                text = path.read_text(encoding='utf-8', errors='replace')
                get = lambda key: (re.search(rf'^{key}:(.*)$', text, re.M) or [None, ''])[1].strip()  # noqa: E731
                title, artist = get('TitleUnicode') or get('Title'), get('ArtistUnicode') or get('Artist')
            else:
                with zipfile.ZipFile(path) as z:
                    raw = z.read('info.yml')
                if yaml is not None:
                    info = yaml.safe_load(raw) or {}
                else:
                    text = raw.decode('utf-8', errors='replace')
                    info = {k: (re.search(rf'^{k}:\s*(.*)$', text, re.M) or [None, ''])[1].strip().strip('\'"')
                            for k in ('name', 'composer')}
                title, artist = str(info.get('name') or ''), str(info.get('composer') or '')
        except (OSError, KeyError, zipfile.BadZipFile, yaml_error):
            continue
        if title:
            return title, artist or 'unknown'
    return song['song_id'], 'unknown'


def playback_shift(args, chart_id):
    """Model clock -> game playback clock for a song: the constant model-clock shift of its human reference
    chart (target receipt model_shift_us = offset decision + codec constant; osu mostly +14 ms). 0 if unknown."""
    lookup = getattr(args, 'clock_shifts', None)
    if lookup is None or chart_id is None:
        return 0, 'no receipt lookup: model clock written as is'
    shift, why = lookup(chart_id)
    return (0, f'no receipt shift ({why}): model clock written as is') if shift is None else (int(shift), 'receipt')


def export_song(folder, stem, song, events, audio, *, title, artist, version, creator='BeatFCN (model draft, unreviewed)',
                timing_points=None, bookmarks_ms=None, preview_time_ms=None, clock_shift_us=0, clock_source=None):
    """One editor-openable chart (+ audio) into ``folder``. Returns a receipt; never raises on export errors.

    Drafts live on the MODEL clock (tape targets = raw chart + model_shift_us); editors play the audio on the
    raw chart clock, so every time is written as t - ``clock_shift_us`` (notes that would fall before 0 are
    dropped and counted). ``timing_points`` / bookmarks / preview must already be on the playback clock."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(audio={k: v for k, v in audio.items() if k not in ('data', 'path')})
    shift = int(clock_shift_us or 0)
    moved = [dict(e, onset_us=int(e['onset_us']) - shift, end_us=int(e['end_us']) - shift) for e in events]
    events = [e for e in moved if e['onset_us'] >= 0]
    receipt['clock'] = dict(written='playback (raw chart) clock = model clock - shift', shift_us=shift,
                            source=clock_source, dropped_before_zero=len(moved) - len(events))
    audio_name = f"audio{audio['ext']}" if audio['ok'] else None
    if audio['ok']:
        write_audio(audio, folder/audio_name)
    try:
        if song['mode'] == 'mania_4k':
            receipt['chart'] = dict(skipped='mania drafts are lane-free timing extraction; no .osu '
                                            '(use the draft events directly)')
        elif audio['ok']:
            from beatfcn.events.rhythm_preview import write_phira_preview
            receipt['chart'] = write_phira_preview(folder.parent/f'{stem}.zip', events, folder/audio_name,
                                                   title=title, artist=artist, audio_name=audio_name)
            receipt['chart']['path'] = str(folder.parent/f'{stem}.zip')
        else:
            # No audio -> no Phira package; still write the rhythm as an RPE chart json.
            from beatfcn.events.rhythm_preview import display_positions
            from scripts.build_phira_preview import rpe
            doc = rpe(display_positions(events), 'DRAFT · unrated · display positions only')
            doc['META'].update(name=title, composer=artist, song='MISSING_AUDIO')
            path = folder/f'{stem}.rpe.json'
            path.write_text(json.dumps(doc, ensure_ascii=False))
            receipt['chart'] = dict(path=str(path), notes=len(events), package=False)
    except (ExportError, ValueError, FileExistsError) as exc:
        receipt['failed'] = f'{type(exc).__name__}: {exc}'
        receipt.pop('chart', None)
    return receipt


def export_drafts(args, drafts, panel, raw_root, *, readout, refs=None):
    """Per song the SELECTED draft (``readout``; never the oracle) as chart + audio under --export-dir.

    Written on the playback clock (model time - the primary reference's model-clock shift). Mania: the
    primary human chart's BPM/offset when its clock matches the target (the charter's timing in a real
    workflow), else one placeholder 120 BPM point."""
    songs = {s['song_id']: s for s in panel['songs']}
    out = dict(readout=readout, dir=str(args.export_dir), songs={},
               layout='<stem>/ holds the .osu (+audio.<ext>); <stem>.osz (mania) or <stem>.zip (Phira preview) '
                      'sit next to it. Clock: playback (raw chart) clock. Timing: primary human chart BPM/offset '
                      'when available, else one placeholder 120 BPM point.')
    for song_id, d in drafts.items():
        song = songs[song_id]
        stem = song_id.replace(':', '-')
        title, artist = song_metadata(raw_root, song)
        audio = resolve_audio(raw_root, song, args.audio_dir)
        primary = primary_reference(song)['chart_id'] if song.get('references') else None
        points = reason = timing_shift = None
        if song['mode'] == 'mania_4k' and refs and primary:
            notes = next((r['notes'] for r in refs.get(song_id, ()) if r['chart_id'] == primary), None)
            if notes is not None:
                points, reason, timing_shift = charter_timing(raw_root, primary, notes,
                                                              clock_shifts=getattr(args, 'clock_shifts', None))
        shift, source = ((timing_shift, 'charter timing (resolved against the target)') if timing_shift is not None
                         else playback_shift(args, primary))
        receipt = export_song(args.export_dir/stem, stem, song, d['events'], audio, title=title, artist=artist,
                              version=f"BeatFCN draft ({readout}, seed {d['seed']})", timing_points=points,
                              clock_shift_us=shift, clock_source=source)
        out['songs'][song_id] = dict(receipt, seed=d['seed'], complete=d['complete'], failure=d['failure'],
                                     timing='charter (primary human chart BPM/offset)' if points else
                                            f'placeholder 120 BPM @ 0 ms [{reason}]' if reason else 'placeholder 120 BPM @ 0 ms')
    return out


def primary_reference(song):
    return min(song['references'], key=lambda r: stable_rank(PANEL_SEED, 'primary', r['chart_id']))


def segment_window(song, notes, seed):
    """Draft-agnostic 30-45 s window: chosen from the HUMAN primary chart's head density (never the draft),
    among 1 s-step windows with at least the median head count (skips intros/breaks), hash-picked."""
    duration = int(song['duration_us'])
    length = min(PILOT_SEGMENT_US, duration)
    if length < PILOT_MIN_SEGMENT_US:
        return None
    heads = np.unique(np.asarray([n[0] for n in notes], np.int64))
    starts = list(range(0, duration-length+1, 1_000_000))
    counts = [int(((heads >= t) & (heads < t+length)).sum()) for t in starts]
    floor = max(1., float(np.median(counts)))
    candidates = [t for t, c in zip(starts, counts) if c >= floor] or starts
    start = min(candidates, key=lambda t: stable_rank(seed, 'pilot-window', song['song_id'], t))
    return start, start+length


def choose_pilot_segments(panel, refs, k, seed, *, audio_ok):
    """K/2 pairs of segments from DIFFERENT songs; each pair = same mode (+ same coarse stratum when possible).

    Pairs alternate across modes and round-robin over coarse strata; songs with resolvable audio
    first, then hash order. One extra unused song (mania preferred) becomes the practice segment."""
    by_mode = defaultdict(lambda: defaultdict(list))
    for song in panel['songs']:
        primary = primary_reference(song)
        notes = next(r['notes'] for r in refs[song['song_id']] if r['chart_id'] == primary['chart_id'])
        window = segment_window(song, notes, seed)
        if window is None:
            continue
        entry = dict(song=song, window=window, stratum=primary['coarse_stratum'], primary=primary['chart_id'])
        by_mode[song['mode']][primary['coarse_stratum']].append(entry)
    for strata in by_mode.values():
        for queue in strata.values():
            queue.sort(key=lambda e: (not audio_ok[e['song']['song_id']], stable_rank(seed, 'pilot-song', e['song']['song_id'])))
    pairs, modes, cursor = [], sorted(by_mode), defaultdict(int)
    while len(pairs) < k//2:
        progressed = False
        for mode in modes:
            if len(pairs) >= k//2:
                break
            strata = by_mode[mode]
            names = sorted(strata)
            full = [n for n in names if len(strata[n]) >= 2]
            if full:
                name = full[cursor[mode] % len(full)]
                cursor[mode] += 1
                pair = [strata[name].pop(0), strata[name].pop(0)]
            else:
                pool = sorted((e for n in names for e in strata[n]),
                              key=lambda e: (not audio_ok[e['song']['song_id']], stable_rank(seed, 'pilot-song', e['song']['song_id'])))
                if len(pool) < 2:
                    continue
                pair = pool[:2]
                for e in pair:
                    strata[e['stratum']].remove(e)
            pairs.append(pair)
            progressed = True
        if not progressed:
            break
    leftovers = sorted((e for m in modes for q in by_mode[m].values() for e in q),
                       key=lambda e: (e['song']['mode'] != 'mania_4k', not audio_ok[e['song']['song_id']],
                                      stable_rank(seed, 'pilot-practice', e['song']['song_id'])))
    return pairs, (leftovers[0] if leftovers else None)


def charter_timing(raw_root, chart_id, notes, clock_shifts=None):
    """(timing points [(ms, bpm)] on the raw chart's PLAYBACK clock, reason, model-clock shift us) from the raw
    human chart if its clock matches the target (after the target's model-clock shift, see reference_grid);
    points None (with the reason) otherwise. Exports write draft notes at model time - shift, so notes and
    these points share the clock the editor plays the audio on."""
    from beatfcn.events.chart_export import timing_points_from_map
    grid, info = reference_grid(raw_root, chart_id, notes, clock_shifts=clock_shifts)
    if grid is None:
        return None, info.get('reason'), None
    chart = load_raw_chart(raw_chart_path(raw_root, chart_id), chart_id)
    return [(ms, bpm) for ms, bpm in timing_points_from_map(chart.timing)], None, int(info.get('clock_shift_us', 0))


def write_pilot_kit(args, drafts, panel, refs, raw_root, *, readout):
    """Pilot kit under <export-dir>/pilot (see design/PILOT_KIT_v1.md).

    Per segment: the selected draft cut to the window (heads inside [start, end)), an EMPTY template
    with identical audio/timing/bookmarks, and the full-song audio (notes stay on the original audio
    clock; segment = editor bookmarks + PreviewTime). Same timing for both conditions: the primary
    human chart's BPM/offset when its clock matches the tape, else one placeholder point.
    Assignment: two counterbalanced participant orders; within each pair one segment is draft and
    the other scratch (hash-randomised), flipped for the other order; condition order ABBA over pairs.
    """
    import csv
    pilot = args.export_dir/'pilot'
    seed = args.pilot_seed
    songs = panel['songs']
    audio = {s['song_id']: resolve_audio(raw_root, s, args.audio_dir) for s in songs}
    pairs, practice = choose_pilot_segments(panel, refs, args.pilot_segments, seed,
                                            audio_ok={sid: a['ok'] for sid, a in audio.items()})
    segments = []
    numbered = [(f'seg-{i*2+j+1:02d}', i, e) for i, pair in enumerate(pairs) for j, e in enumerate(pair)]
    if practice is not None:
        numbered.append(('practice', None, practice))
    for seg_id, pair_index, e in numbered:
        song = e['song']
        sid = song['song_id']
        start, end = e['window']
        notes = next(r['notes'] for r in refs[sid] if r['chart_id'] == e['primary'])
        timing_points, timing_reason, timing_shift = (charter_timing(raw_root, e['primary'], notes,
                                                                     clock_shifts=getattr(args, 'clock_shifts', None))
                                                      if song['mode'] == 'mania_4k' else (None, None, None))
        if timing_shift is not None:
            shift, shift_source = timing_shift, 'charter timing (resolved against the target)'
        else:
            shift, shift_source = playback_shift(args, e['primary'])
        title, artist = song_metadata(raw_root, song)
        cut = [ev for ev in drafts[sid]['events'] if start <= int(ev['onset_us']) < end]
        files = {}
        for condition, events in (('draft', cut), ('scratch', [])):
            if seg_id == 'practice' and condition == 'scratch':
                continue
            stem = f'{seg_id}-{condition}'
            files[condition] = export_song(pilot/'segments'/seg_id/condition, stem, song, events, audio[sid],
                title=f'{title} (pilot {seg_id})', artist=artist, version=seg_id, creator='pilot',
                timing_points=timing_points if song['mode'] == 'mania_4k' else None,
                bookmarks_ms=[(start-shift)/1000, (end-shift)/1000], preview_time_ms=(start-shift)/1000,
                clock_shift_us=shift, clock_source=shift_source)
        segments.append(dict(segment_id=seg_id, pair=pair_index, song_id=sid, mode=song['mode'],
            coarse_stratum=e['stratum'], pair_stratum_matched=None if pair_index is None else
                len({x['stratum'] for x in pairs[pair_index]}) == 1, primary_reference=e['primary'], start_us=start, end_us=end,
            start_mmss=_mmss(start), end_mmss=_mmss(end), draft_notes=len(cut),
            clock_shift_us=shift, clock_shift_source=shift_source,
            human_primary_notes_in_window=sum(1 for n in notes if start <= n[0] < end),
            timing='charter (primary human chart BPM/offset)' if timing_points else
                   ('placeholder 120 BPM @ 0 ms' if song['mode'] == 'mania_4k' else 'Phira preview: fixed 60 BPM beat grid')
                   + (f' [{timing_reason}]' if timing_reason and song['mode'] == 'mania_4k' else ''),
            audio_ok=audio[sid]['ok'], audio=dict((k, v) for k, v in audio[sid].items() if k not in ('data', 'path')),
            files=files, draft_seed=drafts[sid]['seed']))
    paired = [s for s in segments if s['pair'] is not None]
    flips = {i: int(stable_rank(seed, 'pilot-flip', i)[:8], 16) % 2 for i in range(len(pairs))}

    def condition(order, seg):
        first_is_draft = (seg['segment_id'] == f"seg-{seg['pair']*2+1:02d}") != bool(flips[seg['pair']])
        return 'draft' if first_is_draft == (order == 'P1') else 'scratch'

    assignment = []
    for order in PILOT_ORDERS:
        sequence = []
        if practice is not None:
            sequence.append((next(s for s in segments if s['segment_id'] == 'practice'), 'draft (practice, not analysed)'))
        for i in range(len(pairs)):
            members = [s for s in paired if s['pair'] == i]
            # ABBA: P1 starts with draft on even pairs, scratch on odd; P2 mirrors.
            want_first = 'draft' if (i % 2 == 0) == (order == 'P1') else 'scratch'
            members.sort(key=lambda s: condition(order, s) != want_first)
            sequence += [(s, condition(order, s)) for s in members]
        for position, (seg, cond) in enumerate(sequence):
            kind = 'draft' if cond.startswith('draft') else 'scratch'
            ext = '.osz' if seg['mode'] == 'mania_4k' else '.zip'
            src = pilot/'segments'/seg['segment_id']/f"{seg['segment_id']}-{kind}{ext}"
            name = f"{position:02d}-{seg['segment_id']}-{kind}{ext}"
            dest = pilot/order/name
            if src.is_file():
                import shutil
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            code = 'R' + stable_rank(seed, 'pilot-blind', order, seg['segment_id'])[:6].upper()
            assignment.append(dict(participant_order=order, position=position, pair='' if seg['pair'] is None else seg['pair'],
                segment_id=seg['segment_id'], mode=seg['mode'], song_id=seg['song_id'], start=seg['start_mmss'],
                end=seg['end_mmss'], condition=cond, file=f'{order}/{name}' if src.is_file() else
                f"MISSING ({seg['audio'].get('reason', 'export failed')}) -- open segments/{seg['segment_id']}/{kind}/",
                blind_code='' if seg['pair'] is None else code))
    pilot.mkdir(parents=True, exist_ok=True)
    columns = list(assignment[0]) if assignment else []
    with open(pilot/'assignment.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(assignment)
    log_cols = ['participant', 'participant_order', 'position', 'segment_id', 'condition', 'started_at', 'finished_at',
                'minutes', 'notes_added', 'notes_deleted', 'notes_moved', 'final_note_count', 'self_quality_1to5',
                'fatigue_1to5', 'would_use_draft_again_1to5', 'comments']
    with open(pilot/'log_template.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(log_cols)
        for r in assignment:
            w.writerow(['', r['participant_order'], r['position'], r['segment_id'], r['condition']] + ['']*(len(log_cols)-5))
    with open(pilot/'rating_template.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['blind_code', 'rater', 'quality_1to5', 'publishable_yes_no', 'playtested_yes_no', 'comments'])
        for code in sorted(r['blind_code'] for r in assignment if r['blind_code']):
            w.writerow([code, '', '', '', '', ''])
    kit = dict(dir=str(pilot), seed=seed, readout=readout, segments=segments,
               pairs=[[s['segment_id'] for s in paired if s['pair'] == i] for i in range(len(pairs))],
               practice='practice' if practice is not None else None, requested_segments=args.pilot_segments,
               delivered_segments=len(paired), flips=flips, assignment=assignment,
               analysis_rule='WORKFLOW_BENCHMARK_v1 section 7: per pair r = (draft minutes / draft final notes) / '
                             '(scratch minutes / scratch final notes); pass = r <= 0.8 in EVERY pair and blind rating '
                             'difference (draft - scratch) >= -0.5; together with offline gates G0-G5 supports >= 80%.')
    (pilot/'pilot-manifest.json').write_text(json.dumps(kit, indent=1, default=str)+'\n')
    return kit


if __name__ == '__main__':
    main()

"""AGENTS Rule 1 probes for recipe v2 (local, real data, small config).

Recipe v2 = EventVocabulary(relative_blocks, tail_query) + FullEventConfig(
timed_intensity, timed_memory_values, onset_head) with mel on, counterfactual
loader conditions (own-target 2 s curve on the 100 ms grid + recomputed
controls, frozen dropout), objective = joint CE + 0.1 type CE + 1e-4 z +
1.0 first-token CE + 0.1 onset BCE.

1. Data probe: real train requests through the real loader path
   (CounterfactualExposureDataset.sample): shapes, NaN/Inf, min/max/mean/std
   of every input/target tensor, onset-target/onset alignment, curve clock vs
   mel/local clocks, curve values vs the achieved target density, controls.
2. Overfit probe: one fixed batch of real train requests (phira with holds +
   mania windows, conditions shown and fully dropped), dropout 0, N optimizer
   steps; per-term losses, per-field CE, gradient norms per module group, the
   onset BCE minus its soft-target lower bound; then free generation (greedy
   and T=1 per-row seeds) on the same fitted requests vs targets; teacher CE
   with real vs foreign vs zeroed local+mel audio.

  PYTHONPATH=. .venv/bin/python -m scripts.probe_recipe_v2_overfit --out reports/recipe-v2-probe

Opt-in ``--history-context`` mode (separate outputs, default reports/recipe-v2-probe/history):
the same recipe, but windows start >= 16 s and the human chart's preceding 16 s is attached
as visible reference rows (role human_input) through the real history_context path.
Data probe (reference rows / stats / leak check), overfit with the prefix attached, and a
usage check (teacher CE with real / no / foreign prefix, plus a synthetic repeated-pattern
case). See ``history_main``.
"""
import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS, encode_events
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train import requirement_diagnostics as D
from beatfcn.train.audio_dependence import intervene
from beatfcn.train.checkpoint_resume import file_sha
from beatfcn.train.counterfactual_exposure import CONTROL_FIELDS, build_exposure, sample_conditions
from beatfcn.train.event_reference import REFERENCE_COLUMNS, TIME_SCALE_US
from beatfcn.train.exposure_loader import (CounterfactualExposureDataset, _base_inputs, onset_supervision,
                                           request_sequence)
from beatfcn.train.history_context import (chart_arrays, draw_history, history_sources, reference_inputs,
                                           shift_arrays, visible_source)
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.training_objective import objective_terms, onset_presence_auc
from beatfcn.train.training_panel import build_training_panel

ROOT = Path(__file__).resolve().parents[1]
BASE_PLAN = 'reports/data-contract-v2-final/probe-plan.json'
CONFIG = dict(d=256, heads=4, kv_heads=2, ffn=768, encoder_layers=2, decoder_layers=4,
              dropout=0., mel_bins=128, intensity_dim=3, qk_norm=True, learnable_audio_scale=True,
              timed_intensity=True, timed_memory_values=True, onset_head=True)
VOCABULARY = dict(relative_blocks=True, tail_query=True)
LOSS = dict(type_weight=.1, z_weight=1e-4, first_token_weight=1., onset_weight=.1)
# (record index, conditions shown?) -- phira 11/13/14 carry holds (14 also an
# incoming hold), mania 16/33/58 hold-heavy windows (33 with 3 incoming holds).
BATCH = ((11, True), (13, False), (14, True), (16, False), (33, True), (58, False))
EXTRA_DATA = (0, 5, 15, 24, 59)   # + data probe only: empty/short tail windows, dense mania
CODE = ('scripts/probe_recipe_v2_overfit.py', 'beatfcn/model/full_event_model.py',
        'beatfcn/model/full_event_codec.py', 'beatfcn/model/full_event_generate.py',
        'beatfcn/train/training_objective.py', 'beatfcn/train/exposure_loader.py',
        'beatfcn/train/counterfactual_exposure.py', 'beatfcn/train/counterfactual_density.py',
        'beatfcn/train/density_curve.py', 'beatfcn/train/audio_dependence.py',
        'beatfcn/train/mert_features.py', 'beatfcn/train/mel_normalization.py',
        'beatfcn/train/history_context.py', 'beatfcn/train/event_reference.py')
GROUPS = (
    ('onset_head', ('onset_norm', 'onset_projection')),
    ('intensity_projection', ('intensity_projection', 'intensity_source', 'intensity_unspecified')),
    ('mel_projection', ('mel_projection', 'mel_source')),
    ('audio_projection', ('local_projection', 'song_projection', 'layer_mix', 'local_input_scale',
                          'song_input_scale', 'band_embedding', 'band_queries')),
    ('encoder_local', ('encoders.0.',)), ('encoder_song', ('encoders.1.',)),
    ('encoder_reference', ('encoders.2.',)),
    ('decoder', ('decoder.', 'token_embedding', 'field_embedding', 'final_norm', 'output')),
    ('conditions_other', ('control_projection', 'control_ids', 'reference_projection', 'empty_reference',
                          'source_ids', 'mode_ids')),
)


def group_of(name):
    for group, prefixes in GROUPS:
        if any(name == p or name.startswith(p if p.endswith('.') else p+'.') or name == p.rstrip('.')
               for p in prefixes):
            return group
    return 'unassigned'


def grad_norms(model):
    sums = Counter()
    for name, p in model.named_parameters():
        if p.grad is not None:
            sums[group_of(name)] += float(p.grad.detach().float().square().sum())
    return {k: math.sqrt(v) for k, v in sorted(sums.items())}


def stats(value, valid=None):
    a = np.asarray(value)
    out = dict(shape=list(a.shape), dtype=str(a.dtype))
    if a.dtype == bool:
        out.update(true=int(a.sum()), false=int((~a).sum()))
        return out
    f = a.astype(np.float64)
    out.update(nan=int(np.isnan(f).sum()), inf=int(np.isinf(f).sum()))
    finite = f[np.isfinite(f)]
    if finite.size:
        out.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
                   std=float(finite.std()))
    if valid is not None:
        v = np.asarray(valid, bool)
        sel = f[v] if f.shape[:v.ndim] == v.shape else None
        if sel is not None and sel.size:
            out['valid_only'] = dict(rows=int(v.sum()), min=float(sel.min()), max=float(sel.max()),
                                     mean=float(sel.mean()), std=float(sel.std()))
    return out


def identity(panel, plan, seed):
    audios = sorted({r['audio_sha256'] for r in panel.records})
    feature_root = Path(plan['mert_features'])
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        head, dirty = None, None
    return dict(seed=seed, tape=plan['tape'], tape_sha256=file_sha(ROOT/plan['tape']),
        mert_receipts_sha256={a[:12]: file_sha(feature_root/a/'receipt.json') for a in audios
                              if (feature_root/a/'receipt.json').is_file()},
        mel_normalization=plan['mel_normalization'], base_plan=BASE_PLAN, base_plan_sha256=file_sha(ROOT/BASE_PLAN),
        code_sha256={p: file_sha(ROOT/p) for p in CODE}, git_head=head, git_dirty_tracked=dirty,
        torch=torch.__version__, python=platform.python_version(), checkpoint=None)


HISTORY = dict(p=1., p_infill=0., lengths_s=[16], roles={'human_input': 1.})   # always a 16 s human prefix


def make_dataset(panel, exposure, history_context=None):
    return CounterfactualExposureDataset(panel, exposure, microbatch=4, cell_us=100_000,
        curve_cell_us=2_000_000, vocabulary=VOCABULARY, onset=dict(scope='window'),
        mel_bins=CONFIG['mel_bins'], history_context=history_context)


def build(seed, history_context=None):
    plan = json.loads((ROOT/BASE_PLAN).read_text())
    plan = copy.deepcopy(plan)
    plan.pop('intensity', None)       # recipe v2: conditions come from the counterfactual loader
    plan['config'] = dict(plan['config'], **CONFIG)
    plan['vocabulary'] = VOCABULARY
    panel = build_training_panel(ROOT, plan)
    train = [i for i, r in enumerate(panel.records) if r['split'] == 'train']
    exposure = build_exposure(train, steps=4, seed=seed)
    exposure['pool_identity'] = 'probe'
    dataset = make_dataset(panel, exposure, history_context)
    dataset.probe_exposure = exposure
    return plan, panel, dataset


def sample(dataset, index, shown):
    fields = list(range(len(CONTROL_FIELDS))) if shown else []
    return dataset.sample(index, None, edit_seed=0, curve_specified=shown, specified_fields=fields)


# ------------------------------------------------------------------ data probe
def alignment(row, audit):
    inp, seq, request = row['inputs'], row['sequence'], audit['request']
    span = (request.end_us-request.start_us)/1e6
    mel_t = np.asarray(inp['mel_times'], np.float64); mel_v = np.asarray(inp['mel_valid'], bool)
    loc_t = np.asarray(inp['local_times'], np.float64); loc_v = np.asarray(inp['local_valid'], bool)
    targets = np.asarray(seq['onset_targets']); valid = np.asarray(seq['onset_valid'], bool)
    hop = float(np.median(np.diff(mel_t)))
    heads = [((e.onset_us-request.start_us)/1e6, e.type) for e in audit['events']]
    misses, offsets, type_misses = 0, [], 0
    for t, kind in heads:
        idx = int(np.abs(mel_t-t).argmin())
        offsets.append(abs(mel_t[idx]-t))
        if not (valid[idx] and targets[idx, 0] == 1.):
            misses += 1
        if targets[idx, 1+TYPES.index(kind)] != 1.:
            type_misses += 1
    # Every exact-1 frame in the core must sit within one hop of a true head.
    ones = np.flatnonzero((targets[:, 0] == 1.) & valid)
    core_ones = [i for i in ones if 0 <= mel_t[i] < span]
    head_times = np.asarray([t for t, _ in heads]) if heads else np.zeros(0)
    stray = [float(mel_t[i]) for i in core_ones
             if not len(head_times) or np.abs(head_times-mel_t[i]).min() > hop+1e-6]
    curve = np.asarray(inp['intensity']); ct = np.asarray(inp['intensity_times']); cv = np.asarray(inp['intensity_valid'])
    edges_ok = bool(np.allclose(curve[1:, 1], curve[:-1, 2], atol=1e-6) and abs(curve[0, 1]) < 1e-6
                    and abs(curve[-1, 2]-span) < 1e-5 and np.allclose(ct, (curve[:, 1]+curve[:, 2])/2, atol=1e-5))
    density_err = None
    if audit['curve_segments']:
        decoded = np.expm1(curve[:, 0]*math.log1p(16.))
        expected = []
        for c in ct:
            s = next(s for s in audit['curve_segments']
                     if (s['start_us']-request.start_us)/1e6 <= c < (s['end_us']-request.start_us)/1e6)
            expected.append(s['density'])
        density_err = float(np.max(np.abs(decoded-np.asarray(expected))))
    controls = np.asarray(inp['controls'])
    return dict(core_heads=len(heads), heads_without_exact_peak=misses, heads_type_channel_missing=type_misses,
        max_head_to_nearest_mel_frame_s=float(max(offsets)) if offsets else None, mel_hop_s=hop,
        stray_core_peaks=len(stray), mel_core_frames=int(((mel_t >= 0) & (mel_t < span) & mel_v).sum()),
        expected_core_frames=round(span/hop), mel_range_s=[float(mel_t[mel_v].min()), float(mel_t[mel_v].max())],
        local_range_s=[float(loc_t[loc_v].min()), float(loc_t[loc_v].max())],
        local_hop_s=float(np.median(np.diff(loc_t))), span_s=span,
        curve_cells=len(ct), curve_cell_s=float(np.median(curve[:, 2]-curve[:, 1])),
        curve_edges_consistent=edges_ok, curve_specified_cells=int(cv.sum()),
        curve_max_abs_density_error_hz=density_err,
        controls_specified_fields=int(controls[:, 2].sum()), controls_applicable_fields=int(controls[:, 3].sum()),
        onset_target_frames=len(targets), onset_valid_frames=int(valid.sum()),
        onset_frames_equal_mel_frames=len(targets) == len(mel_t))


def data_probe(dataset, indices):
    rows = []
    for index, shown in indices:
        row, audit = sample(dataset, index, shown)
        record = dataset.panel.records[index]
        inputs, seq = row['inputs'], row['sequence']
        rows.append(dict(index=index, mode=record['mode'], chart_id=record['chart_id'],
            request=dict(start_us=audit['request'].start_us, end_us=audit['request'].end_us,
                         known_carry=len(audit['request'].known_carry)),
            conditions_shown=shown, types=dict(Counter(e.type for e in audit['events'])),
            inputs={k: stats(v, inputs.get(k+'_valid')) for k, v in inputs.items()},
            sequence={k: stats(v) for k, v in seq.items()},
            alignment=alignment(row, audit)))
    totals = Counter()
    for r in rows:
        for part in ('inputs', 'sequence'):
            for s in r[part].values():
                totals['nan'] += s.get('nan', 0); totals['inf'] += s.get('inf', 0)
        a = r['alignment']
        totals['heads'] += a['core_heads']; totals['heads_without_exact_peak'] += a['heads_without_exact_peak']
        totals['type_channel_missing'] += a['heads_type_channel_missing']; totals['stray_core_peaks'] += a['stray_core_peaks']
        totals['curve_edges_inconsistent'] += int(not a['curve_edges_consistent'])
    return dict(rows=rows, totals=dict(totals))


# ------------------------------------------------------------------ overfit
def per_field_ce(logits, seq):
    legal = logits.float().masked_fill(~seq['allowed'], -1e9)
    ce = torch.nn.functional.cross_entropy(legal.flatten(0, 1), seq['targets'].flatten(),
                                           reduction='none').view_as(seq['targets'])
    free = seq['valid'] & (seq['allowed'].sum(-1) > 1)
    out = {}
    for i, field in enumerate(FIELDS):
        mask = free & (seq['fields'] == i)
        if mask.any():
            out[field] = dict(ce=float(ce[mask].mean()), tokens=int(mask.sum()))
    out['_all_free_tokens'] = dict(ce=float(ce[free].mean()), tokens=int(free.sum()),
                                   forced_tokens=int((seq['valid'] & ~free).sum()))
    return out


def onset_bce_lower_bound(aux, seq):
    """Min of the same positive-weighted soft BCE (batch pos_weight) over predictions:
    soft Gaussian targets make it > 0; the head can at best reach this value."""
    targets = seq['onset_targets'].cpu().double()
    valid = (seq['onset_valid'].bool() & aux['onset_valid'].bool()).cpu()
    weight = valid[..., None].double()
    positive = (targets*weight).sum((0, 1)); negative = ((1-targets)*weight).sum((0, 1))
    w = (negative/positive.clamp_min(1e-6)).clamp(1., 100.)
    a, b = w*targets, 1-targets
    p = a/(a+b).clamp_min(1e-300)
    # xlogy: 0*log(0) = 0 for the hard 0/1 frames.
    bound = -(torch.special.xlogy(a, p)+torch.special.xlogy(b, 1-p))
    count = weight.sum((1, 2))
    per = (bound*weight).sum((1, 2))/count.clamp_min(1)
    return float(per.sum()/(count > 0).sum().clamp_min(1))


def scalars(terms):
    return {k: float(v) for k, v in terms.items()}


def teacher_eval(model, inp, seq):
    model.eval()
    with torch.no_grad():
        logits, aux = model(inp, seq, return_aux=True)
        terms = scalars(objective_terms(logits, seq, aux=aux, **LOSS))
        fields = per_field_ce(logits, seq)
        auc = onset_presence_auc(aux, seq)
        bound = onset_bce_lower_bound(aux, seq) if 'onset_targets' in seq else None
    return dict(terms=terms, per_field_ce=fields, onset_presence_auc=auc, onset_bce_lower_bound=bound,
                onset_bce_excess=None if bound is None else terms['onset_bce']-bound)


def to(device, mapping):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in mapping.items()}


def event_key(e):
    return (int(e['onset_us']), int(e['end_us']), e['type'], int(e['lane']))


def compare(prediction, truth, request):
    truth_keys = Counter(map(event_key, truth))
    pred_keys = Counter(map(event_key, prediction['events']))
    acc = D.window_accuracy(prediction, truth, request)
    return dict(complete=prediction['complete'], failure=prediction.get('failure'),
        token_limit=prediction.get('token_limit_reached'), events_pred=len(prediction['events']),
        events_true=len(truth), exact_multiset=pred_keys == truth_keys,
        missing=sum((truth_keys-pred_keys).values()), extra=sum((pred_keys-truth_keys).values()),
        onset_f1_20ms=D.f1_of(acc['full']['onset_20ms']), onset_f1_50ms=D.f1_of(acc['full']['onset_50ms']),
        note_f1_50ms=D.f1_of(acc['full']['note_50ms']), hold_joint_f1_50ms=D.f1_of(acc['full']['hold_joint_50ms']),
        rng_mode=prediction.get('rng_mode'))


def generation(model, inp, requests, truths, request_dicts, vocab, teacher_tokens, device, seeds):
    model.eval()
    max_tokens = int(2*max(len(t) for t in teacher_tokens))+8
    out = {}
    readouts = [('greedy', None, None)] + [(f'sample_t1_seed{s}', 1., s) for s in seeds]
    for name, temperature, seed in readouts:
        tick = time.monotonic()
        kwargs = {}
        if temperature is not None:
            gdev = device.type if device.type in ('cuda', 'mps') else 'cpu'
            kwargs = dict(temperature=temperature, generator=[torch.Generator(device=gdev).manual_seed(
                int(hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:12], 16)) for i in range(len(requests))])
        predictions = generate_batch(model, inp, requests, max_tokens, **kwargs)
        rows = []
        for p, truth, req, tokens in zip(predictions, truths, request_dicts, teacher_tokens):
            c = compare(p, truth, req)
            c['tokens_exact'] = list(p['tokens']) == list(tokens)
            rows.append(c)
        out[name] = dict(rows=rows, wall_s=time.monotonic()-tick, max_tokens=max_tokens,
            exact_rows=sum(r['exact_multiset'] for r in rows), token_exact_rows=sum(r['tokens_exact'] for r in rows),
            mean_onset_f1_20ms=float(np.mean([r['onset_f1_20ms'] for r in rows])),
            mean_onset_f1_50ms=float(np.mean([r['onset_f1_50ms'] for r in rows])),
            mean_note_f1_50ms=float(np.mean([r['note_f1_50ms'] for r in rows])))
    return out


def audio_dependence(model, dataset, rows, audits, device):
    """Teacher CE of the fitted batch with real / foreign (other song, same times) / zeroed local+mel."""
    panel = dataset.panel
    audios = sorted({panel.records[i]['audio_sha256'] for i, _ in BATCH})
    conditions = {}
    for condition in ('real', 'foreign', 'zero_local_mel'):
        modified = []
        for (index, _), row, audit in zip(BATCH, rows, audits):
            inputs = row['inputs']
            if condition == 'foreign':
                record = panel.records[index]
                donor = next(a for a in audios if a != record['audio_sha256'])
                request = audit['request']
                span = request.end_us-request.start_us
                duration = panel.features.records[donor]['duration_us']
                start = min(request.start_us, duration-span)
                foreign = panel.features.window(donor, start, start+span)
                inputs = intervene(inputs, 'foreign', foreign=foreign)
                # Onset targets belong to the real audio frames; not defined here.
                row = dict(row, sequence={k: v for k, v in row['sequence'].items()
                                          if k not in ('onset_targets', 'onset_valid')})
            elif condition == 'zero_local_mel':
                inputs = intervene(inputs, 'zero_local_mel')
            modified.append(dict(row, inputs=inputs))
        inp, seq = collate(modified)
        inp, seq = to(device, inp), to(device, seq)
        result = teacher_eval(model, inp, seq)
        conditions[condition] = dict(joint_ce=result['terms']['joint_ce'], type_ce=result['terms']['type_ce'],
            first_token_ce=result['terms']['first_token_ce'],
            onset_bce=result['terms']['onset_bce'] if result['terms']['onset_active'] else None,
            onset_presence_auc=result['onset_presence_auc'],
            per_field_ce={k: v['ce'] for k, v in result['per_field_ce'].items()})
    real = conditions['real']
    gaps = {c: dict(joint_ce=conditions[c]['joint_ce']-real['joint_ce'],
                    per_field_ce={k: conditions[c]['per_field_ce'].get(k, float('nan'))-v
                                  for k, v in real['per_field_ce'].items()})
            for c in ('foreign', 'zero_local_mel')}
    return dict(conditions=conditions, gaps_minus_real=gaps,
                note='foreign = the other local song (other mode) at the same request times; '
                     'only two songs are available locally')


# ------------------------------------------------------------------ history-context mode
# (record index, conditions shown?) -- start >= 16 s; phira 11/13/14 (14: incoming hold), mania 28/33/58
# (28 two, 33 three incoming holds). Mania 16 of the plain probe starts at 0 so 28 replaces it.
HISTORY_BATCH = ((11, True), (13, False), (14, True), (28, False), (33, True), (58, False))
HISTORY_EXTRA_DATA = ((3, True), (15, False), (18, True), (24, False), (59, True))   # 18 starts at exactly 16 s
# Repeated-pattern windows: no incoming hold, previous window non-empty (target := previous window + 8 s).
REPEAT_EVAL = ((10, True), (12, False), (41, True), (44, False))     # held out from every fit
REPEAT_TRAIN = ((3, True), (5, False), (7, True), (9, False))        # only used by --copy-train-steps
PREFIX_US = 16_000_000
FOREIGN_GAP_US = 40_000_000
REFERENCE_PARAM_GROUPS = (('reference_projection', ('reference_projection.',)),
                          ('empty_reference', ('empty_reference',)),
                          ('reference_encoder', ('encoders.2.',)))
REFERENCE_KEYS = ('reference', 'reference_times', 'reference_valid')


def reference_grad_norms(model):
    sums = Counter()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        for group, prefixes in REFERENCE_PARAM_GROUPS:
            if any(name == q or name.startswith(q) for q in prefixes):
                sums[group] += float(p.grad.detach().float().square().sum())
    return {k: math.sqrt(sums.get(k, 0.)) for k, _ in REFERENCE_PARAM_GROUPS}


def reference_stats(reference):
    """Row counts, role/kind counts and value stats of an encoded [R,16] reference block."""
    ref = np.asarray(reference)
    col = {n: i for i, n in enumerate(REFERENCE_COLUMNS)}
    out = dict(rows=len(ref), nan=int(np.isnan(ref).sum()), inf=int(np.isinf(ref).sum()),
               event_rows=int(ref[:, col['event']].sum()), known_region_rows=int(ref[:, col['known_region']].sum()),
               unknown_region_rows=int(ref[:, col['unknown_region']].sum()),
               human_input_rows=int(ref[:, col['human_input']].sum()),
               model_history_rows=int(ref[:, col['model_history']].sum()),
               kinds={t: int(ref[:, col[t]].sum()) for t in ('tap', 'hold', 'flick', 'drag')},
               columns={n: stats(ref[:, i]) for n, i in col.items() if n in ('start_relative', 'end_relative')})
    ev = ref[:, col['event']] == 1
    if ev.any():
        out['event_start_s'] = [float(ref[ev, 0].min()*TIME_SCALE_US/1e6), float(ref[ev, 0].max()*TIME_SCALE_US/1e6)]
    return out


def verify_reference_leak(reference, provenance, request, arrays, prefix_us=PREFIX_US):
    """Independent leak assertion on the encoded block + exact provenance.

    No event row may start inside the target window; an event already running at the window start must be
    exactly one of the request's authorized known_carry holds (full tail shown). The reference must hold
    exactly the chart events with onset in [start-L, start)."""
    start, end = request.start_us, request.end_us
    carry = {(e.onset_us, e.end_us, e.lane if request.mode == 'mania_4k' else -1) for e in request.known_carry}
    col = {n: i for i, n in enumerate(REFERENCE_COLUMNS)}
    ref = np.asarray(reference)
    crossing, event_rows, inside = 0, 0, 0
    for row in provenance:                          # exact integers
        if row['event_id'] is None:
            continue
        event_rows += 1
        on, off = row['interval_us']
        inside += int(start <= on < end)
        crossing += int(on < start < off)
    is_event = ref[:, col['event']] == 1            # float32 tensor (~1 us rounding)
    on_us = np.rint(ref[is_event, 0].astype(np.float64)*TIME_SCALE_US) + start
    off_us = np.rint(ref[is_event, 1].astype(np.float64)*TIME_SCALE_US) + start
    tensor_inside = int(((on_us >= start+5) & (on_us < end)).sum())
    lanes = ref[is_event, 6:10].argmax(1) if request.mode == 'mania_4k' else np.full(int(is_event.sum()), -1)
    tensor_crossing = 0
    for a, b, lane in zip(on_us, off_us, lanes):
        if a < start-5 and b > start+5:
            tensor_crossing += 1
            if not any(abs(a-c0) <= 5 and abs(b-c1) <= 5 and lane == cl for c0, c1, cl in carry):
                raise AssertionError('reference hold crossing the window start is not an authorized carry')
    expected = int(((arrays['onset_us'] >= start-prefix_us) & (arrays['onset_us'] < start)).sum())
    if inside or tensor_inside:
        raise AssertionError(f'reference shows target-window events: exact={inside} tensor={tensor_inside}')
    if event_rows != expected:
        raise AssertionError(f'reference holds {event_rows} events, chart has {expected} in the prefix')
    unknown = ref[:, col['unknown_region']] == 1
    unk = [(int(round(r[0]*TIME_SCALE_US))+start, int(round(r[1]*TIME_SCALE_US))+start) for r in ref[unknown]]
    if len(unk) != 1 or abs(unk[0][0]-start) > 5 or abs(unk[0][1]-end) > 5:
        raise AssertionError(f'unknown region is not the target window: {unk}')
    return dict(event_rows=event_rows, expected_chart_events_in_prefix=expected, events_inside_window=inside,
                tensor_events_inside_window=tensor_inside, crossing_holds=crossing, tensor_crossing_holds=tensor_crossing,
                authorized_carry_holds=len(carry), unknown_region_matches_window=True)


def history_sample(dataset, index, shown, *, key=0):
    fields = list(range(len(CONTROL_FIELDS))) if shown else []
    return dataset.sample(index, None, edit_seed=0, curve_specified=shown, specified_fields=fields, history_key=key)


def plain_sample(dataset, index, shown):
    fields = list(range(len(CONTROL_FIELDS))) if shown else []
    return dataset.sample(index, None, edit_seed=0, curve_specified=shown, specified_fields=fields)


def foreign_sources(panel, record, request, *, gap_us=FOREIGN_GAP_US, length_us=PREFIX_US):
    """A prefix from the OTHER side of the same chart (>= gap away), retimed onto this window's clock.

    Only two songs (two modes) exist locally, so a same-mode other-song prefix is impossible; this is a
    different-content, same-mode, same-audio-id prefix. Holds running past the donor boundary are dropped."""
    arrays = chart_arrays(panel, record['chart_index'])
    s = request.start_us
    d = s-gap_us if s-gap_us >= length_us else s+gap_us
    if d > request.duration_us:
        raise ValueError('no donor region available')
    keep = arrays['end_us'] <= d
    donor = shift_arrays({k: v[keep] for k, v in arrays.items()}, s-d)
    source = visible_source(donor, mode=request.mode, audio_sha256=record['audio_sha256'], role='human_input',
                            region=(s-length_us, s), source_id=f"foreign-prefix:{record['chart_id']}")
    return (source,), dict(donor_end_us=d, shift_us=s-d, events=len(source.events))


def foreign_reference(panel, record, request):
    sources, info = foreign_sources(panel, record, request)
    reference, _ = reference_inputs(request, record['audio_sha256'], sources)
    return reference, info


def repeated_events(panel, index):
    """Target of window ``index`` := the human chart's previous window events shifted by one window length.

    Legal by construction: the window has no incoming hold and the previous window is the adjacent one, so
    every copied event (holds included) starts and ends inside the target window."""
    record, prev = panel.records[index], panel.records[index-1]
    request, prev_request = event_request(record['request']), event_request(prev['request'])
    if prev['chart_index'] != record['chart_index'] or prev_request.end_us != request.start_us:
        raise ValueError('previous record is not the adjacent window of the same chart')
    if request.known_carry:
        raise ValueError('repeated window must have no incoming holds')
    shift = request.start_us-prev_request.start_us
    events = []
    for e in panel.targets(index-1):
        moved = replace(e, onset_us=e.onset_us+shift, end_us=e.end_us+shift)
        if not (request.start_us <= moved.onset_us and moved.end_us <= request.end_us):
            raise ValueError('repeated event escapes the target window')
        events.append(moved)
    if not events:
        raise ValueError('previous window is empty')
    return tuple(events)


def repeated_row(dataset, index, shown):
    """Like dataset.sample for the native treatment, with the target replaced by repeated_events()."""
    panel, record = dataset.panel, dataset.panel.records[index]
    request = event_request(record['request'])
    events = repeated_events(panel, index)
    sequence = request_sequence(panel, index, request, dataset.vocab, events=events)
    fields = list(range(len(CONTROL_FIELDS))) if shown else []
    conditions, audit = sample_conditions(request.mode, events, request, curve_specified=shown,
        specified_fields=fields, render_cell_us=dataset.cell_us, curve_cell_us=dataset.curve_cell_us,
        scale=dataset.density_scale)
    inputs = _base_inputs(panel, index, request, conditions['controls'])
    inputs.update(conditions)
    sequence = dict(sequence, **onset_supervision(panel, record, request, events, inputs,
                                                  mel_bins=dataset.mel_bins, scope='window'))
    audit.update(request=request, events=events, native=panel.targets(index), rate=None)
    return dict(inputs=inputs, sequence=sequence), audit


def with_reference(row, reference):
    inputs = dict(row['inputs'])
    for key in REFERENCE_KEYS:
        inputs[key] = reference[key]
    return dict(row, inputs=inputs)


def real_reference(dh, index, request):
    record = dh.panel.records[index]
    draw = draw_history(dh.history, dh.history_seed, index, 0)
    sources, _ = history_sources(chart_arrays(dh.panel, record['chart_index']), request, draw, dh.history,
        chart_id=record['chart_id'], audio_sha256=record['audio_sha256'])
    return reference_inputs(request, record['audio_sha256'], sources)


def history_data_probe(dh, dp, indices):
    rows, totals = [], Counter()
    for index, shown in indices:
        row_h, audit_h = history_sample(dh, index, shown)
        row_p, _ = plain_sample(dp, index, shown)
        record, request = dh.panel.records[index], audit_h['request']
        info = audit_h['history']
        if not (info['applied'] and info['prefix_s'] == 16 and info['role'] == 'human_input' and not info['infill']):
            raise AssertionError(f'prefix not attached for record {index}: {info}')
        reference, provenance = real_reference(dh, index, request)
        if not np.array_equal(reference['reference'], row_h['inputs']['reference']):
            raise AssertionError('loader reference differs from the re-derived reference')
        leak = verify_reference_leak(reference['reference'], provenance, request,
                                     chart_arrays(dh.panel, record['chart_index']))
        same_other = all(np.array_equal(np.asarray(row_h['inputs'][k]), np.asarray(row_p['inputs'][k]))
                         for k in row_h['inputs'] if k not in REFERENCE_KEYS)
        same_seq = all(np.array_equal(np.asarray(row_h['sequence'][k]), np.asarray(row_p['sequence'][k]))
                       for k in row_h['sequence'])
        if not (same_other and same_seq):
            raise AssertionError('attaching the prefix changed non-reference inputs or the target sequence')
        entry = dict(index=index, mode=record['mode'], start_s=request.start_us/1e6, conditions_shown=shown,
            known_carry=len(request.known_carry), target_events=len(audit_h['events']),
            history=dict(info), with_prefix=reference_stats(reference['reference']),
            no_prefix=reference_stats(row_p['inputs']['reference']), leak_check=leak,
            non_reference_inputs_and_targets_identical=True,
            reference_tensors={k: stats(row_h['inputs'][k]) for k in REFERENCE_KEYS})
        rows.append(entry)
        for part in ('with_prefix', 'no_prefix'):
            totals['nan'] += entry[part]['nan']; totals['inf'] += entry[part]['inf']
        totals['events_inside_window'] += leak['events_inside_window'] + leak['tensor_events_inside_window']
        totals['crossing_holds'] += leak['crossing_holds']
        totals['reference_event_rows'] += leak['event_rows']
    return dict(rows=rows, totals=dict(totals))


def per_row_ce(model, inp, seq):
    model.eval()
    with torch.no_grad():
        logits = model(inp, seq)
        legal = logits.float().masked_fill(~seq['allowed'], -1e9)
        ce = torch.nn.functional.cross_entropy(legal.flatten(0, 1), seq['targets'].flatten(),
                                               reduction='none').view_as(seq['targets'])
        free = seq['valid'] & (seq['allowed'].sum(-1) > 1)
        return [float(ce[b][free[b]].mean()) if free[b].any() else float('nan') for b in range(ce.shape[0])]


def usage_check(model, rows, references, device):
    """Teacher CE of the same target rows with each named reference swapped in (reference = only change)."""
    out = {}
    for name, refs in references.items():
        inp, seq = collate([with_reference(row, ref) for row, ref in zip(rows, refs)])
        inp, seq = to(device, inp), to(device, seq)
        result = teacher_eval(model, inp, seq)
        out[name] = dict(joint_ce=result['terms']['joint_ce'], type_ce=result['terms']['type_ce'],
            first_token_ce=result['terms']['first_token_ce'],
            free_token_ce=result['per_field_ce']['_all_free_tokens']['ce'],
            per_row_free_token_ce=per_row_ce(model, inp, seq),
            per_field_ce={k: v['ce'] for k, v in result['per_field_ce'].items()})
    real = out['real_prefix']
    for name in ('no_reference', 'foreign_prefix'):
        out[name]['gap_minus_real_joint_ce'] = out[name]['joint_ce']-real['joint_ce']
        out[name]['gap_minus_real_free_token_ce'] = out[name]['free_token_ce']-real['free_token_ce']
        out[name]['gap_minus_real_per_field'] = {
            k: out[name]['per_field_ce'].get(k, float('nan'))-v for k, v in real['per_field_ce'].items()}
    return out


def pick_ref(row):
    return {k: row[k] for k in REFERENCE_KEYS} if 'reference' in row else {k: row['inputs'][k] for k in REFERENCE_KEYS}


def fit_history(model, inp, seq, steps, lr, warmup):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=0.)
    curve, norms = [], {}
    for step in range(1, steps+1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = lr*min(1., step/max(1, warmup))
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(inp, seq, return_aux=True)
        terms = objective_terms(logits, seq, aux=aux, **LOSS)
        terms['total'].backward()
        ref_norm = reference_grad_norms(model)
        if step in (1, 2, 10, 25, 50, 75, steps):
            norms[step] = dict(grad_norms(model), **{'ref/'+k: v for k, v in ref_norm.items()})
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        point = dict(step=step, grad_norm_preclip=total_norm, ref_grad=ref_norm,
                     **{k: round(float(v), 6) for k, v in terms.items() if k != 'onset_active'})
        curve.append(point)
        if step % 10 == 0 or step == 1:
            print(json.dumps(dict(phase='fit', **point)), flush=True)
    return curve, norms


def reference_sets(dh, dp, items, audits, rows=None):
    """real / blank / foreign reference triples for the given (index, shown) items."""
    panel = dh.panel
    real = [{k: v[k] for k in REFERENCE_KEYS} for v in (real_reference(dh, i, a['request'])[0]
                                                       for (i, _), a in zip(items, audits))]
    blank = [pick_ref(plain_sample(dp, i, s)[0]) for i, s in items]
    foreign, info = [], []
    for (i, _), a in zip(items, audits):
        ref, finfo = foreign_reference(panel, panel.records[i], a['request'])
        foreign.append({k: ref[k] for k in REFERENCE_KEYS}); info.append(finfo)
    return dict(real_prefix=real, no_reference=blank, foreign_prefix=foreign), info


def history_main(args):
    started = time.monotonic()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    out = ROOT/(args.out or 'reports/recipe-v2-probe/history')
    out.mkdir(parents=True, exist_ok=True)
    plan, panel, dh = build(args.seed, HISTORY)
    dp = make_dataset(panel, dh.probe_exposure, None)
    vocab = dh.vocab
    result = dict(schema='recipe-v2-rule1-probe-history-context', identity=identity(panel, plan, args.seed),
        recipe=dict(config=CONFIG, vocabulary=VOCABULARY, loss=LOSS, onset_supervision=dict(scope='window'),
                    curve_cell_us=2_000_000, render_cell_us=100_000, dropout=0., augmentation='none (native treatment)',
                    history_context=dh.history, history_key=0,
                    foreign_prefix=f'same-chart region >= {FOREIGN_GAP_US/1e6:.0f}s away, retimed (only 2 songs/2 modes local)'),
        device=str(device))
    # 1. data probe (windows start >= 16 s)
    tick = time.monotonic()
    items = list(HISTORY_BATCH)+list(HISTORY_EXTRA_DATA)
    assert all(panel.records[i]['request']['start_us'] >= PREFIX_US for i, _ in items)
    result['data_probe'] = history_data_probe(dh, dp, items)
    result['data_probe']['wall_s'] = time.monotonic()-tick
    # 2. overfit with the prefix attached
    rows, audits = [], []
    for index, shown in HISTORY_BATCH:
        row, audit = history_sample(dh, index, shown)
        rows.append(row); audits.append(audit)
    inp_cpu, seq_cpu = collate(rows)
    inp, seq = to(device, inp_cpu), to(device, seq_cpu)
    model = FullEventModel(FullEventConfig(**dict(plan['config'])), vocab).to(device)
    start_eval = teacher_eval(model, inp, seq)
    tick = time.monotonic()
    curve, norms = fit_history(model, inp, seq, args.steps, args.lr, args.warmup)
    train_s = time.monotonic()-tick
    end_eval = teacher_eval(model, inp, seq)
    batch_desc = [dict(index=i, mode=panel.records[i]['mode'], conditions_shown=s, start_us=a['request'].start_us,
                       carry=len(a['request'].known_carry), reference_rows=int(r['inputs']['reference'].shape[0]),
                       types=dict(Counter(e.type for e in a['events'])), teacher_tokens=int(r['sequence']['valid'].sum()))
                  for (i, s), a, r in zip(HISTORY_BATCH, audits, rows)]
    result['overfit'] = dict(batch=batch_desc, steps=args.steps, train_s=train_s, start=start_eval, end=end_eval,
        optimizer=dict(name='AdamW', lr=args.lr, warmup_steps=args.warmup, betas=[.9, .95], weight_decay=0., clip=1.),
        grad_norms_by_group=norms, curve=curve,
        input_shapes={k: list(v.shape) for k, v in inp_cpu.items() if torch.is_tensor(v)})
    # generation exactness on the fitted batch (prefix attached)
    requests = [a['request'] for a in audits]
    truths = [[asdict(e) for e in a['events']] for a in audits]
    request_dicts = [dict(mode=r.mode, start_us=r.start_us, end_us=r.end_us) for r in requests]
    teacher_tokens = [encode_events(a['events'], a['request'], vocab) for a in audits]
    seeds = [int(s) for s in args.sample_seeds.split(',') if s]
    tick = time.monotonic()
    result['generation'] = generation(model, inp, requests, truths, request_dicts, vocab, teacher_tokens, device, seeds)
    result['generation_wall_s'] = time.monotonic()-tick
    # 3a. usage check on the fitted batch
    refs, finfo = reference_sets(dh, dp, HISTORY_BATCH, audits)
    result['usage_fitted_batch'] = dict(usage_check(model, rows, refs, device), foreign_info=finfo)
    # 3b. repeated pattern, windows never fitted: target = previous window + 8 s, prefix = real human prefix
    rep_rows, rep_audits = [], []
    for index, shown in REPEAT_EVAL:
        row, audit = repeated_row(dh, index, shown)
        rep_rows.append(with_reference(row, real_reference(dh, index, audit['request'])[0])); rep_audits.append(audit)
    rep_refs, rep_finfo = reference_sets(dh, dp, REPEAT_EVAL, rep_audits)
    result['usage_repeated_unfitted'] = dict(
        windows=[dict(index=i, mode=panel.records[i]['mode'], start_s=a['request'].start_us/1e6,
                      target_events=len(a['events'])) for (i, _), a in zip(REPEAT_EVAL, rep_audits)],
        **usage_check(model, rep_rows, rep_refs, device), foreign_info=rep_finfo)
    (out/'probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str)+'\n')
    # 3c. optional: a fresh model fitted on the batch plus repeated train windows
    if args.copy_train_steps:
        tr_rows, tr_audits = [], []
        for index, shown in REPEAT_TRAIN:
            row, audit = repeated_row(dh, index, shown)
            tr_rows.append(with_reference(row, real_reference(dh, index, audit['request'])[0])); tr_audits.append(audit)
        # phira-only (short sequences): the dense mania windows are too slow/heavy for the extra fit.
        sel = [i for i, (idx, _) in enumerate(HISTORY_BATCH) if panel.records[idx]['mode'] == 'phira']
        rsel = [i for i, (idx, _) in enumerate(REPEAT_EVAL) if panel.records[idx]['mode'] == 'phira']
        fit_rows = [rows[i] for i in sel]+tr_rows
        torch.manual_seed(args.seed+1)
        model2 = FullEventModel(FullEventConfig(**dict(plan['config'])), vocab).to(device)
        inp2, seq2 = collate(fit_rows)
        inp2, seq2 = to(device, inp2), to(device, seq2)
        tick = time.monotonic()
        curve2, norms2 = fit_history(model2, inp2, seq2, args.copy_train_steps, args.lr, args.warmup)
        tr_refs, _ = reference_sets(dh, dp, REPEAT_TRAIN, tr_audits)
        sub = lambda refs, ids: {k: [v[i] for i in ids] for k, v in refs.items()}
        result['copy_variant'] = dict(steps=args.copy_train_steps, train_s=time.monotonic()-tick,
            fit_windows=[HISTORY_BATCH[i][0] for i in sel]+[i for i, _ in REPEAT_TRAIN],
            heldout_repeated=[REPEAT_EVAL[i][0] for i in rsel], grad_norms_by_group=norms2, curve_tail=curve2[-3:],
            train_repeated=usage_check(model2, tr_rows, tr_refs, device),
            heldout_repeated_usage=usage_check(model2, [rep_rows[i] for i in rsel], sub(rep_refs, rsel), device),
            fitted_phira_usage=usage_check(model2, [rows[i] for i in sel], sub(refs, sel), device))
    result['wall_s'] = time.monotonic()-started
    path = out/'probe.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str)+'\n')
    u, r = result['usage_fitted_batch'], result['usage_repeated_unfitted']
    print(json.dumps(dict(phase='done', path=str(path), wall_s=result['wall_s'],
        start_joint_ce=start_eval['terms']['joint_ce'], end_joint_ce=end_eval['terms']['joint_ce'],
        greedy_exact=result['generation']['greedy']['exact_rows'],
        fitted_ce={k: u[k]['joint_ce'] for k in ('real_prefix', 'no_reference', 'foreign_prefix')},
        repeated_ce={k: r[k]['joint_ce'] for k in ('real_prefix', 'no_reference', 'foreign_prefix')})), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None)
    parser.add_argument('--history-context', action='store_true',
                        help='opt-in: windows start >= 16 s with a 16 s human_input prefix reference')
    parser.add_argument('--copy-train-steps', type=int, default=0,
                        help='history mode: also fit a fresh model on the batch plus repeated-pattern train '
                             'windows for N steps (0 = skip)')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--device', default=None)
    parser.add_argument('--sample-seeds', default='0,1,2,3')
    args = parser.parse_args(argv)
    if args.history_context:
        return history_main(args)
    args.out = args.out or 'reports/recipe-v2-probe'
    started = time.monotonic()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    plan, panel, dataset = build(args.seed)
    vocab = dataset.vocab
    result = dict(schema='recipe-v2-rule1-probe', identity=identity(panel, plan, args.seed),
        recipe=dict(config=CONFIG, vocabulary=VOCABULARY, loss=LOSS, onset_supervision=dict(scope='window'),
                    curve_cell_us=2_000_000, render_cell_us=100_000, dropout=0., augmentation='none (native treatment)',
                    soft_targets='onset head: Gaussian soft targets (sigma = 1 mel hop), nonzero BCE lower bound reported'),
        device=str(device))
    # 1. data probe
    tick = time.monotonic()
    result['data_probe'] = data_probe(dataset, list(BATCH)+[(i, i % 2 == 0) for i in EXTRA_DATA])
    result['data_probe']['wall_s'] = time.monotonic()-tick
    # 2. overfit probe
    rows, audits = [], []
    for index, shown in BATCH:
        row, audit = sample(dataset, index, shown)
        rows.append(row); audits.append(audit)
    inp_cpu, seq_cpu = collate(rows)
    inp, seq = to(device, inp_cpu), to(device, seq_cpu)
    config = FullEventConfig(**dict(plan['config']))
    model = FullEventModel(config, vocab).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    unassigned = sorted({group_of(n) for n, _ in model.named_parameters()} - {g for g, _ in GROUPS})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=0.)
    batch_desc = [dict(index=i, mode=panel.records[i]['mode'], conditions_shown=s,
                       start_us=a['request'].start_us, carry=len(a['request'].known_carry),
                       types=dict(Counter(e.type for e in a['events'])), teacher_tokens=int(r['sequence']['valid'].sum()))
                  for (i, s), a, r in zip(BATCH, audits, rows)]
    start_eval = teacher_eval(model, inp, seq)
    curve, norms = [], {}
    tick = time.monotonic()
    for step in range(1, args.steps+1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = args.lr*min(1., step/max(1, args.warmup))
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(inp, seq, return_aux=True)
        terms = objective_terms(logits, seq, aux=aux, **LOSS)
        terms['total'].backward()
        if step in (1, 2, 10, 25, 50, 75, args.steps):
            norms[step] = grad_norms(model)
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        point = dict(step=step, grad_norm_preclip=total_norm, lr=optimizer.param_groups[0]['lr'],
                     **{k: round(float(v), 6) for k, v in terms.items() if k != 'onset_active'})
        curve.append(point)
        if step % 10 == 0 or step == 1:
            print(json.dumps(dict(phase='overfit', **point)), flush=True)
    train_s = time.monotonic()-tick
    end_eval = teacher_eval(model, inp, seq)
    # 3. free generation on the fitted requests
    requests = [a['request'] for a in audits]
    truths = [[asdict(e) for e in a['events']] for a in audits]
    request_dicts = [dict(mode=r.mode, start_us=r.start_us, end_us=r.end_us) for r in requests]
    teacher_tokens = [encode_events(a['events'], a['request'], vocab) for a in audits]
    seeds = [int(s) for s in args.sample_seeds.split(',') if s]
    tick = time.monotonic()
    result['generation'] = generation(model, inp, requests, truths, request_dicts, vocab, teacher_tokens,
                                      device, seeds)
    result['generation_wall_s'] = time.monotonic()-tick
    # 4. audio dependence on the fitted batch
    result['audio_dependence'] = audio_dependence(model, dataset, rows, audits, device)
    result['overfit'] = dict(batch=batch_desc, parameters=parameters, unassigned_param_groups=unassigned,
        optimizer=dict(name='AdamW', lr=args.lr, warmup_steps=args.warmup, betas=[.9, .95], weight_decay=0.,
                       clip=1., schedule='linear warmup then constant'),
        steps=args.steps, train_s=train_s, start=start_eval, end=end_eval, grad_norms_by_group=norms, curve=curve,
        input_shapes={k: list(v.shape) for k, v in inp_cpu.items() if torch.is_tensor(v)},
        sequence_shapes={k: list(v.shape) for k, v in seq_cpu.items() if torch.is_tensor(v)})
    result['wall_s'] = time.monotonic()-started
    path = out/'probe.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str)+'\n')
    print(json.dumps(dict(phase='done', path=str(path), wall_s=result['wall_s'],
        start_joint_ce=start_eval['terms']['joint_ce'], end_joint_ce=end_eval['terms']['joint_ce'],
        greedy_exact=result['generation']['greedy']['exact_rows'])), flush=True)
    return result


if __name__ == '__main__':
    main()

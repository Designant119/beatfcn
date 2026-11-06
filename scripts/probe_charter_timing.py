"""AGENTS Rule 1 mini check for the optional charter-timing input (FullEventConfig.beat_features).

Same local real data / small recipe-v2 config as scripts/probe_recipe_v2_overfit.py, plus
beat_features=True and the windows' own chart timing (reports/chart-timing/<probe tape>.json,
scripts/build_chart_timing.py) attached through the real CounterfactualExposureDataset path
(dropout 0 for the fitted batch). Reports:

1. data probe of ``beat`` (shape, NaN/Inf, min/max/mean/std, presence per row) and the human
   target's on-grid fraction against the same timing;
2. 100-step overfit on one fixed batch with timing: per-term loss, gradient norm of
   beat_projection (zero-initialised: its weight gradient must be nonzero from step 1);
3. on the fitted batch: teacher CE and greedy (+ T=1) outputs with timing / without timing
   (absent state) / timing shifted by +20 ms; on-grid fraction (onset within 5 ms of a 1/1..1/16
   or triplet point of the chart timing) of the generated onsets vs the human and chance.

  PYTHONPATH=. .venv/bin/python -m scripts.probe_charter_timing --out reports/charter-timing-probe
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.charter_timing import ChartTimingTable, window_beat_features
from beatfcn.train.checkpoint_resume import file_sha
from beatfcn.train.exposure_loader import CounterfactualExposureDataset
from beatfcn.train.local_panel import collate
from beatfcn.train.training_objective import objective_terms
from beatfcn.train.workflow_metrics import chance_on_grid, beat_length_at, grid_profile
from scripts.probe_recipe_v2_overfit import (BATCH, CONFIG, LOSS, VOCABULARY, build, compare, grad_norms,
                                             identity, stats, teacher_eval, to)

ROOT = Path(__file__).resolve().parents[1]
TABLE = 'reports/chart-timing/data-contract-v2-final-probe.json'
SHIFT_US = 20_000


def grid_of(segments, shift_us=0):
    return [(float(s)+shift_us, float(b)) for s, b, _ in segments]


def on_grid(events, grid, start_us, end_us):
    notes = [(int(e['onset_us']),) for e in events if start_us <= int(e['onset_us']) < end_us]
    profile = grid_profile(notes, grid)
    return profile['on_grid_fraction'], profile['onsets']


def chance(grid, events):
    heads = np.unique([int(e['onset_us']) for e in events])
    return float(np.mean([chance_on_grid(b) for b in beat_length_at(heads, grid)])) if len(heads) else None


def with_beat(inp_cpu, rows, audits, table, *, mode):
    """Batch inputs with the charts' timing ('with'), absent ('without') or shifted by +SHIFT_US."""
    out = dict(inp_cpu)
    if mode == 'without':
        out['beat'] = torch.zeros_like(inp_cpu['beat'])
        return out
    beat = torch.zeros_like(inp_cpu['beat'])
    for i, (row, audit) in enumerate(zip(rows, audits)):
        segments = table.segments(audit['chart_id']).copy()
        if mode == 'shifted':
            segments[:, 0] += SHIFT_US
        values = window_beat_features(row['inputs'], audit['request'].start_us, segments)
        beat[i, :len(values)] = torch.from_numpy(values)
    out['beat'] = beat
    return out


def readout(model, inp, requests, truths, teacher_tokens, audits, table, device, *, seeds):
    model.eval()
    max_tokens = int(2*max(len(t) for t in teacher_tokens))+8
    out = {}
    for name, temperature, seed in [('greedy', None, None)]+[(f'sample_t1_seed{s}', 1., s) for s in seeds]:
        kwargs = {}
        if temperature is not None:
            gdev = device.type if device.type in ('cuda', 'mps') else 'cpu'
            kwargs = dict(temperature=temperature, generator=[torch.Generator(device=gdev).manual_seed(1000*seed+i)
                                                              for i in range(len(requests))])
        predictions = generate_batch(model, inp, requests, max_tokens, **kwargs)
        rows = []
        for p, truth, req, audit in zip(predictions, truths, requests, audits):
            grid = grid_of(table.segments(audit['chart_id']))
            c = compare(p, truth, dict(mode=req.mode, start_us=req.start_us, end_us=req.end_us))
            c['on_grid_fraction'], c['onsets'] = on_grid(p['events'], grid, req.start_us, req.end_us)
            c['on_grid_shifted_grid'], _ = on_grid(p['events'], grid_of(table.segments(audit['chart_id']), SHIFT_US),
                                                   req.start_us, req.end_us)
            rows.append(c)
        pooled = sum((r['on_grid_fraction'] or 0)*r['onsets'] for r in rows)/max(1, sum(r['onsets'] for r in rows))
        pooled_shift = (sum((r['on_grid_shifted_grid'] or 0)*r['onsets'] for r in rows)
                        / max(1, sum(r['onsets'] for r in rows)))
        out[name] = dict(rows=rows, onsets=sum(r['onsets'] for r in rows), on_grid_fraction_pooled=pooled,
                         on_grid_vs_shifted_grid_pooled=pooled_shift,
                         exact_rows=sum(r['exact_multiset'] for r in rows),
                         mean_onset_f1_20ms=float(np.mean([r['onset_f1_20ms'] for r in rows])),
                         mean_note_f1_50ms=float(np.mean([r['note_f1_50ms'] for r in rows])))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='reports/charter-timing-probe')
    p.add_argument('--table', default=TABLE)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--device', default=None)
    p.add_argument('--sample-seeds', default='0,1')
    args = p.parse_args(argv)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    plan, panel, base = build(args.seed)
    timing = dict(path=args.table, sha256=file_sha(ROOT/args.table), dropout=0.)
    dataset = CounterfactualExposureDataset(panel, base.probe_exposure, microbatch=4, cell_us=100_000,
        curve_cell_us=2_000_000, vocabulary=VOCABULARY, onset=dict(scope='window'), mel_bins=CONFIG['mel_bins'],
        charter_timing=timing)
    table = dataset.timing_table
    config = FullEventConfig(**dict(plan['config'], beat_features=True))
    result = dict(schema='charter-timing-rule1-probe', identity=dict(identity(panel, plan, args.seed),
                  timing_table=timing, code_sha256_extra={k: file_sha(ROOT/k) for k in (
                      'beatfcn/train/charter_timing.py', 'scripts/probe_charter_timing.py')}),
                  config=asdict(config), loss=LOSS, device=str(device))
    rows, audits = [], []
    for index, shown in BATCH:
        fields = list(range(8)) if shown else []
        row, audit = dataset.sample(index, None, edit_seed=0, curve_specified=shown, specified_fields=fields,
                                    timing_key=0)
        audit['chart_id'] = panel.records[index]['chart_id']
        rows.append(row)
        audits.append(audit)
    inp_cpu, seq_cpu = collate(rows)
    # 1. data probe
    data = dict(beat=stats(inp_cpu['beat'].numpy()), rows=[])
    for (index, shown), row, audit in zip(BATCH, rows, audits):
        beat = row['inputs']['beat']
        grid = grid_of(table.segments(audit['chart_id']))
        truth = [asdict(e) for e in audit['events']]
        req = audit['request']
        human, n = on_grid(truth, grid, req.start_us, req.end_us)
        data['rows'].append(dict(index=index, mode=panel.records[index]['mode'], chart_id=audit['chart_id'],
            timing=audit['charter_timing'], frames=int(len(beat)),
            present_frames=int(beat[:, -1].sum()), mel_valid_frames=int(np.asarray(row['inputs']['mel_valid']).sum()),
            nan=int(np.isnan(beat).sum()), human_on_grid=human, human_onsets=n, chance_on_grid=chance(grid, truth),
            first_bpm=round(60e6/grid[0][1], 3)))
    result['data_probe'] = data
    print(json.dumps(dict(phase='data', rows=[{k: r[k] for k in ('index', 'mode', 'present_frames', 'frames',
                                                               'human_on_grid', 'chance_on_grid')} for r in data['rows']]),
                     default=str), flush=True)
    # 2. overfit with timing
    inp, seq = to(device, inp_cpu), to(device, seq_cpu)
    model = FullEventModel(config, dataset.vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=0.)
    start_eval = teacher_eval(model, inp, seq)
    curve, beat_grad = [], {}
    tick = time.monotonic()
    for step in range(1, args.steps+1):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = args.lr*min(1., step/max(1, args.warmup))
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(inp, seq, return_aux=True)
        terms = objective_terms(logits, seq, aux=aux, **LOSS)
        terms['total'].backward()
        g = model.beat_projection.weight.grad
        beat_grad[step] = dict(weight=float(g.norm()), bias=float(model.beat_projection.bias.grad.norm()),
                               weight_param_norm=float(model.beat_projection.weight.detach().norm()))
        if step in (1, 10, 50, args.steps):
            beat_grad[step]['all_groups'] = grad_norms(model)
        total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        point = dict(step=step, grad_norm_preclip=total, beat_projection_grad=beat_grad[step]['weight'],
                     **{k: round(float(v), 6) for k, v in terms.items() if k != 'onset_active'})
        curve.append(point)
        if step % 10 == 0 or step == 1:
            print(json.dumps(dict(phase='overfit', **point)), flush=True)
    train_s = time.monotonic()-tick
    end_eval = teacher_eval(model, inp, seq)
    result['overfit'] = dict(steps=args.steps, train_s=train_s, start=start_eval, end=end_eval, curve=curve,
                             beat_projection=dict(grad=beat_grad, final_weight_norm=float(model.beat_projection.weight.norm()),
                                                  final_bias_norm=float(model.beat_projection.bias.norm())))
    # 3. with / without / shifted timing on the fitted batch
    requests = [a['request'] for a in audits]
    truths = [[asdict(e) for e in a['events']] for a in audits]
    teacher_tokens = [encode_events(a['events'], a['request'], dataset.vocab) for a in audits]
    seeds = [int(s) for s in args.sample_seeds.split(',') if s]
    variants = {}
    for mode in ('with', 'without', 'shifted'):
        x = to(device, with_beat(inp_cpu, rows, audits, table, mode=mode))
        variants[mode] = dict(teacher=teacher_eval(model, x, seq)['terms'],
                              generation=readout(model, x, requests, truths, teacher_tokens, audits, table, device,
                                                 seeds=seeds))
        g = variants[mode]['generation']['greedy']
        print(json.dumps(dict(phase='readout', timing=mode, joint_ce=variants[mode]['teacher']['joint_ce'],
                              greedy_on_grid=g['on_grid_fraction_pooled'],
                              greedy_on_shifted_grid=g['on_grid_vs_shifted_grid_pooled'], onsets=g['onsets'],
                              exact_rows=g['exact_rows'], onset_f1_20ms=g['mean_onset_f1_20ms'])), flush=True)
    human_onsets = sum(r['human_onsets'] for r in data['rows'])
    result['readouts'] = variants
    result['summary'] = dict(
        human_on_grid_pooled=sum((r['human_on_grid'] or 0)*r['human_onsets'] for r in data['rows'])/max(1, human_onsets),
        chance_on_grid_mean=float(np.mean([r['chance_on_grid'] for r in data['rows'] if r['chance_on_grid'] is not None])),
        joint_ce=dict(start=start_eval['terms']['joint_ce'], end=end_eval['terms']['joint_ce'],
                      **{m: v['teacher']['joint_ce'] for m, v in variants.items()}),
        greedy={m: dict(on_grid=v['generation']['greedy']['on_grid_fraction_pooled'],
                        on_shifted_grid=v['generation']['greedy']['on_grid_vs_shifted_grid_pooled'],
                        onsets=v['generation']['greedy']['onsets'], exact_rows=v['generation']['greedy']['exact_rows'],
                        onset_f1_20ms=v['generation']['greedy']['mean_onset_f1_20ms'])
                for m, v in variants.items()},
        beat_projection_grad_step1=beat_grad[1]['weight'], beat_projection_grad_last=beat_grad[args.steps]['weight'])
    result['wall_s'] = time.monotonic()-started
    path = out/'probe.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str)+'\n')
    print(json.dumps(dict(phase='done', path=str(path), wall_s=result['wall_s'], summary=result['summary']), default=str),
          flush=True)
    return result


if __name__ == '__main__':
    main()

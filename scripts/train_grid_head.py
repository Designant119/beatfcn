"""Self-timing grid head (Track B1): from the FROZEN BeatFCN encoder's 10 ms frames, predict the charter grid.

The encoded mel frames (the input of the model's onset head, captured with a forward pre-hook; encoded WITHOUT
any charter timing, i.e. the cold condition) feed a small MLP that regresses, per frame, the first 8 charter
beat features (sin/cos of the beat, bar, half-beat and quarter-beat phase; beatfcn.train.charter_timing). Labels
come from the charter timing table of osu! charts only (Phira BPM lists can be tempo multiples). The model is
never updated; a later pass fits a global grid to the predicted phases and feeds it back as timing input.

  python -m scripts.train_grid_head --plan PLAN --checkpoint CK --weights ema --device cuda --out-dir DIR \
      [--train-windows 24000 --val-windows 1200 --steps 3000 --batch 16]
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_DIMS = 8


def pick_windows(panel, table, split, count, seed):
    rows = [i for i, r in enumerate(panel.records) if r['split'] == split and r['chart_id'].startswith('osu:')
            and table.status(r['chart_id']) == 'valid']
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:count]


class FrameTap:
    """Captures the onset head's input (encoded 10 ms frames) during model.encode."""

    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        self.value = args[0]


def make_head(d):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, TARGET_DIMS))


def batch_frames(model, panel, table, indices, device, tap):
    """(features [B,T,d] float32, labels [B,T,8], mask [B,T]) for a batch of windows, cold encode
    (unconditional controls / curve, no charter timing input)."""
    import torch
    from beatfcn.train.charter_timing import window_beat_features
    from beatfcn.train.counterfactual_exposure import unconditional_inputs
    from beatfcn.train.exposure_loader import _base_inputs
    from beatfcn.train.local_panel import collate
    from beatfcn.train.local_request_panel import event_request
    rows, labels, masks = [], [], []
    for i in indices:
        rec = panel.records[i]
        req = event_request(rec['request'])
        cond = unconditional_inputs(req.mode, req)
        inp = _base_inputs(panel, i, req, cond['controls'])
        inp.update(cond)
        lab = window_beat_features(inp, req.start_us, table.segments(rec['chart_id']))
        labels.append(lab[:, :TARGET_DIMS]); masks.append(lab[:, -1] > 0.5)
        rows.append(dict(inputs=inp, sequence=panel.sequence(i)))
    inputs, _ = collate(rows)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        model.encode(inputs)
    feats = tap.value.float()
    T = feats.shape[1]
    lab = np.zeros((len(indices), T, TARGET_DIMS), np.float32); m = np.zeros((len(indices), T), bool)
    for k, (l, mk) in enumerate(zip(labels, masks)):
        n = min(T, len(l)); lab[k, :n] = l[:n]; m[k, :n] = mk[:n]
    return feats, torch.from_numpy(lab).to(feats.device), torch.from_numpy(m).to(feats.device)


def phase_errors(pred, lab, mask):
    """Beat-phase absolute error (fraction of a beat) on masked frames."""
    import torch
    p = torch.atan2(pred[..., 0], pred[..., 1]); t = torch.atan2(lab[..., 0], lab[..., 1])
    d = torch.remainder(p - t + math.pi, 2 * math.pi) - math.pi
    return (d.abs() / (2 * math.pi))[mask]


def main():
    import torch
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--weights', choices=('raw', 'ema'), default='ema')
    a.add_argument('--device', default='cuda')
    a.add_argument('--root', type=Path, default=Path('/'))
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--train-windows', type=int, default=24000)
    a.add_argument('--val-windows', type=int, default=1200)
    a.add_argument('--steps', type=int, default=3000)
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--seed', type=int, default=20261103)
    a.add_argument('--max-wall-s', type=float, default=3000.)
    args = a.parse_args()
    started = time.monotonic()
    from scripts.run_continuation_benchmark import checkpoint_setup
    from beatfcn.train.charter_timing import ChartTimingTable, normalize_charter_timing
    model, panel, identity, _, _ = checkpoint_setup(args)
    for p in model.parameters():
        p.requires_grad_(False)
    plan = json.loads(args.plan.read_text())
    table = ChartTimingTable.from_settings(normalize_charter_timing(plan['charter_timing']), root=args.root)
    train = pick_windows(panel, table, 'train', args.train_windows, args.seed)
    val = pick_windows(panel, table, 'val', args.val_windows, args.seed + 1)
    tap = FrameTap(model.onset_norm)
    head = make_head(model.c.d).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(args.seed)
    log = []
    for step in range(args.steps):
        if time.monotonic() - started > args.max_wall_s:
            print(json.dumps(dict(phase='wall_limit', step=step)), flush=True); break
        idx = rng.sample(train, args.batch)
        feats, lab, mask = batch_frames(model, panel, table, idx, args.device, tap)
        pred = head(feats)
        loss = ((pred - lab) ** 2).mean(-1)[mask].mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            err = phase_errors(pred.detach(), lab, mask)
            rec = dict(phase='train', step=step, loss=round(float(loss), 4), beat_err_median=round(float(err.median()), 4),
                       wall_s=round(time.monotonic() - started))
            log.append(rec); print(json.dumps(rec), flush=True)
    head.eval(); errs = []
    with torch.no_grad():
        for k in range(0, len(val), args.batch):
            feats, lab, mask = batch_frames(model, panel, table, val[k:k + args.batch], args.device, tap)
            errs.append(phase_errors(head(feats), lab, mask).cpu())
    e = torch.cat(errs)
    report = dict(schema='beatfcn.grid-head.v1', model=identity, train_windows=len(train), val_windows=len(val),
                  steps=args.steps, val_beat_phase_error=dict(median=float(e.median()), p90=float(e.quantile(.9)),
                  within_5pct=float((e <= .05).float().mean()), within_10pct=float((e <= .10).float().mean())),
                  log=log, wall_s=round(time.monotonic() - started))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=head.state_dict(), d=model.c.d, target_dims=TARGET_DIMS, report=report), args.out_dir / 'grid-head.pt')
    (args.out_dir / 'grid-head.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'log'}, indent=1))


if __name__ == '__main__':
    main()

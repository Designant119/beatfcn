"""Zero-training teacher CE of a checkpoint on the frozen held-out panel, per field and mode, for gross kills.

Reuses the benchmark v2 teacher screen (audio_dependence.run_teacher_screen) on real audio only, and adds a Phira kind
calibration: teacher-forced marginal vs target share, at T 1 and T .3. Windows without features are skipped and reported
(BEATFCN_ALLOW_MISSING_MERT=1 on a machine holding a subset).

  BEATFCN_ALLOW_MISSING_MERT=1 python -m scripts.teacher_ce_probe --checkpoint CAND.pt --plan /data/round5-level/plan.json \
      --protocol /data/evolve/teacher-panel.json --out probe/CAND.json [--baseline probe/r5.json]
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beatfcn.events.canonical import TYPES  # noqa: E402
from beatfcn.model.full_event_codec import FIELDS  # noqa: E402
from beatfcn.train.audio_dependence import (FIELD_GROUPS, base_inputs, run_teacher_screen,  # noqa: E402
                                            teacher_target)
from beatfcn.train.local_panel import collate  # noqa: E402
from scripts.evolve_surgery import load_checkpoint_model, macs_per_audio_second  # noqa: E402

CALIBRATION_T = (1., .3)


def real_audio_protocol(protocol, panel, max_windows=None):
    """Real-audio-only copy: drops the counterfactual donors, and windows whose song has no features."""
    missing = set(getattr(panel, 'missing_audio', ()))
    windows = [{k: v for k, v in w.items() if k not in ('foreign_index', 'foreign_audio_sha256', 'roll_s')}
               for w in protocol['windows'] if w['audio_sha256'] not in missing]
    if max_windows:
        windows = windows[:max_windows]
    return dict(protocol, windows=windows, conditions=['real']), len(protocol['windows'])-len(windows)


@torch.no_grad()
def kind_calibration(model, panel, protocol, *, device, batch_size=8, autocast=True):
    """Phira kind tokens under teacher forcing: does the model's marginal match the target mix?"""
    vocab, device = model.vocab, torch.device(device)
    kinds = torch.as_tensor([vocab.token('kind', i) for i in range(len(TYPES))], device=device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if autocast and device.type == 'cuda' else nullcontext()
    windows = [w for w in protocol['windows'] if w['mode'] == 'phira']
    target = torch.zeros(len(TYPES))
    predicted = {t: torch.zeros(len(TYPES)) for t in CALIBRATION_T}
    own = torch.zeros(len(TYPES))
    for first in range(0, len(windows), batch_size):
        rows = []
        for window in windows[first:first+batch_size]:
            inputs, request = base_inputs(panel, window['index'], cell_us=protocol.get('cell_us', 100_000))
            _, sequence = teacher_target(panel, window['index'], request, vocab)
            rows.append(dict(inputs=inputs, sequence=sequence))
        inp, seq = collate(rows)
        inp = {k: v.to(device) for k, v in inp.items()}
        seq = {k: v.to(device) for k, v in seq.items()}
        with amp:
            logits = model(inp, seq).float()
        keep = seq['valid'] & (seq['fields'] == FIELDS.index('kind')) & (seq['allowed'].sum(-1) > 1)
        if not keep.any():
            continue
        selected = logits[keep].masked_fill(~seq['allowed'][keep], -1e9)[:, kinds]
        labels = (seq['targets'][keep][:, None] == kinds).float().argmax(-1)
        target += torch.bincount(labels, minlength=len(TYPES)).float().cpu()
        for t in CALIBRATION_T:
            predicted[t] += (selected/t).softmax(-1).sum(0).cpu()
        own += torch.zeros(len(TYPES), device=device).index_add_(
            0, labels, selected.softmax(-1).gather(1, labels[:, None])[:, 0]).cpu()
    count = float(target.sum())
    if not count:
        return None
    return dict(tokens=int(count), target_share=dict(zip(TYPES, (target/count).tolist())),
                teacher_forced_marginal={str(t): dict(zip(TYPES, (predicted[t]/count).tolist())) for t in CALIBRATION_T},
                mean_p_of_target_class=dict(zip(TYPES, (own/target.clamp_min(1)).tolist())))


def field_table(summary):
    """{'all'|mode: {field group|joint: mean CE, real audio}} from the screen summary."""
    blocks = dict(all=summary['field_ce'], **{m: b['field_ce'] for m, b in summary['by_mode'].items()})
    return {name: {g: block['real'][g]['mean'] for g in (*FIELD_GROUPS, 'joint')} for name, block in blocks.items()}


def compare(table, baseline, kill_nats):
    delta = {name: {g: (v-baseline[name][g]) if v is not None and baseline[name].get(g) is not None else None
                    for g, v in row.items()} for name, row in table.items() if name in baseline}
    worst = max(row['joint'] for row in delta.values() if row['joint'] is not None)
    return dict(delta=delta, worst_joint_delta_nats=worst, kill_nats=kill_nats, gross_kill=bool(worst > kill_nats))


def probe(model, panel, protocol, *, device, batch_size=8, autocast=True, max_wall_s=None, max_windows=None,
          baseline=None, kill_nats=.5):
    real, skipped = real_audio_protocol(protocol, panel, max_windows)
    deadline = None if max_wall_s is None else time.monotonic()+float(max_wall_s)
    result = run_teacher_screen(model, panel, real, device=device, batch_size=batch_size, deadline=deadline,
                                autocast=autocast)
    table = field_table(result['summary'])
    out = dict(windows=result['summary']['measured_windows'], expected_windows=len(real['windows']),
               skipped_no_features=skipped, complete=result['summary']['complete'],
               windows_by_mode=result['summary']['windows_by_mode'], field_ce=table,
               kind_calibration=kind_calibration(model, panel, real, device=device, batch_size=batch_size,
                                                 autocast=autocast),
               parameters=sum(p.numel() for p in model.parameters()),
               decoder_layers=model.c.decoder_layers, decoder_ffn=model.c.decoder_ffn or model.c.ffn,
               elapsed_s=result['summary']['elapsed_s'])
    if baseline is not None:
        out['versus_baseline'] = compare(table, baseline['field_ce'], kill_nats)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--plan', type=Path, required=True, help='r5 plan: tape, features, mel normalization')
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--baseline', type=Path, help='an earlier probe of r5; adds deltas and the gross kill verdict')
    a.add_argument('--kill-nats', type=float, default=.5)
    a.add_argument('--device', default='cuda')
    a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--fp32', action='store_true', help='no bf16 autocast')
    a.add_argument('--max-wall-s', type=float, default=900.)
    a.add_argument('--max-windows', type=int)
    args = a.parse_args()
    from beatfcn.train.training_panel import build_training_panel
    plan = json.loads(args.plan.read_text())
    panel = build_training_panel(Path('/'), plan)
    model = load_checkpoint_model(args.checkpoint, args.device)
    result = probe(model, panel, json.loads(args.protocol.read_text()), device=args.device, batch_size=args.batch_size,
                   autocast=not args.fp32, max_wall_s=args.max_wall_s, max_windows=args.max_windows,
                   baseline=json.loads(args.baseline.read_text()) if args.baseline else None, kill_nats=args.kill_nats)
    result.update(checkpoint=str(args.checkpoint), autocast=not args.fp32,
                  macs_per_audio_second=macs_per_audio_second(vars(model.c))['macs_per_audio_second'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=1)+'\n')
    temporary.replace(args.out)
    print(json.dumps(dict(field_ce={k: dict(joint=v['joint'], kind=v['kind']) for k, v in result['field_ce'].items()},
                          kind_calibration=result['kind_calibration'], versus=result.get('versus_baseline', {}).get(
                              'worst_joint_delta_nats'), gross_kill=result.get('versus_baseline', {}).get('gross_kill'),
                          windows=result['windows'], skipped=result['skipped_no_features']), indent=1), flush=True)
    if not np.isfinite(result['field_ce']['all']['joint']):
        raise SystemExit('Non-finite teacher CE')


if __name__ == '__main__':
    main()

"""Difficulty-spread teacher probe (beatfcn/train/spread_benchmark.py): does the model use ANOTHER
human chart of the song (role human_alternative) when deriving this chart?

  .venv/bin/python -m scripts.run_spread_probe --plan PLAN.json --checkpoint step-X.pt --weights ema \
      --device cuda --out-dir reports/<run>/spread-probe [--windows 200] [--protocol frozen.json]

Teacher-forced only (no sampling, never trains, never schedules cloud work). Gaps in nats, paired over
windows: sibling_vs_none (uses the other chart), alignment_specific (uses it at the right time).
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from beatfcn.train import spread_benchmark as sb  # noqa: E402
from scripts.run_continuation_benchmark import checkpoint_setup, sha_file  # noqa: E402

FIELDS = ('head_time', 'kind', 'mania', 'tails', 'group', 'joint')


def markdown(report):
    s = report['summary']
    lines = ['# Difficulty-spread teacher probe', '', f"model: `{report['model']['path']}` ({report['model']['weights']}, "
             f"updates {report['model'].get('updates')})", '', s['meaning'], '',
             f"windows {s['measured_windows']}/{s['expected_windows']} {s['windows_by_mode']}", '']
    for mode, block in s.get('by_mode', {}).items():
        lines += [f'## {mode}', '', '| comparison | ' + ' | '.join(FIELDS) + ' |', '|---|' + '---|' * len(FIELDS)]
        for name, gaps in block['gaps'].items():
            cells = []
            for f in FIELDS:
                g = gaps.get(f) or {}
                m, se = g.get('paired_mean_nats'), g.get('paired_se_nats')
                cells.append('—' if m is None else f'{m:+.3f} ± {se:.3f}' if se is not None else f'{m:+.3f}')
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        ce = block['field_ce']
        lines += ['', 'joint CE: ' + ', '.join(f"{c} {ce[c]['joint']['mean']:.3f}" for c in ce
                                               if ce[c]['joint']['mean'] is not None), '']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--weights', choices=('raw', 'ema'), default='ema')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--protocol', type=Path, help='frozen spread panel (default: freeze from the tape)')
    p.add_argument('--windows', type=int, default=200)
    p.add_argument('--panel-seed', type=int, default=20261102)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-wall-s', type=float, default=3000.)
    args = p.parse_args()
    started = time.monotonic()
    model, panel, identity, _, _ = checkpoint_setup(args)
    plan = json.loads(args.plan.read_text())
    tape_sha = plan.get('inputs_sha256', {}).get(plan['tape'])
    if args.protocol:
        protocol = json.loads(args.protocol.read_text())
    else:
        protocol = sb.freeze_spread_panel(panel, windows=args.windows, seed=args.panel_seed, tape_sha256=tape_sha)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'spread-panel.json').write_text(json.dumps(protocol, indent=1) + '\n')
    result = sb.run_spread_teacher(model, panel, protocol, device=args.device, batch_size=args.batch_size,
                                   deadline=started + args.max_wall_s)
    report = dict(schema='beatfcn.spread-probe.v1', model=identity, protocol_sha256=sha_file(args.out_dir / 'spread-panel.json'),
                  summary=result['summary'], records=result['records'], wall_s=time.monotonic() - started)
    out = args.out_dir / 'spread-probe.json'
    out.write_text(json.dumps(report, indent=1, default=str) + '\n')
    (args.out_dir / 'summary.md').write_text(markdown(report))
    print(markdown(report))
    print(f'wrote {out} ({sha_file(out)})')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One-shot status for the 0.6B training run (no长驻进程, 不会卡在旧容器上).

Reads `modal app logs`, keeps the last contiguous segment of increasing steps
(that is the currently running container), and prints a compact summary.

Usage:
    .venv/bin/python -m scripts.train_status
    .venv/bin/python -m scripts.train_status --app stage0-cuda-training --hours 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / '.venv' / 'bin' / 'modal'


def records(app: str):
    proc = subprocess.run([str(MODAL), 'app', 'logs', app], capture_output=True,
                          text=True, timeout=90, cwd=ROOT)
    out = []
    for line in proc.stdout.splitlines():
        start = line.find('{"phase"')
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except ValueError:
            continue
        if record.get('phase') == 'train':
            out.append(record)
    return out


def last_run(rows):
    """Trailing segment whose step strictly increases: the live container."""
    if not rows:
        return []
    end = len(rows)
    start = end - 1
    while start > 0 and rows[start - 1].get('step', 0) < rows[start].get('step', 0):
        start -= 1
    return rows[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', default='stage0-cuda-training')
    parser.add_argument('--total', type=int, default=None)
    args = parser.parse_args()

    rows = last_run(records(args.app))
    if not rows:
        print('no train records yet (container starting, or nothing running)')
        return 1
    head, tail = rows[0], rows[-1]
    steps = tail['step'] - head['step'] + 1
    seconds = (tail.get('elapsed') or 0) - (head.get('elapsed') or 0)
    rate = seconds / steps if steps > 1 else None
    total = args.total
    print(f"step        {tail['step']:,}" + (f" / {total:,}" if total else ''))
    if rate:
        print(f'speed       {rate:.3f} s/step   ({3600/rate:,.0f} steps/h)')
        if total:
            print(f'eta         {(total-tail["step"])*rate/3600:.1f} h')
    print(f"loss(avg25) {tail.get('loss')}   last micro {tail.get('loss_now')}")
    print(f"tok-ce      {tail.get('tok_ce')}   tokens {tail.get('tokens')}")
    print(f"objective   {tail.get('objective')}   grad {tail.get('grad')}   lr {tail.get('lr')}")
    print(f"gpu         {tail.get('gpu')}  util {tail.get('gpu_util')}%  "
          f"peak {tail.get('mem_peak_mib')} MiB  alloc {tail.get('mem_alloc_mib')} MiB")
    print(f"window      {len(rows)} records")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

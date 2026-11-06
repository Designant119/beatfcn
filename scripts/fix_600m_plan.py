#!/usr/bin/env python3
"""Normalize the 0.6B plan on the Volume before a training run.

The plan builder rewrites relative health/features paths and stale code hashes.
This script restores absolute Volume paths, keeps the checkpoint cadence, and
refreshes the frozen implementation hashes after local edits.

Usage:
    .venv/bin/python -m scripts.fix_600m_plan [--plan 600m-native-cf/plan.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / '.venv' / 'bin' / 'modal'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='600m-native-cf/plan.json')
    parser.add_argument('--light-every', type=int, default=1000)
    parser.add_argument('--full-every', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / 'plan.json'
        subprocess.run([str(MODAL), 'volume', 'get', 'beatfcn-data', args.plan,
                        str(local), '--force'], check=True, capture_output=True)
        plan = json.loads(local.read_text())

        plan['health'] = '/data/data-contract-v2-preflight.json'
        plan['features'] = '/data/data-contract-v2-features.json'
        plan['log_every'] = 1
        plan['light_checkpoint_every'] = args.light_every
        plan['light_checkpoint_keep'] = 2
        plan['checkpoint_every'] = args.full_every
        plan['checkpoint_keep'] = 1
        plan['checkpoints'] = []
        plan['counterfactual_relative'] = True
        if args.steps:
            plan['steps'] = args.steps
        plan['code_sha256'] = {path: sha(ROOT / path) for path in plan['code_sha256']
                               if (ROOT / path).is_file()}

        out = Path(tmp) / 'plan-out.json'
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + '\n')
        subprocess.run([str(MODAL), 'volume', 'put', '-f', 'beatfcn-data', str(out),
                        args.plan], check=True, capture_output=True)

        print(json.dumps(dict(steps=plan['steps'],
                              micro=plan.get('physical_microbatch_requests'),
                              accum=plan.get('accum_steps'),
                              rates=plan.get('counterfactual_rates'),
                              relative=plan.get('counterfactual_relative'),
                              light=plan['light_checkpoint_every'],
                              full=plan['checkpoint_every'],
                              code=sorted(plan['code_sha256'])), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Dispatch stage-0 CUDA training plans on Modal and collect results.

Usage:
  python -m scripts.run_stage0_modal --plans a,b,c [--smoke]
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'reports/stage0_modal_run_v1.json'
GPU_RATE_USD_S = 0.000164   # T4 0.59/h
FUNCTION_TIMEOUT_S = 2400
MAX_CONTAINERS = 8


def plan_path(tag: str) -> str:
    if tag == 'smoke':
        return 'bench/rhythm-v2/stage0-smoke-cpu-v1.json'
    if tag == 'smoke-mert':
        return 'bench/rhythm-v2/stage0-mert-smoke-v1.json'
    if tag.startswith('mert-'):
        short = tag[5:]
        candidate = ROOT / f'bench/rhythm-v2/stage0-mert-{short}-v1.json'
        if candidate.is_file():
            return f'bench/rhythm-v2/stage0-mert-{short}-v1.json'
        return f'bench/rhythm-v2/stage0-mert-lr-{short}-v1.json'
    return f'bench/rhythm-v2/stage0-fresh-lr-{tag}-v1.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plans', default='smoke')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpu', default='T4')
    arguments = parser.parse_args()
    tags = [t.strip() for t in arguments.plans.split(',') if t.strip()]
    rels = [plan_path(t) for t in tags]
    for rel in rels:
        if not (ROOT / rel).is_file():
            raise SystemExit(f'Missing plan: {rel}')
    from beatfcn.cloud import train_stage0_app
    started = time.time()
    receipt = dict(id='stage0-modal-run-v1', started_at_unix=started,
        plans=rels, gpu=arguments.gpu, max_containers=MAX_CONTAINERS,
        function_timeout_s=FUNCTION_TIMEOUT_S,
        worst_case_upper_usd=str(round(len(rels) * FUNCTION_TIMEOUT_S * GPU_RATE_USD_S, 4)),
        results=[])
    with train_stage0_app.app.run():
        calls = [train_stage0_app.run_plan.with_options(gpu=arguments.gpu).spawn(rel) for rel in rels]
        for rel, call in zip(rels, calls):
            result = call.get()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            receipt['results'].append(result)
    receipt['finished_at_unix'] = time.time()
    receipt['wall_s'] = receipt['finished_at_unix'] - started
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    failed = [r for r in receipt['results'] if r.get('returncode')]
    print(json.dumps(dict(plans=rels, failed=len(failed), wall_s=receipt['wall_s'],
        worst_case_upper_usd=receipt['worst_case_upper_usd']), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

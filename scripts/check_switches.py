"""Refuse a dispatch whose plan misses a required switch (reports/goal-100usd-20261101/switches.json).

  .venv/bin/python -m scripts.check_switches --plan PLAN.json [--round 3] [--exposure EXPOSURE.json]
Exit 1 with the list of violations; prints the checked values otherwise.
"""
import argparse
import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1]/'reports/goal-100usd-20261101/switches.json'


def lookup(plan, dotted, exposure=None):
    if dotted.startswith('exposure.'):
        return (exposure or {}).get(dotted.split('.', 1)[1])
    value = plan
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check(plan, spec, round_no, exposure=None):
    problems, seen = [], {}
    for key, want in spec['training'].items():
        if isinstance(want, dict):
            if want.get('from_round', 0) > round_no:
                continue
            want = want['value']
        got = lookup(plan, key, exposure)
        seen[key] = got
        if key.startswith('exposure.') and exposure is None:
            continue
        ok = (got is not None and got is not False) if want == 'present' else got == want
        if not ok:
            problems.append(f'{key}: want {want!r}, got {got!r}')
    return problems, seen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--exposure', type=Path)
    p.add_argument('--round', type=int, default=3)
    p.add_argument('--spec', type=Path, default=DEFAULT)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    exposure = json.loads(a.exposure.read_text()) if a.exposure else None
    if exposure is not None:
        exposure = {k: v for k, v in exposure.items() if k != 'steps'}
    problems, seen = check(plan, json.loads(a.spec.read_text()), a.round, exposure)
    print(json.dumps(seen, indent=1, default=str))
    if problems:
        print('MISSING SWITCHES:\n  ' + '\n  '.join(problems))
        raise SystemExit(1)
    print('all required switches on')


if __name__ == '__main__':
    main()

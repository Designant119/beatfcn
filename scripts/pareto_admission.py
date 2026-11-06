"""Apply recorded hard gates before turning objective wins into promotion."""
from copy import deepcopy


def qualify_comparison(report, objectives=('W_nearest_exact', 'H', 'M')):
    result = deepcopy(report)
    raw = result['verdict']
    result['objective_verdict'] = raw
    side = 'a' if raw == 'A dominates B' else 'b' if raw == 'B dominates A' else None
    blockers = []
    for mode, row in result['modes'].items():
        missing = set(objectives) - row['axes'].keys()
        if missing:
            blockers.append(f'{mode}: unmeasured objectives {sorted(missing)}')
        if side:
            constraints = row['constraints']
            for gate in ('pass', 'f1_no_harm', 'playability_not_worse'):
                if not constraints[f'{side}_{gate}']:
                    blockers.append(f'{mode}: {side.upper()} failed {gate}')
    if 'cost' not in result:
        blockers.append('Comparable end-to-end cost is unmeasured')
    result['promotion_blockers'] = blockers
    result['promotion_allowed'] = side is not None and not blockers
    if side and blockers:
        result['verdict'] = 'No admissible Pareto promotion; ' + '; '.join(blockers)
    return result

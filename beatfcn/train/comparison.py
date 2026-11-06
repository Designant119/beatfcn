"""Keep compute measurements separate from end-to-end quality/cost acceptance.

Resident-data timing can shortlist hardware, but cannot establish a training Pareto
improvement. In particular, cheap compute does not account for paid data-wait time.
"""
from __future__ import annotations

import math


def compare_compute_benchmarks(measurements, resource_rates_per_second):
    """Describe the resource tradeoff on identical batches; never approve a variant.

    Rates include the allocated GPU, CPU and RAM. Estimated cost here covers ONLY
    timed resident computation, not container startup, loading, validation or idle.
    The fastest frontier candidate is a provisional experiment order, not a winner.
    """
    if not measurements:
        raise ValueError('no measurements')
    first = next(iter(measurements.values()))
    identity = ('rows_sha256', 'sample_sequence', 'config', 'effective_batch', 'steps')
    points = {}
    for name, result in measurements.items():
        if not result.get('rows_sha256') or not any(result.get('sample_sequence', [])):
            raise ValueError(f'{name}: missing data identity or sample_sequence')
        for key in identity:
            if key not in result or result[key] != first[key]:
                raise ValueError(f'{name}: mismatched {key}')
        seconds = result['seconds_per_step']
        rate = resource_rates_per_second[name]
        if any(not math.isfinite(x) or x <= 0 for x in (seconds, rate, result['effective_batch'], result['steps'])):
            raise ValueError(f'{name}: invalid timing, resource rate or sample count')
        points[name] = {'seconds_per_step': seconds, 'compute_usd_per_step': seconds * rate}
    frontier = []
    for name, point in points.items():
        dominated = any(other != name
                        and all(points[other][k] <= point[k] for k in point)
                        and any(points[other][k] < point[k] for k in point)
                        for other in points)
        if not dominated:
            frontier.append(name)
    provisional = min(frontier, key=lambda name: (points[name]['seconds_per_step'], points[name]['compute_usd_per_step']))
    return {'points': points, 'resource_frontier': frontier, 'provisional_gpu': provisional,
            'quality_evaluated': False, 'pareto_improvement_established': False,
            'scope': 'resident compute only; not total billed runtime or time/cost to quality',
            'selection_reason': 'fastest measured frontier candidate for the next experiment; within reserved budget'}

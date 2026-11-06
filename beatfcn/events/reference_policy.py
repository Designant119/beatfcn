"""Versioned reference capacity and controlled-swap policy; never drops positives."""
from copy import deepcopy
import hashlib
import json

REFERENCE_ROWS = 256
POLICY = 'positive_preserving_reference_v2'


def content_fingerprint(evidence):
    """Semantic selected content, including multiplicity; IDs are not style evidence."""
    spans = []
    for ref in evidence:
        notes = []
        for event in ref['selected']:
            notes.append({k: event[k] for k in ('onset_us', 'end_us', 'type', 'location')})
        notes.sort(key=lambda x: json.dumps(x, sort_keys=True))
        spans.append({'span': list(ref['span']), 'coverage': ref['coverage'], 'notes': notes})
    return hashlib.sha256(json.dumps(spans, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bound_evidence(evidence, max_rows=REFERENCE_ROWS):
    """Share remaining capacity between negative rows. Positive events/spans are unchanged.

    Negative evidence is a sampled representation of a complete span, not its
    definition. A positive overflow is an explicit failure, never silent truncation.
    """
    if max_rows < 0:
        raise ValueError('reference row budget must be nonnegative')
    out = deepcopy(list(evidence))
    positive = sum(len(r['selected']) for r in out)
    before = sum(len(r['unselected_us']) for r in out)
    if positive > max_rows:
        raise ValueError(f'reference positives {positive} exceed row capacity {max_rows}')
    remaining = max_rows - positive
    quotas = [0] * len(out)
    while remaining:
        changed = False
        for i, ref in enumerate(out):
            if remaining and quotas[i] < len(ref['unselected_us']):
                quotas[i] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    for ref, quota in zip(out, quotas):
        times, sources = ref['unselected_us'], ref['unselected_source']
        if len(times) != len(sources):
            raise ValueError('negative reference provenance length mismatch')
        # Spread selected rows across the complete span, deterministically.
        indices = [i * len(times) // quota for i in range(quota)]
        ref['unselected_us'] = [times[i] for i in indices]
        ref['unselected_source'] = [sources[i] for i in indices]
    return out, {'policy': POLICY, 'capacity': max_rows, 'positive_rows': positive,
                 'negative_rows_before': before, 'negative_rows_after': sum(quotas),
                 'negative_rows_removed': before - sum(quotas), 'positive_rows_removed': 0}


def controlled_pair(a, b):
    """Positive-only swap: identical negative treatment, real content contrast required.

    This is a diagnostic distribution, not the default complete-reference task.
    """
    if [(r['span'], r['coverage']) for r in a] != [(r['span'], r['coverage']) for r in b]:
        raise ValueError('reference contrast needs identical spans and coverage')
    if content_fingerprint(a) == content_fingerprint(b):
        raise ValueError('reference contrast contains identical positive content')
    result = []
    for evidence in (a, b):
        clean = deepcopy(list(evidence))
        for ref in clean:
            ref['unselected_us'], ref['unselected_source'] = [], []
        result.append(bound_evidence(clean)[0])
    return tuple(result)

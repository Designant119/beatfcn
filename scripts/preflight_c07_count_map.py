"""Check modal count support in saved C06 tensors; no models or new inference."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.count_readout import expected_tail_rank


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/c07-count-map-support-preflight-v1.json'
KINDS = ('tap', 'hold', 'flick', 'drag')


def main():
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    for path, expected in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    logits = torch.load(ROOT / 'artifacts/c06-reset-unified-count-law-v1/count-logits.pt',
                        map_location='cpu', weights_only=False)
    rows = json.loads((ROOT / 'artifacts/c06-reset-unified-count-law-v1/diagnostic-events.json').read_text())
    results = []
    for row in rows:
        if time.monotonic() - started > plan['resource']['wall_s']:
            raise TimeoutError('Frozen count prerequisite deadline')
        index, request = row['index'], row['request']
        inferred = {}
        # Infer without supplying targets; original event output is only an identity control.
        for model in ('old', 'new'):
            x = logits[index][model]['count'].numpy().astype(np.float64)
            assert x.shape == (1, 200, 4, 17) and np.isfinite(x).all()
            p = np.exp(x[0] - x[0].max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            r0, _ = expected_tail_rank(p.reshape(-1, 17), event_limit=request['max_events'])
            r0 = r0.reshape(200, 4)
            r1 = p.argmax(-1)
            if r1.sum() > request['max_events']:
                raise ValueError('MAP event resource limit, no truncation allowed')
            observed = np.zeros((200, 4), dtype=np.int64)
            for event in row['conditions'][model]['prediction']['events']:
                cell = (event['onset_us'] - request['start_us']) // 40000
                assert 0 <= cell < 200
                observed[cell, KINDS.index(event['type'])] += 1
            assert np.array_equal(r0, observed), (index, model, 'R0 identity failure')
            inferred[model] = {'R0': r0, 'R1': r1}
        truth = np.zeros((200, 4), dtype=np.int64)
        for event in row['target']:
            cell = (event['onset_us'] - request['start_us']) // 40000
            assert 0 <= cell < 200
            truth[cell, KINDS.index(event['type'])] += 1
        condition_scores = {}
        for model, rules in inferred.items():
            condition_scores[model] = {}
            for rule, prediction in rules.items():
                per_type = {}
                for k, kind in enumerate(KINDS):
                    a, b = prediction[:, k], truth[:, k]
                    per_type[kind] = dict(pred=int(a.sum()), true=int(b.sum()),
                        intersection=int(np.minimum(a, b).sum()), l1=int(abs(a-b).sum()),
                        max_multiplicity=int(a.max()))
                condition_scores[model][rule] = per_type
        results.append(dict(index=index, scores=condition_scores))
    totals = {}
    for model in ('old', 'new'):
        totals[model] = {}
        for rule in ('R0', 'R1'):
            per_type = {kind: {metric: sum(r['scores'][model][rule][kind][metric] for r in results)
                for metric in ('pred', 'true', 'intersection', 'l1')} for kind in KINDS}
            non_tap = {metric: sum(per_type[kind][metric] for kind in KINDS[1:])
                       for metric in ('pred', 'true', 'intersection', 'l1')}
            non_tap['f1'] = 2 * non_tap['intersection'] / max(non_tap['pred'] + non_tap['true'], 1)
            totals[model][rule] = dict(per_type=per_type, non_tap=non_tap,
                predicted_events=sum(t['pred'] for t in per_type.values()),
                nonempty_windows=sum(sum(r['scores'][model][rule][k]['pred'] for k in KINDS) > 0 for r in results),
                max_multiplicity=max(r['scores'][model][rule][k]['max_multiplicity'] for r in results for k in KINDS))
    new_intersection = totals['new']['R1']['non_tap']['intersection']
    result = dict(id=plan['id'], status='completed', identity_replays=len(results)*2,
        optimizer_updates=0, model_forward_calls=0, cloud_calls=0,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        scope=plan['scope'], totals=totals, per_window=results,
        proposed_map_repair_prerequisite=new_intersection >= 20,
        rejection_scope='Only this per-cell MAP repair prerequisite; not an exhaustive readout or architecture family test.',
        wall_s=time.monotonic()-started)
    (ROOT / plan['output']).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'per_window'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

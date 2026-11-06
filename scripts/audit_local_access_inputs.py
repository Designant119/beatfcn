"""Read-only actual-tensor audit before the new access comparison's health run."""
import json
from pathlib import Path

import torch

from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_mamba_health import sha, write
from scripts.run_local_portfolio_scale import row_factory


ROOT = Path(__file__).resolve().parents[1]


def statistics(tensor):
    value = tensor.detach().cpu().double()
    finite = value[torch.isfinite(value)]
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
        nan=int(torch.isnan(value).sum()), inf=int(torch.isinf(value).sum()),
        minimum=float(finite.min()) if finite.numel() else None,
        maximum=float(finite.max()) if finite.numel() else None,
        mean=float(finite.mean()) if finite.numel() else None,
        std=float(finite.std(unbiased=False)) if finite.numel() else None)


def main():
    torch.set_num_threads(2)
    path = ROOT / 'reports/local_access_input_audit_v1.json'
    if path.exists():
        raise FileExistsError('Preserve actual-input audit')
    plan_path = ROOT / 'bench/rhythm-v2/local-access-comparison-v1.json'
    plan = json.loads(plan_path.read_text())
    panel, store = setup(json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row = row_factory(panel, store)
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    records = []
    for index, view in specs:
        example = row(index, view)
        inputs, target = relative_batch([example])
        inputs = {k: v for k, v in inputs.items() if not k.startswith(('local', 'song'))}
        stats = {name: {k: statistics(v) for k, v in values.items()}
                 for name, values in [('inputs', inputs), ('targets', target)]}
        mel_times = inputs['mel_times'][0, inputs['mel_valid'][0]]
        covered = bool(mel_times.min() <= .005 + 1e-6 and mel_times.max() >= 7.995 - 1e-6)
        finite = all(v['nan'] == v['inf'] == 0 for group in stats.values() for v in group.values())
        records.append(dict(index=index, view=view, target_events=len(example['targets']),
            query_audio_coverage=covered, all_tensors_finite=finite, statistics=stats))
    report = dict(id='local-access-input-audit-v1', plan_sha256=sha(plan_path),
        audit_source_sha256=sha(Path(__file__)), records=records,
        passed=all(r['query_audio_coverage'] and r['all_tensors_finite'] for r in records),
        optimizer_updates=0, new_model_forward_calls=0, cloud_compute_calls=0,
        confirmation_scored=False, boundary='This audits data and clocks, not model quality.')
    write(path, report)
    print(json.dumps(dict(passed=report['passed'], windows=len(records),
                          audit_path=str(path.relative_to(ROOT)))))
    if not report['passed']:
        raise ValueError('Actual input/target audit failed')


if __name__ == '__main__':
    main()

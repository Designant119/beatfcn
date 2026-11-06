"""Check the actual D95 step2-to4 handoff before any continued quality run."""
from collections import Counter
import torch

from scripts.prepare_local_capacity import ROOT
from scripts.prepare_local_data_scale_probe import read, sha, write_new
from scripts.prepare_local_expansion_exposure import records


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def run():
    path = 'bench/rhythm-v2/d95-capacity-quality-v1.json'
    plan = read(path)
    report = read(plan['output'])
    initial = plan['initialize_from']
    parent = torch.load(ROOT / initial['checkpoint'], map_location='cpu', weights_only=False)
    checkpoint = f"{plan['checkpoint_dir']}/step-0004.pt"
    saved = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)
    exposure = read(plan['exposure'])['steps'][:4]
    pool = records(read(plan['tape'])['charts'])
    rows = [pool[i] for step in exposure for i in step['indices']]
    expected_tokens = sum(r['tokens'] for r in rows)
    modes = dict(Counter(r['mode'] for r in rows))
    changed = sum(not torch.equal(v, parent['model'][k]) for k, v in saved['model'].items())
    states = {float(v['step']) for v in saved['optimizer']['state'].values()}
    checks = dict(
        bound_to_frozen_plan=saved['plan_sha256'] == report['plan_sha256'] == sha(path),
        actual_step4=saved['updates'] == report['updates'] == 4 and report['new_updates'] == 2,
        actual_exposure=report['requests'] == 16 and report['valid_tokens'] == expected_tokens
            and report['mode_exposure'] == modes,
        same_width=saved['config'] == parent['config'] == plan['config'],
        adam_steps=states == {4.},
        optimizer_parameter_ids=saved['optimizer']['state'].keys() == parent['optimizer']['state'].keys(),
        optimizer_hyperparameters=saved['optimizer']['param_groups'] == parent['optimizer']['param_groups'],
        all_values_finite=finite(saved['model']) and finite(saved['optimizer']),
        parent_unchanged=sha(initial['checkpoint']) == plan['inputs_sha256'][initial['checkpoint']],
        not_diagnostic_checkpoint=parent.get('diagnostic_only') is False,
        parameter_update=changed > 0,
        cpu_rng_unchanged=torch.equal(saved['cpu_rng'], parent['cpu_rng']),
        mps_rng_unchanged=torch.equal(saved['mps_rng'], parent['mps_rng']),
        report_checkpoint=report['checkpoint'] == checkpoint and report['checkpoint_sha256'] == sha(checkpoint),
        no_error=not report.get('error'))
    result = dict(id='D95-driver4-verification-v1', passed=all(checks.values()), checks=checks,
        updates=4, new_updates=2, requests=16, valid_tokens=report['valid_tokens'],
        expected_tokens=expected_tokens, mode_counts=modes, parameter_tensors_changed=changed,
        checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
        source_sha256={'scripts/verify_local_capacity_driver.py': sha('scripts/verify_local_capacity_driver.py')},
        new_cloud_compute_calls=0, quality_passed=False)
    write_new('reports/d95_driver4_verification_v1.json', result)
    if not result['passed']:
        raise ValueError('Capacity driver handoff failed: ' + str(checks))
    print('D95 actual step4 verified:16 requests, full token exposure, Adam, RNG, finite parameters.')


if __name__ == '__main__':
    run()

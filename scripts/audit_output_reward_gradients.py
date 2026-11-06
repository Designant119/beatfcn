"""Actual saved-trajectory SCST replay and gradient scale audit, zero updates."""
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.condition_outcome import score_condition_outcome
from beatfcn.train.condition_scst import _versions, replay_log_probabilities, scst_loss
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.audit_local_candidate_conditions import request_legality
from scripts.prepare_local_data_scale_probe import padding_shapes, sha
from scripts.run_local_song_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d100-output-gradient-audit-v1.json'


def gradient_vector(model):
    pieces = []
    for _, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise ValueError('Actual gradient is not finite')
        pieces.append(torch.zeros(p.numel()) if p.grad is None else p.grad.detach().cpu().reshape(-1).clone())
    return torch.cat(pieces)


def run():
    plan = json.loads((ROOT / PLAN).read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen replay source changed: ' + path)
    source = json.loads((ROOT / plan['trajectories']).read_text())
    if not source['complete'] or not source['signal_gate_passed'] or len(source['rows']) != 32:
        raise ValueError('Original actual32 support gate required')
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    models = {}
    for device in ('cpu', 'mps'):
        model = FullEventModel(FullEventConfig(**saved['config'])).to(device).eval()
        model.load_state_dict(saved['model'])
        if any(not torch.equal(v.detach().cpu(), saved['model'][k]) for k, v in model.state_dict().items()):
            raise ValueError('Replay device does not have exact archived policy weights')
        models[device] = model
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    support = {r['record_index']: r for r in json.loads((ROOT / plan['support_panel']).read_text())['records']}
    size = sum(p.numel() for p in models['cpu'].parameters())
    aggregates = {device: {kind: torch.zeros(size) for kind in ('ce', 'scst')} for device in models}
    report = dict(id=plan['id'], complete=False, passed=False, rows=[], optimizer_updates=0,
        new_cloud_compute_calls=0, normalizer=plan['normalizer'], source_sha256=plan['source_sha256'],
        plan_sha256=sha(PLAN), no_new_samples=True, training_allowed=False, quality_passed=False,
        model_weights_verified_by_name_on_both_devices=True,
        durable_trajectory_identity='Archived checkpoint SHA and exact all-parameter values verified before rebinding process-local version guards.',
        limitations=['Scale calibration from32 training requests is not a condition or music-quality result.',
            'Per-field output variation does not establish that all controls receive useful training updates.',
            'No optimizer is created and no policy update or new rollout is performed.'])
    started = time.monotonic()

    def save():
        report['wall_s'] = time.monotonic() - started
        write_json(output, report)

    try:
        for row in source['rows']:
            if time.monotonic() - started > plan['max_wall_s'] or time.time() >= plan['deadline_unix']:
                raise TimeoutError('Original replay resource/deadline bound')
            case = support[row['record_index']]
            request = event_request(case['request'])
            if row['request'] != case['request'] or row['bands'] != case['bands']:
                raise ValueError('Sample request or reward condition changed')
            outcomes = []
            for variant in ('greedy', 'sampled'):
                trajectory = row[variant]
                events = [RhythmEvent(**e) for e in trajectory['events']]
                outcomes.append(score_condition_outcome(request.mode,
                    canonical([*request.known_carry, *events], request.mode),
                    (request.start_us, request.end_us), row['bands'], complete=trajectory['complete'],
                    legality_errors=request_legality(events, request)))
            if outcomes != row['outcomes']:
                raise ValueError('Actual output reward did not reproduce exactly')
            inp, teacher = panel.batch([row['record_index']], controls=[case['controls']])
            inp = pad_input_shapes(inp, padding_shapes(inp['song'].shape[1]))
            gradients, probabilities = {}, {}
            for device, model in models.items():
                inputs = {k: v.to(device) for k, v in inp.items()}
                seq = {k: v.to(device) for k, v in teacher.items()}
                # Saved trajectories do not have live object IDs. The archived
                # checkpoint was verified above; no mutation occurs in this job.
                trajectory = {**row['sampled'], 'parameter_versions': _versions(model)}
                model.zero_grad(set_to_none=True)
                logp = replay_log_probabilities(model, inputs, request, trajectory)
                probabilities[device] = logp.detach().cpu()
                loss = scst_loss(logp, outcomes[1]['reward'], outcomes[0]['reward'], normalizer=plan['normalizer'])
                loss.backward()
                policy = gradient_vector(model)
                model.zero_grad(set_to_none=True)
                ce = event_loss(model(inputs, seq), seq)
                ce.backward()
                supervised = gradient_vector(model)
                gradients[device] = dict(scst=policy, ce=supervised)
                for kind, value in gradients[device].items():
                    aggregates[device][kind].add_(value / len(source['rows']))
                del inputs, seq, logp, loss, ce
            online = torch.tensor(row['sampled']['online_log_probabilities'])
            online_error = float((probabilities['cpu'] - online).abs().max())
            cross_error = float((probabilities['cpu'] - probabilities['mps']).abs().max())
            errors = {kind: float((gradients['cpu'][kind] - gradients['mps'][kind]).abs().max()) for kind in ('ce', 'scst')}
            if max(online_error, cross_error, *errors.values()) > plan['parity_atol']:
                raise ValueError('Actual policy probabilities/gradients differ beyond the original tolerance')
            if row['advantage'] == 0 and any(bool(g['scst'].any()) for g in gradients.values()):
                raise ValueError('Zero actual advantage acquired a gradient')
            report['rows'].append(dict(index=row['record_index'], mode=case['mode'], fields=list(row['bands']),
                tokens=len(row['sampled']['tokens']), advantage=row['advantage'],
                sampled_log_probability_sum=float(probabilities['cpu'].sum()),
                online_replay_max_abs=online_error, cpu_mps_logp_max_abs=cross_error,
                gradient_max_abs=errors,
                cpu_gradient_norms={k: float(v.norm()) for k, v in gradients['cpu'].items()}))
            save()
        ce, policy = aggregates['cpu']['ce'], aggregates['cpu']['scst']
        ce_norm, policy_norm = float(ce.norm()), float(policy.norm())
        if min(ce_norm, policy_norm) <= 0:
            raise ValueError('No aggregate objective gradient: cannot calibrate lambda')
        offsets, module_norms = 0, defaultdict(lambda: dict(ce_sq=0., scst_sq=0.))
        for name, param in models['cpu'].named_parameters():
            end = offsets + param.numel()
            key = name.split('.')[0]
            module_norms[key]['ce_sq'] += float(ce[offsets:end].square().sum())
            module_norms[key]['scst_sq'] += float(policy[offsets:end].square().sum())
            offsets = end
        norms = {key: {k.removesuffix('_sq'): v**.5 for k, v in values.items()} for key, values in module_norms.items()}
        if norms['control_projection']['scst'] <= 0:
            raise ValueError('No actual outcome gradient reaches the condition projection')
        unchanged = all(torch.equal(v.detach().cpu(), saved['model'][k])
                        for model in models.values() for k, v in model.state_dict().items())
        if not unchanged:
            raise ValueError('Zero-update audit changed policy weights')
        lam = ce_norm / policy_norm
        report.update(complete=True, passed=True, weights_unchanged=True,
            calibration=dict(rule='lambda=norm(mean CE gradients)/norm(mean actual SCST gradients); no clipping or quality-based tuning',
                lambda_value=lam, ce_norm=ce_norm, scst_norm=policy_norm,
                ce_scst_cosine=float(torch.dot(ce, policy) / (ce.norm() * policy.norm())),
                module_gradient_norms=norms,
                backend_lambda_relative_difference=abs(float(aggregates['mps']['ce'].norm() / aggregates['mps']['scst'].norm()) - lam) / lam),
            checkpoint_unchanged=sha(plan['checkpoint']) == plan['source_sha256'][plan['checkpoint']])
        save()
        print(json.dumps(dict(complete=True, passed=True, wall_s=report['wall_s'],
                             calibration={k: v for k, v in report['calibration'].items() if k != 'module_gradient_norms'})))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', status='failed')
        save()
        raise


if __name__ == '__main__':
    run()

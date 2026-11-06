"""D95 startup comparability: replay two control updates, then bootstrap width256.

This bounded gate creates no quality curve and never resumes diagnostic weights.
The original eight requests, unpadded microbatch-two execution, and optimizer
are retained. The expanded panel must supply byte-identical tensors for them.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from scripts.prepare_local_data_scale_probe import read, sha, write_new
from scripts.run_local_event_experiment import memory
from scripts.run_local_song_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d95-initialization-v1.json'
OLD_PLAN = 'bench/rhythm-v2/d68-local-baseline-v1.json'
BASE = 'bench/rhythm-v2/d94-data-scale-quality-v1.json'
LARGE = 'bench/rhythm-v2/d65-local-8m-v1.json'
PARENT = 'artifacts/d68-local-baseline/checkpoints/step-0002.pt'


def width_only(small, large):
    changed = {k for k in set(small) | set(large) if small.get(k) != large.get(k)}
    if changed != {'d', 'ffn'} or (small['d'], small['ffn'], large['d'], large['ffn']) != (128, 384, 256, 768):
        raise ValueError('Only the registered width and corresponding FFN may change')


def state_difference(actual, expected):
    """Check structure as well as values; never hide absent Adam states."""
    errors = []

    def visit(a, b, path):
        if isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Tensor structure differs: ' + path)
            a, b = a.detach().cpu(), b.detach().cpu()
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError('Nonfinite state: ' + path)
            errors.append(float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.)
        elif isinstance(b, dict):
            if not isinstance(a, dict) or a.keys() != b.keys():
                raise ValueError('State keys differ: ' + path)
            for k in b:
                visit(a[k], b[k], path + '/' + str(k))
        elif isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError('State sequence differs: ' + path)
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, path + '/' + str(i))
        elif a != b:
            raise ValueError('Scalar state differs: ' + path)

    visit(actual, expected, '')
    return dict(tensors=len(errors), max_abs=max(errors, default=0.), exact=not any(errors))


def freeze():
    old, base, large = read(OLD_PLAN), read(BASE), read(LARGE)
    width_only(old['config'], large['config'])
    if base['config'] != old['config'] or large['seed'] != old['seed'] or base['seed'] != old['seed']:
        raise ValueError('Baseline configuration or seed differs')
    old_exposure, expanded = read(old['exposure']), read(base['exposure'])
    if old_exposure['steps'][:2] != expanded['steps'][:2]:
        raise ValueError('The two-update request prefix differs')
    compatible = ['beatfcn/model/full_event_model.py', 'beatfcn/model/full_event_codec.py',
        'beatfcn/train/local_request_panel.py', 'beatfcn/train/local_panel.py',
        'beatfcn/train/local_features.py', 'beatfcn/train/event_reference.py']
    for path in compatible:
        if sha(path) != old['code_sha256'][path]:
            raise ValueError('Original startup implementation changed: ' + path)
    health = read('reports/d65_local_8m_v1.json')
    if not health['passed'] or health['sources_sha256'][compatible[0]] != sha(compatible[0]):
        raise ValueError('Old width256 health does not cover the same model implementation')
    panel = LocalRequestPanel(ROOT, base['tape'], base['features'])
    sources = [OLD_PLAN, BASE, LARGE, PARENT, old['tape'], old['features'], old['exposure'],
        base['tape'], base['features'], base['exposure'], 'reports/d65_local_8m_v1.json',
        'reports/d68_driver_probe_v1.json', 'design/LOCAL_CAPACITY_D95.md',
        'scripts/prepare_local_capacity.py', 'scripts/prepare_local_data_scale_probe.py',
        'scripts/run_local_event_experiment.py', 'scripts/run_local_song_baseline.py',
        'tests/test_local_capacity_startup.py']
    identities = {p: sha(p) for p in sources}
    identities.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'beatfcn').rglob('*.py')})
    for step in expanded['steps'][:2]:
        for index in step['indices']:
            row = panel.records[index]
            chart = panel.charts[row['chart_index']]
            for key in ('target', 'token'):
                identities[chart[key + '_path']] = chart[key + '_sha256']
            identities.update(panel.features.records[row['audio_sha256']]['sha256'])
    identities[read(base['features'])['scales']] = sha(read(base['features'])['scales'])
    for path, expected in identities.items():
        if sha(path) != expected:
            raise ValueError('Source/asset identity differs: ' + path)
    plan = dict(id='D95-initialization-v1', seed=old['seed'], small_config=old['config'],
        config=large['config'], steps=expanded['steps'][:2], original_plan=OLD_PLAN,
        original_parent=PARENT, tape=base['tape'], features=base['features'],
        learning_rate=.001, clip=1., weight_decay=0., physical_microbatch_requests=2,
        effective_batch_requests=4, fixed_input_shapes=None, max_wall_s=180,
        replay_max_abs=1e-6, source_sha256=identities,
        output='reports/d95_initialization_v1.json',
        checkpoint='artifacts/d95-capacity-v1/initialization/step-0002.pt',
        scope='Two-update execution parity only; different widths share initialization rules, not weights',
        quality_passed=False, cloud_compute_calls=0)
    write_new(PLAN, plan)
    print(json.dumps(dict(plan=PLAN, frozen=True, updates_started=False)))


def run(plan_path):
    plan = read(plan_path)
    output = ROOT / plan['output']
    if output.exists() or (ROOT / plan['checkpoint']).exists():
        raise FileExistsError('Inspect existing startup evidence; do not repeat it')
    for p, expected in plan['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Frozen startup source changed: ' + p)
    width_only(plan['small_config'], plan['config'])
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS is required')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, quality_passed=False,
        plan_sha256=sha(plan_path), status='preparing', tensor_batches=[],
        replay_updates=0, candidate_updates=0, new_cloud_compute_calls=0)

    def save():
        report.update(wall_s=time.monotonic() - started, memory=memory('mps'))
        write_json(output, report)
        print(json.dumps({k: report[k] for k in ('status', 'replay_updates', 'candidate_updates', 'wall_s')}), flush=True)

    def deadline():
        if time.monotonic() - started > plan['max_wall_s']:
            raise TimeoutError('D95 initialization wall limit')

    try:
        old_plan = read(plan['original_plan'])
        old_panel = LocalRequestPanel(ROOT, old_plan['tape'], old_plan['features'])
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        batches = []
        for step in plan['steps']:
            for offset in (0, 2):
                ids = step['indices'][offset:offset + 2]
                controls = []
                for i, fields in zip(ids, step['specified_condition_fields'][offset:offset + 2], strict=True):
                    row = panel.records[i]
                    if row['split'] != 'train' or row != old_panel.records[i]:
                        raise ValueError('Original eight training requests differ')
                    c = np.asarray(row['controls'], np.float32).copy()
                    c[[j for j in range(len(c)) if j not in fields], :3] = 0
                    controls.append(c)
                pair = panel.batch(ids, controls=controls)
                parity = state_difference(pair, old_panel.batch(ids, controls=controls))
                if not parity['exact']:
                    raise ValueError('Actual old/new startup tensors differ')
                for j, i in enumerate(ids):
                    expected = list(encode_events(sorted(panel.targets(i)), event_request(panel.records[i]['request'])))
                    actual = pair[1]['targets'][j][pair[1]['valid'][j]].tolist()
                    if actual != expected:
                        raise ValueError('Full target alignment differs')
                batches.append(pair)
                report['tensor_batches'].append(dict(update=step['effective_update'], indices=ids,
                    parity=parity, target_alignment_exact=True,
                    tensors={section: {k: tensor_health(v.numpy()) for k, v in values.items()}
                        for section, values in zip(('inputs', 'sequence'), pair, strict=True)}))
        del panel
        saved = torch.load(ROOT / plan['original_parent'], map_location='cpu', weights_only=False)
        if saved['updates'] != 2 or saved['plan_sha256'] != sha(plan['original_plan']):
            raise ValueError('Original step2 checkpoint identity differs')

        def bootstrap(config, key):
            torch.manual_seed(plan['seed'])
            torch.mps.manual_seed(plan['seed'])
            model = FullEventModel(FullEventConfig(**config)).to('mps')
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
            if plan.get('restore_original_validation_preflight'):
                model.eval()
                values = []
                with torch.no_grad():
                    for index in old_plan['validation_indices']:
                        deadline()
                        row = old_panel.records[index]
                        if row['split'] != 'val' or np.asarray(row['controls'])[:, 2].any():
                            raise ValueError('Initial diagnostic validation contains hidden conditions')
                        inp, seq = old_panel.batch([index], device='mps')
                        values.append(dict(index=index, request_ce=float(event_loss(model(inp, seq), seq))))
                        del inp, seq
                torch.mps.synchronize()
                report[key + '_initial_validation'] = values
            model.train()
            curve = []
            for step in range(2):
                deadline()
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                tick = time.monotonic()
                for pair in batches[step * 2:step * 2 + 2]:
                    inp, seq = ({k: v.to('mps') for k, v in values.items()} for values in pair)
                    loss = event_loss(model(inp, seq), seq) / 2
                    loss.backward()
                    total += float(loss.detach())
                    del inp, seq, loss
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.mps.synchronize()
                curve.append(dict(step=step + 1, request_mean_ce=total, preclip_gradient_norm=float(norm),
                                  wall_s=time.monotonic() - tick))
                report[key + '_updates'] = step + 1
            report[key + '_curve'] = curve
            return model, optimizer

        model, optimizer = bootstrap(plan['small_config'], 'replay')
        proof = dict(model=state_difference(model.state_dict(), saved['model']),
            optimizer=state_difference(optimizer.state_dict(), saved['optimizer']),
            cpu_rng_exact=torch.equal(torch.get_rng_state(), saved['cpu_rng']),
            mps_rng_exact=torch.equal(torch.mps.get_rng_state(), saved['mps_rng']))
        report['control_replay'] = proof
        if (max(proof['model']['max_abs'], proof['optimizer']['max_abs']) > plan['replay_max_abs']
                or not proof['cpu_rng_exact'] or not proof['mps_rng_exact']):
            raise ValueError('Two-update control replay differs; capacity attribution is blocked')
        del model, optimizer, saved
        torch.mps.empty_cache()
        report['status'] = 'control_replay_passed'
        save()
        model, optimizer = bootstrap(plan['config'], 'candidate')
        if any(float(s['step']) != 2 for s in optimizer.state.values()):
            raise ValueError('Candidate optimizer has inconsistent update counts')
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError('Candidate parameters became nonfinite')
        checkpoint = ROOT / plan['checkpoint']
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('xb') as stream:
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), config=asdict(model.c), updates=2,
                plan_sha256=sha(plan_path), cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state(),
                diagnostic_only=False, scope='Fresh width256 two-update common request prefix'), stream)
        report.update(complete=True, passed=True, status='completed',
            parameters=sum(p.numel() for p in model.parameters()),
            requests=8, valid_tokens=sum(int(pair[1]['valid'].sum()) for pair in batches),
            checkpoint=plan['checkpoint'], checkpoint_sha256=sha(checkpoint),
            original_parent_unchanged=sha(plan['original_parent']) == plan['source_sha256'][plan['original_parent']])
        if not report['original_parent_unchanged']:
            raise ValueError('Original control checkpoint changed')
        save()
    except Exception as exc:
        report.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    run(args.plan) if args.run else freeze()

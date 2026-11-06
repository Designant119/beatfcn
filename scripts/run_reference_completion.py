"""Bounded, isolated reference-completion experiment with a strict real-data gate."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_panel import collate
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.reference_completion import TASKS, completion_row

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/reference-completion-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(path, payload):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare():
    if (ROOT / PLAN).exists():
        raise FileExistsError(PLAN)
    started = time.monotonic()
    tape, features = 'reports/d94_expanded_target_tape_v1.json', 'reports/d94_merged_feature_admission_v1.json'
    panel = LocalRequestPanel(ROOT, tape, features)
    exposure = read('inputs/d94_local_requests/learning-exposure-1000-v1.json')
    exposed = {i for step in exposure['steps'] for i in step['indices']}
    candidates = [i for i in exposed if panel.records[i]['mode'] == 'phira'
                  and 6 <= panel.records[i]['target_stop'] - panel.records[i]['target_start'] <= 60]
    candidates.sort(key=lambda i: hashlib.sha256(f'20261026:{i}'.encode()).hexdigest())
    used, cases = set(), []
    # Different songs and actual newly constructed targets, never the old four-request panel.
    for task, required in zip(TASKS * 2, ('hold', 'flick', 'drag', 'tap', 'hold', 'flick')):
        for i in candidates:
            row = panel.records[i]
            if row['component'] in used:
                continue
            value, request, metadata = completion_row(panel, i, task, reference=True, seed=20261026)
            kinds = Counter(e['type'] for e in metadata['hidden'])
            if (kinds[required] and 3 <= len(metadata['hidden']) <= 30
                    and len(metadata['visible']) >= 3 and len(value['sequence']['tokens']) <= 512
                    and len(value['inputs']['reference']) <= 128):
                cases.append(dict(index=i, task=task, required=required,
                                  hidden_types=dict(kinds), chart_id=row['chart_id'],
                                  component=row['component'], hidden_count=len(metadata['hidden']),
                                  visible_count=len(metadata['visible']), tokens=len(value['sequence']['tokens'])))
                used.add(row['component'])
                break
        else:
            raise ValueError('Insufficient actual health support for ' + task + '/' + required)
    train_components = {r['component'] for r in panel.records if r['split'] == 'train'}
    val_components = {r['component'] for r in panel.records if r['split'] == 'val'}
    if train_components & val_components:
        raise ValueError('Train/validation component leakage')
    sources = {}
    for p in ('scripts/run_reference_completion.py', 'beatfcn/train/reference_completion.py',
              'beatfcn/train/event_reference.py', 'beatfcn/train/local_request_panel.py',
              'beatfcn/train/local_features.py', 'beatfcn/model/full_event_model.py',
              'beatfcn/model/full_event_codec.py', 'beatfcn/model/full_event_generate.py',
              'design/CAUSAL_DIAGNOSIS_AND_COMPLETION_v1.md', tape, features,
              'artifacts/d68-local-baseline/checkpoints/step-0002.pt',
              'inputs/d94_local_requests/learning-exposure-1000-v1.json'):
        sources[p] = sha(p)
    for case in cases:
        row = panel.records[case['index']]
        chart = panel.charts[row['chart_index']]
        sources.update(panel.features.records[row['audio_sha256']]['sha256'])
        sources.update({chart['target_path']: chart['target_sha256'], chart['token_path']: chart['token_sha256']})
    write(PLAN, dict(id='reference-completion-v1', tape=tape, features=features,
          parent='artifacts/d68-local-baseline/checkpoints/step-0002.pt', seed=20261026,
          cases=cases, source_sha256=sources, steps=100, max_wall_s=600,
          deadline_unix=1793037775., learning_rate=.001, max_request_ce=.05,
          token_cap=512, padding=dict(local=400, mel=1600, song=1024, reference=128),
          report='reports/reference_completion_probe_v1.json',
          artifacts='artifacts/reference-completion-v1/health',
          quality_training_allowed_only_if_probe_passes=True,
          preparation_wall_s=time.monotonic() - started))
    print(json.dumps(dict(prepared=True, cases=cases, wall_s=time.monotonic() - started)), flush=True)


def transfer(pair, device):
    return tuple({k: v.to(device) for k, v in section.items()} for section in pair)


def probe():
    plan = read(PLAN)
    if (ROOT / plan['report']).exists():
        raise FileExistsError(plan['report'])
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, passed=False, status='verifying',
                  optimizer_updates=0, quality_updates=0, new_cloud_compute_calls=0,
                  plan_sha256=sha(PLAN), cases=[], curve=[], final=[], generation=[])
    def save():
        report['wall_s'] = time.monotonic() - started
        write(plan['report'], report)
    def deadline():
        if time.monotonic() - started >= plan['max_wall_s'] or time.time() >= plan['deadline_unix']:
            raise TimeoutError('Frozen completion health limit')
    try:
        for path, expected in plan['source_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen identity changed: ' + path)
        if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise RuntimeError('Native MPS required without fallback')
        torch.set_num_threads(2)
        torch.mps.set_per_process_memory_fraction(.6)
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        cpu, requests, metadata_list = [], [], []
        for case in plan['cases']:
            deadline()
            value, request, metadata = completion_row(panel, case['index'], case['task'], reference=True, seed=plan['seed'])
            blank, other, _ = completion_row(panel, case['index'], case['task'], reference=False, seed=plan['seed'])
            if panel.records[case['index']]['split'] != 'train' or metadata['chart_id'] != case['chart_id']:
                raise ValueError('Health case identity changed')
            if other != request:
                raise ValueError('Reference changed request/grammar')
            for k, v in value['inputs'].items():
                if not k.startswith('reference') and not np.array_equal(v, blank['inputs'][k]):
                    raise ValueError('Unmatched non-reference input: ' + k)
            for k, v in value['sequence'].items():
                if not np.array_equal(v, blank['sequence'][k]):
                    raise ValueError('Unmatched target: ' + k)
            inputs, seq = collate([value])
            inputs = pad_input_shapes(inputs, plan['padding'])
            if not seq['allowed'].gather(2, seq['targets'][..., None])[seq['valid']].all():
                raise ValueError('Illegal target')
            if inputs['controls'][..., 2].any():
                raise ValueError('Hidden condition labels supplied')
            report['cases'].append(dict(**case, metadata=metadata,
                tensors={s: {k: tensor_health(v.numpy()) for k, v in values.items()}
                         for s, values in (('inputs', inputs), ('sequence', seq))},
                nonreference_inputs_and_targets_equal=True))
            cpu.append((inputs, seq)); requests.append(request); metadata_list.append(metadata)
        saved = torch.load(ROOT / plan['parent'], map_location='cpu', weights_only=False)
        if saved['updates'] != 2 or saved['config']['dropout'] != 0:
            raise ValueError('Common parent identity differs')
        model = FullEventModel(FullEventConfig(**saved['config'])).to('mps')
        model.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
        optimizer.load_state_dict(deepcopy(saved['optimizer']))
        if any(g['lr'] != plan['learning_rate'] or g['weight_decay'] != 0 for g in optimizer.param_groups):
            raise ValueError('Parent optimizer hyperparameters differ')
        torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
        del saved
        # New reference lengths use the same model; check actual CPU/MPS forward parity.
        model.eval()
        with torch.no_grad():
            inp, seq = transfer(cpu[0], 'mps')
            mps_logits = model(inp, seq).cpu()[cpu[0][1]['valid']]
            model.to('cpu')
            cpu_logits = model(*cpu[0])[cpu[0][1]['valid']]
            report['cpu_mps_valid_logit_max_abs'] = float((cpu_logits - mps_logits).abs().max())
            if report['cpu_mps_valid_logit_max_abs'] > 1e-4:
                raise ValueError('New reference input CPU/MPS parity failed')
            model.to('mps')
        model.train()
        report.update(status='actual_input_gate_passed', initial=[])
        with torch.no_grad():
            for pair in cpu:
                inp, seq = transfer(pair, 'mps')
                report['initial'].append(float(event_loss(model(inp, seq), seq)))
        save()
        print(json.dumps(dict(status=report['status'], initial=report['initial'])), flush=True)
        for step in range(1, plan['steps'] + 1):
            deadline()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for pair in cpu:
                inp, seq = transfer(pair, 'mps')
                loss = event_loss(model(inp, seq), seq) / len(cpu)
                loss.backward()
                total += float(loss.detach())
                del inp, seq, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1:
                gradients = Counter()
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise ValueError('Nonfinite real gradient')
                        gradients[name.split('.')[0]] += float(p.grad.square().sum())
                report['gradient_norms'] = {k: v**.5 for k, v in gradients.items()}
                for k in ('reference_projection', 'local_projection', 'mel_projection', 'token_embedding', 'decoder', 'output'):
                    if gradients[k] <= 0:
                        raise ValueError('No real gradient for ' + k)
            optimizer.step()
            report['optimizer_updates'] = step
            if step % 10 == 0:
                report.update(status='fixed_batch_training')
                report['curve'].append(dict(step=step, ce=total, preclip_norm=float(norm)))
                save()
                print(json.dumps(dict(step=step, ce=total, wall_s=report['wall_s'])), flush=True)
        model.eval()
        with torch.no_grad():
            for case, pair in zip(plan['cases'], cpu):
                inp, seq = transfer(pair, 'mps')
                logits = model(inp, seq)
                probs = logits.masked_fill(~seq['allowed'], -1e9)
                heads = {}
                for field_id in seq['fields'][seq['valid']].unique().tolist():
                    mask = seq['valid'] & (seq['fields'] == field_id)
                    heads[str(field_id)] = dict(count=int(mask.sum()), accuracy=float((probs.argmax(-1)[mask] == seq['targets'][mask]).float().mean()))
                report['final'].append(dict(index=case['index'], task=case['task'], ce=float(event_loss(logits, seq)), heads=heads))
        # CPU cached greedy generation is an independent exact-reconstruction check.
        del optimizer
        model.to('cpu')
        for case, pair, request, metadata in zip(plan['cases'], cpu, requests, metadata_list):
            deadline()
            prediction = generate_batch(model, pair[0], [request], plan['token_cap'],
                        deadline=min(started + plan['max_wall_s'], time.monotonic() + plan['deadline_unix'] - time.time()))[0]
            target_tokens = pair[1]['targets'][0][pair[1]['valid'][0]].tolist()
            exact = prediction['complete'] and prediction['tokens'] == target_tokens
            path = f"{plan['artifacts']}/{case['index']}-{case['task']}.json"
            write(path, dict(metadata=metadata, prediction=prediction, exact=exact))
            report['generation'].append(dict(index=case['index'], task=case['task'],
                complete=prediction['complete'], exact=exact, artifact=path,
                predicted_types=dict(Counter(e['type'] for e in prediction['events']))))
            save()
        report.update(complete=True, status='completed',
            passed=all(r['ce'] <= plan['max_request_ce'] for r in report['final'])
                   and len(report['generation']) == len(plan['cases'])
                   and all(r['exact'] for r in report['generation']),
            parent_unchanged=sha(plan['parent']) == plan['source_sha256'][plan['parent']])
        # Health weights never become a quality-training initialization.
        write(f"{plan['artifacts']}/disposition.json", dict(probe_only=True,
              passed=report['passed'], quality_training_started=False, auto_resume=False))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(dict(complete=report['complete'], passed=report['passed'], wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else probe()

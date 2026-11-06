"""Bounded real fixed-batch probe after the complete-song loader migration."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    out = ROOT/plan['report']
    if out.exists():
        raise FileExistsError(out)
    for path, expected in plan['inputs_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Probe input changed: '+path)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0' or not torch.backends.mps.is_available():
        raise ValueError('Native MPS required')
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    torch.mps.manual_seed(plan['seed'])
    torch.mps.set_per_process_memory_fraction(.6)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'], allow_partial_features=True)
    indices = []
    for case in plan['cases']:
        found = [i for i, r in enumerate(panel.records)
                 if r['chart_id'] == case['chart_id'] and r['request']['start_us'] == case['start_us'] and r['split'] == 'train']
        if len(found) != 1:
            raise ValueError('Registered training probe request missing or ambiguous')
        indices.extend(found)
    inp, seq = panel.batch(indices)
    selected_charts = {panel.records[i]['chart_index'] for i in indices}
    for i in selected_charts:
        row = panel.charts[i]
        for field in ('target', 'token'):
            if sha(ROOT/row[field+'_path']) != row[field+'_sha256']:
                raise ValueError('Prepared target changed')
        feature = panel.features.records[row['audio_sha256']]
        for path, expected in feature['sha256'].items():
            if sha(ROOT/path) != expected:
                raise ValueError('Prepared feature changed')
    report = dict(id=plan['id'], plan_sha256=sha(plan_path), completed=False, passed=False,
        optimizer_updates=0, quality_passed=False, cloud_compute_calls=0, scope='fixed batch implementation health only',
        tensor_stats={section: {k: tensor_health(v.numpy()) for k, v in tensors.items()}
                      for section, tensors in [('inputs', inp), ('sequence', seq)]},
        cases=[dict(chart_id=panel.records[i]['chart_id'], request=panel.records[i]['request'],
                    human_model_view=[asdict(e) for e in panel.targets(i)],
                    condition_source=panel.records[i]['condition_source']) for i in indices],
        code_sha256={p: sha(ROOT/p) for p in ['scripts/probe_local_request_panel.py',
            'beatfcn/train/local_request_panel.py','beatfcn/train/local_features.py',
            'beatfcn/model/full_event_model.py','beatfcn/model/full_event_generate.py']}, curve=[])
    model = FullEventModel(FullEventConfig(**plan['config'])).to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    inp, seq = ({k: v.to('mps') for k, v in d.items()} for d in (inp, seq))
    started = time.monotonic()

    def save():
        report.update(wall_s=time.monotonic()-started, mps_bytes=torch.mps.current_allocated_memory())
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp.replace(out)

    def measure(step):
        model.eval()
        with torch.no_grad():
            logits = model(inp, seq)
            pred = logits.masked_fill(~seq['allowed'], -1e9).argmax(-1)
            correct = (pred == seq['targets']) & seq['valid']
            fields = {}
            for i, name in enumerate(FIELDS):
                mask = (seq['fields'] == i) & seq['valid']
                if bool(mask.any()):
                    fields[name] = float(correct[mask].float().mean())
            point = dict(step=step, event_ce=float(event_loss(logits, seq)), accuracy=fields)
        report['curve'].append(point)
        model.train()
        save()
        return point

    try:
        measure(0)
        for step in range(1, plan['steps']+1):
            if time.monotonic()-started > plan['max_wall_s']:
                raise TimeoutError('Local probe deadline')
            optimizer.zero_grad(set_to_none=True)
            loss = event_loss(model(inp, seq), seq)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report.update(optimizer_updates=step, gradient_norm=float(grad))
            if step in (50, 100):
                point = measure(step)
        model.eval()
        outputs = generate_batch(model, inp, [event_request(panel.records[i]['request']) for i in indices],
            max_tokens=plan['generation_token_limit'], deadline=time.monotonic()+plan['generation_max_s'])
        for case, prediction in zip(report['cases'], outputs):
            case['prediction'] = prediction
            case['exact_multiset'] = Counter(tuple(sorted(e.items())) for e in case['human_model_view']) == Counter(
                tuple(sorted(e.items())) for e in prediction['events'])
        report.update(completed=True, passed=point['event_ce'] <= plan['maximum_ce']
            and min(point['accuracy'].values()) >= plan['minimum_field_accuracy']
            and all(c['prediction']['complete'] and c['exact_multiset'] for c in report['cases']))
        save()
        print(json.dumps(dict(passed=report['passed'], final=point,
            exact=[c['exact_multiset'] for c in report['cases']], wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', completed=True)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

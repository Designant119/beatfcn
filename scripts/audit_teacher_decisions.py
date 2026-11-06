"""Bounded zero-update teacher-decision diagnosis of an archived checkpoint."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.prepare_local_data_scale_probe import padding_shapes

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/d98-teacher-decisions-v1.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(path, result):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def run():
    plan = read(PLAN)
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    for p, expected in plan['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Frozen source changed: ' + p)
    torch.set_num_threads(2)
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    if saved['updates'] != 1000 or saved['plan_sha256'] != sha(plan['training_plan']):
        raise ValueError('Checkpoint provenance changed')
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    del saved
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    report = dict(id='D98-D97-teacher-decision-audit-v1', complete=False,
        rows=[], plan_sha256=sha(PLAN), source_sha256=dict(plan['source_sha256']),
        optimizer_updates=0, model_forward_calls=0, new_cloud_compute_calls=0,
        quality_passed=False, backend='cpu', limitations=plan['limitations'])
    verified = set()
    try:
        for selected in plan['records']:
            if time.monotonic() - started > plan['max_wall_s'] or time.time() >= plan['deadline_unix']:
                raise TimeoutError('Original diagnostic wall/deadline bound')
            index = selected['index']
            row = panel.records[index]
            if row['split'] != selected['split']:
                raise ValueError('Diagnostic split changed')
            chart = panel.charts[row['chart_index']]
            sources = {**panel.features.records[row['audio_sha256']]['sha256'],
                       chart['target_path']: chart['target_sha256'], chart['token_path']: chart['token_sha256']}
            for path, digest in sources.items():
                if path not in verified:
                    if sha(path) != digest:
                        raise ValueError('Actual feature/target changed: ' + path)
                    verified.add(path)
                    report['source_sha256'][path] = digest
            controls = np.asarray(row['controls'], np.float32).copy()
            fields = selected['specified_fields']
            controls[[k for k in range(len(controls)) if k not in fields], :3] = 0
            if row['split'] == 'val' and controls[:, 2].any():
                raise ValueError('Hidden held-out condition supplied')
            inputs, sequence = panel.batch([index], controls=[controls], device='cpu')
            logits = model(pad_input_shapes(inputs, padding_shapes(inputs['song'].shape[1])), sequence)[0]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite real teacher logits')
            allowed, targets, field_ids, valid = (sequence[k][0] for k in ('allowed', 'targets', 'fields', 'valid'))
            legal = logits.float().masked_fill(~allowed, -1e9)
            predicted = legal.argmax(-1)
            nll = -legal.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            supports = allowed.sum(-1)
            statistics = {}
            for f, name in enumerate(FIELDS):
                mask = valid & (field_ids == f)
                if not mask.any():
                    continue
                ys, ps = targets[mask], predicted[mask]
                truth_values = [model.vocab.decision(int(y))[1] for y in ys]
                pred_values = [model.vocab.decision(int(y))[1] for y in ps]
                statistics[name] = dict(count=int(mask.sum()), correct=int((ys == ps).sum()),
                    nll_sum=float(nll[mask].sum()), forced=int((supports[mask] == 1).sum()),
                    nonforced_count=int((supports[mask] > 1).sum()),
                    nonforced_correct=int(((ys == ps) & (supports[mask] > 1)).sum()),
                    confusion=dict(Counter(f'{a}->{b}' for a, b in zip(truth_values, pred_values, strict=True))))
            item = dict(index=index, split=row['split'], mode=row['mode'], chart_id=row['chart_id'],
                component=row['component'], stratum=row['stratum'], scope_us=[row['request']['start_us'], row['request']['end_us']],
                specified_fields=fields, selection_reason=selected['selection_reason'],
                valid_tokens=int(valid.sum()), teacher_ce=float(nll[valid].mean()), fields=statistics)
            report['rows'].append(item)
            report['model_forward_calls'] += 1
            report['wall_s'] = time.monotonic() - started
            write(output, report)
            del inputs, sequence, logits, legal
        groups = defaultdict(list)
        for row in report['rows']:
            groups[row['split'] + '/' + row['mode']].append(row)
        summary = {}
        for key, rows in groups.items():
            summary[key] = dict(requests=len(rows), songs=len({r['component'] for r in rows}),
                request_mean_teacher_ce=sum(r['teacher_ce'] for r in rows) / len(rows), fields={})
            for field in FIELDS:
                metrics = [r['fields'][field] for r in rows if field in r['fields']]
                if not metrics:
                    continue
                totals = {k: sum(m[k] for m in metrics) for k in
                          ('count', 'correct', 'nll_sum', 'forced', 'nonforced_count', 'nonforced_correct')}
                confusion = Counter()
                for m in metrics:
                    confusion.update(m['confusion'])
                totals.update(accuracy=totals['correct'] / totals['count'],
                    nll_mean=totals['nll_sum'] / totals['count'], confusion=dict(confusion))
                summary[key]['fields'][field] = totals
        report.update(complete=True, summary=summary, wall_s=time.monotonic() - started,
                      type_value_order=list(TYPES), checkpoint_unchanged=sha(plan['checkpoint']) == plan['source_sha256'][plan['checkpoint']])
        if not report['checkpoint_unchanged']:
            raise ValueError('Archived checkpoint changed')
        write(output, report)
        print(json.dumps(dict(complete=True, wall_s=report['wall_s'], forward_calls=report['model_forward_calls'],
            kind={k: v['fields'].get('kind') for k, v in summary.items()})))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic() - started)
        write(output, report)
        raise


if __name__ == '__main__':
    run()

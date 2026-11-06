"""CPU prototype: train on counterfactual density variants and test control.

Same audio/context, target edited to the requested heads/s.  This directly tests
whether the sample factory fixes the failure of target-derived curves.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.counterfactual_density import edit_events_to_density
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.training_panel import build_training_panel
from beatfcn.train.training_objective import training_loss
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/data-contract-v2-final'
CELL_US = 500_000


def make_row(panel, index, rate):
    record = panel.records[index]
    request = event_request(record['request'])
    start, end = request.start_us, request.end_us
    events, actual = edit_events_to_density(panel.targets(index), start, end,
        rate, request.mode, request.duration_us)
    sequence = teacher_sequence(encode_events(events, request), request)
    curve = grid_density_curve(
        [dict(start_us=start, end_us=end, density=float(rate))], start, end,
        cell_us=100_000, scale=16.)
    row = panel.row(index)
    row['inputs'] = dict(row['inputs'])
    row['inputs'].update(curve)
    return dict(inputs=row['inputs'], sequence=sequence), actual


def objective(model, inputs, sequence, settings, first_weight=5.):
    logits = model(inputs, sequence)
    base = training_loss(logits, sequence, settings)
    valid = sequence['valid']
    legal = logits.float().masked_fill(~sequence['allowed'], -1e9)
    token = F.cross_entropy(legal.flatten(0, 1), sequence['targets'].flatten(),
        reduction='none').view_as(valid)
    first = (token*valid)[:, 0].mean()
    return base + first_weight*first


def rows_for(panel, pairs, rates):
    rows, actual = [], []
    for index, rate in zip(pairs, rates):
        row, values = make_row(panel, index, rate)
        rows.append(row)
        actual.append(values)
    return collate(rows), actual


def main():
    plan = json.loads((REPORT/'probe-plan.json').read_text())
    config = FullEventConfig(**plan['config'])
    panel = build_training_panel(ROOT, plan)
    indices = list(plan['tensor_probe_indices'])
    torch.set_num_threads(4)
    random.seed(20261101)
    torch.manual_seed(20261101)
    model = FullEventModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    train_rates = [0., 2., 4., 8.]
    model.train()
    losses = []
    for step in range(600):
        batch_indices = [random.choice(indices) for _ in range(4)]
        rates = [random.choice(train_rates) for _ in range(4)]
        (inputs, sequence), _ = rows_for(panel, batch_indices, rates)
        loss = objective(model, inputs, sequence, plan.get('loss'))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            print(json.dumps(dict(step=step, ce=losses[-1])), flush=True)
    model.eval()
    results = []
    with torch.no_grad():
        for index in indices:
            for rate in (0., 1., 2., 3., 4., 6., 8.):
                (inputs, sequence), actual = rows_for(panel, [index], [rate])
                correct = float(event_loss(model(inputs, sequence), sequence))
                other = 8. if rate != 8. else 0.
                (wrong_inputs, wrong_sequence), _ = rows_for(panel, [index], [other])
                # Keep the target fixed; only the curve differs.
                request = event_request(panel.records[index]['request'])
                wrong_inputs = dict(inputs)
                wrong_curve = grid_density_curve(
                    [dict(start_us=request.start_us, end_us=request.end_us,
                          density=other)], request.start_us,
                    request.end_us, cell_us=100_000, scale=16.)
                wrong_inputs.update({key: torch.as_tensor(value)[None]
                    for key, value in wrong_curve.items()})
                shuffled = float(event_loss(model(wrong_inputs, sequence), sequence))
                prediction = generate_batch(model, inputs, [request], max_tokens=512)[0]
                counts = np.zeros(max(1, math.ceil((request.end_us-request.start_us)/CELL_US)))
                for event in prediction['events']:
                    cell = (event['onset_us']-request.start_us)//CELL_US
                    if 0 <= cell < len(counts):
                        counts[cell] += 1
                results.append(dict(index=index, rate=rate, actual_density=float(np.mean(actual[0])),
                    correct_ce=correct, shuffled_ce=shuffled, delta=shuffled-correct,
                    generated_events=len(prediction['events']),
                    generated_mean=float(counts.mean()/(CELL_US/1e6)),
                    generated_cells=counts.tolist(), complete=prediction['complete']))
    report = dict(steps=600, train_rates=train_rates, final_train_ce=losses[-1],
        results=results,
        interpretation='If correct CE is lower than shuffled and generated density tracks rate, the sample factory works.')
    (REPORT/'counterfactual-density-probe.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

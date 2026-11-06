"""CPU causal diagnostic: does the v2 intensity condition change the loss?

Trains the bounded v2 probe through the real driver, then holds the audio,
prefix and target fixed and swaps only the intensity curve.  A near-zero delta
means the model ignores the condition regardless of generation quality.
"""
import copy
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.density_curve import absent_curve, requested_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.training_panel import build_training_panel
from scripts.run_local_song_baseline import ROOT, run

REPORT = ROOT/'reports/data-contract-v2-final'


def loss_for(model, panel, index, curve=None):
    row = panel.row(index)
    if curve is not None:
        row['inputs'] = dict(row['inputs'])
        row['inputs'].update(curve)
    inp, sequence = collate([row])
    return event_loss(model(inp, sequence), sequence)


def grad_norm(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    weight = model.intensity_projection.weight.grad
    bias = model.intensity_projection.bias.grad
    return dict(weight=float(weight.norm()) if weight is not None else None,
                bias=float(bias.norm()) if bias is not None else None)


def main():
    plan = json.loads((REPORT/'probe-plan.json').read_text())
    plan = copy.deepcopy(plan)
    plan['id'] = 'data-contract-v2-density-diagnostic'
    plan['output'] = 'reports/data-contract-v2-final/diagnostic-training.json'
    plan['checkpoint_dir'] = 'inputs/data-contract-v2-final-probe/diagnostic-checkpoints'
    plan['note'] = 'Causal condition diagnostic; no generation quality claim'
    plan_path = REPORT/'diagnostic-plan.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1)+'\n')
    if (ROOT/plan['output']).exists():
        raise FileExistsError('Diagnostic output already exists')
    run(plan_path)
    training = json.loads((ROOT/plan['output']).read_text())
    checkpoint = torch.load(ROOT/training['checkpoint'], map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**plan['config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    panel = build_training_panel(ROOT, plan)
    results = []
    with torch.no_grad():
        for index in plan['tensor_probe_indices']:
            row = panel.row(index)
            start = int(panel.records[index]['request']['start_us'])
            end = int(panel.records[index]['request']['end_us'])
            scale = float(plan['intensity']['scale'])
            base_curve = row['inputs']['intensity'].copy()
            shuffled = base_curve.copy()
            shuffled[:, 0] = base_curve[::-1, 0]
            curves = dict(
                correct=row['inputs'],
                zero=requested_density_curve([dict(start_us=start, end_us=end, density=0.)],
                                             start, end, scale=scale),
                shuffled_curve=dict(intensity=shuffled,
                    intensity_times=row['inputs']['intensity_times'],
                    intensity_valid=row['inputs']['intensity_valid']),
                absent=absent_curve(start, end, int(plan['intensity']['cell_us'])))
            values = {}
            for name, curve in curves.items():
                if name == 'correct':
                    loss = loss_for(model, panel, index)
                else:
                    loss = loss_for(model, panel, index, curve)
                values[name] = loss
            correct = values['correct']
            with torch.enable_grad():
                grad = grad_norm(model, loss_for(model, panel, index))
            results.append(dict(index=index,
                correct_ce=float(correct),
                zero_ce=float(values['zero']),
                shuffled_ce=float(values['shuffled_curve']),
                absent_ce=float(values['absent']),
                delta_zero=float(values['zero']-correct),
                delta_shuffled=float(values['shuffled_curve']-correct),
                delta_absent=float(values['absent']-correct),
                correct_intensity_grad=grad,
                curve_cells=int(len(base_curve))))
    report = dict(plan=plan_path.name, training=plan['output'],
        updates=training['updates'], parameters=training['parameters'],
        results=results,
        interpretation=('A near-zero delta means the model ignores the intensity condition; '
                        'generation differences are not evidence of control.'))
    (REPORT/'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

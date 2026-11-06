"""Train the bounded v2 probe with condition-contrast loss, then test control.

This is still a CPU wiring/optimization probe: two training windows, 1000
updates.  The question is whether the loss can make the correct curve beat a
shuffled curve and change free generation.  It is not a quality claim.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.control_metrics import density_request_metrics, readout_metrics, type_metrics
from beatfcn.train.density_curve import requested_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.training_panel import build_training_panel
from scripts.run_local_song_baseline import ROOT, run

REPORT = ROOT/'reports/data-contract-v2-final'
CACHE = ROOT/'inputs/data-contract-v2-final-probe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_exposure():
    source = json.loads((CACHE/'exposure.json').read_text())
    steps = []
    for index in range(1000):
        steps.append(dict(effective_update=index+1,
            indices=list(source['steps'][index % len(source['steps'])]['indices']),
            specified_condition_fields=[[], [], [], []]))
    exposure = dict(pool_identity=source['pool_identity'], steps=steps)
    path = CACHE/'contrast-exposure-1000.json'
    path.write_text(json.dumps(exposure, indent=1)+'\n')
    return path


def main():
    plan = copy.deepcopy(json.loads((REPORT/'probe-plan.json').read_text()))
    exposure_path = make_exposure()
    plan.update(id='data-contract-v2-contrast-probe', steps=1000,
        exposure=str(exposure_path.relative_to(ROOT)),
        output='reports/data-contract-v2-final/contrast-training.json',
        checkpoint_dir='inputs/data-contract-v2-final-probe/contrast-checkpoints',
        checkpoint_every=1000, checkpoints=[], max_train_wall_s=900,
        condition_contrast=dict(weight=1.0, margin=0.),
        note='Condition-contrast wiring probe; CPU only; no quality claim')
    plan['inputs_sha256'][str(exposure_path.relative_to(ROOT))] = digest(exposure_path)
    plan['code_sha256'] = {path: digest(path) for path in plan['code_sha256']}
    plan_path = REPORT/'contrast-plan.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1)+'\n')
    if (ROOT/plan['output']).exists():
        raise FileExistsError('Contrast probe already exists')
    run(plan_path)
    training = json.loads((ROOT/plan['output']).read_text())
    saved = torch.load(ROOT/training['checkpoint'], map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**plan['config']))
    model.load_state_dict(saved['model'])
    model.eval()
    panel = build_training_panel(ROOT, plan)
    details = []
    with torch.no_grad():
        for index in plan['tensor_probe_indices']:
            row = panel.row(index)
            start = int(panel.records[index]['request']['start_us'])
            end = int(panel.records[index]['request']['end_us'])
            scale = float(plan['intensity']['scale'])
            base = row['inputs']['intensity'].copy()
            shuffled = base.copy()
            shuffled[:, 0] = base[::-1, 0]
            values = {}
            for name, curve in (
                    ('correct', None),
                    ('zero', requested_density_curve([dict(start_us=start, end_us=end, density=0.)],
                        start, end, scale=scale)),
                    ('shuffled', dict(intensity=shuffled,
                        intensity_times=row['inputs']['intensity_times'],
                        intensity_valid=row['inputs']['intensity_valid']))):
                probe_row = dict(row)
                probe_row['inputs'] = dict(row['inputs'])
                if curve is not None:
                    probe_row['inputs'].update(curve)
                inp, sequence = collate([probe_row])
                values[name] = float(event_loss(model(inp, sequence), sequence))
            details.append(dict(index=index, **values,
                delta_zero=values['zero']-values['correct'],
                delta_shuffled=values['shuffled']-values['correct']))
    outputs = []
    index = plan['tensor_probe_indices'][0]
    request = event_request(panel.records[index]['request'])
    for density in (0., 32.):
        segments = [dict(start_us=request.start_us, end_us=request.end_us, density=density)]
        row = panel.row(index)
        row['inputs'].update(requested_density_curve(segments, request.start_us, request.end_us))
        inp, _ = collate([row])
        prediction = generate_batch(model, inp, [request], max_tokens=512,
            deadline=None)[0]
        outputs.append(dict(requested_density=density,
            prediction=prediction,
            curve=density_request_metrics(prediction, segments),
            types=type_metrics(prediction, panel.targets(index)),
            readout=readout_metrics([prediction])))
    identical = outputs[0]['prediction']['events'] == outputs[1]['prediction']['events']
    result = dict(training=plan['output'], updates=training['updates'],
        parameters=training['parameters'], free_control_identical=identical,
        condition_ablation=details, free_control=outputs,
        interpretation='Correct curve must beat shuffled and change free generation; otherwise contrast did not work.')
    (REPORT/'contrast-diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'free_control'}, ensure_ascii=False, indent=1))
    for output in outputs:
        print(json.dumps(dict(density=output['requested_density'],
            events=len(output['prediction']['events']),
            curve=output['curve'], types=output['types']['predicted_types']), ensure_ascii=False))


if __name__ == '__main__':
    main()

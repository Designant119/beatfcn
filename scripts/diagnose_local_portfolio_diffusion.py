"""Zero-update decomposition of masked reconstruction versus full-MASK generation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from beatfcn.model.canvas_diffusion import AbsorbingCanvasDiffusion, ancestral_sample, diffusion_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.relative_canvas import relative_batch
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory
from scripts.run_local_typed_canvas import score


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'bench/rhythm-v2/local-portfolio-diffusion-health-v1.json'
REPORT = ROOT / 'reports/local_portfolio_diffusion_health_v1.json'
CHECKPOINT = ROOT / 'artifacts/local-portfolio-v1/diffusion-health/health-0100.pt'
DEST = ROOT / 'reports/local_portfolio_diffusion_reverse_diagnostic_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/diffusion-health/reverse-diagnostic.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss_at(model, example, inputs, target, t, seed):
    noisy = model.corrupt(target, t, generator=torch.Generator().manual_seed(seed))
    time_tensor = torch.tensor([t])
    with torch.no_grad():
        output = model(inputs, noisy, time_tensor)
        loss, detail = diffusion_loss(output, target, noisy, inputs, [example['request']], time_tensor)
    return float(loss), {key: float(value) for key, value in detail.items()}


def main():
    torch.set_num_threads(2)
    plan = json.loads(PLAN.read_text())
    health = json.loads(REPORT.read_text())
    if health['status'] != 'health_passed' or health['health_updates'] != 100:
        raise ValueError('Expected the completed diffusion health checkpoint')
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    batches = []
    for index, view in specs:
        example = row(index, view)
        batches.append((example, *relative_batch([example], 'cpu')))
    torch.manual_seed(plan['seed'])
    initial = AbsorbingCanvasDiffusion(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], steps=plan['steps'])
    initial_t32 = []
    for i, (example, inputs, target) in enumerate(batches):
        loss, detail = loss_at(initial, example, inputs, target, 32, plan['seed'] + 5000 + i)
        initial_t32.append(dict(index=example['index'], loss=loss, **detail))
    del initial
    model = AbsorbingCanvasDiffusion(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], steps=plan['steps'])
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if saved['plan_sha256'] != sha(PLAN) or saved['updates'] != 100:
        raise ValueError('Wrong diffusion checkpoint')
    model.load_state_dict(saved['model'])
    model.eval()
    times = [1, 2, 4, 8, 16, 24, 31, 32]
    curves = []
    for i, (example, inputs, target) in enumerate(batches):
        for t in times:
            loss, detail = loss_at(model, example, inputs, target, t, plan['seed'] + 5000 + i)
            curves.append(dict(index=example['index'], t=t, loss=loss, **detail))
    sensitivity = []
    for i, (example, inputs, target) in enumerate(batches):
        other = batches[(i + 1) % len(batches)][1]
        noisy = model.corrupt(target, 32, generator=torch.Generator().manual_seed(plan['seed'] + 5000 + i))
        t = torch.tensor([32])
        wrong_inputs = dict(inputs)
        for key in ('local', 'local_times', 'local_valid', 'mel', 'mel_times', 'mel_valid', 'song', 'song_times', 'song_valid'):
            wrong_inputs[key] = other[key]
        with torch.no_grad():
            correct_output = model(inputs, noisy, t)
            wrong_output = model(wrong_inputs, noisy, t)
        difference = (correct_output['offset0'] - wrong_output['offset0']).abs()
        valid = inputs['canvas_valid']
        correct_probability = correct_output['offset0'].softmax(-1)
        wrong_probability = wrong_output['offset0'].softmax(-1)
        correct_expected = ((1 - correct_probability[..., 0]).sum((2, 3)) * valid).sum()
        wrong_expected = ((1 - wrong_probability[..., 0]).sum((2, 3)) * valid).sum()
        sensitivity.append(dict(index=example['index'], wrong_index=batches[(i + 1) % len(batches)][0]['index'],
            logits_mean_absolute=float(difference.mean()), logits_max_absolute=float(difference.max()),
            correct_expected_events=float(correct_expected), wrong_expected_events=float(wrong_expected)))
    reconstructions = []
    started = time.monotonic()
    for i in (0, 2, 4):
        example, inputs, target = batches[i]
        for t in (4, 16, 31, 32):
            noisy = model.corrupt(target, t, generator=torch.Generator().manual_seed(plan['seed'] + 5000 + i))
            prediction = ancestral_sample(model, inputs, example['request'], seed=plan['seed'] + 8000 + i,
                initial_noisy=noisy, start_step=t)
            events = [RhythmEvent(**event) for event in prediction['events']]
            reconstructions.append(dict(index=example['index'], start_t=t, prediction=prediction,
                target=[asdict(event) for event in example['targets']], scores=score(events, example['targets'])))
    result = dict(id='local-portfolio-diffusion-reverse-diagnostic-v1', updates=0,
        source_health_sha256=sha(REPORT), checkpoint_sha256=sha(CHECKPOINT),
        fixed_diagnostic_times=sorted({row['t'] for row in health['diagnostic_initial']['rows']}),
        training_t32_updates_per_example=[sum(1 + (step * 7 + i * 5) % plan['steps'] == plan['steps']
            for step in range(1, plan['health_steps'] + 1)) for i in range(len(batches))],
        initial_t32=initial_t32, trained_curves=curves, audio_sensitivity_t32=sensitivity,
        reverse_reconstructions=reconstructions, wall_s=time.monotonic() - started,
        note='All inputs are training health windows. Partial reverse chains use authorized noised targets for diagnosis only; no development or confirmation scoring.')
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(reconstructions, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(fixed_diagnostic_times=result['fixed_diagnostic_times'],
        training_t32_updates_per_example=result['training_t32_updates_per_example'],
        initial_t32=[{'index': row['index'], 'auxiliary': row['auxiliary']} for row in initial_t32],
        trained_t32=[{'index': row['index'], 'auxiliary': row['auxiliary']} for row in curves if row['t'] == 32],
        audio_sensitivity_t32=sensitivity,
        reconstructions=[{'index': row['index'], 't': row['start_t'], 'count': row['prediction']['predicted_count'],
            'target': len(row['target']), 'full': row['scores']['full_note_f1'],
            'onset10': (row['scores']['onset_10ms'] or {}).get('f1')} for row in reconstructions]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

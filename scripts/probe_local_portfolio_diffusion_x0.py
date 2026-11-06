"""Compare one-shot x0/argmax with the already saved ancestral failures."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.model.canvas_diffusion import AbsorbingCanvasDiffusion, reconstruction_probe
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
DEST = ROOT / 'reports/local_portfolio_diffusion_x0_probe_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/diffusion-health/x0-probes.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    plan = json.loads(PLAN.read_text())
    comparison = json.loads((ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text())
    panel, store = setup(comparison)
    row = row_factory(panel, store)
    model = AbsorbingCanvasDiffusion(FullEventConfig(**plan['model']),
        density_reference=plan['density_reference'], steps=plan['steps'])
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if saved['plan_sha256'] != sha(PLAN):
        raise ValueError('Checkpoint plan identity drift')
    model.load_state_dict(saved['model'])
    model.eval()
    specs = list(zip(plan['health_indices'], ('draft', 'single_condition', 'random_groups', 'middle')))
    specs.append((plan['empty_health_index'], 'draft'))
    rows = []
    for i, (index, view) in enumerate(specs):
        example = row(index, view)
        inputs, target = relative_batch([example], 'cpu')
        for t in (1, 16, 32):
            noisy = model.corrupt(target, t, generator=torch.Generator().manual_seed(plan['seed'] + 5000 + i))
            prediction = reconstruction_probe(model, inputs, noisy, torch.tensor([t]), example['request'])
            events = [RhythmEvent(**event) for event in prediction['events']]
            rows.append(dict(index=index, t=t, prediction=prediction,
                target=[asdict(event) for event in example['targets']], scores=score(events, example['targets'])))
    result = dict(id='local-portfolio-diffusion-x0-probe-v1', updates=0,
        source_health_sha256=sha(REPORT), checkpoint_sha256=sha(CHECKPOINT), rows=rows,
        note='One-shot x0/argmax reconstruction probes, not ancestral quality results; training health windows only.')
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    ART.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([dict(index=row['index'], t=row['t'], count=row['prediction']['predicted_count'],
        target=len(row['target']), full=row['scores']['full_note_f1'],
        onset10=(row['scores']['onset_10ms'] or {}).get('f1')) for row in rows], indent=2))


if __name__ == '__main__':
    main()

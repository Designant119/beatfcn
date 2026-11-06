"""Real loss/gradient equivalence across physical microbatch sizes; no updates."""
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes

ROOT = Path(__file__).resolve().parents[1]


def run():
    output = ROOT/'reports/d68_microbatch_parity_v1.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.6)
    plan = json.loads((ROOT/'bench/rhythm-v2/d68-local-baseline-v2.json').read_text())
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    saved = torch.load(ROOT/'artifacts/d68-local-baseline-v2/checkpoints/step-0525.pt', map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**plan['config'])).to('mps').train()
    model.load_state_dict(saved['model'])
    update = json.loads((ROOT/plan['exposure']).read_text())['steps'][537]
    versions = []
    for micro in (2, 1):
        model.zero_grad(set_to_none=True); torch.mps.empty_cache()
        values, total = [], 0.
        for offset in range(0, 4, micro):
            ids = update['indices'][offset:offset+micro]
            controls = []
            for index, fields in zip(ids, update['specified_condition_fields'][offset:offset+micro]):
                control = np.asarray(panel.records[index]['controls'], np.float32).copy()
                control[[i for i in range(len(control)) if i not in fields], :3] = 0
                controls.append(control)
            inputs, sequence = panel.batch(ids, controls=controls, device='cpu')
            inputs = pad_input_shapes(inputs, plan['input_shapes'])
            inputs = {k: v.to('mps') for k, v in inputs.items()}
            sequence = {k: v.to('mps') for k, v in sequence.items()}
            logits = model(inputs, sequence)
            loss = event_loss(logits, sequence)*len(ids)/4
            loss.backward(); torch.mps.synchronize()
            total += float(loss.detach())
            values.append(logits.detach().cpu()[sequence['valid'].cpu()])
            del inputs, sequence, logits, loss
        versions.append(dict(loss=total, logits=torch.cat(values),
            grads={k: v.grad.detach().cpu().clone() for k, v in model.named_parameters()}))
    a, b = versions
    errors = dict(loss=abs(a['loss']-b['loss']), logits=float((a['logits']-b['logits']).abs().max()),
        gradient=max(float((a['grads'][k]-b['grads'][k]).abs().max()) for k in a['grads']))
    report = dict(passed=max(errors.values()) <= 1e-3, max_error=1e-3, errors=errors,
        indices=update['indices'], microbatches=[2,1], effective_requests=4, optimizer_updates=0,
        criterion='Existing native backend equivalence gate; equal-request CE and full gradient accumulation')
    output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))
    if not report['passed']:
        raise AssertionError('Microbatch parity failed')


if __name__ == '__main__':
    run()

"""Real-context output and all-parameter gradient CPU/MPS parity; zero updates."""
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_model import event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.probe_fullstack_mamba import make_pair
from scripts.run_completion_quality import batches
from scripts.run_reference_completion import read, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = 'reports/fullstack_mamba_backend_parity_v1.json'
    if (ROOT / output).exists():
        raise FileExistsError(output)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4); torch.mps.set_per_process_memory_fraction(.6)
    plan = read('bench/rhythm-v2/fullstack-mamba-quality-v1.json')
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    t, model, _ = make_pair(); del t
    items = [x for x in plan['steps'][0] if x['task'] is not None][:2]
    rows = []; started = time.monotonic()
    for item in items:
        inp, seq = batches(panel, item, plan)['reference']
        versions = []
        for device in ('cpu', 'mps'):
            model.to(device); model.zero_grad(set_to_none=True)
            values, sequence = ({k: v.to(device) for k, v in section.items()} for section in (inp, seq))
            tick = time.monotonic()
            logits = model(values, sequence); loss = event_loss(logits, sequence); loss.backward()
            versions.append(dict(loss=float(loss.detach()), logits=logits.detach().cpu()[seq['valid']],
                gradients={n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in model.named_parameters()},
                wall_s=time.monotonic() - tick))
            del values, sequence, logits, loss
        a, b = versions
        if any((a['gradients'][k] is None) != (b['gradients'][k] is None) for k in a['gradients']):
            raise ValueError('Backend changed gradient connectivity')
        gradient = max(float((v - b['gradients'][k]).abs().max()) for k, v in a['gradients'].items() if v is not None)
        row = dict(index=item['index'], logits_max_abs=float((a['logits'] - b['logits']).abs().max()),
                   loss_max_abs=abs(a['loss'] - b['loss']), all_parameters_gradient_max_abs=gradient,
                   cpu_wall_s=a['wall_s'], mps_wall_s=b['wall_s'])
        rows.append(row)
        if max(row['logits_max_abs'], row['loss_max_abs'], gradient) > 1e-4:
            raise ValueError('Actual backend parity exceeded frozen tolerance')
    result = dict(passed=True, rows=rows, optimizer_updates=0, cloud_calls=0,
                  tolerance=1e-4, wall_s=time.monotonic() - started)
    write(output, result); print(json.dumps(result))


if __name__ == '__main__':
    main()

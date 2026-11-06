"""Real MPS logits/gradient parity for bounded audio padding, without updates."""
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes

ROOT = Path(__file__).resolve().parents[1]


def run():
    output = ROOT/'reports/d68_shape_parity_v1.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.6)
    plan = json.loads((ROOT/'bench/rhythm-v2/d68-local-baseline-v1.json').read_text())
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    saved = torch.load(ROOT/'artifacts/d68-local-baseline/checkpoints/step-0002.pt', map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**plan['config'])).to('mps').train()
    model.load_state_dict(saved['model'])
    records = []
    for indices in ([1448, 2528], [0, 3]):
        versions = []
        for padded in (False, True):
            model.zero_grad(set_to_none=True); torch.mps.empty_cache()
            inputs, sequence = panel.batch(indices, device='cpu')
            if padded:
                inputs = pad_input_shapes(inputs, dict(local=400, mel=1600, song=896, reference=1))
            inputs = {k: v.to('mps') for k, v in inputs.items()}
            sequence = {k: v.to('mps') for k, v in sequence.items()}
            logits = model(inputs, sequence); loss = event_loss(logits, sequence); loss.backward()
            torch.mps.synchronize()
            versions.append(dict(loss=float(loss), logits=logits.detach().cpu()[sequence['valid'].cpu()],
                grads={k: v.grad.detach().cpu().clone() for k, v in model.named_parameters()}))
            del inputs, sequence, logits, loss
        a, b = versions
        records.append(dict(indices=indices, loss_max_abs=abs(a['loss']-b['loss']),
            valid_logits_max_abs=float((a['logits']-b['logits']).abs().max()),
            gradient_max_abs=max(float((a['grads'][k]-b['grads'][k]).abs().max()) for k in a['grads'])))
    report = dict(passed=all(max(r[k] for k in ('loss_max_abs','valid_logits_max_abs','gradient_max_abs')) <= 1e-3 for r in records),
        max_error=1e-3, threshold_source='Existing D65 native backend parity gate; not fitted after results',
        records=records, optimizer_updates=0, native_mps=True,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['scripts/probe_local_shape_parity.py','beatfcn/train/local_shapes.py','beatfcn/model/full_event_model.py']})
    output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))
    if not report['passed']:
        raise AssertionError('Real padding parity failed')


if __name__ == '__main__':
    run()

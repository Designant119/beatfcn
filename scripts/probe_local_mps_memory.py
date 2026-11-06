"""Replay the frozen real training requests to isolate MPS allocation failure."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_event_experiment import memory

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    output = ROOT/args.output
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT/args.plan).read_text())
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposure = json.loads((ROOT/plan['exposure']).read_text())
    saved = torch.load(ROOT/args.checkpoint, map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**plan['config'])).to('mps').train()
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
    report = dict(start_step=args.start_step, end_step=args.end_step, clear_cache=args.clear_cache,
        fixed_audio_shapes=args.fixed_audio_shapes,
        microbatch=args.microbatch,
        plan=args.plan, checkpoint=args.checkpoint, complete=False, trace=[], cloud_cost_usd=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Allocation correctness probe only; weights discarded and not quality exposure')
    started = time.monotonic()

    def record(phase, step, offset=None):
        if phase in ('backward', 'update'):
            torch.mps.synchronize()
        report['trace'].append(dict(phase=phase, step=step, microbatch=offset, **memory('mps')))

    try:
        for update in exposure['steps'][args.start_step-1:args.end_step]:
            step = update['effective_update']
            if time.monotonic()-started > 180:
                raise TimeoutError('Bounded memory probe exceeded 180 seconds')
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, 4, args.microbatch):
                record('before', step, offset)
                if args.clear_cache:
                    torch.mps.empty_cache()
                    record('after_empty_cache', step, offset)
                ids = update['indices'][offset:offset+args.microbatch]
                controls = []
                for index, fields in zip(ids, update['specified_condition_fields'][offset:offset+args.microbatch]):
                    control = np.asarray(panel.records[index]['controls'], np.float32).copy()
                    control[[i for i in range(len(control)) if i not in fields], :3] = 0
                    controls.append(control)
                inp, sequence = panel.batch(ids, controls=controls, device='cpu')
                if args.fixed_audio_shapes:
                    inp = pad_input_shapes(inp, dict(local=400, mel=1600, song=896, reference=1))
                inp = {k: v.to('mps') for k, v in inp.items()}
                sequence = {k: v.to('mps') for k, v in sequence.items()}
                torch.mps.synchronize()
                loss = event_loss(model(inp, sequence), sequence)*len(ids)/4
                record('forward', step, offset)
                loss.backward()
                report['last_loss'] = float(loss.detach())
                del inp, sequence, loss
                record('backward', step, offset)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record('update', step)
            report['successful_updates'] = step-args.start_step+1
            if step % 25 == 0:
                print(json.dumps(dict(step=step, **memory('mps'))), flush=True)
        report['complete'] = True
        if args.save_state:
            state_path = output.with_suffix('.pt')
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, state_path)
            report['state'] = str(state_path.relative_to(ROOT))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_s'] = time.monotonic()-started
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'trace'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='bench/rhythm-v2/d68-local-baseline-v1.json')
    parser.add_argument('--checkpoint', default='artifacts/d68-local-baseline/checkpoints/step-0002.pt')
    parser.add_argument('--start-step', type=int, default=51)
    parser.add_argument('--end-step', type=int, default=51)
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--fixed-audio-shapes', action='store_true')
    parser.add_argument('--microbatch', type=int, choices=[1, 2], default=2)
    parser.add_argument('--save-state', action='store_true')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())

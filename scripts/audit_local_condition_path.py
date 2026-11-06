"""Read-only cross-attention instrumentation on frozen real requests.

Attention mass diagnoses a path; it does not establish causal generation quality.
No model parameter, input mask or output is changed by the hooks.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import CONTROL_FIELDS, encode_controls, request_log_probability
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import LocalRequestPanel, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def sections(model, inp):
    lengths = [inp['local'].shape[2] * model.c.band_slots + inp['mel'].shape[1],
               inp['song'].shape[1], inp['reference'].shape[1] + 1, model.c.controls]
    ends = np.cumsum([0] + lengths)
    return {name: slice(int(a), int(b)) for name, a, b in
            zip(('local_with_mel', 'song', 'reference', 'controls'), ends[:-1], ends[1:])}


def run():
    plan_path = ROOT / 'bench/rhythm-v2/d69-condition-path-v1.json'
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for path, expected in plan['inputs_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Diagnostic input changed: ' + path)
    torch.set_num_threads(2)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    probe = json.loads((ROOT / plan['probe']).read_text())
    previous = json.loads((ROOT / plan['generation_audit']).read_text())
    cases = []
    for item in probe['cases']:
        row = panel.records[item['index']]
        control = np.asarray(row['controls'], np.float32).copy()
        field = CONTROL_FIELDS.index(item['field'])
        control[[i for i in range(len(control)) if i != field], :3] = 0
        inp, seq = panel.batch([item['index']], controls=[control])
        cases.append((dict(chart_id=row['chart_id'], split='train', field=item['field'],
                           prefix='human training target'), inp, seq,
                      torch.tensor(item['opposing_controls'])[None]))
    for item in previous['rows']:
        if item['field'] != 'density_notes_per_second':
            continue
        a, b = item['scope_us']
        request = EventRequest('phira', a, b, panel.features.records[item['audio_sha256']]['duration_us'])
        control = encode_controls('phira', {item['field']: item['low']}, (a, b), origin_us=a)
        opposing = encode_controls('phira', {item['field']: item['high']}, (a, b), origin_us=a)
        inp, _ = request_inputs(panel.features, item['audio_sha256'], request, control)
        events = [RhythmEvent(**event) for event in item['predictions'][0]['events']]
        seq = teacher_sequence(encode_events(events, request), request)
        inp, seq = collate([dict(inputs=inp, sequence=seq)])
        cases.append((dict(chart_id=item['chart_id'], split='val', field=item['field'],
                           prefix='D68 free-generated low-condition prefix; no human target'),
                      inp, seq, torch.from_numpy(opposing)[None]))
    report = dict(id=plan['id'], complete=False, plan_sha256=sha(plan_path),
                  optimizer_updates=0, new_cloud_compute_calls=0, rows=[],
                  scope='Diagnostic first 128 valid queries, same fixed prefixes; not generation or quality validation')
    started = time.monotonic()
    with torch.no_grad():
        for checkpoint in plan['checkpoints']:
            saved = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=False)
            model = FullEventModel(FullEventConfig(**saved['config'])).eval()
            model.load_state_dict(saved['model'])
            for metadata, raw, seq, opposing in cases:
                if time.monotonic() - started > plan['max_wall_s']:
                    raise TimeoutError('Read-only diagnostic deadline')
                inp = pad_input_shapes(raw, plan['input_shapes'])
                seq = {k: v[:, :plan['max_queries']] for k, v in seq.items()}
                mem = model.encode(inp)
                slices = sections(model, inp)
                norm = {}
                for name, part in slices.items():
                    values = mem['values'][:, part][mem['valid'][:, part]]
                    norms = values.square().mean(-1).sqrt()
                    norm[name] = dict(tokens=len(norms), rms_mean=float(norms.mean()),
                                      rms_min=float(norms.min()), rms_max=float(norms.max()))
                args = [seq[k] for k in ('tokens', 'fields', 'times', 'valid')]
                baseline = model.decode(*args, mem)
                layers = []

                def capture(module, args):
                    x, times, positions, (k, _), mask = args
                    q = module.q(x).view(x.shape[0], x.shape[1], module.c.heads, -1).transpose(1, 2)
                    q = module._position(q, times, positions)
                    k = k.repeat_interleave(module.c.heads // module.c.kv_heads, 1)
                    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                    scores = scores.masked_fill(~mask, float('-inf'))
                    probabilities = scores.softmax(-1)
                    masses = {name: float(probabilities[..., part].sum(-1).mean())
                              for name, part in slices.items()}
                    field_index = slices['controls'].start + CONTROL_FIELDS.index(metadata['field'])
                    layers.append(dict(source_mass=masses,
                        specified_field_mass=float(probabilities[..., field_index].mean()),
                        max_score=float(scores.max()),
                        minimum_source_max={name: float(scores[..., part].max(-1).values.mean())
                                            for name, part in slices.items()}))

                handles = [layer.cross_attn.register_forward_pre_hook(capture) for layer in model.decoder]
                try:
                    observed = model.decode(*args, mem)
                finally:
                    for handle in handles:
                        handle.remove()
                delta = float((observed - baseline).abs().max())
                if delta != 0.:
                    raise ValueError('Read-only instrumentation changed logits')
                altered = model.replace_controls(mem, opposing)
                negative = model.decode(*args, altered)
                difference = (baseline - negative).abs()
                mask = seq['allowed'] & seq['valid'][..., None]
                report['rows'].append(dict(**metadata, checkpoint=checkpoint, updates=saved['updates'],
                    checkpoint_sha256=sha(ROOT / checkpoint), valid_queries=int(seq['valid'].sum()),
                    memory_sources=norm, uniform_field_mass=1 / int(mem['valid'].sum()),
                    memory_replacement_difference=float((mem['values'] - altered['values']).abs().max()),
                    control_input_difference=float((inp['controls'] - opposing).abs().max()),
                    logit_abs_difference_max=float(difference[mask].max()),
                    logit_abs_difference_mean=float(difference[mask].mean()),
                    probability_gap=float((request_log_probability(baseline, seq) -
                                           request_log_probability(negative, seq))[0]),
                    instrumentation_difference=delta, layers=layers))
                write_json(output, report)
            del model, saved
    report.update(complete=True, wall_s=time.monotonic()-started)
    write_json(output, report)
    print(json.dumps(dict(complete=True, rows=len(report['rows']), wall_s=report['wall_s'])))


if __name__ == '__main__':
    run()

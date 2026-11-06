"""Read-only type supervision and teacher-prefix diagnosis of D68 collapse."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.model.full_event_codec import FIELDS, EventRequest, EventVocabulary, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import LocalRequestPanel, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run():
    plan_path = ROOT/'bench/rhythm-v2/d68-local-baseline-v3.json'
    plan = json.loads(plan_path.read_text())
    state = json.loads((ROOT/plan['output']).read_text())
    output = ROOT/'reports/d68_field_diagnosis_v1.json'
    if output.exists():
        raise FileExistsError(output)
    if state['updates'] != 1000 or sha(ROOT/state['checkpoint']) != state['checkpoint_sha256']:
        raise ValueError('Unexpected checkpoint')
    for p, h in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        if sha(ROOT/p) != h:
            raise ValueError('Frozen source changed: '+p)
    torch.set_num_threads(2)
    start = time.monotonic()
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposure = json.loads((ROOT/plan['exposure']).read_text())
    repeated = Counter(i for step in exposure['steps'] for i in step['indices'])
    raw, weighted = Counter(), Counter()
    field_tokens = Counter()
    vocab = EventVocabulary()
    for index, count in repeated.items():
        row = panel.records[index]
        if row['mode'] != 'phira':
            continue
        values, targets = panel._arrays(row['chart_index'])
        tokens = values[row['token_start']:row['token_stop']]
        for f, name in enumerate(FIELDS):
            lower = vocab.token(name, 0)
            field_tokens[name] += int(((tokens >= lower) & (tokens < lower+vocab.sizes[f])).sum())*count
        kinds = targets['kind'][row['target_start']:row['target_stop']]
        for k, n in Counter(kinds.tolist()).items():
            raw[TYPES[k]] += n*count
            weighted[TYPES[k]] += n*count/row['tokens']
    saved = torch.load(ROOT/state['checkpoint'], map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    by_field = defaultdict(lambda:dict(count=0, correct=0, nll=0.))
    confusion, probabilities, truth_count, rows = Counter(), defaultdict(lambda:np.zeros(4)), Counter(), []
    kind_id = FIELDS.index('kind')
    kind_tokens = [vocab.token('kind', i) for i in range(4)]
    for index in plan['validation_indices']:
        row = panel.records[index]
        if row['split'] != 'val' or np.asarray(row['controls'])[:,2].any():
            raise ValueError('Hidden validation condition or training request')
        inp, seq = panel.batch([index])
        inp = pad_input_shapes(inp, plan['input_shapes'])
        logits = model(inp, seq).masked_fill(~seq['allowed'], -1e9)
        pred = logits.argmax(-1)
        losses = F.cross_entropy(logits.flatten(0,1), seq['targets'].flatten(), reduction='none').view_as(pred)
        for i, name in enumerate(FIELDS):
            mask = seq['valid'] & (seq['fields'] == i)
            key = row['mode']+'/'+name
            by_field[key]['count'] += int(mask.sum())
            by_field[key]['correct'] += int(((pred == seq['targets']) & mask).sum())
            by_field[key]['nll'] += float(losses[mask].sum())
        mask = seq['valid'] & (seq['fields'] == kind_id)
        local = Counter()
        if mask.any():
            correct = seq['targets'][mask]-kind_tokens[0]
            guesses = pred[mask]-kind_tokens[0]
            probs = logits[mask][:,kind_tokens].softmax(-1).numpy()
            for a, b, prob in zip(correct.tolist(), guesses.tolist(), probs):
                key = TYPES[a]+'->'+TYPES[b]
                confusion[key] += 1; local[key] += 1
                probabilities[TYPES[a]] += prob; truth_count[TYPES[a]] += 1
        rows.append(dict(index=index, chart_id=row['chart_id'], scope_us=[row['request']['start_us'],row['request']['end_us']], kind_confusion=dict(local)))
    # Same actual generated prefix: full teacher pass versus incremental cache.
    generated_path = ROOT/'artifacts/d68-cpu-generation-1000-v1/phira-28312.json'
    generated = json.loads(generated_path.read_text())
    core = next(c for c in generated['cores'] if c['start_us'] == 80_000_000)
    request = EventRequest('phira', core['start_us'], core['end_us'], generated['duration_us'])
    controls = encode_controls('phira', {}, (request.start_us,request.end_us), origin_us=request.start_us)
    inputs, _ = request_inputs(panel.features, generated['audio_sha256'], request, controls)
    inp, seq = collate([dict(inputs=inputs, sequence=teacher_sequence(core['tokens'],request))])
    inp = pad_input_shapes(inp, plan['input_shapes'])
    memory = model.encode(inp)
    keys = ('tokens','fields','times','valid')
    full = model.decode(*(seq[k] for k in keys),memory)
    cache = model.start_cache(memory)
    max_error, mismatch = 0., 0
    for i in range(seq['tokens'].shape[1]):
        incremental = model.decode(*(seq[k][:,i:i+1] for k in keys),memory,cache=cache)
        max_error = max(max_error,float((full[:,i:i+1]-incremental).abs().max()))
        allowed = seq['allowed'][:,i:i+1]
        mismatch += int((full[:,i:i+1].masked_fill(~allowed,-1e9).argmax(-1) != incremental.masked_fill(~allowed,-1e9).argmax(-1)).sum())
    report = dict(complete=True, optimizer_updates=0, new_cloud_compute_calls=0,
        checkpoint=state['checkpoint'], checkpoint_sha256=state['checkpoint_sha256'],
        script_sha256=sha(Path(__file__)), training_plan_sha256=sha(plan_path),
        training_phira_note_exposure=dict(raw), training_kind_loss_weight=dict(weighted),
        training_phira_field_tokens=dict(field_tokens),
        validation_fields={k:dict(**v, accuracy=v['correct']/v['count'] if v['count'] else None,
            mean_nll=v['nll']/v['count'] if v['count'] else None) for k,v in by_field.items()},
        kind_confusion=dict(confusion),kind_probability_order=list(TYPES),
        kind_mean_probability_given_truth={k:(v/truth_count[k]).tolist() for k,v in probabilities.items()}, rows=rows,
        generated_prefix_cache_parity=dict(chart_id='phira:28312',scope_us=[80_000_000,88_000_000],
            tokens=seq['tokens'].shape[1],maximum_logit_difference=max_error,legal_argmax_mismatches=mismatch,
            tolerance=1e-3,passed=max_error<1e-3 and mismatch==0,raw_sha256=sha(generated_path)),
        wall_s=time.monotonic()-start,scope='Actual exposure, 29 heldout teacher-prefix requests, and one actual generated-prefix cache equivalence. Not a quality experiment.')
    write_json(output,report)
    print(json.dumps({k:report[k] for k in ['training_phira_note_exposure','kind_confusion','generated_prefix_cache_parity','wall_s']}))


if __name__ == '__main__':
    run()

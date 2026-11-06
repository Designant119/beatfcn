"""Check frozen inputs, saved outputs, and one restored R25 checkpoint."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.action_state import ManiaAction, PhiraAction, replay
from scripts.acoustic_probe_metrics import event_metrics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def key(event):
    return event['onset_us'], event['end_us'], event['type'], event['lane']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    audit_plan = json.loads(args.plan.read_text())
    for name, expected in audit_plan['input_sha256'].items():
        assert digest(name) == expected, name
    plan = json.loads((ROOT / audit_plan['execution_plan']).read_text())
    result = json.loads((ROOT / plan['result']).read_text())
    assert result['plan_sha256'] == digest(audit_plan['execution_plan'])
    for name, expected in plan['input_sha256'].items():
        assert digest(name) == expected, name
    assert result['complete'] and result['optimizer_updates'] == plan['updates'] == 1000
    manifest = ROOT / plan['manifest']
    panel = json.loads(manifest.read_text())
    rows = {r['id']: r for r in panel['rows']}
    expected_ids = set(plan['train_ids'] + plan['heldout_ids'])
    assert not ({rows[k]['audio_sha256'] for k in plan['train_ids']}
                & {rows[k]['audio_sha256'] for k in plan['heldout_ids']})
    expected_exposure = Counter()
    for step in range(plan['updates']):
        expected_exposure.update(plan['batch_cycle'][step % len(plan['batch_cycle'])])
    assert dict(expected_exposure) == result['exposure']
    recomputed, sequences, assets = 0, 0, 0
    all_records = {}
    for step, checkpoint in result['checkpoints'].items():
        for name, expected in checkpoint['assets_sha256'].items():
            assert digest(name) == expected, name
            assets += 1
        path = next(n for n in checkpoint['assets_sha256'] if Path(n).name.startswith('predictions-'))
        records = json.loads((ROOT / path).read_text())
        assert {r['core_id'] for r in records} == expected_ids
        for r in records:
            row = rows[r['core_id']]
            assert Counter(map(key, r['truth'])) == Counter(map(key, row['target_events']))
            assert r['partition'] == ('train' if r['core_id'] in plan['train_ids'] else 'heldout')
            generation = r['generation']
            assert r['metrics'] == event_metrics(generation['events'], row['target_events'], row['mode'])
            recomputed += 1
            actions = [ManiaAction(**a) if row['mode'] == 'mania_4k' else
                       PhiraAction(a['time_us'], tuple(a['starts']), tuple(tuple(p) for p in a['releases']))
                       for a in generation['actions']]
            decoded = replay(actions, row['mode'])
            assert Counter(map(key, generation['events'])) == Counter(key(asdict(e)) for e in decoded.events)
            assert decoded.state.active_holds == generation['active_holds'] == 0
            sequences += 1
        for partition in ['train', 'heldout']:
            selected = [r for r in records if r['partition'] == partition]
            for mode in ['all', 'mania_4k', 'phira']:
                subset = selected if mode == 'all' else [r for r in selected if r['mode'] == mode]
                average = sum(r['metrics']['full_note_f1'] for r in subset) / len(subset)
                assert abs(average - checkpoint['summary'][partition][mode]['full_note_f1']) < 1e-12
        all_records[step] = records
    assert set(result['checkpoints']) == {'100', '300', '1000'}

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('r25_frozen_verified', ROOT / plan['model_source'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.StateModel()
    checkpoint = torch.load(ROOT / plan['artifacts'] / 'step-1000.pt', map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 1000 and checkpoint['diagnostic_only']
    assert checkpoint['plan_sha256'] == result['plan_sha256']
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    weight_digest = sha256()
    for name, value in model.state_dict().items():
        weight_digest.update(name.encode())
        weight_digest.update(value.numpy().tobytes())
    assert weight_digest.hexdigest() == result['checkpoints']['1000']['weight_sha256']
    saved_logits = torch.load(ROOT / plan['artifacts'] / 'teacher-logits-1000.pt', weights_only=True)
    with np.load(manifest.parent / 'inputs.npz', allow_pickle=False) as z:
        inputs = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    indices = {r['id']: i for i, r in enumerate(panel['rows'])}
    largest_error = 0.
    with torch.no_grad():
        for record in all_records['1000']:
            core_id = record['core_id']
            source = {k: v[indices[core_id]:indices[core_id] + 1] for k, v in inputs.items()}
            prefix = module.prefix_batch([rows[core_id]])
            out = model.teacher(model.encode_audio(source), prefix, False)
            for name, value in out.items():
                error = float((value - saved_logits[core_id][name]).abs().max())
                largest_error = max(largest_error, error)
                assert error <= 1e-6, (core_id, name, error)
            loss, fields = module.loss_and_metrics(out, prefix, source['mode'])
            assert abs(float(loss) - record['teacher']['loss']) <= 1e-6
            assert fields == record['teacher']['fields']
            generated = module.generate(model, source, False, maximum_actions=plan['maximum_actions'])
            # JSON encodes tuple-valued action/state fields as lists.
            # Compare the same serialized contract, without changing numbers.
            assert json.loads(json.dumps(generated)) == record['generation'], core_id
    evidence = dict(id='R25-verification', verified=True, verified_at_unix=time.time(),
        frozen_execution_inputs=len(plan['input_sha256']), checkpoint_assets=assets,
        saved_event_metrics_recomputed=recomputed, action_streams_replayed=sequences,
        restored_checkpoint_teacher_sequences=11, restored_checkpoint_generated_sequences=11,
        maximum_saved_logit_difference=largest_error, song_split_disjoint=True,
        training_exposure_verified=dict(expected_exposure), optimizer_updates=0, new_cloud_spend_usd=0,
        model_promotion=False, source_sha256={plan['result']: digest(plan['result']),
            audit_plan['execution_plan']: digest(audit_plan['execution_plan'])},
        limitation='Saved predictions and checkpoint reproducibility only; five development holdout songs do not establish population, conditions or whole-song utility.')
    (ROOT / audit_plan['result']).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()

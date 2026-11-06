"""Frozen-checkpoint local/global clock perturbations, without optimization."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.acoustic_probe_metrics import event_metrics
from scripts.probe_training_horizon import load_model_module, distribution_metrics, statistics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def shifted_inputs(source, frames, move_global):
    out = {k: v.clone() for k, v in source.items()}
    out['audio'][:, :frames] = 0
    out['audio'][:, frames:] = source['audio'][:, :-frames]
    out['audio_valid'][:, :frames] = False
    out['audio_valid'][:, frames:] = source['audio_valid'][:, :-frames] & source['audio_valid'][:, frames:]
    if move_global:
        out['song_times_us'] += frames * 10_000
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['input_sha256'].items():
        assert digest(path) == expected, path
    assert not (ROOT / plan['result']).exists()
    torch.set_num_threads(1)
    module = load_model_module(plan['model_source'])
    model = module.StateModel()
    ck = torch.load(ROOT / plan['checkpoint'], weights_only=True, map_location='cpu')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    weight_before = sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    panel_path = ROOT / plan['manifest']
    panel = json.loads(panel_path.read_text())
    rows = {r['id']: r for r in panel['rows']}
    indices = {r['id']: i for i, r in enumerate(panel['rows'])}
    with np.load(panel_path.parent / 'inputs.npz', allow_pickle=False) as z:
        inputs = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    saved = {r['core_id']: r for r in json.loads((ROOT / plan['factual_predictions']).read_text())}
    delta = plan['shift_frames'] * 10_000
    lo, hi = 8_000_000 + delta, 16_000_000
    result = dict(id='D53', complete=False, optimizer_updates=0, new_cloud_spend_usd=0,
        model_promotion=False, plan_sha256=digest(str(args.plan)), shift_us=delta,
        scored_head_interval_us=[lo, hi], cases=[])
    started = time.time()
    with torch.no_grad():
        for core_id in plan['core_ids']:
            row = rows[core_id]
            i = indices[core_id]
            source = {k: v[i:i+1] for k, v in inputs.items()}
            assert not source['condition_known'].any()
            versions = {'factual': source,
                        'local_shift_only': shifted_inputs(source, plan['shift_frames'], False),
                        'local_and_global_clock_shift': shifted_inputs(source, plan['shift_frames'], True)}
            original = [e for e in row['target_events'] if lo <= e['onset_us'] < hi]
            translated = [dict(e, onset_us=e['onset_us'] + delta, end_us=e['end_us'] + delta)
                          for e in row['target_events'] if lo <= e['onset_us'] + delta < hi]
            for variant, data in versions.items():
                generation = module.generate(model, data, False, maximum_actions=256)
                if variant == 'factual':
                    assert json.loads(json.dumps(generation)) == saved[core_id]['generation']
                pred = [e for e in generation['events'] if lo <= e['onset_us'] < hi]
                result['cases'].append(dict(core_id=core_id, mode=row['mode'], variant=variant,
                    audio=row['audio'], source_clip_start_us=row['source_clip_start_us'], generation=generation,
                    actual_input_statistics=statistics(data),
                    original_target=original, translated_target=translated,
                    scores_to_original=dict(full=event_metrics(pred, original, row['mode']),
                        distribution=distribution_metrics(pred, original)),
                    scores_to_translated=dict(full=event_metrics(pred, translated, row['mode']),
                        distribution=distribution_metrics(pred, translated))))
    assert weight_before == sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    consistent = [r for r in result['cases'] if r['variant'] == 'local_and_global_clock_shift']
    result.update(complete=True, model_weights_unchanged=True, generated_sequences=len(result['cases']),
        translation_screen_passed=all(r['scores_to_translated']['full']['full_note_f1'] >= .8
            and r['scores_to_translated']['distribution']['timing_point_f1'] >= .8
            and r['generation']['ended'] and r['generation']['failure'] is None for r in consistent),
        wall_s=time.time()-started)
    (ROOT / plan['result']).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','generated_sequences','translation_screen_passed','wall_s']}))


if __name__ == '__main__':
    main()

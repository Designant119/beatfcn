"""Generate reference / shuffled / zero intensity arms from one checkpoint.

No training.  Used for control-response evaluation: the same checkpoint and the
same songs, only the intensity condition changes.
"""
import argparse
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import generate_complete_songs
from beatfcn.train.intensity_curve import audio_intensity_curve

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', default='reference,shuffle,zero')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    device = args.device or ('cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    if plan.get('mert_features'):
        from beatfcn.train.mert_panel import MertRequestPanel
        panel = MertRequestPanel(ROOT, plan['tape'], plan['mert_features'])
    else:
        from beatfcn.train.local_request_panel import LocalRequestPanel
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    records = json.loads((ROOT / plan['evaluation20']).read_text())['records'][:args.limit]
    songs = [dict(chart_id=row['chart_id'], audio_sha256=row['audio_sha256'],
                  mode=row['mode'],
                  duration_us=panel.features.records[row['audio_sha256']]['duration_us'])
             for row in records]
    model = FullEventModel(FullEventConfig(**plan['config']))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    model.to(device).eval()
    settings = plan.get('reference_intensity') or plan.get('intensity') or {}
    cell_us = int(settings.get('cell_us', 500_000))
    scale = float(settings.get('scale', 16.0))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in args.arms.split(','):
        arm = arm.strip()
        if arm not in ('reference', 'shuffle', 'zero'):
            raise ValueError('Unknown arm: ' + arm)
        arm_dir = args.out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        def progress(song):
            path = arm_dir / (song['chart_id'].replace(':', '-') + '.json')
            path.write_text(json.dumps(song, ensure_ascii=False, indent=1) + '\n')
        provider = lambda audio, start, end: audio_intensity_curve(
            panel.features, audio, start, end, cell_us=cell_us, scale=scale, mode=arm)
        generated = generate_complete_songs(model, panel.features, songs, device=device,
            batch_size=plan['generation_batch_size'], token_cap=plan['generation_token_cap'],
            max_wall_s=plan['max_generation_wall_s'], history_reference=False,
            on_progress=progress, input_shapes=plan.get('input_shapes'),
            intensity_provider=provider)
        summary[arm] = dict(complete=generated['complete'], wall_s=generated['wall_s'],
            songs=[dict(chart_id=s['chart_id'], complete=s['complete'],
                        failure=s['failure'], events=len(s['events']),
                        cores=len(s['cores'])) for s in generated['songs']])
        (args.out / f'{arm}-summary.json').write_text(
            json.dumps(summary[arm], ensure_ascii=False, indent=1) + '\n')
        print(json.dumps(summary[arm], ensure_ascii=False), flush=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()

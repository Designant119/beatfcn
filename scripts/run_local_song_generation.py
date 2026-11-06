"""Standalone CPU whole-song inference from a frozen trained checkpoint."""
import argparse
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import generate_complete_songs
from beatfcn.train.local_features import LocalFeatureStore
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run(path):
    plan = json.loads(path.read_text())
    target = ROOT/plan['output']
    if target.exists():
        raise FileExistsError('Generation attempt already exists; inspect its terminal state')
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen CPU generation input changed: '+source)
    torch.set_num_threads(plan['cpu_threads'])
    saved = torch.load(ROOT/plan['checkpoint'], map_location='cpu', weights_only=False)
    if saved['updates'] != 1000:
        raise ValueError('This review requires the actual 1000-update checkpoint')
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    features = LocalFeatureStore(ROOT, plan['features'])
    evaluation = json.loads((ROOT/plan['evaluation20']).read_text())['records']
    if len(evaluation) != 20 or len({r['component'] for r in evaluation}) != 20:
        raise ValueError('Need twenty frozen independent evaluation songs')
    songs = [dict(chart_id=r['chart_id'], audio_sha256=r['audio_sha256'], mode=r['mode'],
        duration_us=features.records[r['audio_sha256']]['duration_us']) for r in evaluation]
    identities = {p:h for r in songs for p,h in features.records[r['audio_sha256']]['sha256'].items()}
    for source, expected in identities.items():
        if sha(ROOT/source) != expected:
            raise ValueError('Admitted generation cache changed: '+source)
    report = dict(id=plan['id'], backend='cpu', complete=False, execution_complete=False,
        checkpoint=plan['checkpoint'], checkpoint_sha256=sha(ROOT/plan['checkpoint']),
        plan_sha256=sha(path), expected_songs=20, optimizer_updates=1000, inference_optimizer_updates=0,
        evaluation20=plan['evaluation20'], song_files={}, song_progress={}, completed_songs=0,
        human_review_complete=False, new_cloud_compute_calls=0, started_at_unix=time.time())
    write_json(target, report)

    def progress(song):
        output = ROOT/plan['generation_dir']/(song['chart_id'].replace(':','-')+'.json')
        write_json(output, song)
        report['song_files'][song['chart_id']] = str(output.relative_to(ROOT))
        report['song_progress'][song['chart_id']] = dict(complete=song['complete'], failure=song['failure'],
            generated_notes=len(song['events']), completed_until_us=song['next_start_us'], duration_us=song['duration_us'])
        completed = sum(s['complete'] for s in report['song_progress'].values())
        if completed > report['completed_songs']:
            print(json.dumps(dict(completed=completed, chart_id=song['chart_id'])), flush=True)
        report.update(completed_songs=completed, last_progress_at_unix=time.time())
        write_json(target, report)

    try:
        result = generate_complete_songs(model, features, songs, device='cpu', batch_size=plan['batch_size'],
            token_cap=plan['token_cap'], max_wall_s=plan['max_wall_s'], history_reference=False,
            input_shapes=plan['input_shapes'], on_progress=progress)
        report.update(complete=result['complete'], execution_complete=True, wall_s=result['wall_s'],
            failures={s['chart_id']:s['failure'] for s in result['songs'] if s['failure']})
        write_json(target, report)
        print(json.dumps(dict(complete=report['complete'], completed=report['completed_songs'], wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        write_json(target, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

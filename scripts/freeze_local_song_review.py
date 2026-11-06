"""Freeze a completed generation attempt and all twenty raw outputs for CPU review."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            h.update(block)
    return h.hexdigest()


def run(training_plan, output, artifact_dir, report, generation_plan=None):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    training = json.loads(training_plan.read_text())
    state = json.loads((ROOT/training['output']).read_text())
    inference = json.loads(generation_plan.read_text()) if generation_plan else None
    generation_path = inference['output'] if inference else training['generation_output']
    generation = json.loads((ROOT/generation_path).read_text())
    terminal = generation.get('execution_complete') if inference else state['complete']
    if (not terminal or generation.get('error') or state.get('error') or state['updates'] != 1000
            or state['plan_sha256'] != sha(training_plan)
            or generation['checkpoint_sha256'] != state['checkpoint_sha256']
            or generation['optimizer_updates'] != 1000 or len(generation['song_files']) != 20):
        raise ValueError('Need terminal 1000-update baseline and all twenty attempted raw outputs')
    if inference and (generation['plan_sha256'] != sha(generation_plan)
            or inference['evaluation20'] != training['evaluation20'] or inference['features'] != training['features']):
        raise ValueError('Standalone generation has another identity or evaluation panel')
    fields = dict(generation=generation_path, evaluation20=training['evaluation20'],
        tape=training['tape'], features=training['features'], clock='reports/d66_audio_clock_v1.json',
        manifest='inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl')
    paths = [*fields.values(), training['output'], str(training_plan.relative_to(ROOT)),
             *generation['song_files'].values(), 'scripts/review_local_song_rollouts.py',
             'scripts/build_phira_preview.py', 'scripts/freeze_local_song_review.py',
             'reports/d66_native_audit_v2.json', 'reports/d66_source_gates_v1.json',
             *training['code_sha256']]
    if inference:
        paths += [str(generation_plan.relative_to(ROOT)), *inference['inputs_sha256']]
    plan = dict(id=training['id']+'-review20', **fields, expected_optimizer_updates=1000,
        output_dir=artifact_dir, report=report, inputs_sha256={p: sha(ROOT/p) for p in paths})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(plan=str(output.relative_to(ROOT)), expected_songs=20,
                         complete_generation=generation['complete'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--generation-plan', type=Path)
    args = parser.parse_args()
    run(args.training_plan.resolve(), args.output, args.artifact_dir, args.report,
        args.generation_plan.resolve() if args.generation_plan else None)

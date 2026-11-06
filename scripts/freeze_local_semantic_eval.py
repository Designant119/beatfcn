"""Bind unchanged historical condition/audio protocols to D97's actual result."""
import json
from pathlib import Path

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    training_path = 'bench/rhythm-v2/d97-semantic-quality-v1.json'
    training = json.loads((ROOT/training_path).read_text())
    result = json.loads((ROOT/training['output']).read_text())
    if (not result.get('complete') or result.get('error') or result['updates'] != 1000
            or result['plan_sha256'] != sha(ROOT/training_path)):
        raise ValueError('Require real terminal1000 and original whole-song generation')
    for path, expected in {**training['inputs_sha256'], **training['code_sha256']}.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen training source changed: ' + path)
    checkpoint = result['checkpoint']
    if sha(ROOT/checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Actual checkpoint identity changed')
    prepared_path = 'reports/d87_eval_preparation_v1.json'
    prepared = json.loads((ROOT/prepared_path).read_text())
    for path, expected in prepared['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Original evaluation source changed: ' + path)
    outputs = {kind: ROOT/f'bench/rhythm-v2/d97-semantic-{kind}-response-v1.json'
               for kind in ('condition', 'audio')}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError('Evaluation plans are immutable')
    for kind, destination in outputs.items():
        protocol = prepared['protocols'][kind]
        paths = [prepared_path, training_path, training['output'], checkpoint, protocol,
                 *prepared['source_sha256'], 'scripts/freeze_local_semantic_eval.py']
        plan = dict(id=f'D97-semantic-{kind}-1000-v1', training_plan=training_path,
            protocol=protocol, checkpoint=checkpoint,
            output=f'reports/d97_sequence_{kind}_response_v1.json',
            inputs_sha256={p: sha(ROOT/p) for p in paths})
        write_json(destination, plan)
    print(json.dumps(dict(checkpoint_sha256=result['checkpoint_sha256'],
        protocols=prepared['protocols'], new_inference_calls=0)))


if __name__ == '__main__':
    run()

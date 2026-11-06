"""Bind preregistered D83 protocols to the completed candidate, never launch work."""
import json
from pathlib import Path

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    preparation_path = ROOT/'reports/d83_eval_preparation_v2.json'
    preparation = json.loads(preparation_path.read_text())
    for source, expected in preparation['source_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Preregistered evaluation source changed: '+source)
    training_path = 'bench/rhythm-v2/d83-independent-song-candidate-v1.json'
    training = json.loads((ROOT/training_path).read_text())
    result = json.loads((ROOT/training['output']).read_text())
    if not result.get('complete') or result.get('error') or result['updates'] != 1000:
        raise ValueError('Wait for successful terminal training/generation before binding')
    if result['plan_sha256'] != sha(ROOT/training_path) or sha(ROOT/result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Candidate identity changed')
    plans = {}
    for kind in ('condition', 'audio'):
        protocol = f'bench/rhythm-v2/d83-{kind}-protocol-v1.json'
        script = f'scripts/audit_local_candidate_{"conditions" if kind == "condition" else "audio"}.py'
        paths = {*training['code_sha256'], training_path, training['output'], result['checkpoint'],
            training['features'], training['tape'], training['normalizer'], training['normalizer_module'],
            training['key_module'], training['model_module'], protocol, script,
            'scripts/audit_local_candidate_conditions.py', 'scripts/freeze_local_candidate_audits.py',
            str(preparation_path.relative_to(ROOT))}
        destination = ROOT/f'bench/rhythm-v2/d83-{kind}-response-v1.json'
        if destination.exists():
            raise FileExistsError(destination)
        plans[destination] = dict(id=f'D83-{kind}-response-v1', protocol=protocol,
            checkpoint=result['checkpoint'], output=f'reports/d83_{kind}_response_v1.json',
            inputs_sha256={p:sha(ROOT/p) for p in sorted(paths)})
    for destination, plan in plans.items():
        write_json(destination, plan)
    print(json.dumps(dict(plans=[str(p.relative_to(ROOT)) for p in plans], optimizer_updates=0, launched=False)))


if __name__ == '__main__':
    main()

"""Bind D95 adapters to the same pre-existing condition and audio requests."""
import argparse
import json
from pathlib import Path

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
PREP = 'reports/d95_eval_preparation_v1.json'
TRAINING = 'bench/rhythm-v2/d95-capacity-quality-v1.json'


def prepare():
    if (ROOT/PREP).exists():
        raise FileExistsError(ROOT/PREP)
    protocols = {kind:f'bench/rhythm-v2/d87-{kind}-protocol-v1.json' for kind in ['condition', 'audio']}
    paths = [*protocols.values(), 'scripts/audit_local_sequence_conditions.py',
        'scripts/audit_local_sequence_audio.py', 'scripts/freeze_local_capacity_eval.py',
        'scripts/audit_local_candidate_conditions.py', 'tests/test_local_sequence_conditions.py',
        'beatfcn/model/full_event_generate.py', 'beatfcn/model/full_event_codec.py']
    report = dict(id='D95-eval-preparation-v1', complete=True, optimizer_updates=0,
        new_cloud_compute_calls=0, protocols=protocols,
        source_sha256={p:sha(ROOT/p) for p in paths},
        scope='Unchanged D87 condition and whole-audio counterfactual requests; same absolute event generator/model; only width/FFN capacity and its independently seeded initialization.')
    write_json(ROOT/PREP, report)
    print(json.dumps(dict(prepared=True, protocols=protocols)))


def bind():
    prep = json.loads((ROOT/PREP).read_text())
    for source, expected in prep['source_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Prepared evaluation source changed: '+source)
    training = json.loads((ROOT/TRAINING).read_text())
    result = json.loads((ROOT/training['output']).read_text())
    if (not result.get('complete') or result.get('error') or result['updates'] != 1000
            or result['plan_sha256'] != sha(ROOT/TRAINING)
            or result.get('objective') != 'request_mean'):
        raise ValueError('Need completed real1000 and whole-song attempt')
    checkpoint = result['checkpoint']
    if sha(ROOT/checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    for kind, protocol in prep['protocols'].items():
        output = ROOT/f'bench/rhythm-v2/d95-capacity-{kind}-response-v1.json'
        if output.exists():
            raise FileExistsError(output)
        sources = [PREP, TRAINING, training['output'], checkpoint, protocol, *prep['source_sha256']]
        plan = dict(id=f'D95-capacity-{kind}-1000-v1', training_plan=TRAINING, protocol=protocol,
            checkpoint=checkpoint, output=f'reports/d95_sequence_{kind}_response_v1.json',
            inputs_sha256={p:sha(ROOT/p) for p in sources})
        write_json(output, plan)
    print(json.dumps(dict(bound=True, checkpoint_sha256=result['checkpoint_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', action='store_true')
    args = parser.parse_args()
    bind() if args.bind else prepare()

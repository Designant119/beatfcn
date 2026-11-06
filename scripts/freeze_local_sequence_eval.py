"""Prepare immutable D87 protocols now, bind actual1000 checkpoints only later."""
import argparse
import json
from pathlib import Path

from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]
PREP='reports/d87_eval_preparation_v1.json'


def prepare():
    if (ROOT/PREP).exists():
        raise FileExistsError(ROOT/PREP)
    original=json.loads((ROOT/'bench/rhythm-v2/d83-condition-protocol-v1.json').read_text())
    condition=dict(id='D87-condition-protocol-v1',songs=original['songs'],fields=original['fields'],
        cpu_threads=2,token_cap=4096,max_wall_s=600,
        gate='Each field:3/4 sufficiently directed and6/8 in band, legal complete. Stop at first failing field; unmeasured cannot pass.',
        reference='Unknown; no style-continuation claim.')
    audio=dict(id='D87-audio-protocol-v1',songs=original['songs'],donor_positions=[1,2,3,0],
        cpu_threads=2,token_cap=4096,max_wall_s=180,
        limits='Move all audio values/clocks/masks; changed output alone is not music understanding.')
    paths={}
    for kind,value in [('condition',condition),('audio',audio)]:
        path=f'bench/rhythm-v2/d87-{kind}-protocol-v1.json'
        if (ROOT/path).exists():raise FileExistsError(path)
        write_json(ROOT/path,value);paths[kind]=path
    sources=['scripts/audit_local_sequence_conditions.py','scripts/audit_local_sequence_audio.py',
        'scripts/freeze_local_sequence_eval.py','scripts/audit_local_candidate_conditions.py',
        'tests/test_local_sequence_conditions.py',*paths.values()]
    write_json(ROOT/PREP,dict(id='D87-eval-preparation-v1',complete=True,optimizer_updates=0,
        protocols=paths,source_sha256={p:sha(ROOT/p)for p in sources},new_cloud_compute_calls=0))
    print(json.dumps(dict(prepared=paths)))


def bind(arm):
    prepared=json.loads((ROOT/PREP).read_text())
    for source,expected in prepared['source_sha256'].items():
        if sha(ROOT/source)!=expected:raise ValueError('Prepared evaluation code changed: '+source)
    training_path=f'bench/rhythm-v2/d87-sequence-{arm}-v1.json'
    training=json.loads((ROOT/training_path).read_text())
    result=json.loads((ROOT/training['output']).read_text())
    if not result.get('complete') or result.get('error') or result['updates']!=1000 or result['plan_sha256']!=sha(ROOT/training_path):
        raise ValueError('Wait for the real1000 training and full-song attempt to finish')
    checkpoint=result['checkpoint']
    if sha(ROOT/checkpoint)!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    for kind,protocol in prepared['protocols'].items():
        output=f'bench/rhythm-v2/d87-{arm}-{kind}-response-v1.json'
        if (ROOT/output).exists():raise FileExistsError(output)
        sources=[PREP,training_path,training['output'],checkpoint,protocol,*prepared['source_sha256']]
        plan=dict(id=f'D87-{arm}-{kind}-1000-v1',training_plan=training_path,protocol=protocol,
            checkpoint=checkpoint,output=f'reports/d87_{arm}_{kind}_response_v1.json',
            inputs_sha256={p:sha(ROOT/p)for p in sources})
        write_json(ROOT/output,plan)
    print(json.dumps(dict(bound=arm,checkpoint_sha256=result['checkpoint_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bind',choices=('control','candidate'))
    args=parser.parse_args()
    if args.bind:bind(args.bind)
    else:prepare()

"""Locate ambiguous true-prefix decisions without adding exposure or labels."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def first_fork(a, b):
    size = min(len(a['targets']), len(b['targets']))
    differing = np.flatnonzero(a['targets'][:size] != b['targets'][:size])
    if not len(differing):
        return None
    index = int(differing[0])
    for key in ('tokens', 'fields', 'times', 'valid', 'allowed'):
        if not np.array_equal(a[key][:index+1], b[key][:index+1]):
            raise ValueError('Non-condition signal distinguishes the proposed common prefix')
    return index


def run():
    output = ROOT / 'inputs/d71_condition_forks/exposure-v1.json'
    if output.exists():
        raise FileExistsError(output)
    baseline = ROOT / 'bench/rhythm-v2/d68-local-baseline-v3.json'
    plan = json.loads(baseline.read_text())
    for p, h in plan['inputs_sha256'].items():
        if sha(ROOT/p) != h:
            raise ValueError('Original exposure source changed: '+p)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposure = json.loads((ROOT/plan['exposure']).read_text())
    groups = defaultdict(list)
    for i, row in enumerate(panel.records):
        if row['split'] == 'train':
            # request_inputs is a pure function of these metadata and controls.
            # Exact request equality also preserves carries and decoder grammar.
            identity = (row['mode'], row['audio_sha256'], json.dumps(row['request'], sort_keys=True))
            groups[identity].append(i)
    choices = defaultdict(list)
    for indices in groups.values():
        for a in indices:
            for b in indices:
                ra, rb = panel.records[a], panel.records[b]
                if a == b or ra['chart_id'] == rb['chart_id']:
                    continue
                fields = [field for field in CONTROL_FIELDS if field in (ra['bands'] or {})
                          and field in (rb['bands'] or {}) and ra['bands'][field] != rb['bands'][field]]
                if not fields:
                    continue
                fork = first_fork(panel.sequence(a), panel.sequence(b))
                if fork is None:
                    continue
                for field in fields:
                    choices[(a, field)].append(dict(index=a, partner=b, fork=fork, field=field,
                        chart_id=ra['chart_id'], partner_chart_id=rb['chart_id'], component=ra['component'],
                        target=int(panel.sequence(a)['targets'][fork]),
                        partner_target=int(panel.sequence(b)['targets'][fork]), mode=ra['mode']))
    steps, counts, songs, skipped = [], Counter(), defaultdict(set), Counter()
    for step in exposure['steps']:
        forks = []
        for index, specified in zip(step['indices'], step['specified_condition_fields']):
            if step['effective_update'] <= 2 or len(specified) != 1:
                forks.append(None); skipped['common_parent_or_not_single_field'] += 1
                continue
            field = CONTROL_FIELDS[specified[0]]
            candidates = choices[(index, field)]
            if not candidates:
                forks.append(None); skipped['no_real_same_input_pair'] += 1
                continue
            item = min(candidates, key=lambda r: r['partner'])
            forks.append(item); counts[item['mode']+'/'+field] += 1
            songs[item['mode']+'/'+field].add(item['component'])
        steps.append(dict(effective_update=step['effective_update'], indices=step['indices'], forks=forks))
    sources = [str(baseline.relative_to(ROOT)), plan['tape'], plan['exposure'],
               'scripts/prepare_local_condition_forks.py', 'beatfcn/train/local_request_panel.py']
    result = dict(id='D71-condition-forks-v1', complete=True, steps=steps, counts=dict(counts),
        songs_by_field={k: len(v) for k, v in songs.items()}, skipped=dict(skipped),
        pool_identity=exposure['pool_identity'], exposure_sha256=sha(ROOT/plan['exposure']),
        inputs_sha256={p:sha(ROOT/p) for p in sources},
        scope='Extra weight only on an existing true target token; no new request, paired target exposure, artificial label, or changed audio.')
    write_json(output, result)
    write_json(ROOT/'reports/d71_condition_fork_support_v1.json', {k:v for k,v in result.items() if k!='steps'})
    print(json.dumps({k:result[k] for k in ('counts','songs_by_field','skipped')}))


if __name__ == '__main__':
    run()

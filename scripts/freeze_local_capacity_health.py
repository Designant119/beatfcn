"""Reuse D94's frozen eight new-song requests for the missing width256 health gate."""
from copy import deepcopy

from scripts.prepare_local_data_scale_probe import read, sha, write_new
from scripts.prepare_local_capacity import width_only


def run():
    initialization = 'reports/d95_initialization_v2.json'
    parent_plan = 'bench/rhythm-v2/d95-initialization-v2.json'
    proof, startup = read(initialization), read(parent_plan)
    if (not proof['complete'] or not proof['passed'] or proof['candidate_updates'] != 2
            or proof['plan_sha256'] != sha(parent_plan)
            or proof['checkpoint_sha256'] != sha(proof['checkpoint'])
            or not proof['original_parent_unchanged']):
        raise ValueError('Width256 startup has not passed its original execution gate')
    for path, expected in startup['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Verified initialization identity changed: ' + path)
    source = 'bench/rhythm-v2/d94-new-data-probe-v1.json'
    old = read(source)
    width_only(old['config'], startup['config'])
    plan = deepcopy(old)
    plan.update(id='D95-new-data-width256-probe-v1', parent=proof['checkpoint'],
        config=startup['config'], output='reports/d95_new_data_probe_v1.json',
        checkpoint='artifacts/d95-probe/diagnostic-step100.pt',
        scope='Same eight full new-song requests as D94; width256 implementation health only. Never quality or condition-response evidence.',
        reused_case_selection=source, initialization=initialization)
    del plan['source_sha256'][old['parent']]
    for path in (source, initialization, parent_plan, proof['checkpoint'],
                 'scripts/freeze_local_capacity_health.py', 'scripts/prepare_local_capacity.py',
                 'design/LOCAL_CAPACITY_D95.md'):
        plan['source_sha256'][path] = sha(path)
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('New health input or source changed: ' + path)
    write_new('bench/rhythm-v2/d95-new-data-probe-v1.json', plan)
    print('Frozen D95 eight-request health; no training started.')


if __name__ == '__main__':
    run()

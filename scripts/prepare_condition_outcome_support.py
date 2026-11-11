"""Freeze real training requests for a later SCST signal gate; no model runs.

Use existing exposure views, not validation-derived conditions or newly invented
attribute combinations. Sixteen single-field plus sixteen all-applicable views
cover 32 distinct training songs at the unchanged 24 Phira / 8 mania ratio.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def run():
    destination = ROOT / 'bench/rhythm-v2/d97-scst-support-panel-v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    plan_path = 'bench/rhythm-v2/d97-semantic-quality-v1.json'
    audit_path = 'reports/d97_condition_outcome_audit_v1.json'
    plan = json.loads((ROOT / plan_path).read_text())
    audit = json.loads((ROOT / audit_path).read_text())
    if not audit['complete'] or not audit['training_labels_all_satisfied']:
        raise ValueError('Actual scope/label validation is required')
    exposure = json.loads((ROOT / plan['exposure']).read_text())
    for path in (plan['tape'], plan['features'], plan['exposure']):
        if sha(path) != audit['source_sha256'][path]:
            raise ValueError('Validated data identity changed: ' + path)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    available = {}
    all_support = defaultdict(set)
    for update in exposure['steps']:
        for index, specified in zip(update['indices'], update['specified_condition_fields'], strict=True):
            row = panel.records[index]
            if row['split'] != 'train':
                raise ValueError('Training support includes a held-out song')
            fields = tuple(CONTROL_FIELDS[i] for i in specified if row['controls'][i][2] == 1)
            if not fields:
                continue
            for field in fields:
                all_support[field].add(row['component'])
            key = (index, fields)
            available.setdefault(key, dict(record_index=index, fields=fields, row=row,
                observed_at_update=update['effective_update'],
                ordering=hashlib.sha256(json.dumps(key).encode()).hexdigest()))
    if any(len(all_support[field]) < 8 for field in CONTROL_FIELDS):
        raise ValueError('Fewer than eight real training songs support a requested field')
    candidates = sorted(available.values(), key=lambda x: x['ordering'])
    selected, used = [], set()

    def select(mode, fields, count):
        matches = [r for r in candidates if r['row']['mode'] == mode and r['fields'] == fields]
        for r in matches:
            if r['row']['component'] in used:
                continue
            used.add(r['row']['component'])
            selected.append(r)
            count -= 1
            if count == 0:
                return
        raise ValueError('Insufficient distinct observed views: ' + repr((mode, fields, count)))

    for field in CONTROL_FIELDS:  # 单字段视图：每个字段两条 phira 观察
        select('phira', (field,), 2)
    # jack_ratio 退休后 mania 侧少了两条单字段视图，用 chord_ratio 补足模式配额（24 phira / 8 mania）
    select('mania_4k', ('chord_ratio',), 2)
    for mode, count in [('phira', 10), ('mania_4k', 6)]:
        select(mode, tuple(f for f in CONTROL_FIELDS if f in MODE_FIELDS[mode]), count)
    support = defaultdict(set)
    records = []
    for picked in selected:
        row = picked['row']
        control = np.asarray(row['controls'], dtype=np.float32).copy()
        control[[i for i, field in enumerate(CONTROL_FIELDS) if field not in picked['fields']], :3] = 0
        for field in picked['fields']:
            support[field].add(row['component'])
        records.append(dict(record_index=picked['record_index'], component=row['component'],
            chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], split=row['split'],
            mode=row['mode'], stratum=row['stratum'], request=row['request'], controls=control.tolist(),
            bands={field: row['bands'][field] for field in picked['fields']},
            condition_source='Actual previously exposed training-target labels, never validation targets',
            original_exposure_update=picked['observed_at_update'], selection_hash=picked['ordering'],
            reference='unknown_region'))
    if len(records) != 32 or any(len(support[field]) < 8 for field in CONTROL_FIELDS):
        raise ValueError('The fixed panel itself must support every field on eight training songs')
    mode_counts = dict(Counter(r['mode'] for r in records))
    if mode_counts != {'phira': 24, 'mania_4k': 8}:
        raise ValueError('Actual mode balance differs')
    report = dict(id='D97-SCST-real-support-panel-v1', complete=True, created_at_unix=time.time(),
        source_sha256={path: sha(path) for path in [plan_path, audit_path, plan['tape'],
            plan['features'], plan['exposure'], 'scripts/prepare_condition_outcome_support.py']},
        records=records, mode_counts=mode_counts, distinct_training_songs=len(used),
        global_training_song_support={f: len(all_support[f]) for f in CONTROL_FIELDS},
        panel_training_song_support={f: len(support[f]) for f in CONTROL_FIELDS},
        sample_temperature=1., token_cap=4096, random_seed=20261026,
        minimum_nonzero_advantage_requests=8,
        additional_gates=['Eligible A generation checkpoint required before sampling',
            'At least one legal complete nonempty sample; differences not solely failed or empty outcomes',
            'All eight fields audited separately; type and completion support retained',
            'CPU/MPS replay and actual gradient gate before any B training'],
        no_filtering_failed_rollouts=True, model_calls=0, optimizer_updates=0,
        signal_gate_passed=None, training_enabled=False, checkpoint=None, normalizer=None,
        lambda_value=None, new_cloud_spend_usd=0,
        purpose='A frozen training-signal probe, not a product benchmark or condition response test')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ['mode_counts', 'distinct_training_songs',
        'global_training_song_support', 'panel_training_song_support', 'signal_gate_passed']}))


if __name__ == '__main__':
    run()

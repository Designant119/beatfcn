"""Audit only known training exposures; never fit statistics to held-out songs."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from beatfcn.events.controls import BAND_EDGES
from beatfcn.train.event_conditions import CONTROL_FIELDS, encode_controls
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    output = ROOT / 'reports/d75_condition_scale_audit_v1.json'
    normalizer_path = ROOT / 'inputs/d75_condition_standardization/normalizer-v1.json'
    if output.exists() or normalizer_path.exists():
        raise FileExistsError('Frozen statistics already exist')
    plan_path = ROOT / 'bench/rhythm-v2/d68-local-baseline-v3.json'
    plan = json.loads(plan_path.read_text())
    for path in (plan['tape'], plan['exposure'], plan['features']):
        if path in plan['inputs_sha256'] and sha(ROOT / path) != plan['inputs_sha256'][path]:
            raise ValueError('Frozen source changed: ' + path)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    exposure = json.loads((ROOT / plan['exposure']).read_text())
    values, songs, bands, mode_counts = defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(lambda: defaultdict(int))
    scopes, known_total = [], 0
    for step in exposure['steps']:
        for index, specified in zip(step['indices'], step['specified_condition_fields']):
            row = panel.records[index]
            if row['split'] != 'train':
                raise ValueError('Cannot fit a normalizer on heldout requests')
            controls = np.asarray(row['controls'], dtype=np.float32)
            if not np.isfinite(controls).all():
                raise ValueError('Nonfinite source controls')
            for i in specified:
                field = CONTROL_FIELDS[i]
                if controls[i, 2] != 1 or controls[i, 3] != 1 or field not in row['bands']:
                    raise ValueError('Specified view lacks a true applicable condition')
                values[field].extend(controls[i, :2].astype(np.float64).tolist())
                songs[field].add(row['component']); bands[field].add(tuple(row['bands'][field]))
                mode_counts[field][row['mode']] += 1
                scopes.append(controls[i, 4:].astype(np.float64).tolist()); known_total += 1
    fields = []
    for field in CONTROL_FIELDS:
        x = np.asarray(values[field], np.float64)
        if not len(x) or not np.isfinite(x).all() or x.std() == 0:
            raise ValueError('Unsupported or constant field cannot be standardized: ' + field)
        fields.append(dict(field=field, endpoint_count=len(x), request_exposures=len(x)//2,
            songs=len(songs[field]), mode_counts=dict(mode_counts[field]), distinct_bands=len(bands[field]),
            normalized_min=float(x.min()), normalized_max=float(x.max()), mean=float(x.mean()),
            std=float(x.std()), quantiles=np.quantile(x, [0., .1, .5, .9, 1.]).tolist(),
            physical_scale=BAND_EDGES[field][-1], affine_gain=float(1/x.std()),
            standardized_min=float((x.min()-x.mean())/x.std()), standardized_max=float((x.max()-x.mean())/x.std())))
    sources = [str(plan_path.relative_to(ROOT)), plan['tape'], plan['features'], plan['exposure'],
               'scripts/audit_local_condition_scales.py','beatfcn/train/event_conditions.py','beatfcn/events/controls.py']
    normalizer = dict(schema='beatfcn.condition_training_standardization.v1',
        field_order=list(CONTROL_FIELDS), mean=[f['mean'] for f in fields], std=[f['std'] for f in fields],
        fit_split='train', fit_views='actual specified fields in frozen1000 exposure',
        application='Only known and applicable lower/upper; unknown0, flags and scopes unchanged; no clipping',
        inputs_sha256={p: sha(ROOT/p) for p in sources})
    write_json(normalizer_path, normalizer)
    pairs = []
    for field, aa, bb in [('hold_head_ratio', [.8, 1.], [0., .02]),
                           ('density_notes_per_second', [.5, 1.], [3., 4.])]:
        i = CONTROL_FIELDS.index(field)
        a, b = [encode_controls('phira', {field: band}, (0, 8_000_000), origin_us=0)[i, :2]
                for band in (aa, bb)]
        pairs.append(dict(field=field, bands=[aa, bb], original_endpoint_distance=float(np.linalg.norm(a-b)),
                          standardized_endpoint_distance=float(np.linalg.norm((a-b)/fields[i]['std']))))
    report = dict(id='D75-condition-scale-audit-v1', complete=True, known_field_exposures=known_total,
        exposure_requests=sum(len(s['indices']) for s in exposure['steps']), fit_split='train',
        fields=fields, diagnostic_pairs=pairs,
        scopes_normalized_range=[float(np.min(scopes)), float(np.max(scopes))],
        normalizer=str(normalizer_path.relative_to(ROOT)), normalizer_sha256=sha(normalizer_path),
        optimizer_updates=0, cloud_compute_calls=0, quality_passed=False,
        scope='Field scale audit and frozen train-only affine statistics; no input or model changed.')
    write_json(output, report)
    print(json.dumps(dict(fields=[{k:f[k] for k in ('field','request_exposures','mean','std','affine_gain')} for f in fields], pairs=pairs)))


if __name__ == '__main__':
    run()

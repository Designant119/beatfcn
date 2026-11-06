"""D27: verify the title correction, then re-summarize D26 without decoding again."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from beatfcn.ingest.pipeline import assign_splits
from scripts.audit_phigros_clock import calibration, save, sha

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    registered = json.loads((ROOT/'reports/d27_registration.json').read_text())
    before_root = ROOT/'inputs/data_scale_v2/phigros_resource_v1/admission_v2'
    after_root = ROOT/'inputs/data_scale_v2/phigros_resource_v1/admission_v3'
    before = load_rows(before_root/'manifest/candidates.jsonl')
    after = load_rows(after_root/'manifest/candidates.jsonl')
    old = []
    for name, expected in registered['frozen_manifests'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('old manifest changed')
        old.extend(load_rows(ROOT/name))
    if identity(assign_splits(old)) != registered['old_grouping_identity']:
        raise ValueError('old grouping changed')
    snapshot = ROOT/'inputs/code_baselines/d27-title-split/beatfcn/ingest/pipeline.py'
    if sha(snapshot) != registered['owned_files']['beatfcn/ingest/pipeline.py']:
        raise ValueError('baseline grouping source changed')
    spec = importlib.util.spec_from_file_location('beatfcn.ingest._d27_old_pipeline', snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, c2g, g2s = module.assign_splits(old + [dict(row, title=row['name']) for row in before])
    anchors = defaultdict(list)
    for row in old:
        if row.get('split') in ('train', 'val', 'test'):
            anchors[c2g[row['chart_id']]].append(row)
    if len(after) != 665 or {r['chart_id'] for r in before} != {r['chart_id'] for r in after}:
        raise ValueError('chart set changed')
    old_new = {r['chart_id']: r for r in before}
    changes = []
    allowed = {'title', 'split', 'song_group', 'split_evidence', 'split_conflict'}
    for row in after:
        original = old_new[row['chart_id']]
        if {k:v for k,v in row.items() if k not in allowed} != {k:v for k,v in original.items() if k not in allowed}:
            raise ValueError(f'non-split metadata changed: {row["chart_id"]}')
        name = row['native_id']+'.json'
        if (before_root/'canon'/name).read_bytes() != (after_root/'canon'/name).read_bytes():
            raise ValueError(f'canonical changed: {name}')
        group = c2g[row['chart_id']]
        members = anchors[group]
        splits = {m['split'] for m in members}
        if len(splits) > 1:
            expected_split, expected_group = None, group
        elif members:
            anchor = min(members, key=lambda m:m['chart_id'])
            expected_split, expected_group = anchor['split'], anchor['song_group']
        else:
            expected_split, expected_group = g2s[group], group
        if (row['split'], row['song_group'], row['split_conflict']) != (expected_split, expected_group, len(splits)>1):
            raise ValueError(f'golden split mismatch: {row["chart_id"]}')
        if row['split'] != original['split']:
            changes.append(dict(chart_id=row['chart_id'], name=row['name'],
                old=original['split'], corrected=row['split'],
                anchored_existing_ids=[m['chart_id'] for m in members]))
    verification = dict(schema='beatfcn.phigros_split_verification.v1', complete=True, passed=True,
        script_sha256=sha(Path(__file__)), old_manifest_rows_unchanged=len(old),
        old_grouping_unchanged=True, canonical_bytes_unchanged=665, golden_split_matches=665,
        changes=changes, split_changes=dict(Counter(f'{r["old"]}->{r["corrected"]}' for r in changes)),
        corrected_structural_train=dict(Counter(r['difficulty_tier'] for r in after
            if r['structural_audit_pass'] and r['split']=='train')),
        new_paid_compute_usd=0, model_updates=0, training_ready=False)
    save(ROOT/'reports/d27_verification.json', verification)

    # The immutable D26 acoustic results were never applied. Change only admission metadata.
    source_path = ROOT/'reports/d26_phigros_clock.json'
    clock = json.loads(source_path.read_text())
    if not clock['complete'] or clock['labels_changed'] or clock['training_updates']:
        raise ValueError('unexpected previous clock state')
    plan = json.loads((ROOT/'bench/rhythm-v2/d26-phigros-clock-v2.json').read_text())
    rows_by_id = {r['chart_id']:r for r in after}
    for item in clock['records']:
        row = rows_by_id[item['chart_id']]
        if item['audio_sha256'] != row['audio_sha256']:
            raise ValueError('audio identity changed')
        item['split'] = row['split']
        item['calibration_exclusions'] = [s for s in item['calibration_exclusions'] if s != 'not_training_split']
        if row['split'] != 'train':
            item['calibration_exclusions'].append('not_training_split')
        item['calibration_eligible'] = not item['calibration_exclusions']
    clock['calibration'] = calibration(clock['records'], plan)
    clock['summary'].update(calibration_eligible_charts=sum(r['calibration_eligible'] for r in clock['records']),
        eligible_training_audio=len(clock['calibration'].get('qualified_songs', [])),
        exclusions=dict(Counter(x for r in clock['records'] for x in r['calibration_exclusions'])),
        calibration_status=clock['calibration']['status'])
    clock['split_reanalysis'] = dict(id='D27', timestamp_unix=time.time(),
        source_report_sha256=sha(source_path), verification_sha256=sha(ROOT/'reports/d27_verification.json'),
        corrected_manifest_sha256=sha(after_root/'manifest/candidates.jsonl'),
        script_sha256=sha(Path(__file__)), acoustic_results_unchanged=True,
        original_split_calibration_superseded=True)
    clock['limitations'].append('Use this split-corrected report; the original D26 source summary used a name/title mapping with missing song anchors.')
    save(ROOT/'reports/d26_phigros_clock_split_corrected.json', clock)
    print(json.dumps({k:v for k,v in verification.items() if k!='changes'}, ensure_ascii=False))
    print(json.dumps(dict(clock_summary=clock['summary'], calibration={k:v for k,v in clock['calibration'].items()
        if k not in ('qualified_songs', 'excluded')}), ensure_ascii=False))


if __name__ == '__main__':
    main()

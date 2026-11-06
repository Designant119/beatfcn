"""Audit a frozen policy change without mutating source data or admitting new charts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from beatfcn.train.data import phira_rating_eligible, phira_rating_metadata, PHIRA_SELECTION_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'reports/d67_rating_policy_impact_v1.json'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def main():
    output = ROOT / OUTPUT
    if output.exists():
        raise FileExistsError('Policy evidence is immutable; use a new report version.')
    names = {
        'old_audit': 'reports/d57_phira_strata_audit_v2.json',
        'new_audit': 'reports/d67_phira_strata_audit_v1.json',
        'old_rows': 'reports/d57_phira_strata_audit_v2.rows.jsonl',
        'new_rows': 'reports/d67_phira_strata_audit_v1.rows.jsonl',
        'catalog': 'inputs/data_scale_v2/catalog_v1/phira/catalog.jsonl',
        'contract': 'bench/rhythm-v2/fullscale-data-contract-v3.json',
        'source_gates': 'reports/d66_source_gates_v1.json',
        'feature_panel': 'inputs/d66_local_cohort/feature-panels-v1.json',
        'evaluation20': 'inputs/d66_local_cohort/evaluation20-candidates-v1.json',
        'manifest': 'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
        'transfer': 'reports/d66_features_initial_transfer_v1.json',
        'old_plan': 'bench/rhythm-v2/d66-features-initial-v1.json',
    }
    old_audit, new_audit = read(names['old_audit']), read(names['new_audit'])
    if old_audit['input_sha256'][names['catalog']] != new_audit['input_sha256'][names['catalog']]:
        raise ValueError('Policy comparison must use the same catalog snapshot.')
    old = {r['native_id']: r for r in rows(names['old_rows'])}
    new = {r['native_id']: r for r in rows(names['new_rows'])}
    old_ids = {k for k, r in old.items() if r['metadata_candidate']}
    new_ids = {k for k, r in new.items() if r['metadata_candidate']}
    added, removed = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
    manifest = {r['chart_id']: r for r in rows(names['manifest'])}
    source, panel, evaluation = (read(names[k]) for k in ('source_gates', 'feature_panel', 'evaluation20'))

    def inspect(records):
        kept, rejected = [], []
        for row in records:
            audit = new.get(row['chart_id'].removeprefix('phira:'))
            original = manifest.get(row['chart_id'])
            reasons = []
            if row['mode'] == 'phira':
                if not audit:
                    reasons.append('missing_catalog_metadata')
                elif not audit['metadata_candidate']:
                    reasons.extend(audit['selection_reasons'])
                if original is None or not phira_rating_eligible(original):
                    reasons.append('frozen_manifest_rating_or_votes_fail')
            rating, votes = phira_rating_metadata(original or {})
            record = dict(chart_id=row['chart_id'], mode=row['mode'], split=row['split'],
                          component=row['component'], audio_sha256=row['audio_sha256'],
                          stratum=row['stratum'], catalog_rating=audit['rating'] if audit else None,
                          catalog_votes=audit['rating_count'] if audit else None,
                          frozen_rating=rating, frozen_votes=votes, reasons=reasons)
            (rejected if reasons else kept).append(record)
        components = {(r['split'], r['mode'], r['component']) for r in kept}
        return dict(kept=kept, rejected=rejected, retained_charts=len(kept), rejected_charts=len(rejected),
                    songs_by_split_mode=dict(Counter(f'{s}/{m}' for s, m, _ in components)),
                    all_pass=not rejected)

    collections = {'source_screened': inspect(source['eligible']), 'feature_panel': inspect(panel['charts']),
                   'evaluation20': inspect(evaluation['records'])}
    train = set(panel['tiers'][max(panel['tiers'], key=int)])
    val = set(panel['validation'])
    if train & val or len({r['component'] for r in evaluation['records']}) != 20:
        raise ValueError('Frozen song isolation or whole-song evaluation identity failed.')
    if not set(panel['tiers']['100']) <= train:
        raise ValueError('Panel nesting changed.')
    transfer = read(names['transfer'])
    plan = read(names['old_plan'])
    current_initial = set(panel['tiers']['100']) | val
    needed_audio = {r['audio_sha256'] for r in collections['feature_panel']['kept']
                    if r['component'] in current_initial}
    unused_files = [r['path'] for r in plan['files'] if r['audio_sha256'] not in needed_audio]
    for record in transfer['files']:
        path = ROOT / record['local']
        if path.stat().st_size != record['bytes'] or sha(record['local']) != record['sha256']:
            raise ValueError(f'Completed cache changed: {path}')
    added_rows = [new[k] for k in added]
    removed_rows = [new[k] for k in removed]
    report = dict(id='D67-rating-policy-impact-v1', created_at_unix=time.time(),
        selection_version=PHIRA_SELECTION_VERSION,
        policy=read(names['contract'])['phira_rating'],
        inputs={p: sha(p) for p in names.values()},
        code_sha256={p: sha(p) for p in ['scripts/audit_phira_rating_change.py', 'beatfcn/train/data.py']},
        catalog=dict(total=len(new), previous_metadata_candidates=len(old_ids),
                     current_metadata_candidates=len(new_ids), added=len(added), removed=len(removed),
                     net_change=len(new_ids)-len(old_ids),
                     added_by_stratum=dict(Counter(r['stratum']['key'] for r in added_rows)),
                     removed_reasons=dict(Counter(reason for r in removed_rows for reason in r['selection_reasons'])),
                     added_ids=added, removed_ids=removed),
        revalidation=collections, original_song_assignments_unchanged=True,
        all_prepared_panels_policy_pass=all(r['all_pass'] for r in collections.values()),
        stopped_transfer=dict(pid=16134, session_id=47412, exit_code=130,
            reason='user_policy_update', cumulative_bytes=transfer['cumulative_bytes'],
            complete_files_rehashed=len(transfer['files']),
            complete_bytes=sum(r['bytes'] for r in transfer['files']),
            files_outside_revalidated_initial_panel=unused_files, original_maximum_bytes=plan['maximum_transfer_bytes']),
        new_formally_admitted_charts=0, training_ready=False, optimizer_updates=0, cloud_compute_calls=0,
        limitations=['Catalog is frozen, not a fresh live Phira API snapshot.',
                    'Newly eligible metadata still needs native content, target dedup, split, clock and feature gates.',
                    'A ratings gate is not musical quality validation.',
                    'Existing source/clock evidence is reused, not rerun; final feature/tensor gates remain.',
                    'Original D66 panels are immutable; this report binds their exact identities to the new policy.'])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['all_prepared_panels_policy_pass', 'stopped_transfer']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

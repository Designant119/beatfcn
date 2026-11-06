"""Re-audit local native targets under the explicit-isFake-only policy.

No downloads, optimization, source edits, split changes or automatic admission.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from beatfcn.train.data import phira_rating_eligible
from beatfcn.train.local_canonical_audit import audit_document
from scripts.audit_phira_native_content import inspect_chart, read_chart

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'reports/d91_isfake_policy_impact_v1.json'


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    if (ROOT/OUTPUT).exists():
        raise FileExistsError('Use a new version; do not overwrite frozen evidence.')
    started = time.monotonic()
    paths = [
        'reports/d66_source_gates_v1.json', 'reports/d66_native_audit_v2.json',
        'reports/d66_pair_native_audit_v1.json', 'reports/d66_canonical_audit_v2.json',
        'reports/d66_supplement_audit_v1.json', 'reports/d66_combined_canonical_receipt.json',
        'reports/d67_phira_strata_audit_v1.rows.jsonl',
        'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
        'bench/rhythm-v2/fullscale-data-contract-v4.json',
        'scripts/audit_local_isfake_policy.py', 'scripts/audit_phira_native_content.py',
        'beatfcn/events/phira.py', 'beatfcn/events/canonical.py',
        'beatfcn/train/data.py', 'beatfcn/train/local_canonical_audit.py',
    ]
    source_sha = {p:sha(p) for p in paths}
    old = read(paths[0])
    old_native = {r['chart_id']:r for p in paths[1:3] for r in read(p)['records']}
    old_canonical = {r['chart_id']:r for r in read(paths[3])['records'] + read(paths[4])['paired_charts']}
    receipt = {r['path']:r for r in read(paths[5])['files']}
    catalogue = {r['native_id']:r for r in map(json.loads, (ROOT/paths[6]).read_text().splitlines())}
    manifest = {r['chart_id']:r for r in map(json.loads, (ROOT/paths[7]).read_text().splitlines())}
    results = []
    for candidate in old['eligible'] + old['unresolved']:
        if time.monotonic()-started > 180:
            raise TimeoutError('Bounded local audit exceeded180seconds; no training dispatched.')
        cid = candidate['chart_id']
        row = manifest[cid]
        file = receipt[f"canon/{row['source']}/{row['native_id']}.json"]
        if sha(file['local']) != file['sha256']:
            raise ValueError('Canonical receipt drift: '+cid)
        source_sha[file['local']] = file['sha256']
        canonical, _ = audit_document(read(file['local']), row)
        target_unchanged = canonical['model_rhythm_sha256'] == old_canonical[cid]['model_rhythm_sha256']
        if not target_unchanged:
            raise ValueError('Policy audit unexpectedly changed target events: '+cid)
        semantic = None
        if row['mode'] == 'phira':
            prior = old_native[cid]
            package = ROOT / f"inputs/d66_local_cohort/source/raw/phira/zips/{row['native_id']}.zip"
            info, raw, identity = read_chart(package)
            if any(identity[k] != prior[k] for k in ('source_sha256', 'info_sha256', 'chart_member')):
                raise ValueError('Native chart/info changed: '+cid)
            semantic = inspect_chart(raw, info.get('offset'))
            for key in ('canonical_events', 'canonical_fakes', 'raw_objects'):
                if semantic['counts'][key] != prior['native_semantics']['counts'][key]:
                    raise ValueError('Native policy changed target/fake count: '+cid)
            if semantic['canonical_types'] != prior['native_semantics']['canonical_types']:
                raise ValueError('Native policy changed target types: '+cid)
        reasons = list(candidate.get('reasons', []))
        if canonical['canonical_integrity_passed'] and 'canonical_integrity' in reasons:
            reasons.remove('canonical_integrity')
        native_pass = semantic is None or (old_native[cid]['source_identity_passed'] and
                                           semantic['status'] == 'static_screen_only')
        if native_pass and 'native_semantics_unresolved' in reasons:
            reasons.remove('native_semantics_unresolved')
        if not canonical['canonical_integrity_passed'] and 'canonical_integrity' not in reasons:
            reasons.append('canonical_integrity')
        metadata_reasons = []
        if row['mode'] == 'phira':
            cat = catalogue.get(str(row['native_id']))
            if cat is None or not cat['metadata_candidate']:
                metadata_reasons.append('current_catalogue_policy')
            if not phira_rating_eligible(row):
                metadata_reasons.append('frozen_manifest_rating_votes')
        results.append(dict(chart_id=cid, mode=row['mode'], split=candidate['split'],
            component=candidate['component'], stratum=candidate['stratum'],
            old_source_pass=candidate['source_screen_passed'], new_source_pass=not reasons,
            source_reasons=reasons, metadata_reasons=metadata_reasons,
            current_policy_source_pass=not reasons and not metadata_reasons,
            target_events_unchanged=target_unchanged, native_semantics=semantic,
            old_canonical_pass=old_canonical[cid]['canonical_integrity_passed'],
            canonical_pass=canonical['canonical_integrity_passed'],
            canonical_issues=canonical['issues'], playable_notes=canonical['playable_notes'],
            fake_notes=canonical['fake_notes'], model_rhythm_sha256=canonical['model_rhythm_sha256'],
            native_source_sha256=canonical['source_sha256'], training_ready=False))
    restored = [r for r in results if not r['old_source_pass'] and r['current_policy_source_pass']]
    prior_ids = {r['chart_id'] for r in old['eligible']}
    assert all(r['current_policy_source_pass'] for r in results if r['chart_id'] in prior_ids)
    assert all(sha(p) == h for p, h in source_sha.items())
    report = dict(id='D91-isfake-policy-impact-v1', complete=True,
        optimizer_updates=0, new_cloud_compute_calls=0, additional_cloud_usd=0,
        training_ready=False, new_formally_admitted_charts=0,
        source_sha256=source_sha, rows=results, wall_s=time.monotonic()-started,
        summary=dict(charts_checked=len(results), native_phira_packages=len(old_native),
            previous_source_pass=len(prior_ids), current_source_pass=sum(r['new_source_pass'] for r in results),
            current_policy_source_pass=sum(r['current_policy_source_pass'] for r in results),
            newly_source_eligible=len(restored), newly_source_eligible_ids=[r['chart_id'] for r in restored],
            restored_by_split=dict(Counter(r['split'] for r in restored)),
            restored_by_stratum=dict(Counter(r['stratum'] for r in restored)),
            visual_only_review_released=sum(bool(r['native_semantics'] and old_native[r['chart_id']]['native_semantics']['review_signals'])
                                          for r in results),
            stack_quarantine_released=sum(not r['old_canonical_pass'] and r['canonical_pass'] for r in results),
            all_existing_targets_unchanged=all(r['target_events_unchanged'] for r in results),
            explicit_fake_notes_preserved=sum(r['fake_notes'] for r in results)),
        limitations=['Local frozen cohort plus same-song supplement, not whole remote corpus.',
            'Native chart/info bytes rehashed; prior package/audio identity reused, no new audio decoding.',
            'Prior clock and split gates retained; metadata uses frozen catalogue and manifest, not live ratings.',
            'Existing training tapes/evaluation20/checkpoints are immutable. New candidates need a new data version '
            'and feature/tensor admission before training. No model improvement claimed.'])
    (ROOT/OUTPUT).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'], wall_s=report['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    main()

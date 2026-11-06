"""Compare complete pure-rhythm targets across explicitly linked source charts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.canonical import build_model_view
from beatfcn.events.group_split import work_key
from beatfcn.train.data import chart_from_canon
from scripts.audit_phigros_resource import rhythm_hash

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_keys(row):
    return {work_key(row[k]) for k in ('title', 'title_alt', 'name') if row.get(k) and work_key(row[k])}


def linked(a, b):
    return a['audio_sha256'] == b['audio_sha256'] or bool(title_keys(a) & title_keys(b))


def signature(doc):
    chart = chart_from_canon(doc)
    if chart.mode != 'phira':
        raise ValueError('pure-rhythm identity must not discard mania lanes')
    view = build_model_view(chart.mode, (), chart.events)
    events = [e for group in view.target_groups for e in group.events]
    if view.quarantine or not events or len(events) != len(chart.events):
        return dict(status='unresolved', issues=[x.code for x in view.quarantine],
                    raw_events=len(chart.events), target_events=len(events))
    first = min(e.onset_us for e in events)
    relative = sorted((e.onset_us-first, e.end_us-first, e.type) for e in events)
    return dict(status='ok', count=len(events), absolute=rhythm_hash(events),
        relative=hashlib.sha256(json.dumps(relative, separators=(',', ':')).encode()).hexdigest(),
        first_onset_us=first, types=dict(Counter(e.type for e in events)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    registration = json.loads((ROOT/'reports/d28_audit_registration.json').read_text())
    if registration['source_sha256'] != sha(Path(__file__)):
        raise ValueError('audit source changed after registration')
    transfer = json.loads((ROOT/plan['report']).read_text())
    if not transfer['complete'] or transfer['plan_sha256'] != sha(args.plan):
        raise ValueError('complete frozen transfer required')
    for name, expected in plan['input_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'input changed: {name}')
    paths = {r['chart_id']:r['local'] for r in transfer['files']}
    paths.update(plan['existing_canonical'])
    files = {r['local']:r['sha256'] for r in transfer['files']}
    files.update({p:plan['input_sha256'][p] for p in plan['existing_canonical'].values()})
    signatures = {}
    for row in plan['matched_rows']:
        path = ROOT/paths[row['chart_id']]
        if sha(path) != files[str(path.relative_to(ROOT))]:
            raise ValueError('canonical receipt mismatch')
        doc = json.loads(path.read_text())
        if any(doc.get(k) != row[k] for k in ('chart_id', 'audio_sha256', 'mode')):
            raise ValueError('canonical differs from frozen old metadata')
        signatures[row['chart_id']] = signature(doc)
    rows = [json.loads(line) for line in (ROOT/plan['new_manifest']).read_text().splitlines()]
    canon_root = (ROOT/plan['new_manifest']).parent.parent/'canon'
    records = []
    for row in rows:
        path = canon_root/(row['native_id']+'.json')
        if sha(path) != registration['canonical_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('new canonical changed after audit registration')
        files[str(path.relative_to(ROOT))] = sha(path)
        doc = json.loads(path.read_text())
        if any(doc.get(k) != row[k] for k in ('chart_id', 'audio_sha256', 'mode')):
            raise ValueError('new canonical differs from frozen metadata')
        sig = signature(doc)
        if sig['status'] == 'ok' and sig['absolute'] != row['pure_rhythm_hash']:
            raise ValueError('new target signature differs from structural audit')
        matches, candidates, relative_candidates, unresolved = [], [], [], []
        relevant = [old for old in plan['matched_rows'] if linked(row, old)]
        for old in relevant:
            previous = signatures[old['chart_id']]
            if sig['status'] != 'ok' or previous['status'] != 'ok':
                unresolved.append(old['chart_id'])
                continue
            same_audio = row['audio_sha256'] == old['audio_sha256']
            if sig['absolute'] == previous['absolute']:
                (matches if same_audio else candidates).append(old['chart_id'])
            elif sig['relative'] == previous['relative']:
                relative_candidates.append(dict(chart_id=old['chart_id'], same_audio_sha256=same_audio,
                    target_global_difference_us=sig['first_onset_us']-previous['first_onset_us']))
        records.append(dict(chart_id=row['chart_id'], name=row['name'], split=row['split'],
            tier=row['difficulty_tier'], structural_pass=row['structural_audit_pass'],
            signature=sig, old_linked=len(relevant), confirmed_exact_audio_duplicates=matches,
            same_work_identical_target_different_audio=candidates,
            same_work_identical_relative_target=relative_candidates, unresolved_old_ids=unresolved,
            training_ready=False))
    report = dict(schema='beatfcn.phigros_overlap_audit.v1', complete=True,
        plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)),
        transfer_sha256=sha(ROOT/plan['report']), canonical_sha256=files, old_signatures=signatures,
        records=records, new_paid_compute_usd=0, optimizer_updates=0, training_ready=False,
        summary=dict(old_linked_charts=len(plan['matched_rows']), old_canonicals_checked=len(signatures),
            old_signature_ok=sum(s['status']=='ok' for s in signatures.values()), new_charts=len(rows),
            new_with_old_link=sum(r['old_linked']>0 for r in records),
            confirmed_duplicate_new_charts=sum(bool(r['confirmed_exact_audio_duplicates']) for r in records),
            same_work_exact_target_candidates=sum(bool(r['same_work_identical_target_different_audio']) for r in records),
            same_work_relative_target_candidates=sum(bool(r['same_work_identical_relative_target']) for r in records),
            unresolved_new_charts=sum(bool(r['unresolved_old_ids']) for r in records)),
        limitations=plan['limitations']+['Different audio hashes are not automatically merged even when full chart content matches. Unrecognized work aliases remain untested.'])
    out=ROOT/'reports/d28_overlap_audit.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__ == '__main__':
    main()

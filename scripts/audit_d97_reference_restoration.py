"""Inspect a held-out native pair excluded by the previous duplicate-object policy.

This is data support evidence only. It changes no training or evaluation tape.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from beatfcn.events.controls import derive_controls
from beatfcn.events.phira import parse_phira
from beatfcn.ingest.phira_policy import phira_stratum
from beatfcn.train.local_cohort import connected_songs
from scripts.audit_expanded_reference_support import signature, visible_prefix
from scripts.audit_local_cohort_native import audio_identity
from scripts.audit_phira_native_content import inspect_chart, read_chart

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    started = time.monotonic()
    output = ROOT/'reports/d97_reference_restoration_audit_v1.json'
    if output.exists():
        raise FileExistsError('Existing evidence must not be overwritten')
    pp = ROOT/'bench/rhythm-v2/d97-reference-restoration-sources-v1.json'
    plan = json.loads(pp.read_text())
    receipt_path = ROOT/plan['report']
    receipt = json.loads(receipt_path.read_text())
    if not receipt['all_archives_downloaded'] or receipt['plan_sha256'] != sha(pp):
        raise ValueError('Frozen transfer receipt required')
    corpus = list(map(json.loads, (ROOT/plan['corpus']).read_text().splitlines()))
    identity, splits = connected_songs(corpus)
    rows = {r['chart_id']: r for r in corpus}
    tape_path = ROOT/'reports/d94_expanded_target_tape_v1.json'
    tape = json.loads(tape_path.read_text())
    train_components = {r['component'] for r in tape['charts'] if r['split'] == 'train'}
    diagnostic_path = ROOT/'inputs/d65_local/panel.json'
    diagnostic_text = diagnostic_path.read_text()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               (pp, receipt_path, tape_path, diagnostic_path, ROOT/plan['corpus'],
                Path(__file__), ROOT/'scripts/audit_phira_native_content.py',
                ROOT/'beatfcn/events/phira.py', ROOT/'beatfcn/events/canonical.py',
                ROOT/'beatfcn/ingest/phira_policy.py')}
    records, charts = [], []
    for item, rec in zip(plan['items'], receipt['records'], strict=True):
        if rec['id'] != item['id']:
            raise ValueError('Receipt order changed')
        cid = f"phira:{item['id']}"
        row = rows[cid]
        component = identity[cid]
        if splits[component] != {'val'} or component in train_components:
            raise ValueError('Held-out component contamination')
        # D65 is diagnostic-only and must not silently become independent validation.
        related = [r for r in corpus if identity.get(r['chart_id']) == component]
        if any(r['audio_sha256'] in diagnostic_text for r in related):
            raise ValueError('Diagnostic song alias overlaps candidate')
        package = ROOT/rec['path']
        if sha(package) != rec['sha256']:
            raise ValueError('Native package changed')
        sources[rec['path']] = rec['sha256']
        info, raw, native_identity = read_chart(package)
        audio = audio_identity(package, info, {'max_audio_member_bytes': 32 << 20})
        if audio['audio_sha256'] != item['audio_sha256']:
            raise ValueError('Source audio differs from frozen corpus')
        chart = parse_phira(raw, info_offset_sec=info.get('offset'))
        semantic = inspect_chart(raw, info.get('offset'))
        native_stratum = phira_stratum(dict(info, difficulty_label=info.get('level'), source='phira'))
        stacks = Counter(e.signature() for e in chart.events)
        records.append(dict(chart_id=cid, component=component, split='val',
            component_chart_ids=sorted(r['chart_id'] for r in related),
            source_identity=native_identity, audio=audio,
            package_level=info.get('level'), package_difficulty=info.get('difficulty'),
            package_offset_s=info.get('offset'), native_stratum=native_stratum,
            catalog_stratum=item['catalog_stratum'],
            catalog_native_stratum_match=native_stratum['key'] == item['catalog_stratum']['key'],
            original_quarantine=item['original_p0_quarantine'], native=semantic,
            preserved_exact_duplicate_extras=sum(n-1 for n in stacks.values()),
            rhythm_sha256=signature(chart.events), training_ready=False))
        charts.append(chart)
    duration_us = int(min(rows[r['chart_id']]['audio_sec'] for r in records)*1e6)
    windows = []
    for start in range(16_000_000, duration_us-8_000_000+1, 8_000_000):
        end = start+8_000_000
        pair = []
        for chart in charts:
            _, visible, unseen = visible_prefix(chart.events, start)
            target = [e for e in chart.events if start <= e.onset_us < end]
            # A carry difference changes the continuation request itself.
            carry = sorted((e.onset_us, e.end_us, e.type) for e in chart.events
                           if e.is_hold and e.onset_us < start < e.end_us)
            pair.append(dict(visible_sha256=signature(visible), target_sha256=signature(target),
                unseen_incoming_holds=len(unseen), carry=carry,
                target_bands=derive_controls('phira', chart.events, [(start,end)])[0]))
        a,b = pair
        windows.append(dict(start_us=start, end_us=end, records=pair,
            distinct_prefix_and_target=(a['visible_sha256'] != b['visible_sha256']
                                       and a['target_sha256'] != b['target_sha256']),
            equal_carry=a['carry'] == b['carry'],
            all_target_bands_equal=a['target_bands'] == b['target_bands'],
            visible_support=not (a['unseen_incoming_holds'] or b['unseen_incoming_holds'])))
    native_same_stratum = len({r['native_stratum']['key'] for r in records}) == 1
    static_passed = all(r['native']['status'] == 'static_screen_only' for r in records)
    strict = [w for w in windows if w['distinct_prefix_and_target'] and w['equal_carry']
              and w['all_target_bands_equal'] and w['visible_support'] and native_same_stratum and static_passed]
    result = dict(id='D97-native-reference-restoration-audit-v1', complete=True,
        source_sha256=sources, records=records, windows=windows, static_accounting_passed=static_passed,
        distinct_full_rhythm=records[0]['rhythm_sha256'] != records[1]['rhythm_sha256'],
        native_same_stratum=native_same_stratum, strict_support_windows=len(strict),
        added_admitted_validation_songs=0, training_ready=False, optimizer_updates=0,
        new_cloud_compute_calls=0, wall_s=time.monotonic()-started,
        limitations=['Window support uses manifest duration; decoded audio clock and features remain unverified.',
            'Target bands describe data support only and must never construct product evaluation requests.',
            'One related-author song pair cannot establish independent style generalization.',
            'No training/evaluation tape changed and no reference capability has been trained.'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','static_accounting_passed',
        'distinct_full_rhythm','native_same_stratum','strict_support_windows','wall_s')}))


if __name__ == '__main__':
    run()

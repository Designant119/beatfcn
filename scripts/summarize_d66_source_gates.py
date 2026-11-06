"""Freeze source-screened candidates and an ungenerated 20-song review panel."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

from beatfcn.events.canonical import build_model_view
from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.data import chart_from_canon, model_shift_us
from beatfcn.train.event_conditions import in_band, training_labels

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = dict(panel='inputs/d66_local_cohort/candidates-v1.json',
        extra='inputs/d66_local_cohort/paired-native-supplement-v1.json',
        native='reports/d66_native_audit_v2.json', extra_native='reports/d66_pair_native_audit_v1.json',
        canonical='reports/d66_canonical_audit_v2.json', supplement='reports/d66_supplement_audit_v1.json',
        mania='reports/d66_mania_text_transport_v2.json', clock='reports/d66_audio_clock_v1.json',
        receipt='reports/d66_combined_canonical_receipt.json',
        pairs='reports/d66_full_metadata_pair_candidates.json')
    data = {k: json.loads((ROOT/p).read_text()) for k, p in paths.items()}
    for key in ('native', 'extra_native', 'canonical', 'supplement', 'clock', 'receipt'):
        if not data[key]['complete']:
            raise ValueError('Required source gate incomplete: '+key)
    output = ROOT/'reports/d66_source_gates_v1.json'
    panel_output = ROOT/'inputs/d66_local_cohort/evaluation20-candidates-v1.json'
    if output.exists() or panel_output.exists():
        raise FileExistsError('Source-gate summary or review panel already frozen')
    candidates = {c['chart_id']: c for key in ('panel', 'extra')
                  for song in data[key]['records'] for c in song['candidate_charts']}
    native = {r['chart_id']: r for key in ('native', 'extra_native') for r in data[key]['records']}
    canonical = {r['chart_id']: r for r in data['canonical']['records']+data['supplement']['paired_charts']}
    clock = {r['chart_id']: r for r in data['clock']['records']}
    audios = {r['audio_sha256']: r for r in data['clock']['audio']}
    # The transport report independently resolves the four BOM variants; require
    # all original mania records to pass either the first check or that proof.
    mania = {r['chart_id']: r['passed'] for r in data['supplement']['mania']}
    transport = data['mania']
    if not transport['complete'] or transport['base_report_sha256'] != sha(ROOT/paths['supplement']):
        raise ValueError('Mania text transport proof is incomplete or stale')
    for r in transport['additional_records']:
        mania[r['chart_id']] = r['passed']
    eligible, excluded = [], []
    for cid, c in candidates.items():
        reasons = []
        if not canonical[cid]['canonical_integrity_passed']:
            reasons.append('canonical_integrity')
        if c['mode'] == 'phira' and not native[cid]['native_static_candidate']:
            reasons.append('native_semantics_unresolved')
        if c['mode'] == 'mania_4k' and not mania[cid]:
            reasons.append('native_source_identity')
        cl = clock.get(cid)
        if (cl is None or not cl['duration_matches_rounded_manifest']
                or cl['native_bounds']['count'] or cl['model_bounds']['count']):
            reasons.append('decoded_bounds_or_duration')
        record = dict(c, source_screen_passed=not reasons, training_ready=False,
            audio_alignment_verified=False, feature_verified=False,
            model_rhythm_sha256=canonical[cid]['model_rhythm_sha256'])
        if reasons:
            excluded.append(dict(record, reasons=reasons))
        else:
            eligible.append(record)
    components = defaultdict(set)
    for c in eligible:
        components[c['component']].add(c['split'])
    if any(len(v) != 1 for v in components.values()):
        raise ValueError('Source-screened song components cross splits')

    # Preserve original validation rank inside fine strata, then round-robin the
    # strata. No model prediction or matching score participates in selection.
    by_id = {c['chart_id']: c for c in eligible}
    bins = defaultdict(list)
    for song in data['panel']['records']:
        if song['key'] != 'val/phira':
            continue
        choices = [by_id[c['chart_id']] for c in song['candidate_charts'] if c['chart_id'] in by_id]
        if choices:
            c = choices[0]
            bins[c['stratum']].append(dict(c, raw_audio_path=audios[c['audio_sha256']]['raw_path']))
    selected = []
    while bins and len(selected) < 20:
        for key in sorted(tuple(bins)):
            selected.append(bins[key].pop(0))
            if not bins[key]:
                del bins[key]
            if len(selected) == 20:
                break
    if len(selected) != 20 or len({c['component'] for c in selected}) != 20:
        raise ValueError('Twenty independent source-screened validation songs unavailable')

    manifest_path = ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    rows = {r['chart_id']: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    files = {r['path']: r for r in data['receipt']['files']}
    fields = MODE_FIELDS['phira']
    support = {f: dict(songs=set(), scopes=0, isolated_songs=set(), isolated_scopes=0, examples=[]) for f in fields}
    pair_songs, labels = set(), {}
    for group in data['pairs']['groups']:
        distinct = {}
        for c in group['charts']:
            cid = c['chart_id']
            if cid in by_id:
                distinct.setdefault(by_id[cid]['model_rhythm_sha256'], cid)
        members = list(distinct.values())
        if len(members) < 2:
            continue
        component = by_id[members[0]]['component']
        pair_songs.add(component)
        for cid in members:
            if cid in labels:
                continue
            row = rows[cid]
            if row['split'] != 'train':
                raise ValueError('Held-out chart cannot supply training pair labels')
            entry = files[f"canon/phira/{row['native_id']}.json"]
            path = ROOT/entry['local']
            if sha(path) != entry['sha256']:
                raise ValueError('Pair canonical changed')
            chart = chart_from_canon(json.loads(path.read_text()), model_shift_us(row, True))
            view = build_model_view('phira', (), chart.events)
            if view.quarantine:
                raise ValueError('Previously passed pair became quarantined')
            events = tuple(e for group in view.target_groups for e in group.events)
            labels[cid] = [training_labels('phira', events, (start, start+8_000_000))
                for start in range(0, int(row['audio_sec']*1e6)//8_000_000*8_000_000, 8_000_000)]
        for a, b in combinations(members, 2):
            for i, ((ba, va), (bb, vb)) in enumerate(zip(labels[a], labels[b])):
                for field in fields:
                    if field not in ba or field not in bb or in_band(field, va[field], bb[field]) or in_band(field, vb[field], ba[field]):
                        continue
                    item = support[field]
                    item['songs'].add(component); item['scopes'] += 1
                    # This controls only measured descriptor bands, not all human
                    # style attributes or the exact values within each band.
                    isolated = all(k in ba and k in bb and ba[k] == bb[k] for k in fields if k != field)
                    if isolated:
                        item['isolated_songs'].add(component); item['isolated_scopes'] += 1
                    if len(item['examples']) < 3:
                        item['examples'].append(dict(charts=[a, b], scope_us=[i*8_000_000, (i+1)*8_000_000],
                            bands=[ba[field], bb[field]], other_descriptor_bands_equal=isolated))
    identities = {p: sha(ROOT/p) for p in paths.values()}
    identities[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    for p in ('scripts/summarize_d66_source_gates.py', 'beatfcn/train/event_conditions.py',
              'beatfcn/events/controls.py', 'beatfcn/train/data.py'):
        identities[p] = sha(ROOT/p)
    report = dict(id='D66-source-gates-v1', source_sha256=identities, training_ready=False,
        optimizer_updates=0, cloud_compute_calls=0, eligible=eligible, unresolved=excluded,
        source_screened_charts=len(eligible), excluded_chart_reasons=dict(Counter(r for c in excluded for r in c['reasons'])),
        source_screened_song_counts={key: len({c['component'] for c in eligible if c['split']+'/'+c['mode'] == key})
            for key in ('train/phira', 'train/mania_4k', 'val/phira', 'val/mania_4k')},
        distinct_pair_songs=len(pair_songs),
        paired_support={f: dict(songs=len(v['songs']), scopes=v['scopes'],
            other_descriptor_bands_equal_songs=len(v['isolated_songs']),
            other_descriptor_bands_equal_scopes=v['isolated_scopes'], examples=v['examples']) for f, v in support.items()},
        limitations=['Source screen is not complete training admission or native playtesting.',
            'Acoustic alignment, feature values/tails and final request tape remain separate.',
            'Matching the other descriptor bands is not full causal isolation of human style.',
            'Unresolved source objects are preserved; no chart notes were changed.'])
    panel = dict(id='D66-evaluation20-source-candidates-v1', source_sha256=identities,
        generated_charts=0, training_ready=False, feature_verified=False,
        task='Twenty complete songs; music and human charts for independent per-chart review.',
        records=selected, request_tape_frozen=False,
        hidden_target_policy='Human evaluation targets do not supply from-scratch conditions or references.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    panel_output.write_text(json.dumps(panel, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('source_screened_charts', 'source_screened_song_counts',
        'excluded_chart_reasons', 'distinct_pair_songs', 'paired_support')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

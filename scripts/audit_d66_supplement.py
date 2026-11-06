"""Native mania identity and all frozen Phira paired-target support, on CPU."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

from beatfcn.events.canonical import Event
from beatfcn.events.controls import MODE_FIELDS
from beatfcn.events.osu_mania import parse_osu
from beatfcn.train.event_conditions import in_band, training_labels
from beatfcn.train.local_canonical_audit import audit_document

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan_path = ROOT/'bench/rhythm-v2/d66-supplement-audit-v1.json'
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen audit input changed: '+path)
    receipts = [json.loads((ROOT/p).read_text()) for p in plan['receipts']]
    if any(not r['complete'] or r['failures'] for r in receipts):
        raise ValueError('Complete source receipts required')
    files = {r['path']: r for receipt in receipts for r in receipt['files']}

    def read(name):
        entry = files[name]
        path = ROOT/entry['local']
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Downloaded source identity changed: '+name)
        return raw

    frozen = [json.loads(l) for l in (ROOT/plan['selected_rows']).read_text().splitlines()]
    all_rows = {r['chart_id']: r for r in map(json.loads, (ROOT/plan['manifest']).read_text().splitlines())}
    catalog = {str(b['id']): b for s in map(json.loads, (ROOT/plan['mania_catalog']).read_text().splitlines()) for b in s['beatmaps']}
    pairs = json.loads((ROOT/plan['pairs']).read_text())
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, training_ready=False, training_updates=0, cloud_compute_calls=0,
                  plan_sha256=sha(plan_path), source_sha256={p: sha(ROOT/p) for p in
                      ['scripts/audit_d66_supplement.py', 'beatfcn/train/local_canonical_audit.py',
                       'beatfcn/events/osu_mania.py', 'beatfcn/events/canonical.py',
                       'beatfcn/train/event_conditions.py', 'beatfcn/events/controls.py']},
                  mania=[], paired_charts=[], paired_support={})
    for row in frozen:
        if row['source'] != 'osu':
            continue
        raw = read(f"raw/osu/charts/{row['native_id']}.osu")
        doc = json.loads(read(f"canon/osu/{row['native_id']}.json"))
        parsed = parse_osu(raw)
        # The dataset stores text with LF. Only a newline encoding conversion is
        # allowed for matching the official checksum; no metadata/note edits.
        variants = {'raw': raw, 'lf_to_crlf': raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')}
        checksum = catalog[row['native_id']].get('checksum')
        matching = [k for k, value in variants.items() if hashlib.md5(value).hexdigest() == checksum]
        expected = Counter(Event.from_json(e).signature() for e in doc['events'])
        checks = dict(source_sha_match=hashlib.sha256(raw).hexdigest() == doc['source_sha256'],
                      exact_event_multiset=Counter(e.signature() for e in parsed.events) == expected,
                      catalog_checksum_transport_equivalent=bool(matching),
                      normalized_parser_event_multiset=bool(matching) and
                          Counter(e.signature() for e in parse_osu(variants[matching[0]]).events) == expected,
                      no_parser_quarantine=not parsed.quarantined)
        report['mania'].append(dict(chart_id=row['chart_id'], checks=checks, passed=all(checks.values()),
                                    catalog_checksum=checksum, checksum_match_variants=matching,
                                    source_notes=len(parsed.events), audio_identity_verified=False,
                                    limitation='Original music bytes and decoded clock still require verification.'))
    controls, hashes = {}, {}
    for group in pairs['groups']:
        for candidate in group['charts']:
            row = all_rows[candidate['chart_id']]
            row_sha = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if row_sha != candidate['source_row_sha256'] or row['split'] != 'train':
                raise ValueError('Paired chart identity or split changed')
            doc = json.loads(read(f"canon/phira/{row['native_id']}.json"))
            audit, events = audit_document(doc, row)
            audit['stratum'] = candidate['stratum']
            report['paired_charts'].append(audit)
            if audit['canonical_integrity_passed']:
                cid = row['chart_id']; hashes[cid] = audit['model_rhythm_sha256']
                span = plan['scope_us']
                controls[cid] = [training_labels('phira', events, (t, t+span))
                                 for t in range(0, int(row['audio_sec']*1e6)//span*span, span)]
    supported = {f: dict(songs=set(), pairs=set(), scopes=0, examples=[]) for f in MODE_FIELDS['phira']}
    eligible_songs = set()
    duplicates = []
    for group in pairs['groups']:
        members, by_target = [], defaultdict(list)
        for candidate in group['charts']:
            cid = candidate['chart_id']
            if cid in controls:
                by_target[hashes[cid]].append(cid)
        for ids in by_target.values():
            members.append(ids[0])
            if len(ids) > 1:
                duplicates.append(ids)
        if len(members) > 1:
            eligible_songs.add(group['charts'][0]['component'])
        for a, b in combinations(members, 2):
            for index, ((ba, va), (bb, vb)) in enumerate(zip(controls[a], controls[b])):
                for field in supported:
                    if field not in ba or field not in bb or in_band(field, va[field], bb[field]) or in_band(field, vb[field], ba[field]):
                        continue
                    item = supported[field]
                    item['songs'].add(group['charts'][0]['component']); item['pairs'].add((a, b)); item['scopes'] += 1
                    if len(item['examples']) < 3:
                        item['examples'].append(dict(charts=[a, b], scope_us=[index*plan['scope_us'], (index+1)*plan['scope_us']],
                                                     bands=[ba[field], bb[field]], actual=[va[field], vb[field]]))
    report['paired_support'] = {f: dict(songs=len(v['songs']), song_components=sorted(v['songs']),
        chart_pairs=len(v['pairs']), scopes=v['scopes'], examples=v['examples']) for f, v in supported.items()}
    report['duplicate_targets'] = duplicates
    report.update(complete=True, wall_s=time.monotonic()-started,
                  paired_songs_with_distinct_targets=len(eligible_songs),
                  limitations=['Paired support remains a canonical-only upper bound until native source, semantics and audio clocks pass.',
                               'Native mania source proof does not establish audio alignment or feature cache validity.',
                               'Original files are unchanged; checksum newline conversion is used only for identity verification.'])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(mania_charts=len(report['mania']), mania_passed=sum(r['passed'] for r in report['mania']),
        paired_charts=len(report['paired_charts']), paired_integrity_passed=sum(r['canonical_integrity_passed'] for r in report['paired_charts']),
        distinct_pair_songs=len(eligible_songs), pair_support_songs={f: v['songs'] for f, v in report['paired_support'].items()},
        duplicate_groups=len(duplicates), wall_s=report['wall_s'])))


if __name__ == '__main__':
    main()

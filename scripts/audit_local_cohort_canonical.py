"""Audit frozen canonical files and training-only paired condition support.

No inference, optimization, network calls, target repair or training admission.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.event_conditions import in_band, training_labels
from beatfcn.train.local_canonical_audit import audit_document

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    panel_path = ROOT/plan['panel']
    if sha(panel_path) != plan['panel_sha256']:
        raise ValueError('Candidate identity changed')
    panel = json.loads(panel_path.read_text())
    row_path = ROOT/panel['source_rows']
    if sha(row_path) != panel['source_rows_sha256']:
        raise ValueError('Frozen manifest rows changed')
    receipt_path = ROOT/plan['receipt']
    receipt = json.loads(receipt_path.read_text())
    transfer_plan = ROOT/plan['transfer_plan']
    if (not receipt['complete'] or receipt['failures']
            or sha(transfer_plan) != receipt['plan_sha256']):
        raise ValueError('Complete unchanged transfer receipt required')
    files = {r['path']: r for r in receipt['files']}
    rows = {r['chart_id']: r for r in map(json.loads, row_path.read_text().splitlines())}
    if len(rows) != panel['candidate_charts'] or len(files) != len(rows):
        raise ValueError('Candidate/transfer cardinality differs')
    candidates = {r['chart_id']: r for song in panel['records'] for r in song['candidate_charts']}
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, training_ready=False,
                  optimizer_updates=0, new_cloud_compute_calls=0,
                  plan_sha256=sha(plan_path), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                      (panel_path, row_path, receipt_path, transfer_plan, Path(__file__).resolve(),
                       ROOT/'beatfcn/train/local_canonical_audit.py',
                       ROOT/'beatfcn/train/event_conditions.py', ROOT/'beatfcn/events/controls.py',
                       ROOT/'beatfcn/events/canonical.py', ROOT/'beatfcn/train/data.py')},
                  records=[], duplicate_model_targets=[], paired_support={})

    def save():
        report['wall_s'] = time.monotonic()-started
        temp = output.with_suffix('.partial')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    labels, dedup = {}, defaultdict(list)
    for cid, row in sorted(rows.items()):
        if time.monotonic()-started > plan['timeout_s']:
            save()
            raise TimeoutError('Canonical audit time limit reached')
        source = files[f"canon/{row['source']}/{row['native_id']}.json"]
        path = ROOT/source['local']
        if not path.resolve().is_relative_to((ROOT/'inputs/d66_local_cohort/source').resolve()):
            raise ValueError('Receipt file escapes candidate source cache')
        if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
            raise ValueError(f'Canonical file differs from receipt: {cid}')
        row_hash = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if row_hash != candidates[cid]['row_sha256']:
            raise ValueError(f'Candidate row changed: {cid}')
        item = dict(chart_id=cid, mode=row['mode'], split=row['split'],
                    stratum=candidates[cid]['stratum'], canonical_integrity_passed=False)
        try:
            result, events = audit_document(json.loads(path.read_text()), row)
            item.update(result, stratum=candidates[cid]['stratum'])
            dedup[(row['audio_sha256'], row['mode'], result['model_rhythm_sha256'])].append(cid)
            # Labels are support diagnostics. Only training songs enter the pair
            # report; validation targets never become requested user conditions.
            if result['canonical_integrity_passed'] and row['split'] == 'train':
                span = plan['scope_us']
                labels[cid] = [training_labels(row['mode'], events, (start, start+span))
                               for start in range(0, int(row['audio_sec']*1e6)//span*span, span)]
                item['full_scopes'] = len(labels[cid])
                item['unscored_tail_us'] = int(row['audio_sec']*1e6) % span
                item['condition_band_counts'] = {
                    field: dict(Counter(json.dumps(b[field]) if field in b else 'undefined_or_out_of_table'
                                        for b, _ in labels[cid])) for field in MODE_FIELDS[row['mode']]}
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
            item.update(error=f'{type(exc).__name__}: {exc}', canonical_integrity_passed=False)
            labels.pop(cid, None)
        report['records'].append(item)
        if len(report['records']) % 100 == 0:
            save()
    report['duplicate_model_targets'] = [ids for ids in dedup.values() if len(ids) > 1]
    by_id = {r['chart_id']: r for r in report['records']}
    support = {mode: {field: dict(songs=set(), chart_pairs=set(), scopes=0, examples=[])
                      for field in MODE_FIELDS[mode]} for mode in MODE_FIELDS}
    eligible_pair_songs = Counter()
    for song in panel['records']:
        if not song['key'].startswith('train/'):
            continue
        mode = song['key'].split('/')[1]
        # One representative per complete rhythm target; equal-score aliases are
        # not independent paired supervision. Preserve the duplicate report.
        members, seen = [], set()
        for candidate in song['candidate_charts']:
            cid = candidate['chart_id']
            digest = by_id[cid].get('model_rhythm_sha256')
            if cid in labels and digest not in seen:
                members.append(cid); seen.add(digest)
        if len(members) > 1:
            eligible_pair_songs[mode] += 1
        for a, b in combinations(members, 2):
            if rows[a]['audio_sha256'] != rows[b]['audio_sha256']:
                raise ValueError('Pair audio identity differs')
            for index, ((ba, va), (bb, vb)) in enumerate(zip(labels[a], labels[b])):
                for field in MODE_FIELDS[mode]:
                    if (field not in ba or field not in bb
                            or in_band(field, va[field], bb[field]) or in_band(field, vb[field], ba[field])):
                        continue
                    value = support[mode][field]
                    value['songs'].add(song['component']); value['chart_pairs'].add((a, b)); value['scopes'] += 1
                    if len(value['examples']) < 3:
                        value['examples'].append(dict(charts=[a, b],
                            scope_us=[index*plan['scope_us'], (index+1)*plan['scope_us']],
                            actual=[va[field], vb[field]], bands=[ba[field], bb[field]]))
    report['paired_support'] = {
        mode: {field: dict(songs=len(v['songs']), chart_pairs=len(v['chart_pairs']), scopes=v['scopes'],
                          song_components=sorted(v['songs']), examples=v['examples']) for field, v in fields.items()}
        for mode, fields in support.items()}
    report['pair_candidate_songs_after_canonical_checks_and_target_dedup'] = dict(eligible_pair_songs)
    report['summary'] = {}
    for key in sorted({f"{r['split']}/{r['mode']}" for r in report['records']}):
        subset = [r for r in report['records'] if f"{r['split']}/{r['mode']}" == key]
        report['summary'][key] = dict(charts=len(subset), integrity_passed=sum(r['canonical_integrity_passed'] for r in subset),
            playable_notes=sum(r.get('playable_notes', 0) for r in subset), fakes=sum(r.get('fake_notes', 0) for r in subset),
            errors=dict(Counter(r['error'] for r in subset if 'error' in r)),
            issue_charts=dict(Counter(k for r in subset for k in r.get('issues', {}))),
            clock_status=dict(Counter(r.get('internal_clock', {}).get('status', 'not_checked') for r in subset)))
    report.update(complete=True, limitations=[
        'Canonical integrity is not native semantics, renderer parity or independently verified audio alignment.',
        'Pair counts remain candidate upper bounds until source/clock/features pass; supports do not prove learned response.',
        'Only complete non-overlapping 8-second training scopes were counted; empty and out-of-table labels remain missing.',
        'All playable multiplicities and explicit fake sidecars remain unchanged; no artistic drag filtering.',
        'No source/audio features were loaded into an optimizer; no baseline promotion.'])
    save()
    print(json.dumps(dict(complete=report['complete'], summary=report['summary'],
        paired_support={m: {f: {k: v[k] for k in ('songs', 'chart_pairs', 'scopes')}
            for f, v in fs.items()} for m, fs in report['paired_support'].items()},
        duplicate_groups=len(report['duplicate_model_targets']), wall_s=report['wall_s']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

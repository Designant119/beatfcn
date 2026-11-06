"""Audit whether package offsets inflate the expanded prefix-pair inventory."""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/'reports/d97_reference_support_clocks_v1.json'
    if output.exists():
        raise FileExistsError(output)
    paths = ['reports/d97_expanded_reference_support_v1.json',
             'reports/d94_expanded_target_tape_v1.json',
             'reports/d66_native_audit_v2.json', 'reports/d66_pair_native_audit_v1.json']
    audit, tape, *native_reports = [json.loads((ROOT/p).read_text()) for p in paths]
    sources = {p:sha(ROOT/p) for p in paths}
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    native = {r['chart_id']: r for report in native_reports for r in report['records']}
    charts = {c['chart_id']:c for c in tape['charts']}
    groups = defaultdict(list)
    for c in audit['cases']:
        if c['mode'] == 'phira':
            groups[(c['split'], c['audio_sha256'], tuple(c['target_scope_us']),
                    json.dumps(c['request_identity'], sort_keys=True))].append(c)
    cache = {}

    def events(cid, lo, hi, shift=0):
        if cid not in cache:
            chart = charts[cid]
            path = ROOT/chart['target_path']
            if sha(path) != chart['target_sha256']:
                raise ValueError('Frozen target changed')
            sources[chart['target_path']] = chart['target_sha256']
            with np.load(path, allow_pickle=False) as data:
                cache[cid] = list(zip(*(data[k].tolist() for k in ('onset_us','end_us','kind'))))
        return Counter((h+shift,t+shift,k) for h,t,k in cache[cid] if lo <= h+shift < hi)

    records = []
    for group in groups.values():
        for a,b in combinations(group,2):
            if not (a['reference_sha256'] != b['reference_sha256']
                    and a['target_sha256'] != b['target_sha256']
                    and a['stratum'] == b['stratum'] and a['target_bands'] == b['target_bands']):
                continue
            ca,cb = a['chart_id'],b['chart_id']
            if not native[ca]['source_identity_passed'] or not native[cb]['source_identity_passed']:
                raise ValueError('Missing native source identity')
            offsets = [round(float(native[c].get('package_offset') or 0)*1e6) for c in (ca,cb)]
            shift = offsets[0]-offsets[1]
            start,end = a['target_scope_us']
            differences = {}
            for label,lo,hi in [('prefix',start-16_000_000,start),('target',start,end)]:
                first,second = events(ca,lo,hi),events(cb,lo,hi,shift)
                differences[label] = dict(shared=sum((first & second).values()),
                    first_only=sum((first-second).values()), second_only=sum((second-first).values()),
                    distinct=first != second)
            records.append(dict(first=ca,second=cb,component=a['component'],split=a['split'],
                scope_us=[start,end], package_offsets_us=offsets, diagnostic_second_shift_us=shift,
                **differences, survives=all(d['distinct'] for d in differences.values())))
    report = dict(id='D97-expanded-reference-clock-confounds-v1', complete=True, records=records,
        original_strict_pair_windows=len(records), original_songs=len({r['component'] for r in records}),
        surviving_pair_windows=sum(r['survives'] for r in records),
        surviving_songs=len({r['component'] for r in records if r['survives']}),
        removed_as_clock_only_or_identical_prefix_target=sum(not r['survives'] for r in records),
        source_sha256=sources, label_changes=0, optimizer_updates=0, new_cloud_compute_calls=0,
        limitations=['Known package offset normalization is diagnostic only; no native timestamps were changed.',
            'Survival means distinct raw rhythm remains, not a meaningful style effect or independent author.',
            'Internal chart clocks and approximate rhythm duplication are not certified by this audit.',
            'Hidden target attributes remain inventory evidence only, never generated user requests.'])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','source_sha256','limitations')}))


if __name__ == '__main__':
    main()

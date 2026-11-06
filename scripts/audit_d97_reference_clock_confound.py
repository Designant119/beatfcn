"""Separate a known package-offset difference from actual rhythm differences."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from beatfcn.events.phira import parse_phira
from scripts.audit_phira_native_content import read_chart

ROOT = Path(__file__).resolve().parents[1]


def multiset(events, start=None, end=None):
    return Counter((e.onset_us, e.end_us, e.type) for e in events
                   if start is None or start <= e.onset_us < end)


def difference(a, b):
    return dict(shared=sum((a & b).values()), first_only=sum((a-b).values()),
                second_only=sum((b-a).values()), distinct=a != b)


def main():
    output = ROOT/'reports/d97_reference_clock_confound_v1.json'
    if output.exists():
        raise FileExistsError(output)
    audit_path = ROOT/'reports/d97_reference_restoration_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    charts, offsets, sources = [], [], {}
    for cid in (8758, 13082):
        path = ROOT/f'inputs/d97_reference_restoration_v1/{cid}.zip'
        info, raw, _ = read_chart(path)
        charts.append(parse_phira(raw, info_offset_sec=info.get('offset')))
        offsets.append(round(float(info.get('offset') or 0)*1e6))
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    shift = offsets[0]-offsets[1]
    normalized = [replace(e, onset_us=e.onset_us+shift, end_us=e.end_us+shift)
                  for e in charts[1].events]
    windows = []
    for w in audit['windows']:
        if not (w['distinct_prefix_and_target'] and w['equal_carry']
                and w['all_target_bands_equal'] and w['visible_support']):
            continue
        start, end = w['start_us'], w['end_us']
        prefix = difference(multiset(charts[0].events, start-16_000_000, start),
                            multiset(normalized, start-16_000_000, start))
        target = difference(multiset(charts[0].events, start, end), multiset(normalized, start, end))
        windows.append(dict(start_us=start, end_us=end, prefix=prefix, target=target,
                            both_distinct_after_clock_diagnostic=prefix['distinct'] and target['distinct']))
    for path in (Path(__file__), audit_path, ROOT/'beatfcn/events/phira.py'):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(id='D97-reference-package-clock-confound-v1', complete=True,
        package_offset_us=offsets, diagnostic_second_shift_us=shift,
        original_rhythm=difference(multiset(charts[0].events), multiset(charts[1].events)),
        package_offset_normalized_diagnostic=difference(multiset(charts[0].events), multiset(normalized)),
        original_strict_inventory_windows=len(windows), windows=windows,
        surviving_distinct_prefix_and_target_windows=sum(w['both_distinct_after_clock_diagnostic'] for w in windows),
        source_sha256=sources, label_changes=0, optimizer_updates=0, added_admitted_validation_songs=0,
        conclusion='A four-millisecond package offset can make shared rhythms appear distinct. Diagnostic normalization changes no stored labels or requests; remaining differences and independent style support require separate evidence.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','windows')}))


if __name__ == '__main__':
    main()

"""D25 independent analytic clocks and bytewise unchanged v3 canonical evidence."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    before = json.loads((ROOT / 'reports/phigros_resource_admission.json').read_text())
    after = json.loads((ROOT / 'reports/phigros_resource_admission_v2.json').read_text())
    if not before['complete'] or not after['complete']:
        raise ValueError('both bounded audits must be complete')
    base = ROOT / 'inputs/data_scale_v2/phigros_resource_v1'
    current = {row['chart_id']: row for row in after['records']}
    unchanged, legacy = [], []
    mismatches = []
    for old in before['records']:
        new = current[old['chart_id']]
        if old['split'] != new['split'] or old['song_group'] != new['song_group']:
            mismatches.append(dict(chart_id=old['chart_id'], reason='split changed'))
        path = new['native_id'] + '.json'
        if old['status'] == 'parsed':
            a = (base / 'admission_v1/canon' / path).read_bytes()
            b = (base / 'admission_v2/canon' / path).read_bytes()
            if a != b:
                mismatches.append(dict(chart_id=old['chart_id'], reason='prior canonical bytes changed'))
            else:
                unchanged.append(dict(chart_id=old['chart_id'], sha256=hashlib.sha256(a).hexdigest()))
            continue
        if new['status'] != 'parsed':
            mismatches.append(dict(chart_id=new['chart_id'], reason='legacy still unparsed'))
            continue
        raw = json.loads((base / 'raw' / new['chart_path']).read_text(), parse_float=Decimal)
        canon = json.loads((base / 'admission_v2/canon' / path).read_text())
        if raw['formatVersion'] != 1 or canon['format_version'] != '1' or canon['source_format'] != 'phira_official_v1':
            raise ValueError('native version identity changed')
        expected = {}
        for line_index, line in enumerate(raw['judgeLineList']):
            seconds_per_tick = Fraction(60, 32) / Fraction(line['bpm'])
            for side in ('above', 'below'):
                for index, note in enumerate(line['notes' + side.title()]):
                    onset = Fraction(note['time']) * seconds_per_tick + Fraction(raw['offset'])
                    end = onset + (Fraction(note['holdTime']) * seconds_per_tick if note['type'] == 3 else 0)
                    # Independent rational reference: nearest microsecond, ties upward.
                    expected[f'pgr:{line_index}:{side}:{index}'] = (
                        math.floor(onset * 1_000_000 + Fraction(1, 2)),
                        math.floor(end * 1_000_000 + Fraction(1, 2)),
                        {1: 'tap', 2: 'drag', 3: 'hold', 4: 'flick'}[note['type']])
        actual = {event['event_id']: (event['onset_us'], event['end_us'], event['type']) for event in canon['events']}
        if actual != expected:
            mismatches.append(dict(chart_id=new['chart_id'], reason='analytic time/type/count mismatch'))
        legacy.append(dict(chart_id=new['chart_id'], source_chart_id=new['source_chart_id'],
                           native_notes=len(expected), canonical_notes=len(actual),
                           analytic_exact=actual == expected, structural_pass=new['structural_audit_pass'],
                           types=dict(Counter(value[2] for value in expected.values()))))
    report = dict(schema='beatfcn.phigros_v1_correctness_verification.v1', complete=True,
        passed=not mismatches and len(unchanged) == 642 and len(legacy) == 23,
        prior_v3_canonical_bytewise_unchanged=len(unchanged), recovered_v1_charts=len(legacy),
        recovered_v1_native_notes=sum(row['native_notes'] for row in legacy),
        recovered_v1_structural_pass=sum(row['structural_pass'] for row in legacy),
        mismatches=mismatches, legacy=legacy, unchanged=unchanged, final_summary=after['summary'],
        research='reports/phigros_legacy_format_research_2026-10-25.md',
        new_paid_compute_usd=0, model_updates=0, training_ready=False,
        limitation='Parser correctness only; source clock calibration, visibility, cross-source content dedup and frozen features remain pending.')
    (ROOT / 'reports/d25_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'prior_v3_canonical_bytewise_unchanged',
        'recovered_v1_charts', 'recovered_v1_native_notes', 'recovered_v1_structural_pass', 'mismatches', 'final_summary')}, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit('D25 compatibility evidence failed; do not promote parser patch')


if __name__ == '__main__':
    main()

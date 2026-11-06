"""Finish native checksum proof for BOM-stripped text, without changing files."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.canonical import Event
from beatfcn.events.osu_mania import parse_osu

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'reports/d66_supplement_audit_v1.json'
    base = json.loads(source.read_text())
    output = ROOT/'reports/d66_mania_text_transport_v2.json'
    if output.exists():
        raise FileExistsError(output)
    corrected = []
    for item in base['mania']:
        if item['passed']:
            continue
        if any(not v for k, v in item['checks'].items()
               if k not in ('catalog_checksum_transport_equivalent', 'normalized_parser_event_multiset')):
            raise ValueError('This verifier cannot repair a source/semantic mismatch')
        native_id = item['chart_id'].split(':')[1]
        path = ROOT/f'inputs/d66_local_cohort/source/raw/osu/charts/{native_id}.osu'
        raw = path.read_bytes()
        canonical = json.loads((ROOT/f'inputs/d66_local_cohort/source/canon/osu/{native_id}.json').read_text())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != canonical['source_sha256']:
            raise ValueError('Original source changed')
        transported = b'\xef\xbb\xbf'+raw.removeprefix(b'\xef\xbb\xbf').replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        checksum_match = hashlib.md5(transported).hexdigest() == item['catalog_checksum']
        expected = Counter(Event.from_json(e).signature() for e in canonical['events'])
        semantic_match = Counter(e.signature() for e in parse_osu(transported).events) == expected
        corrected.append(dict(chart_id=item['chart_id'], original_sha256=digest,
                              text_transport='UTF-8 BOM plus CRLF; no content or trailing-line changes',
                              catalog_checksum=item['catalog_checksum'], checksum_match=checksum_match,
                              full_event_multiset_unchanged=semantic_match, passed=checksum_match and semantic_match))
    report = dict(complete=True, base_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  prior_verified=sum(r['passed'] for r in base['mania']), additional_records=corrected,
                  native_identity_passed_total=sum(r['passed'] for r in base['mania'])+sum(r['passed'] for r in corrected),
                  original_files_changed=False, training_ready=False,
                  limitation='Music identity, decoded clock and feature checks remain separate.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('prior_verified', 'native_identity_passed_total', 'original_files_changed')}))


if __name__ == '__main__':
    main()

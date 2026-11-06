"""Count real condition-band exposure across the complete frozen training tape."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from beatfcn.events import controls as C
from beatfcn.model import features as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    plan_path = parser.parse_args().plan
    plan = json.loads(plan_path.read_text())
    for filename, expected in plan['input_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen input changed: {filename}')
    manifest = json.loads((ROOT / plan['manifest']).read_text())
    panel = json.loads((ROOT / plan['panel']).read_text())
    heldout = {x['row']['audio_sha256'] for x in panel['cases']}
    batches = [json.loads(line) for line in (ROOT / plan['samples']).read_text().splitlines()]
    if len(batches) != manifest['batches'] or any(len(b) != manifest['micro_batch'] for b in batches):
        raise ValueError('complete tape exposure count differs')
    counts, complete_counts = Counter(), Counter()
    audio = defaultdict(set)
    complete_cores = Counter()
    for batch in batches:
        for row in batch:
            if row['song'] in heldout:
                raise ValueError('held-out benchmark audio in training exposure')
            if row['kind'] != 'density_window':
                continue
            labels = row['full_window_controls']
            if len(labels) != len(F.CONTROL_FIELDS):
                raise ValueError('control label columns differ')
            complete = row['density_scope_us'] == [row['c0_us'], row['c0_us'] + F.CORE_US]
            if complete:
                complete_cores[row['mode']] += 1
            for field, band in zip(F.CONTROL_FIELDS, labels, strict=True):
                if type(band) is not int or not -1 <= band < len(C.BAND_EDGES[field])-1:
                    raise ValueError('condition band outside defined vocabulary')
                if field not in C.MODE_FIELDS[row['mode']] and band != -1:
                    raise ValueError('inapplicable condition present')
                key = f'{row["mode"]}:{field}:{band}'
                counts[key] += 1
                if complete and band >= 0:
                    complete_counts[key] += 1
                    audio[key].add(row['song'])
    if dict(counts) != manifest['full_window_bands']:
        raise ValueError('full tape label counts differ from preparation manifest')
    supported, fields = {}, {}
    for mode in F.MODES:
        supported[mode] = {}
        for field in C.MODE_FIELDS[mode]:
            edges = C.BAND_EDGES[field]
            rows = [dict(index=band, band=[edges[band], edges[band+1]],
                complete_cores=complete_counts[f'{mode}:{field}:{band}'],
                distinct_audio=len(audio[f'{mode}:{field}:{band}'])) for band in range(len(edges)-1)]
            selected = [r['index'] for r in rows if r['distinct_audio'] >= plan['minimum_distinct_audio_per_band']]
            supported[mode][field] = selected
            requested = [edges.index(band[0]) for band in panel['request_bands'][field]]
            fields[f'{mode}:{field}'] = dict(bands=rows, supported_bands=selected,
                diagnostic_request_indices=requested, unsupported_diagnostic_requests=[i for i in requested if i not in selected])
    result = dict(schema='beatfcn.condition_binding_support.v1', complete=True,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        actual_batches=len(batches), actual_cores=sum(map(len,batches)), complete_window_cores=dict(complete_cores),
        preparation_counts_match=True, support=supported, fields=fields,
        cloud_calls=0, optimizer_updates=0, model_promotion=False,
        limitation='Observed whole-window condition feasibility only. This log lacks masks, so it does not prove these cores are eligible for the auxiliary pair; dependency eligibility is audited separately. Native labels are checked against stored preparation counts, not reconstructed from every original chart.')
    (ROOT / 'reports/r08_condition_binding_support.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    missing = {key: value['unsupported_diagnostic_requests'] for key, value in fields.items() if value['unsupported_diagnostic_requests']}
    print(json.dumps(dict(actual_cores=result['actual_cores'], complete_window_cores=dict(complete_cores),
        unsupported_diagnostic_requests=missing, preparation_counts_match=True)))


if __name__ == '__main__':
    main()

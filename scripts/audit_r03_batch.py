"""Reconstruct the prepared batch's labels from canonical and quantized events."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from beatfcn.events import controls as C
from beatfcn.events.canonical import build_model_view
from beatfcn.events.requests import resolve
from beatfcn.model.batch import unpack_batch
from beatfcn.train.data import chart_from_canon, model_shift_us, dup_only, check_stacks
from beatfcn.train.window_controls import encode_window_controls

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'reports/r03_window_raw'
    manifest = json.loads((root / 'manifest.json').read_text())
    payload = (root / 'probe-batch.pt').read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['probe_batch_sha256']:
        raise ValueError('batch fingerprint mismatch')
    bt = unpack_batch(torch.load(root / 'probe-batch.pt', map_location='cpu', weights_only=False))
    records = []
    for i, (row, fingerprint) in enumerate(zip(manifest['probe_rows'], manifest['probe_canon_sha256'])):
        raw = (root / 'canon' / row['source'] / f'{row["native_id"]}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != fingerprint:
            raise ValueError('canonical fingerprint mismatch')
        doc = json.loads(raw)
        if doc['audio_sha256'] != row['audio_sha256']:
            raise ValueError('canonical/audio mismatch')
        chart = chart_from_canon(doc, model_shift_us(row, True))
        if dup_only(row):
            check_stacks(chart.events)
        p = bt['_provenance'][i]; a, b = p['density_scope_us']; duration = round(row['audio_sec'] * 1e6)
        regions = ([(0, a, 'complete')] if a else []) + [(a, b, 'editable')]
        if b < duration:
            regions.append((b, duration, 'complete'))
        observed, hidden, _ = resolve(chart.events, regions)
        view = build_model_view(row['mode'], observed, hidden)
        if not view.ok:
            raise ValueError('model view quarantined')
        quantized = [e for g in view.target_groups for e in g.events]
        native_values = C.raw_controls(row['mode'], chart.events, [(a, b)])
        model_values = C.raw_controls(row['mode'], list(observed) + quantized, [(a, b)])
        native = encode_window_controls(row['mode'], native_values)
        quant = encode_window_controls(row['mode'], model_values)
        labels = bt['full_window_controls'][i].tolist()
        type_counts = Counter(e.type for e in quantized)
        record = dict(chart_id=row['chart_id'], mode=row['mode'], scope_us=[a, b],
            native_bands=native, quantized_bands=quant, stored_bands=labels,
            native_labels_agree=native == labels, model_labels_agree=quant == labels,
            targets=len(quantized), declared_targets=p['notes'], target_types=dict(type_counts),
            target_count_agrees=len(quantized) == p['notes'],
            source_agrees=p['song'] == bt['_audio']['keys'][i] == row['audio_sha256'])
        records.append(record)
    passed = len(records) == 8 and all(all(r[key] for key in
        ('native_labels_agree', 'model_labels_agree', 'target_count_agrees', 'source_agrees')) for r in records)
    result = dict(schema='beatfcn.r03_batch_alignment.v1', passed=passed, records=records,
                  scope='Selected actual tape batch, native and 1ms model targets; not an all-corpus assertion.')
    (ROOT / 'reports/r03_window_batch_alignment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, cores=len(records), targets=sum(r['targets'] for r in records),
                         mismatches=[r['chart_id'] for r in records if not r['model_labels_agree']])))
    if not passed:
        raise SystemExit('alignment gate failed; do not start formal training')


if __name__ == '__main__':
    main()

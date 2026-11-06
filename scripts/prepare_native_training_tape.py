"""Build a new native-clock, strict-drag-v2 request tape from hashed canonicals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.canonical import build_model_view, TYPES
from beatfcn.train.data import chart_from_canon
from beatfcn.train.event_density_drag_filter import RULE, dense_event_drag_mask
from beatfcn.train.local_canonical_audit import rhythm_fingerprint
from scripts.prepare_local_request_tape import requests_for_chart

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_native_tape(root, tape_path, receipt_path, output_dir, *, chart_ids=None):
    root, output = Path(root), Path(output_dir)
    source = json.loads((root/tape_path).read_text())
    receipt = json.loads((root/receipt_path).read_text())
    if not source['complete'] or source['failures'] or not receipt['complete'] or receipt['failures']:
        raise ValueError('Source tape/canonical receipt is not admitted')
    selected = [c for c in source['charts'] if chart_ids is None or c['chart_id'] in chart_ids]
    if not selected or (chart_ids is not None and set(chart_ids) != {c['chart_id'] for c in selected}):
        raise ValueError('Requested charts are missing')
    files = {r['sha256']: r for r in receipt['files']}
    output.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, failures=[], clock_contract='native-v1',
        drag_policy_id=RULE['id'], source_tape_sha256=sha(root/tape_path),
        canonical_receipt_sha256=sha(root/receipt_path), charts=[], removals=[])
    try:
        for chart in selected:
            entry = files[chart['canonical_sha256']]
            path = root/entry['local']
            if sha(path) != chart['canonical_sha256']:
                raise ValueError('Canonical bytes changed')
            native = chart_from_canon(json.loads(path.read_text()))
            if native.mode != chart['mode']:
                raise ValueError('Canonical mode differs from admitted chart')
            removed, _ = dense_event_drag_mask([e.onset_us for e in native.events],
                [TYPES.index(e.type) for e in native.events])
            retained = tuple(e for e, drop in zip(native.events, removed) if not drop)
            view = build_model_view(native.mode, (), retained)
            events = tuple(e for g in view.target_groups for e in g.events)
            if view.quarantine or len(events) != len(retained):
                raise ValueError('Native model view is not lossless under the event codec')
            rows, tokens, targets = requests_for_chart(events, native.mode,
                chart['duration_us'], chart['split'])
            stem = chart['chart_id'].replace(':', '-')
            tp, kp = output/(stem+'-targets.npz'), output/(stem+'-tokens.npy')
            np.savez(tp, **targets)
            np.save(kp, tokens, allow_pickle=False)
            report['charts'].append(dict(chart, requests=rows,
                target_path=str(tp.relative_to(root)), token_path=str(kp.relative_to(root)),
                target_sha256=sha(tp), token_sha256=sha(kp),
                model_rhythm_sha256=rhythm_fingerprint(native.mode, events)))
            report['removals'].append(dict(chart_id=chart['chart_id'],
                raw_events=len(native.events), retained_events=len(events),
                removed_drag=int(removed.sum()), removed_indices=np.flatnonzero(removed).tolist(),
                types_before=dict(Counter(e.type for e in native.events)),
                types_after=dict(Counter(e.type for e in events))))
        report['complete'] = True
    except Exception as exc:
        report['failures'].append(str(exc))
        raise
    finally:
        (output/'tape.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tape', required=True)
    p.add_argument('--canonical-receipt', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepare_native_tape(root, a.tape, a.canonical_receipt, root/a.output)

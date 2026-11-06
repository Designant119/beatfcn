"""Read-only native-clock bounds and split audit over an existing local tape."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from beatfcn.events.canonical import TYPES
from beatfcn.train.data import chart_from_canon
from beatfcn.train.event_density_drag_filter import dense_event_drag_mask


def run():
    root = Path(__file__).resolve().parents[1]
    source = root/'reports/d94_expanded_target_tape_v1.json'
    receipt = root/'reports/d66_combined_canonical_receipt.json'
    tape = json.loads(source.read_text())
    files = {r['sha256']: r for r in json.loads(receipt.read_text())['files']}
    tick = time.monotonic()
    counts, failed, splits, valid = Counter(), [], defaultdict(set), []
    for chart in tape['charts']:
        if time.monotonic()-tick > 120:
            break
        counts['inspected'] += 1
        splits[('audio', chart['audio_sha256'])].add(chart['split'])
        splits[('component', chart['component'])].add(chart['split'])
        try:
            entry = files[chart['canonical_sha256']]
            path = root/entry['local']
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != chart['canonical_sha256']:
                raise ValueError('Canonical SHA mismatch')
            native = chart_from_canon(json.loads(data))
            bad = [e.event_id for e in native.events if e.onset_us < 0
                or e.onset_us >= chart['duration_us'] or e.end_us > chart['duration_us']]
            removed, _ = dense_event_drag_mask([e.onset_us for e in native.events],
                [TYPES.index(e.type) for e in native.events])
            counts['native_events'] += len(native.events)
            counts['removed_drag_v2'] += int(removed.sum())
            if bad:
                failed.append(dict(chart_id=chart['chart_id'], reason='native_audio_bounds',
                    objects=len(bad), examples=bad[:5]))
            else:
                valid.append(chart['chart_id'])
        except (ValueError, KeyError, OSError) as exc:
            failed.append(dict(chart_id=chart['chart_id'], reason=str(exc)))
    result = dict(scope='Local D94 subset; canonical bytes/native bounds/splits only; not full cloud corpus or codec/quality admission',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        canonical_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        counts=dict(counts), failures=failed, native_bounds_pass_chart_ids=valid,
        cross_split_keys=sum(len(v)>1 for v in splits.values()),
        complete=counts['inspected']==len(tape['charts']),
        wall_s=time.monotonic()-tick, files_modified=False, cloud_spend_usd=0)
    target = root/'reports/data-contract-v2-final/native-readiness-audit.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(dict(result['counts'], failure_charts=len(failed),
        cross_split_keys=result['cross_split_keys'], complete=result['complete']))


if __name__ == '__main__':
    run()

"""Validate frozen native targets and relative edits through the production codec."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.train.counterfactual_density import edit_events_to_multiplier
from beatfcn.train.local_request_panel import event_request


def audit_chart(job):
    chart, offset, directory, before_path = job
    target = Path(directory) / Path(chart['target_path']).name
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != chart['target_sha256']:
        raise ValueError(f'Target identity mismatch: {target}')
    spec = importlib.util.spec_from_file_location('before', before_path)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    counts = Counter()
    failures = []
    before_examples = []
    with np.load(target, allow_pickle=False) as values:
        for local_index, row in enumerate(chart['requests']):
            index = offset + local_index
            request = replace(event_request(row['request']), known_carry=())
            sl = slice(row['target_start'], row['target_stop'])
            events = [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l))
                      for h, t, k, l in zip(*(values[key][sl]
                      for key in ('onset_us', 'end_us', 'kind', 'lane')))]
            for rate in (None, .5, 1.5, 2.):
                counts['variants'] += 1
                for label, edit in [('before', before.edit_events_to_multiplier),
                                    ('after', edit_events_to_multiplier)]:
                    modified = events if rate is None else edit(events, request.start_us,
                        request.end_us, rate, request.mode, request.duration_us, seed=index)[0]
                    try:
                        encode_events(modified, request)
                        counts[label + '_passed'] += 1
                        if label == 'after' and rate in (1.5, 2.):
                            if Counter(events) - Counter(modified):
                                raise AssertionError('Original event changed during upsampling')
                    except (ValueError, AssertionError) as exc:
                        counts[label + '_failed'] += 1
                        example = dict(index=index, chart_id=chart['chart_id'], rate=rate,
                                       mode=chart['mode'], error=str(exc))
                        if label == 'after':
                            failures.append(example)
                        elif len(before_examples) < 2:
                            before_examples.append(example)
    return dict(counts=counts, failures=failures, before_examples=before_examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tape', required=True)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--before', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    tape = json.loads(Path(args.tape).read_text())
    jobs, offset = [], 0
    for chart in tape['charts']:
        jobs.append((chart, offset, args.targets, args.before))
        offset += len(chart['requests'])
    counts, failures, examples = Counter(), [], []
    with Pool(args.workers) as pool:
        for number, result in enumerate(pool.imap_unordered(audit_chart, jobs), 1):
            counts.update(result['counts'])
            failures.extend(result['failures'])
            examples.extend(result['before_examples'])
            if number % 50 == 0:
                print(json.dumps(dict(charts=number, counts=counts)), flush=True)
    result = dict(complete=True, charts=len(jobs), requests=offset, counts=counts,
        failures=failures, before_examples=examples[:30], elapsed_s=time.monotonic()-started,
        tape_sha256=hashlib.sha256(Path(args.tape).read_bytes()).hexdigest(),
        after_sha256=hashlib.sha256(Path('beatfcn/train/counterfactual_density.py').read_bytes()).hexdigest())
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(counts=counts, failures=len(failures), elapsed_s=result['elapsed_s'])))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())

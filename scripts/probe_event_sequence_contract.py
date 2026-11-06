"""R17 actual D32 input/sequence audit. No network or optimizer is constructed."""
import argparse
from collections import Counter
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_sequence import (
    FIELDS, EventSequenceState, SequenceSpec, encode_sequence, decode_sequence,
)
from beatfcn.model.group_canvas import decode_group_targets
from scripts.matched_acoustic_data import make_example, common_physical_targets, metadata_sha
from scripts.r09_local_data import ROOT, file_sha, verify_files


def stats(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    numbers = array.astype(np.float64, copy=False)
    finite = np.isfinite(numbers)
    valid = numbers[finite]
    return dict(shape=list(array.shape), dtype=str(array.dtype), nonfinite=int((~finite).sum()),
                min=float(valid.min()) if valid.size else None, max=float(valid.max()) if valid.size else None,
                mean=float(valid.mean()) if valid.size else None, std=float(valid.std()) if valid.size else None)


def spec_for(inputs):
    permitted = tuple(base + int(i) * 1000
                      for (base, _, _), mask in zip(inputs.blank.layout.blocks, inputs.allowed_heads)
                      for i in np.flatnonzero(mask.numpy()))
    types = tuple(t for t, allowed in zip(('tap', 'hold', 'flick', 'drag'), inputs.allowed_types) if allowed)
    return SequenceSpec(inputs.blank.layout, inputs.blank.mode, inputs.duration_ms * 1000,
                        copy_times=inputs.blank.copy_times, permitted_heads_us=permitted, allowed_types=types)


def audit(inputs, labels, identity):
    truth = decode_group_targets(labels)
    spec = spec_for(inputs)
    sequence = encode_sequence(truth, spec)
    if decode_sequence(sequence, spec) != truth:
        raise ValueError('real complete events do not roundtrip')
    measurements = {f.name: stats(getattr(inputs, f.name)) for f in fields(inputs)
                    if torch.is_tensor(getattr(inputs, f.name))}
    states = EventSequenceState(spec)
    syntax, a_occupancy, b_occupancy, legal_sizes = [], [], [], []
    prefix_checks = 0
    for decision in sequence:
        a = states.model_view(history_enabled=False)
        b = states.model_view(history_enabled=True)
        if a['completed_events'] or a['active_hold_total'] or a['occupancy'].any():
            raise ValueError('completed past payload leaked into A')
        if b['completed_events'] != tuple(states.events):
            raise ValueError('B history differs from strictly committed events')
        if not np.array_equal(a['syntax'], b['syntax']) or a['allowed_values'] != b['allowed_values']:
            raise ValueError('A/B grammar differs')
        # Perturb past completed lane/tail content in place, leaving only the
        # public grammar clock/position. A must be bitwise invariant.
        before = states.events
        states.events = [replace(e, lane=(e.lane+1)%4 if spec.mode == 'mania_4k' else -1,
                                end_us=spec.duration_us if e.type == 'hold' else e.end_us) for e in before]
        altered = states.model_view(history_enabled=False)
        states.events = before
        if (not np.array_equal(a['syntax'], altered['syntax']) or
                not np.array_equal(a['occupancy'], altered['occupancy']) or
                a['completed_events'] != altered['completed_events'] or
                a['allowed_values'] != altered['allowed_values']):
            raise ValueError('A contains a completed-event side channel')
        prefix_checks += 1
        syntax.append(a['syntax']);a_occupancy.append(a['occupancy']);b_occupancy.append(b['occupancy'])
        legal_sizes.append(len(a['allowed_values']))
        states.apply(decision)
    if not states.done or tuple(sorted(states.events)) != truth:
        raise ValueError('stepwise decode changed targets')
    measurements.update(sequence_decisions=stats([(FIELDS.index(d.field), d.value) for d in sequence]),
                        strict_prefix_syntax=stats(syntax), A_occupancy=stats(a_occupancy),
                        B_occupancy=stats(b_occupancy), legal_vocab_sizes=stats(legal_sizes),
                        target_events=stats([(e.onset_us, e.end_us, ('tap','hold','flick','drag').index(e.type), e.lane) for e in truth]))
    if any(m['nonfinite'] for m in measurements.values()):
        raise ValueError('nonfinite actual tensor')
    return dict(identity=identity, target_events=[asdict(e) for e in truth],
                sequence=[asdict(d) for d in sequence], decision_counts=dict(Counter(d.field for d in sequence)),
                sequence_length=len(sequence), event_count=len(truth), types=dict(Counter(e.type for e in truth)),
                tensor_stats=measurements, exact_roundtrip=True, strict_prefix_checks=prefix_checks,
                A_completed_attribute_invariance=True, non_acoustic_metadata_sha256=metadata_sha(inputs))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--plan', required=True)
    args = parser.parse_args();path = ROOT / args.plan;plan = json.loads(path.read_text())
    verify_files(plan)
    data_plan = json.loads((ROOT / plan['panel_plan']).read_text())
    manifest = json.loads((ROOT / plan['panel_manifest']).read_text())
    output = ROOT / plan['result']
    if output.exists():
        raise FileExistsError('preserve previous probe report')
    start = time.perf_counter()
    report = dict(id='R17-contract', complete=False, passed=False, optimizer_updates=0,
                  model_constructed=False, gradient_probe_passed=False, overfit_probe_passed=False,
                  model_promotion=False, training_ready=False, new_paid_compute_usd=0,
                  plan_sha256=file_sha(path), rows=[], physical_alignment=[])
    def save():
        report['wall_s'] = time.perf_counter() - start
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    save()
    try:
        torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1)
        for item in manifest['items']:
            original = make_example(item, data_plan)
            shifted = make_example(item, data_plan, translated=True)
            same, interval = common_physical_targets(original[1], shifted[1], original[2], shifted[2], data_plan)
            if not same:
                raise ValueError('physical targets differ across translated views')
            report['physical_alignment'].append(dict(panel_id=item['panel_id'], common_interval_us=interval, exact=True))
            for example in (original, shifted):
                row = audit(*example);report['rows'].append(row);save()
                print(json.dumps({k:row[k] for k in ('identity','event_count','sequence_length','exact_roundtrip')}, ensure_ascii=False), flush=True)
        for mode in ('mania_4k', 'phira'):
            signatures = {r['non_acoustic_metadata_sha256'] for r in report['rows'] if r['identity']['mode'] == mode}
            if signatures != {manifest['non_acoustic_metadata_sha256'][mode]}:
                raise ValueError('non-acoustic identity changed from D32')
        original = [r for r in report['rows'] if not r['identity']['translated']]
        types = Counter()
        for row in original:
            types.update({f"{row['identity']['mode']}:{k}": v for k, v in row['types'].items()})
        report['summary'] = dict(original_cores=len(original), translated_cores=len(report['rows'])-len(original),
                               original_events=sum(r['event_count'] for r in original), original_types=dict(types),
                               original_decisions=sum(r['sequence_length'] for r in original),
                               maximum_sequence_length=max(r['sequence_length'] for r in report['rows']),
                               strict_prefix_checks=sum(r['strict_prefix_checks'] for r in report['rows']),
                               exact_roundtrips=len(report['rows']))
        report.update(complete=True, passed=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}';save();raise
    save()
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

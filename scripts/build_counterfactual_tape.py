"""Build a counterfactual density tape from a v2 native tape.

Each source request becomes one free sample plus one sample per requested
heads/s, with the target edited to that density.  Curves are stored by chart id
so they do not depend on panel row order.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.train.counterfactual_density import edit_events_to_density
from beatfcn.train.local_request_panel import event_request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_events(target_path, start, stop):
    with np.load(ROOT/target_path, allow_pickle=False) as archive:
        return [RhythmEvent(int(onset), int(end), TYPES[int(kind)], int(lane))
                for onset, end, kind, lane in zip(
                    archive['onset_us'][start:stop], archive['end_us'][start:stop],
                    archive['kind'][start:stop], archive['lane'][start:stop])]


def write_variant(chart, request_row, events, output, name, rate, curves, scope):
    request = dict(request_row['request'])
    request['start_us'] = int(request['start_us'])
    request['end_us'] = int(request['end_us'])
    # Counterfactual windows are edited independently; carried holds from a
    # different source request must not constrain the new target grammar.
    request['known_carry'] = []
    tokens = encode_events(events, event_request(request))
    target_path = output/f'{name}-targets.npz'
    token_path = output/f'{name}-tokens.npy'
    np.savez(target_path,
        onset_us=np.asarray([event.onset_us for event in events], np.int64),
        end_us=np.asarray([event.end_us for event in events], np.int64),
        kind=np.asarray([TYPES.index(event.type) for event in events], np.int64),
        lane=np.asarray([event.lane for event in events], np.int64))
    np.save(token_path, np.asarray(tokens, np.int64), allow_pickle=False)
    variant = dict(chart_id=name, component=chart['component'], split=chart['split'],
        mode=chart['mode'], stratum=chart['stratum'], audio_sha256=chart['audio_sha256'],
        canonical_sha256=chart['canonical_sha256'], model_rhythm_sha256=chart['model_rhythm_sha256'],
        duration_us=chart['duration_us'], target_path=str(target_path.relative_to(ROOT)),
        token_path=str(token_path.relative_to(ROOT)), target_sha256=sha(target_path),
        token_sha256=sha(token_path),
        requests=[dict(request_row, request=request, target_start=0, target_stop=len(events),
            token_start=0, token_stop=len(tokens), tokens=len(tokens),
            event_count=len(events), types=dict(Counter(event.type for event in events)),
            controls=request_row.get('controls') or [[0.]*6]*8,
            bands=None, condition_source=scope,
            reference='unknown_region', scope_role=scope)])
    curves[name] = ([] if rate is None else
        [dict(start_us=request['start_us'], end_us=request['end_us'], density=float(rate))])
    return variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rates', default='0,1,2,3,4,5,6,7,8')
    parser.add_argument('--name', default='counterfactual-density-v1')
    args = parser.parse_args()
    source = json.loads((ROOT/args.tape).read_text())
    if not source.get('complete') or source.get('failures'):
        raise ValueError('Source tape is not complete')
    output = ROOT/args.output
    if output.exists():
        raise FileExistsError('Output directory already exists')
    output.mkdir(parents=True)
    rates = [float(value) for value in args.rates.split(',')]
    curves = {}
    charts = []
    for chart in source['charts']:
        for request_row in chart['requests']:
            start = int(request_row['request']['start_us'])
            end = int(request_row['request']['end_us'])
            original = load_events(chart['target_path'], request_row['target_start'],
                                   request_row['target_stop'])
            if not original and request_row['event_count'] == 0:
                continue
            base = chart['chart_id'].replace(':', '-')
            name = f'{base}-{start}-free'
            charts.append(write_variant(chart, request_row, original, output, name,
                                        None, curves, 'free'))
            for rate in rates:
                edited, actual = edit_events_to_density(original, start, end, rate,
                    chart['mode'], chart['duration_us'])
                if not edited and rate > 0:
                    continue
                name = f'{base}-{start}-d{rate:g}'
                charts.append(write_variant(chart, request_row, edited, output, name,
                                            rate, curves, 'counterfactual_curve'))
    tape = {key: source[key] for key in ('clock_contract', 'drag_policy_id',
        'source_tape_sha256', 'canonical_receipt_sha256')}
    tape.update(complete=True, failures=[], charts=charts,
        counterfactual=dict(rates=rates, seed=args.name))
    (output/'tape.json').write_text(json.dumps(tape, ensure_ascii=False, indent=1)+'\n')
    (output/'curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(dict(charts=len(charts), requests=sum(len(c['requests']) for c in charts),
        curves=len(curves), output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()

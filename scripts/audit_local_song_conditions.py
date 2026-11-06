"""CPU-only, paired condition interventions with no hidden chart attributes."""
import argparse
import json
from pathlib import Path
import time

import torch

from beatfcn.events.controls import raw_controls
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.train.event_conditions import encode_controls, in_band
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(path):
    plan = json.loads(path.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Frozen intervention input changed: ' + source)
    torch.set_num_threads(plan['cpu_threads'])
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    if saved['updates'] != plan['checkpoint_updates']:
        raise ValueError('Unexpected checkpoint exposure')
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    features = LocalFeatureStore(ROOT, plan['features'])
    if len({s['component'] for s in plan['songs']}) != len(plan['songs']):
        raise ValueError('Intervention songs must be independent')
    for song in plan['songs']:
        for source, expected in features.records[song['audio_sha256']]['sha256'].items():
            if sha(ROOT / source) != expected:
                raise ValueError('Admitted audio feature changed: ' + source)
    report = dict(id=plan['id'], plan_sha256=sha(path), complete=False, rows=[],
        checkpoint=plan['checkpoint'], checkpoint_sha256=sha(ROOT / plan['checkpoint']),
        checkpoint_updates=saved['updates'], optimizer_updates=0, new_cloud_compute_calls=0,
        backend='cpu', scope='Two single-field interventions on four independent audio-selected 8s cores; no human target, no reference, no teacher carry. Not whole-song or seven-field validation.')
    start = time.monotonic()
    deadline = start + plan['max_wall_s']
    write_json(output, report)
    try:
        for field in plan['fields']:
            for song in plan['songs']:
                a, b = song['scope_us']
                request = EventRequest('phira', a, b, features.records[song['audio_sha256']]['duration_us'])
                rows = []
                for band in (field['low'], field['high']):
                    controls = encode_controls('phira', {field['name']: band}, (a, b), origin_us=a)
                    inputs, provenance = request_inputs(features, song['audio_sha256'], request, controls)
                    rows.append(dict(inputs=inputs, sequence=teacher_sequence(encode_events([], request), request)))
                inputs, _ = collate(rows)
                inputs = pad_input_shapes(inputs, plan['input_shapes'])
                if any(not torch.equal(v[0], v[1]) for k, v in inputs.items() if k != 'controls'):
                    raise ValueError('A non-condition input changed within the intervention')
                predictions = generate_batch(model, inputs, [request, request], plan['token_cap'], deadline=deadline)
                values = [raw_controls('phira', canonical([RhythmEvent(**e) for e in p['events']], 'phira'), [(a, b)]) for p in predictions]
                complete = all(p['complete'] for p in predictions)
                realized = [v[field['name']] for v in values]
                delta = realized[1] - realized[0] if all(v is not None for v in realized) else None
                report['rows'].append(dict(chart_id=song['chart_id'], component=song['component'],
                    audio_sha256=song['audio_sha256'], scope_us=[a, b], field=field['name'],
                    low=field['low'], high=field['high'], complete=complete, predictions=predictions,
                    realized=values, direction_delta=delta,
                    identical_token_stream=complete and predictions[0]['tokens'] == predictions[1]['tokens'],
                    in_band=[in_band(field['name'], v, band) for v, band in zip(realized, (field['low'], field['high']))],
                    independent_legality=[independent_legality([RhythmEvent(**e) for e in p['events']], 'phira', request.duration_us) for p in predictions],
                    reference_provenance=provenance))
                report['wall_s'] = time.monotonic() - start
                write_json(output, report)
                if not complete:
                    break
            if not report['rows'][-1]['complete']:
                break
        report['complete'] = len(report['rows']) == len(plan['songs']) * len(plan['fields']) and all(r['complete'] for r in report['rows'])
        report['by_field'] = {}
        for field in plan['fields']:
            rows = [r for r in report['rows'] if r['field'] == field['name']]
            direction = sum(r['complete'] and r['direction_delta'] is not None and r['direction_delta'] >= field['minimum_delta'] for r in rows)
            hits = sum(sum(r['in_band']) for r in rows)
            report['by_field'][field['name']] = dict(pairs=len(rows), identical_pairs=sum(r['identical_token_stream'] for r in rows),
                sufficiently_directed_pairs=direction, in_band_outputs=hits,
                sentinel_passed=len(rows) == 4 and all(r['complete'] and not any(r['independent_legality']) for r in rows) and direction >= 3 and hits >= 6)
        report['stop_reason'] = ('Both sentinel fields failed preregistered response gates; do not expand this checkpoint to remaining fields.'
            if report['complete'] and not any(r['sentinel_passed'] for r in report['by_field'].values())
            else 'Bounded audit finished; no general control or model quality claim.')
        write_json(output, report)
        print(json.dumps({k: report[k] for k in ('complete', 'wall_s', 'by_field', 'stop_reason')}))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-start)
        write_json(output, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

"""Frozen D83 held-out condition sentinel; CPU inference, no target chart input."""
import argparse
import json
from pathlib import Path
import time

import torch

from beatfcn.events.controls import raw_controls
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.train.event_conditions import encode_controls, in_band
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_interaction_runtime import prepare_inputs
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from scripts.probe_local_condition_interaction import load_frozen_components
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def request_legality(events, request):
    issues = independent_legality(events, request.mode, request.duration_us)
    issues.extend(dict(code='outside_requested_head_scope', index=i) for i, event in enumerate(events)
                  if not request.start_us <= event.onset_us < request.end_us)
    return issues


def summarize(rows, protocol):
    """Keep D83's preregistered combined gate and D68's stricter per-field gate distinct."""
    expected = {(f['name'], s['audio_sha256']) for f in protocol['fields'] for s in protocol['songs']}
    actual = [(r['field'], r['audio_sha256']) for r in rows]
    complete = len(actual) == len(expected) and set(actual) == expected and all(r['complete'] for r in rows)
    legal = complete and all(not any(r['independent_legality']) for r in rows)
    by_field = {}
    for field in protocol['fields']:
        subset = [r for r in rows if r['field'] == field['name']]
        directed = sum(r['complete'] and not r['identical_token_stream'] and
                       r['direction_delta'] is not None and r['direction_delta'] >= field['minimum_delta'] for r in subset)
        hits = sum(sum(r['in_band']) for r in subset)
        by_field[field['name']] = dict(pairs=len(subset),
            identical_pairs=sum(r['identical_token_stream'] for r in subset),
            sufficiently_directed_pairs=directed, in_band_outputs=hits,
            sentinel_passed=len(subset) == 4 and all(r['complete'] and not any(r['independent_legality']) for r in subset)
                            and directed >= 3 and hits >= 6)
    directed = sum(v['sufficiently_directed_pairs'] for v in by_field.values())
    hits = sum(v['in_band_outputs'] for v in by_field.values())
    gates = protocol['combined_gate']
    passed = legal and directed >= gates['minimum_directed_pairs'] and hits >= gates['minimum_in_band_outputs']
    return dict(complete=complete, by_field=by_field,
        combined_gate=dict(passed=passed, complete_and_legal=legal,
                           sufficiently_directed_pairs=directed, in_band_outputs=hits, thresholds=gates),
        expansion_allowed=passed,
        stop_reason=('Combined D83 sentinel failed; stop further condition expansion for this checkpoint.' if not passed
                     else 'Sentinel passed only; remaining fields, combinations, audio dependence and reference are unproven.'))


def validate_checkpoint(saved, training, training_sha, result, checkpoint_sha):
    if not result.get('complete') or result.get('error'):
        raise ValueError('Wait for the training/generation process to reach a successful terminal state')
    if (saved['updates'] != 1000 or result['updates'] != 1000 or saved['config'] != training['config']
            or saved['plan_sha256'] != training_sha or result['plan_sha256'] != training_sha
            or result['checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('Candidate checkpoint/configuration/exposure binding mismatch')
    expected = {k: training[k] for k in ('model_module', 'key_module', 'normalizer_module', 'normalizer')}
    if saved['candidate_components'] != expected:
        raise ValueError('Candidate component identity mismatch')


@torch.no_grad()
def run(path):
    plan = json.loads(path.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Frozen audit input changed: ' + source)
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    training_path = ROOT / protocol['training_plan']
    training = json.loads(training_path.read_text())
    for source, expected in training['code_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Candidate execution code changed: ' + source)
    torch.set_num_threads(protocol['cpu_threads'])
    result = json.loads((ROOT / training['output']).read_text())
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    validate_checkpoint(saved, training, sha(training_path), result, sha(ROOT / plan['checkpoint']))
    Config, Model, Standardizer = load_frozen_components(training)
    normalizer = Standardizer.load(ROOT / training['normalizer'])
    model = Model(Config(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    del saved
    features = LocalFeatureStore(ROOT, training['features'])
    if len({s['component'] for s in protocol['songs']}) != len(protocol['songs']):
        raise ValueError('Sentinel songs must be independent')
    for song in protocol['songs']:
        for source, expected in features.records[song['audio_sha256']]['sha256'].items():
            if sha(ROOT / source) != expected:
                raise ValueError('Admitted audio feature changed: ' + source)
    report = dict(id=plan['id'], plan_sha256=sha(path), protocol_sha256=sha(ROOT / plan['protocol']),
        complete=False, rows=[], checkpoint=plan['checkpoint'], checkpoint_sha256=sha(ROOT / plan['checkpoint']),
        checkpoint_updates=1000, optimizer_updates=0, new_cloud_compute_calls=0, backend='cpu',
        scope='D83 two single-field interventions on the same four audio-selected held-out 8s cores as D68. No human target, reference or teacher carry. Not general controllability.',
        normalization=dict(path=training['normalizer'], sha256=sha(ROOT / training['normalizer']), applied_once=True))
    start = time.monotonic()
    deadline = start + protocol['max_wall_s']
    write_json(output, report)
    try:
        for field in protocol['fields']:
            for song in protocol['songs']:
                a, b = song['scope_us']
                request = EventRequest('phira', a, b, features.records[song['audio_sha256']]['duration_us'])
                rows = []
                for band in (field['low'], field['high']):
                    controls = encode_controls('phira', {field['name']: band}, (a, b), origin_us=a)
                    inputs, provenance = request_inputs(features, song['audio_sha256'], request, controls)
                    rows.append(dict(inputs=inputs, sequence=teacher_sequence(encode_events([], request), request)))
                inputs, _ = collate(rows)
                inputs = prepare_inputs(inputs, normalizer, training['input_shapes'])
                if any(not torch.equal(v[0], v[1]) for k, v in inputs.items() if k != 'controls'):
                    raise ValueError('A non-condition input changed within the intervention')
                predictions = generate_batch(model, inputs, [request, request], protocol['token_cap'], deadline=deadline)
                events = [[RhythmEvent(**e) for e in p['events']] for p in predictions]
                values = [raw_controls('phira', canonical(es, 'phira'), [(a, b)]) for es in events]
                complete = all(p['complete'] for p in predictions)
                realized = [v[field['name']] for v in values]
                report['rows'].append(dict(chart_id=song['chart_id'], component=song['component'],
                    audio_sha256=song['audio_sha256'], scope_us=[a, b], field=field['name'],
                    low=field['low'], high=field['high'], complete=complete, predictions=predictions,
                    realized=values,
                    direction_delta=realized[1]-realized[0] if all(v is not None for v in realized) else None,
                    identical_token_stream=complete and predictions[0]['tokens'] == predictions[1]['tokens'],
                    in_band=[in_band(field['name'], v, band) for v, band in zip(realized, (field['low'], field['high']))],
                    independent_legality=[request_legality(es, request) for es in events],
                    reference_provenance=provenance))
                report['wall_s'] = time.monotonic() - start
                write_json(output, report)
                if not complete:
                    break
            if not report['rows'][-1]['complete']:
                break
        report.update(summarize(report['rows'], protocol))
        write_json(output, report)
        print(json.dumps({k: report[k] for k in ('complete', 'wall_s', 'by_field', 'combined_gate', 'stop_reason')}))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-start)
        write_json(output, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

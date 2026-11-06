"""Fixed whole-audio-stream swaps on held-out cores; no optimizer or target input."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.benchmark import audio_features, music_metrics
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from scripts.audit_local_candidate_conditions import request_legality
from scripts.audit_local_sequence_conditions import load_model
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
AUDIO_KEYS = ('local', 'local_times', 'local_valid', 'mel', 'mel_times', 'mel_valid',
              'song', 'song_times', 'song_valid')


def swap_audio(inputs, donor_positions):
    n = len(inputs['mode'])
    if sorted(donor_positions) != list(range(n)) or any(i == j for i, j in enumerate(donor_positions)):
        raise ValueError('Audio donors must be a complete permutation without self donors')
    donors = torch.tensor(donor_positions, device=inputs['mode'].device)
    return {key: value.index_select(0, donors) if key in AUDIO_KEYS else value.clone()
            for key, value in inputs.items()}


def f1(scores):
    counts = scores['edit_utility']['full_note']
    denominator = counts['pred'] + counts['true']
    return 2 * counts['matched'] / denominator if denominator else None


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
    training_path = ROOT / plan['training_plan']
    training = json.loads(training_path.read_text())
    for source, expected in training['code_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Candidate execution code changed: ' + source)
    torch.set_num_threads(protocol['cpu_threads'])
    model, training = load_model(training_path, ROOT / plan['checkpoint'])
    features = LocalFeatureStore(ROOT, training['features'])
    songs = protocol['songs']
    if len({s['component'] for s in songs}) != len(songs):
        raise ValueError('Expected independent songs')
    requests, rows = [], []
    for song in songs:
        for source, expected in features.records[song['audio_sha256']]['sha256'].items():
            if sha(ROOT / source) != expected:
                raise ValueError('Audio feature changed: ' + source)
        a, b = song['scope_us']
        request = EventRequest('phira', a, b, features.records[song['audio_sha256']]['duration_us'])
        controls = encode_controls('phira', {}, (a, b), origin_us=a)
        inputs, _ = request_inputs(features, song['audio_sha256'], request, controls)
        rows.append(dict(inputs=inputs, sequence=teacher_sequence(encode_events([], request), request)))
        requests.append(request)
    inputs, _ = collate(rows)
    inputs = pad_input_shapes(inputs, training['input_shapes'])
    altered = swap_audio(inputs, protocol['donor_positions'])
    report = dict(id=plan['id'], plan_sha256=sha(path), checkpoint=plan['checkpoint'],
        checkpoint_sha256=sha(ROOT / plan['checkpoint']), checkpoint_updates=1000,
        complete=False, optimizer_updates=0, new_cloud_compute_calls=0, backend='cpu', rows=[],
        scope='Four frozen held-out cores; swap all audio streams with their relative clocks and masks. Controls, reference, mode, decoder request remain fixed. Content and audio-context layout both change; changed outputs alone do not demonstrate musical understanding.',
        donor_positions=protocol['donor_positions'], audio_keys=list(AUDIO_KEYS))
    start = time.monotonic()
    write_json(output, report)
    try:
        predictions = generate_batch(model, inputs, requests, protocol['token_cap'], deadline=start+protocol['max_wall_s'])
        swaps = generate_batch(model, altered, requests, protocol['token_cap'], deadline=start+protocol['max_wall_s'])
        # Read the human target only after both predictions are final.
        tape = json.loads((ROOT / training['tape']).read_text())
        charts = {c['chart_id']: c for c in tape['charts']}
        for index, (song, correct, wrong) in enumerate(zip(songs, predictions, swaps)):
            chart = charts[song['chart_id']]
            source = ROOT / chart['target_path']
            if chart['split'] != 'val' or sha(source) != chart['target_sha256']:
                raise ValueError('Human comparison target must be frozen validation data')
            a, b = song['scope_us']
            with np.load(source, allow_pickle=False) as archive:
                truth = [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h,t,k,l in
                    zip(*(archive[key] for key in ('onset_us','end_us','kind','lane'))) if a <= h < b]
            human = canonical(truth, 'phira')
            feature = features.records[song['audio_sha256']]
            mel = np.load(ROOT / feature['complete_mel'], mmap_mode='r', allow_pickle=False)[:feature['valid_mel_frames']]
            acoustic = audio_features(mel, len(mel)*10_000)
            acoustic.update(duration_us=feature['duration_us'], coverage_end_us=feature['duration_us'])
            variants = {}
            for name, prediction in [('correct_audio', correct), ('wrong_audio', wrong)]:
                events = [RhythmEvent(**e) for e in prediction['events']]
                model_events = canonical(events, 'phira')
                scores = score_events(model_events, human, 'phira')
                variants[name] = dict(prediction=prediction, full_note_f1=f1(scores),
                    scores=scores, types=dict(Counter(e.type for e in events)),
                    tap_hold_music_against_target_audio=music_metrics([e for e in model_events if e.type in ('tap','hold')], [(a,b)], acoustic),
                    independent_legality=request_legality(events, requests[index]))
            identical = correct['complete'] and wrong['complete'] and correct['tokens'] == wrong['tokens']
            report['rows'].append(dict(chart_id=song['chart_id'], audio_sha256=song['audio_sha256'],
                donor_audio_sha256=songs[protocol['donor_positions'][index]]['audio_sha256'], scope_us=[a,b],
                variants=variants, identical_token_stream=identical,
                complete=correct['complete'] and wrong['complete']))
            report['wall_s'] = time.monotonic()-start
            write_json(output, report)
        report.update(complete=all(r['complete'] for r in report['rows']),
            identical_pairs=sum(r['identical_token_stream'] for r in report['rows']),
            quality_passed=False, interpretation='Audio sensitivity diagnostic only; no architecture promotion or human-reference equivalence claim.')
        write_json(output, report)
        print(json.dumps({k:report[k] for k in ('complete','wall_s','identical_pairs','interpretation')}))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-start)
        write_json(output, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

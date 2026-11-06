"""Compare CPU greedy rollout to existing native-MPS outputs, without GPU work."""
import hashlib
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from beatfcn.train.local_panel import collate
from beatfcn.model.event_canvas import RhythmEvent

ROOT = Path(__file__).resolve().parents[1]


def run():
    path = ROOT/'reports/d68_cpu_generation_probe_v1.json'
    if path.exists():
        raise FileExistsError(path)
    plan = json.loads((ROOT/'bench/rhythm-v2/d68-local-baseline-v3.json').read_text())
    state = json.loads((ROOT/plan['output']).read_text())
    index = json.loads((ROOT/plan['generation_output']).read_text())
    # Snapshot complete raw cores before concurrent MPS progress replaces their files.
    songs = [json.loads((ROOT/file).read_text()) for file in list(index['song_files'].values())[:4]]
    if len(songs) != 4 or any(len(s['cores']) < 3 for s in songs):
        raise ValueError('Need four recorded songs with three actual MPS cores')
    sources = ROOT/'inputs/d68_cpu_generation_probe'; sources.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    saved = torch.load(ROOT/state['checkpoint'], map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval(); model.load_state_dict(saved['model'])
    features = LocalFeatureStore(ROOT, plan['features'])
    report = dict(complete=False, passed=False, optimizer_updates=0, new_cloud_compute_calls=0,
        checkpoint=state['checkpoint'], checkpoint_sha256=state['checkpoint_sha256'], cases=[],
        scope='Backend/greedy stream parity and latency only, not quality improvement')
    started = time.monotonic()
    for core_index in (0, 2):
        rows, requests, expected = [], [], []
        for song in songs:
            core = song['cores'][core_index]
            if not core['complete']:
                raise ValueError('Expected a complete MPS comparison core')
            request = EventRequest(song['mode'], core['start_us'], core['end_us'], song['duration_us'],
                known_carry=tuple(RhythmEvent(**e) for e in core['known_carry']))
            controls = encode_controls(song['mode'], {}, (request.start_us, request.end_us), origin_us=request.start_us)
            inp, _ = request_inputs(features, song['audio_sha256'], request, controls)
            rows.append(dict(inputs=inp, sequence=teacher_sequence(encode_events([], request), request)))
            requests.append(request); expected.append(core['tokens'])
        inputs, _ = collate(rows); inputs = pad_input_shapes(inputs, plan['input_shapes'])
        tick = time.monotonic()
        result = generate_batch(model, inputs, requests, 4096, deadline=tick+90)
        case = dict(core_index=core_index, cpu_wall_s=time.monotonic()-tick,
            mps_batch_wall_s=songs[0]['cores'][core_index]['batch_wall_s'],
            complete=all(x['complete'] for x in result), streams_identical=[x['tokens'] == y for x,y in zip(result,expected)],
            results=result, expected=expected)
        report['cases'].append(case)
        if not case['complete']:
            break
    for song in songs:
        (sources/(song['chart_id'].replace(':','-')+'.json')).write_text(json.dumps(song)+'\n')
    report.update(complete=len(report['cases'])==2, wall_s=time.monotonic()-started,
        passed=len(report['cases'])==2 and all(c['complete'] and all(c['streams_identical']) for c in report['cases']),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.glob('*.json')})
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], cases=[{k:v for k,v in c.items() if k not in ('results','expected')} for c in report['cases']])))


if __name__ == '__main__':
    run()

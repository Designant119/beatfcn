"""Frozen heldout rollout at a real D87 stage, with labels used only afterward."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.model.local_song_rollout import canonical, independent_legality
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import LocalRequestPanel, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(path, step):
    plan = json.loads(path.read_text())
    protocol = json.loads((ROOT/plan['stage_protocol']).read_text())
    output = ROOT/plan['stage_report_pattern'].format(step=step)
    if output.exists():
        raise FileExistsError(output)
    for source, expected in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen stage input changed: '+source)
    checkpoint = ROOT/plan['checkpoint_dir']/f'step-{step:04d}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = json.loads((ROOT/plan['output']).read_text())
    if (saved['updates'] != step or saved['plan_sha256'] != sha(path) or saved['config'] != plan['config']
            or saved.get('objective') != plan['loss_mode'] or state['plan_sha256'] != sha(path)
            or state['updates'] < step or state.get('error')):
        raise ValueError('Stage does not bind to the real objective/checkpoint')
    torch.set_num_threads(protocol['cpu_threads'])
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    del saved
    features = LocalFeatureStore(ROOT, plan['features'])
    for audio in {case['audio_sha256'] for case in protocol['cases']}:
        for source, expected in features.records[audio]['sha256'].items():
            if sha(ROOT/source) != expected:
                raise ValueError('Admitted stage feature changed: '+source)
    report = dict(id=plan['id']+f'-stage-{step}', complete=False, optimizer_updates=0,
        checkpoint_updates=step, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
        plan_sha256=sha(path), protocol_sha256=sha(ROOT/plan['stage_protocol']), rows=[],
        inference_inputs='Audio/request scope only; unknown conditions/reference and no teacher carry',
        new_cloud_compute_calls=0, quality_passed=False)
    started = time.monotonic()
    try:
        for case in protocol['cases']:
            a, b = case['scope_us']
            request = EventRequest(case['mode'], a, b, features.records[case['audio_sha256']]['duration_us'])
            inputs, provenance = request_inputs(features, case['audio_sha256'], request,
                encode_controls(case['mode'], {}, (a, b), origin_us=a))
            inp, seq = collate([dict(inputs=inputs, sequence=teacher_sequence(encode_events([], request), request))])
            inp = pad_input_shapes(inp, plan['input_shapes'])
            initial = model(inp, seq)[0, 0].masked_fill(~seq['allowed'][0, 0], -1e9)
            prediction = generate_batch(model, inp, [request], protocol['token_cap'],
                deadline=started+protocol['max_wall_s'])[0]
            report['rows'].append(dict(**case, prediction=prediction, reference_provenance=provenance,
                bos_end_probability=float(torch.softmax(initial, -1)[2])))
            write_json(output, report)
            if prediction['deadline_reached']:
                raise TimeoutError('Stage rollout exceeded frozen wall-clock limit')
        # The generation loop is complete. Only now inspect target labels.
        panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
        for row in report['rows']:
            chart = panel.charts[panel.records[row['record_index']]['chart_index']]
            if sha(ROOT/chart['target_path']) != chart['target_sha256']:
                raise ValueError('Stage reference target changed')
            truth = panel.targets(row['record_index'])
            predictions = [RhythmEvent(**e) for e in row['prediction']['events']]
            request = panel.records[row['record_index']]['request']
            issues = independent_legality(predictions, row['mode'], request['duration_us'])
            issues.extend(dict(code='outside_requested_scope', index=i) for i,e in enumerate(predictions)
                          if not row['scope_us'][0] <= e.onset_us < row['scope_us'][1])
            tap = sorted({e.onset_us for e in predictions if e.type == 'tap'})
            differences = np.diff(tap)
            row.update(truth_events=[dict(onset_us=e.onset_us,end_us=e.end_us,type=e.type,lane=e.lane) for e in truth],
                legality=issues, predicted_notes=len(predictions), target_notes=len(truth),
                predicted_types=dict(Counter(e.type for e in predictions)), truth_types=dict(Counter(e.type for e in truth)),
                scores=score_events(canonical(predictions,row['mode']),canonical(truth,row['mode']),row['mode']),
                tap_positive_intervals=len(differences), tap_intervals_under10ms=int((differences<10000).sum()),
                tap_short_fraction=float((differences<10000).mean()) if len(differences) else None)
        active = [r for r in report['rows'] if r['role'] == 'active']
        rests = [r for r in report['rows'] if r['role'] == 'reference_rest']
        report.update(complete=True, wall_s=time.monotonic()-started,
            generation_complete=all(r['prediction']['complete'] for r in report['rows']),
            active_all_direct_end=all(r['prediction']['complete'] and len(r['prediction']['tokens'])==1 and not r['predicted_notes'] for r in active),
            active_all_token_capped=all(r['prediction']['token_limit_reached'] for r in active),
            legal=all(not r['legality'] for r in report['rows']),
            rest_mean_predicted_notes=sum(r['predicted_notes'] for r in rests)/len(rests))
        write_json(output, report)
        print(json.dumps({k:report[k] for k in ['complete','wall_s','active_all_direct_end','active_all_token_capped','legal','rest_mean_predicted_notes']}))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        write_json(output, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, choices=(250,500,1000))
    args = parser.parse_args()
    run(args.plan, args.step)

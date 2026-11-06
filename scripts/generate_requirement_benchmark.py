"""Run free / curve / continuation / infill requirement benchmarks from a checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.benchmark_generation import generate_request, source
from beatfcn.train.counterfactual_density import edit_events_to_density
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.requirement_benchmarks import requirement_metrics, summarize
from beatfcn.train.training_panel import build_training_panel

ROOT = Path(__file__).resolve().parents[1]


def generate_panel_row(model, panel, index, device):
    row = panel.row(index)
    request = event_request(panel.records[index]['request'])
    if 'intensity' not in row['inputs']:
        row['inputs'] = dict(row['inputs'])
        row['inputs'].update(grid_density_curve([], request.start_us, request.end_us,
            cell_us=100_000, scale=16.))
    inputs, _ = collate([row])
    inputs = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    request = event_request(panel.records[index]['request'])
    return generate_batch(model, inputs, [request], max_tokens=512)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--tasks', default='free,curve,continuation,infill')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    panel = build_training_panel(ROOT, plan)
    model = FullEventModel(FullEventConfig(**plan['config']))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    free = [index for index, row in enumerate(panel.records)
            if row.get('split') == 'val'
            and row.get('scope_role') == 'free' and row.get('event_count', 0) > 0]
    if not free:
        free = [index for index, row in enumerate(panel.records)
                if row.get('split') == 'val' and row.get('event_count', 0) > 0]
    if not free:
        free = [index for index, row in enumerate(panel.records)
                if row.get('event_count', 0) > 0]
    if not free:
        free = list(range(len(panel.records)))
    selected = free[:args.limit]
    tasks = set(args.tasks.split(','))
    records = []
    with torch.no_grad():
        for index in selected:
            row = panel.records[index]
            request = event_request(row['request'])
            target = panel.targets(index)
            mode = row['mode']
            if 'free' in tasks:
                prediction = generate_panel_row(model, panel, index, device)
                records.append(requirement_metrics(prediction['events'], [dict(
                    onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane) for e in target],
                    mode, request.start_us, request.end_us, task='free'))
            if 'curve' in tasks and row.get('scope_role') != 'free':
                for rate in (0., 2., 4., 8.):
                    edited, _ = edit_events_to_density(target, request.start_us,
                        request.end_us, rate, mode, request.duration_us)
                    segments = [dict(start_us=request.start_us, end_us=request.end_us,
                                     density=rate)]
                    prediction = generate_request(model, panel.features, row['audio_sha256'],
                        request, device=device, curve_segments=segments)
                    records.append(requirement_metrics(prediction['events'],
                        [dict(onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane)
                         for e in edited], mode, request.start_us, request.end_us,
                        requested_curve=segments, task='curve'))
            if 'curve' in tasks and row.get('scope_role') == 'free':
                prefix = row['chart_id'].rsplit('-', 1)[0]
                for variant, other in enumerate(panel.records):
                    if other['scope_role'] != 'counterfactual_curve':
                        continue
                    if not other['chart_id'].startswith(prefix + '-'):
                        continue
                    segments = plan['intensity']['requested_curves'].get(other['chart_id'], [])
                    if not segments:
                        continue
                    rate = segments[0]['density']
                    if rate not in (0., 2., 4., 8.):
                        continue
                    prediction = generate_panel_row(model, panel, variant, device)
                    target_curve = panel.targets(variant)
                    records.append(requirement_metrics(prediction['events'], [dict(
                        onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane) for e in target_curve],
                        mode, request.start_us, request.end_us,
                        requested_curve=segments, task='curve'))
            if 'continuation' in tasks:
                mid = request.start_us + (request.end_us - request.start_us) // 2
                prefix_events = [e for e in target if e.onset_us < mid]
                target_suffix = [e for e in target if e.onset_us >= mid]
                req = EventRequest(mode, mid, request.end_us, request.duration_us)
                prediction = generate_request(model, panel.features, row['audio_sha256'], req,
                    device=device, curve_segments=(),
                    sources=(source(row['audio_sha256'], mode, 'prefix',
                        ((request.start_us, mid),), prefix_events),))
                records.append(requirement_metrics(prediction['events'], [dict(
                    onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane) for e in target_suffix],
                    mode, mid, request.end_us, task='continuation'))
            if 'infill' in tasks:
                gap_start = request.start_us + (request.end_us - request.start_us) // 3
                gap_end = request.start_us + 2 * (request.end_us - request.start_us) // 3
                prefix_events = [e for e in target if e.onset_us < gap_start]
                suffix_events = [e for e in target if e.onset_us >= gap_end]
                target_gap = [e for e in target if gap_start <= e.onset_us < gap_end]
                req = EventRequest(mode, gap_start, gap_end, request.duration_us)
                prediction = generate_request(model, panel.features, row['audio_sha256'], req,
                    device=device, curve_segments=(),
                    sources=(source(row['audio_sha256'], mode, 'prefix',
                        ((request.start_us, gap_start),), prefix_events),
                        source(row['audio_sha256'], mode, 'suffix',
                        ((gap_end, request.end_us),), suffix_events)))
                records.append(requirement_metrics(prediction['events'], [dict(
                    onset_us=e.onset_us, end_us=e.end_us, type=e.type, lane=e.lane) for e in target_gap],
                    mode, gap_start, gap_end, task='infill'))
    result = dict(plan=str(args.plan), checkpoint=str(args.checkpoint),
                  tasks=sorted(tasks), records=records, summary=summarize(records))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

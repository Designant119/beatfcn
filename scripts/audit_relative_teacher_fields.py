"""Zero-update D88 diagnosis on the four already-failed held-out active cores.

Uses the same unknown controls/reference and no carry as the free stage. Only
the decoder prefix is supplied from truth. Per-field conditional accuracy is
not joint-event accuracy, free generation, or evidence of audio understanding.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.model import full_event_codec as absolute
from beatfcn.model import relative_event_codec as relative
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import LocalRequestPanel, request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(mask, losses, correct, choices):
    n = int(mask.sum())
    decision = mask & choices
    m = int(decision.sum())
    return dict(tokens=n, ce_sum=float(losses[mask].sum()),
        ce_mean=float(losses[mask].mean()) if n else None,
        correct=int(correct[mask].sum()), accuracy=float(correct[mask].float().mean()) if n else None,
        forced_tokens=n-m, choice_tokens=m, choice_correct=int(correct[decision].sum()),
        choice_accuracy=float(correct[decision].float().mean()) if m else None)


@torch.no_grad()
def run(path):
    plan = json.loads(path.read_text())
    output = ROOT/plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['source_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Diagnostic source changed: '+source)
    torch.set_num_threads(2)
    started = time.monotonic()
    report = dict(id=plan['id'], complete=False, optimizer_updates=0, new_cloud_compute_calls=0,
        plan_sha256=sha(path), rows=[], quality_passed=False,
        interpretation='True-prefix conditional decisions, not joint time/event accuracy or product quality.')
    try:
        for arm in plan['arms']:
            source = json.loads((ROOT/arm['quality_plan']).read_text())
            panel = LocalRequestPanel(ROOT, source['tape'], source['features'])
            for p, expected in {**source['inputs_sha256'], **source['code_sha256']}.items():
                actual = ('inputs/code_baselines/d87-closed-v1/'+p
                    if p == 'scripts/run_local_sequence_quality.py' else p)
                if sha(ROOT/actual) != expected:
                    raise ValueError('Quality source changed: '+actual)
            codec = relative if arm['representation'] == 'relative_head_v1' else absolute
            for step in plan['steps']:
                stage = json.loads((ROOT/arm['stage_report_pattern'].format(step=step)).read_text())
                checkpoint = ROOT/stage['checkpoint']
                if sha(checkpoint) != stage['checkpoint_sha256']:
                    raise ValueError('Stage checkpoint changed')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['updates'] != step or saved['plan_sha256'] != sha(ROOT/arm['quality_plan']):
                    raise ValueError('Wrong checkpoint or quality plan')
                if arm['representation'] == 'relative_head_v1' and saved.get('representation') != arm['representation']:
                    raise ValueError('Wrong head representation')
                model = FullEventModel(FullEventConfig(**saved['config'])).eval()
                model.load_state_dict(saved['model'])
                del saved
                active = [r for r in stage['rows'] if r['role'] == 'active']
                if len(active) != 4 or not stage['complete'] or not stage['active_all_direct_end']:
                    raise ValueError('Unexpected previously observed failure panel')
                for case in active:
                    if time.monotonic()-started > plan['max_wall_s']:
                        raise TimeoutError('Frozen diagnostic deadline')
                    i = case['record_index']; row = panel.records[i]
                    if row['split'] != 'val' or row['chart_id'] != case['chart_id']:
                        raise ValueError('Wrong held-out core')
                    chart = panel.charts[row['chart_index']]
                    if sha(ROOT/chart['target_path']) != chart['target_sha256']:
                        raise ValueError('Reference target changed')
                    for p, expected in panel.features.records[row['audio_sha256']]['sha256'].items():
                        if sha(ROOT/p) != expected:
                            raise ValueError('Admitted audio changed')
                    a, b = case['scope_us']
                    request = absolute.EventRequest(row['mode'], a, b,
                        panel.features.records[row['audio_sha256']]['duration_us'])
                    inputs, _ = request_inputs(panel.features, row['audio_sha256'], request,
                        encode_controls(row['mode'], {}, (a, b), origin_us=a))
                    tokens = codec.encode_events(panel.targets(i), request)
                    inp, seq = collate([dict(inputs=inputs, sequence=codec.teacher_sequence(tokens, request))])
                    logits = model(pad_input_shapes(inp, source['input_shapes']), seq)[0]
                    valid = seq['valid'][0]; allowed = seq['allowed'][0]
                    target = seq['targets'][0]; fields = seq['fields'][0]
                    assert allowed.gather(-1, target[:, None])[valid].all()
                    assert torch.isfinite(logits[valid]).all()
                    masked = logits.masked_fill(~allowed, -1e9)
                    losses = F.cross_entropy(masked, target, reduction='none')
                    pred = masked.argmax(-1); correct = pred == target; choices = allowed.sum(-1) > 1
                    stats = {field:summarize(valid & (fields==j), losses, correct, choices)
                        for j,field in enumerate(absolute.FIELDS)}
                    vocab = absolute.EventVocabulary()
                    kinds = {kind:summarize(valid & (target==vocab.token('kind', k)), losses, correct, choices)
                        for k,kind in enumerate(TYPES)}
                    report['rows'].append(dict(arm=arm['name'], step=step, chart_id=row['chart_id'],
                        record_index=i, scope_us=[a,b], checkpoint_sha256=stage['checkpoint_sha256'],
                        original_free_bos_end_probability=case['bos_end_probability'],
                        teacher_bos_end_probability=float(masked[0].softmax(-1)[2]),
                        all_fields=summarize(valid, losses, correct, choices), fields=stats, kind_targets=kinds,
                        target_tokens=target[valid].tolist(), predicted_tokens=pred[valid].tolist(),
                        query_fields=fields[valid].tolist(), legal_choice_counts=allowed[valid].sum(-1).tolist()))
                    assert sum(s['tokens'] for s in stats.values()) == len(tokens)
                    write_json(output, report)
                if sha(checkpoint) != stage['checkpoint_sha256']:
                    raise ValueError('Diagnostic modified checkpoint')
        report.update(complete=True, wall_s=time.monotonic()-started, checkpoint_unchanged=True)
        write_json(output, report)
        print(json.dumps(dict(complete=True, rows=len(report['rows']), wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-started)
        write_json(output, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)

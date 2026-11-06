"""D97 CPU audit: actual training weights and saved generated-condition failures."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.condition_outcome import score_condition_outcome
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.semantic_decision_loss import semantic_decision_weights

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def run():
    output = ROOT/'reports/d97_condition_outcome_audit_v1.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    tape = 'reports/d94_expanded_target_tape_v1.json'
    features = 'reports/d94_merged_feature_admission_v1.json'
    exposure_path = 'inputs/d94_local_requests/learning-exposure-1000-v1.json'
    previous_path = 'reports/d95_sequence_condition_response_v1.json'
    exposure = json.loads((ROOT/exposure_path).read_text())
    if sha(tape) != exposure['request_tape_sha256']:
        raise ValueError('Exposure and target tape differ')
    panel = LocalRequestPanel(ROOT, tape, features)
    required = {i for step in exposure['steps'] for i in step['indices']}
    identities = {}
    for chart in panel.charts:
        for p, h in [('target_path', 'target_sha256'), ('token_path', 'token_sha256')]:
            if sha(chart[p]) != chart[h]:
                raise ValueError('Training target changed: '+chart[p])
            identities[chart[p]] = chart[h]
    cache = {}
    for i in sorted(required):
        row = panel.records[i]
        if row['split'] != 'train':
            raise ValueError('Validation targets are forbidden in this training audit')
        seq = panel.sequence(i)
        weights, units = semantic_decision_weights(seq)
        request = event_request(row['request'])
        targets = panel.targets(i)
        cache[i] = dict(weights=weights, units=units, tokens=int(seq['valid'].sum()),
                        empty=not targets, events=canonical([*request.known_carry, *targets], row['mode']))
    aggregates = defaultdict(Counter)
    outcomes = []
    for step in exposure['steps']:
        for i, specified in zip(step['indices'], step['specified_condition_fields'], strict=True):
            row, item = panel.records[i], cache[i]
            requested = {CONTROL_FIELDS[k]: row['bands'][CONTROL_FIELDS[k]]
                         for k in specified if row['controls'][k][2] == 1}
            outcome = score_condition_outcome(row['mode'], item['events'],
                (row['request']['start_us'], row['request']['end_us']), requested,
                complete=True, legality_errors=[])
            if requested and not outcome['all_requested_satisfied']:
                raise ValueError('Actual training labels disagree with the complete target scope')
            a = aggregates[row['mode']]
            a['requests'] += 1
            a['empty_requests'] += item['empty']
            a['tokens'] += item['tokens']
            a['original_start_weight'] += 1/item['tokens']
            a['semantic_start_weight'] += float(item['weights'][0])
            if item['empty']:
                a['original_empty_start_weight'] += 1/item['tokens']
                a['semantic_empty_start_weight'] += float(item['weights'][0])
            outcomes.append(dict(record_index=i, update=step['effective_update'],
                mode=row['mode'], component=row['component'], stratum=row['stratum'],
                specified_fields=list(requested), score=outcome['reward'], units=item['units']))
    summaries = {}
    for mode, counts in aggregates.items():
        summaries[mode] = dict(counts)
        summaries[mode]['original_empty_start_weight_fraction'] = counts['original_empty_start_weight']/counts['original_start_weight']
        summaries[mode]['semantic_empty_start_weight_fraction'] = counts['semantic_empty_start_weight']/counts['semantic_start_weight']
    previous = json.loads((ROOT/previous_path).read_text())
    replay = []
    for row in previous['rows']:
        sides = []
        for side, band in enumerate((row['low'], row['high'])):
            generated = row['predictions'][side]
            events = canonical([RhythmEvent(**e) for e in generated['events']], 'phira')
            sides.append(score_condition_outcome('phira', events, tuple(row['scope_us']),
                {row['field']: band}, complete=generated['complete'],
                legality_errors=row['independent_legality'][side]))
        replay.append(dict(chart_id=row['chart_id'], field=row['field'], outcomes=sides))
    sources = [tape, features, exposure_path, previous_path,
               'scripts/audit_local_condition_outcomes.py', 'beatfcn/train/condition_outcome.py',
               'beatfcn/train/semantic_decision_loss.py', 'beatfcn/model/full_event_codec.py',
               'beatfcn/events/controls.py', 'beatfcn/train/local_request_panel.py']
    report = dict(id='D97-output-score-and-semantic-weight-audit-v1', complete=True,
        model_forward_calls=0, optimizer_updates=0, new_cloud_compute_calls=0,
        training_enabled=False, quality_passed=False, elapsed_s=time.monotonic()-started,
        training_requests=len(outcomes), unique_requests=len(cache), by_mode=summaries,
        training_labels_all_satisfied=True, training_rows=outcomes, saved_output_replay=replay,
        source_sha256={**identities, **{p: sha(p) for p in sources}},
        limitations=['Weights are coefficients, not actual gradients or a convergence claim.',
            'Training-target scores only validate label/score definitions, not model quality.',
            'D95 replay measures density only; other conditions were stopped, not passed.',
            'No hidden validation chart was read to construct a generated request.'])
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(complete=True, elapsed_s=report['elapsed_s'],
        by_mode=summaries, saved_pairs=len(replay)), ensure_ascii=False))


if __name__ == '__main__':
    run()

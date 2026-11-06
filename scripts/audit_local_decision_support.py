"""Audit real causal decision exposure without changing labels or training."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import END, EventState, EventVocabulary
from beatfcn.train.local_request_panel import event_request
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
CELL_US = 40_000


def inspect_request(tokens, request, events):
    """Walk prefix-only supports, verifying the entire target multiset."""
    state = EventState(request)
    fields, forced = Counter(), Counter()
    first_support = state.allowed_values()
    for token in tokens:
        field, value = state.vocab.decision(int(token))
        allowed = state.allowed_values()
        if field != state.phase or value not in allowed:
            raise ValueError('Illegal target or mismatched causal field')
        fields[field] += 1
        forced[field] += len(allowed) == 1
        state.apply(int(token))
    if not state.done or sorted(state.events) != sorted(events):
        raise ValueError('Actual token stream did not preserve the target multiset')
    # Absolute 40ms cells match the existing codec's clock; partial edge cells
    # count once. This is an occupancy audit, not a new cell decoder or loss.
    first_cell = request.start_us // CELL_US
    stop_cell = (request.end_us + CELL_US - 1) // CELL_US
    occupied = Counter(e.onset_us // CELL_US for e in events)
    if any(not request.start_us <= e.onset_us < request.end_us for e in events):
        raise ValueError('Head outside the authorized request')
    if not set(occupied) <= set(range(first_cell, stop_cell)):
        raise ValueError('Cell indexing changed the request clock')
    return dict(tokens=len(tokens), events=len(events), fields=dict(fields),
        forced_fields=dict(forced), forced_tokens=sum(forced.values()),
        choice_tokens=len(tokens) - sum(forced.values()),
        bos_label='END' if state.vocab.decision(int(tokens[0])) == ('group', END) else 'NEW',
        bos_support_size=len(first_support), cells=stop_cell-first_cell,
        occupied_cells=len(occupied), cells_with_multiple_heads=sum(n > 1 for n in occupied.values()),
        max_heads_per_cell=max(occupied.values(), default=0),
        outgoing_holds=sum(e.type == 'hold' and e.end_us > request.end_us for e in events),
        incoming_holds=len(request.known_carry), types=dict(Counter(e.type for e in events)),
        partial_first_cell=request.start_us % CELL_US != 0,
        partial_last_cell=request.end_us % CELL_US != 0)


def summarize(rows):
    """Rows retain multiplicity from the actual frozen training exposure."""
    n = len(rows)
    total_tokens = sum(r['tokens'] for r in rows)
    bos_end = sum(1/r['tokens'] for r in rows if r['bos_label'] == 'END' and r['bos_support_size'] > 1)
    bos_new = sum(1/r['tokens'] for r in rows if r['bos_label'] == 'NEW' and r['bos_support_size'] > 1)
    choices_end = sum(1/r['choice_tokens'] for r in rows if r['bos_label'] == 'END' and r['bos_support_size'] > 1)
    choices_new = sum(1/r['choice_tokens'] for r in rows if r['bos_label'] == 'NEW' and r['bos_support_size'] > 1)
    choice_bos = [r for r in rows if r['bos_support_size'] > 1]
    cells = sum(r['cells'] for r in rows)
    empty_cells = sum(r['cells']-r['occupied_cells'] for r in rows)
    fields, forced_fields = Counter(), Counter()
    for r in rows:
        fields.update(r['fields']); forced_fields.update(r['forced_fields'])
    return dict(requests=n, distinct_requests=len({r['record_index'] for r in rows}),
        songs=len({r['component'] for r in rows}),
        empty_requests=sum(r['events'] == 0 for r in rows), tokens=total_tokens,
        mean_tokens=total_tokens/n,
        token_quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'q95', 'max'),
            np.quantile([r['tokens'] for r in rows], [0, .25, .5, .75, .95, 1]).tolist())),
        forced_tokens=sum(r['forced_tokens'] for r in rows),
        forced_fraction=sum(r['forced_tokens'] for r in rows)/total_tokens,
        fields=dict(fields), forced_fields=dict(forced_fields),
        bos_forced_requests=n-len(choice_bos),
        original_bos_end_weight=bos_end, original_bos_new_weight=bos_new,
        original_constant_bos_end_optimum=bos_end/(bos_end+bos_new) if bos_end+bos_new else None,
        hypothetical_choice_only_mean_bos_end_optimum=choices_end/(choices_end+choices_new) if choices_end+choices_new else None,
        hypothetical_equal_bos_decision_end_fraction=sum(r['bos_label'] == 'END' for r in choice_bos)/len(choice_bos) if choice_bos else None,
        cells=cells, occupied_cells=cells-empty_cells,
        empty_cell_fraction=empty_cells/cells,
        request_mean_empty_cell_fraction=sum((r['cells']-r['occupied_cells'])/r['cells'] for r in rows)/n,
        cells_with_multiple_heads=sum(r['cells_with_multiple_heads'] for r in rows),
        max_heads_per_cell=max(r['max_heads_per_cell'] for r in rows),
        requests_with_outgoing_holds=sum(r['outgoing_holds'] > 0 for r in rows),
        requests_with_partial_tail_cell=sum(r['partial_last_cell'] for r in rows))


def main():
    output = ROOT/'reports/d85_decision_support_v1.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    tape_path = 'reports/d67_local_request_tape_v1.json'
    exposure_path = 'inputs/d67_local_requests/initial100/learning-exposure-1000-v1.json'
    tape = json.loads((ROOT/tape_path).read_text())
    exposure = json.loads((ROOT/exposure_path).read_text())
    required = {index for step in exposure['steps'] for index in step['indices']}
    records, sources, index = {}, {}, 0
    for chart in tape['charts']:
        chart_indices = set(range(index, index+len(chart['requests'])))
        if not chart_indices & required:
            index += len(chart['requests'])
            continue
        for path_field, hash_field in [('token_path', 'token_sha256'), ('target_path', 'target_sha256')]:
            path = chart[path_field]
            if sha(ROOT/path) != chart[hash_field]:
                raise ValueError('Frozen corpus changed: '+path)
            sources[path] = chart[hash_field]
        tokens = np.load(ROOT/chart['token_path'], mmap_mode='r', allow_pickle=False)
        with np.load(ROOT/chart['target_path'], allow_pickle=False) as archive:
            targets = {key: archive[key].copy() for key in archive.files}
        for row in chart['requests']:
            current = index
            index += 1
            if current not in required:
                continue
            if chart['split'] != 'train':
                raise ValueError('Frozen exposure includes validation targets')
            sl = slice(row['target_start'], row['target_stop'])
            events = [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(*(targets[key][sl] for key in ('onset_us', 'end_us', 'kind', 'lane')), strict=True)]
            value = inspect_request(tokens[row['token_start']:row['token_stop']], event_request(row['request']), events)
            if value['tokens'] != row['tokens'] or value['events'] != row['event_count']:
                raise ValueError('Prepared request counts differ from actual arrays')
            records[current] = dict(value, record_index=current, mode=chart['mode'],
                chart_id=chart['chart_id'], component=chart['component'], stratum=chart['stratum'],
                start_us=row['request']['start_us'], end_us=row['request']['end_us'],
                control_known=[bool(c[2]) for c in row['controls']])
    if records.keys() != required:
        raise ValueError('Not all exposed requests were audited')
    groups = defaultdict(list)
    for step in exposure['steps']:
        for index, specified in zip(step['indices'], step['specified_condition_fields'], strict=True):
            row = records[index]
            known = any(row['control_known'][field] for field in specified)
            groups[row['mode']].append(row)
            groups[row['mode']+('/specified' if known else '/unknown')].append(row)
    by_group = {key: summarize(rows) for key, rows in groups.items()}
    prior = 'reports/d84_bos_supervision_weight_v2.json'
    prior_values = json.loads((ROOT/prior).read_text())['by_mode_and_view']
    for key, result in by_group.items():
        if result['tokens'] != prior_values[key]['target_tokens']:
            raise ValueError('Exposure totals disagree with prior audit')
        if not result['bos_forced_requests'] and abs(result['original_constant_bos_end_optimum']-prior_values[key]['weighted_constant_bos_end_optimum']) > 1e-12:
            raise ValueError('Recomputed BOS weights disagree with prior audit')
    for path in [tape_path, exposure_path, prior, 'scripts/audit_local_decision_support.py',
                 'beatfcn/model/full_event_codec.py', 'beatfcn/model/event_canvas.py',
                 'beatfcn/events/canonical.py', 'beatfcn/train/local_request_panel.py',
                 'beatfcn/model/full_event_model.py', 'scripts/run_local_song_baseline.py']:
        sources[path] = sha(ROOT/path)
    report = dict(id='D85-causal-decision-support-v1', complete=True,
        optimizer_updates=0, model_forward_calls=0, new_cloud_compute_calls=0,
        elapsed_s=time.monotonic()-started, source_sha256=sources,
        exposed_requests=sum(len(groups[m]) for m in ('phira', 'mania_4k')),
        unique_requests=len(records), cell_us=CELL_US, by_mode_and_view=by_group,
        records=[records[i] for i in sorted(records)],
        limitations=['Cell occupancy is a label distribution only; no new representation or trained model.',
            'A constant marginal END probability does not determine a conditional neural model optimum.',
            'Choice-only normalization and equal-BOS weighting are hypothetical objectives, not implemented changes.',
            'Mode/song sampling and all legitimate empty, drag, flick, hold targets are unchanged.'])
    write_json(output, report)
    print(json.dumps(dict(elapsed_s=report['elapsed_s'], unique_requests=len(records),
                         by_mode_and_view=by_group), indent=2))


if __name__ == '__main__':
    main()

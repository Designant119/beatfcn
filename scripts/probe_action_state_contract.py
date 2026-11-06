"""D49 real-event state contract; zero model construction or parameter updates."""
import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from beatfcn.model.action_state import State, ManiaAction, apply_action, encode, replay, mania_digits, legal_mania_actions
from beatfcn.model.event_canvas import RhythmEvent

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stats(array):
    a = np.asarray(array)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(a).sum()),
        inf=int(np.isinf(a).sum()), min=float(a.min()) if a.size else None,
        max=float(a.max()) if a.size else None, mean=float(a.mean()) if a.size else None,
        std=float(a.std()) if a.size else None)


def expected_state(events, mode, t, before):
    active = [e for e in events if e.type == 'hold' and
              (e.onset_us < t <= e.end_us if before else e.onset_us <= t < e.end_us)]
    if mode == 'mania_4k':
        lanes = [None] * 4
        for e in active:
            if lanes[e.lane] is not None:
                raise ValueError('source has overlapping holds on one key')
            lanes[e.lane] = e.onset_us
        return tuple(lanes), ()
    return (None,) * 4, tuple(sorted(Counter(e.onset_us for e in active).items()))


def peak_occupancy(events):
    deltas = Counter()
    for e in events:
        deltas[e.onset_us] += 1
        deltas[e.end_us if e.type == 'hold' else e.onset_us + 1] -= 1
    active = peak = 0
    for t in sorted(deltas):
        active += deltas[t]
        peak = max(peak, active)
    return peak


def inspect(events, mode, label, artifact_dir):
    events = tuple(events)
    actions = encode(events, mode)
    restored = replay(actions, mode)
    assert restored.events == tuple(sorted(events)) and not restored.state.active_holds
    state = State(mode)
    traces, numeric, action_counts = [], [], Counter()
    pure_release = same_time_release_start = max_holds = max_groups = max_heads = 0
    for action in actions:
        before = expected_state(events, mode, action.time_us, True)
        assert (state.lane_starts, state.hold_groups) == before
        support = None
        if isinstance(action, ManiaAction):
            support = legal_mania_actions(state)
            assert action.code in support
            digits = mania_digits(action.code)
            releases = sum(x in (3, 4, 5) for x in digits)
            heads = sum(x in (1, 2, 4, 5) for x in digits)
            same_time_release_start += sum(x in (4, 5) for x in digits)
            action_counts[str(action.code)] += 1
        else:
            releases = sum(n for _, n in action.releases)
            heads = sum(action.starts)
            same_time_release_start += int(bool(releases and heads))
            action_counts[str(action.starts)] += 1
        transition = apply_action(state, action)
        after = expected_state(events, mode, action.time_us, False)
        assert (transition.state.lane_starts, transition.state.hold_groups) == after
        pure_release += int(releases > 0 and heads == 0)
        max_heads = max(max_heads, heads)
        max_holds = max(max_holds, transition.state.active_holds)
        max_groups = max(max_groups, len(transition.state.hold_groups))
        numeric.append([action.time_us, state.active_holds, len(state.hold_groups), heads, releases])
        traces.append(dict(before=asdict(state), action=asdict(action),
            legal_support_sha256=digest(support) if support else None,
            completed=[asdict(e) for e in transition.events], after=asdict(transition.state)))
        state = transition.state
    # Cuts are intervals in a supplied complete source, not permissions inferred
    # for an editing request. Carry actual prefix state; never read an end time
    # into the state's incoming hold fields.
    cuts = set(range(0, max((e.end_us for e in events), default=0) + 1, 8_000_000))
    cuts.update(e.onset_us + (e.end_us - e.onset_us) // 2 for e in events if e.type == 'hold' and e.end_us - e.onset_us > 1)
    times = [a.time_us for a in actions]
    chunk_checks = active_cuts = 0
    examples = []
    for cut in sorted(cuts):
        index = bisect_left(times, cut)
        left = replay(actions[:index], mode, require_closed=False)
        right = replay(actions[index:], mode, initial=left.state)
        assert tuple(sorted(left.events + right.events)) == tuple(sorted(events))
        active_cuts += int(left.state.active_holds > 0)
        chunk_checks += 1
        if left.state.active_holds and len(examples) < 4:
            examples.append(dict(cut_us=cut, prefix_actions=index, incoming=asdict(left.state)))
    destination = artifact_dir / (label.replace(':', '-').replace('/', '-') + '.json')
    destination.write_text(json.dumps(dict(mode=mode, labels_only=True, traces=traces), separators=(',', ':')) + '\n')
    return dict(label=label, mode=mode, passed=True, events=len(events), actions=len(actions),
        types=dict(Counter(e.type for e in events)), exact_multiset_roundtrip=True,
        exact_event_sha256=digest([asdict(e) for e in sorted(events)]),
        before_and_after_state_checks=len(actions) * 2, target_in_prefix_support=True,
        chunk_roundtrips=chunk_checks, cuts_with_active_holds=active_cuts,
        carry_examples=examples, pure_release_boundaries=pure_release,
        simultaneous_release_and_attack=same_time_release_start,
        maximum_active_holds=max_holds, maximum_active_hold_start_groups=max_groups,
        maximum_heads_at_boundary=max_heads, action_histogram=dict(action_counts),
        numeric_boundary_statistics=stats(np.array(numeric, dtype=np.int64).reshape(-1, 5)),
        numeric_columns=['time_us', 'prefix_active_holds', 'prefix_hold_groups', 'target_heads', 'target_releases'],
        numeric_scope='Label/context alignment audit, not a neural-network tensor or hidden-target-free model input.',
        trace=str(destination.relative_to(ROOT)), trace_sha256=sha(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest_ in plan['input_sha256'].items():
        if sha(ROOT / path) != digest_:
            raise ValueError('input identity changed: ' + path)
    result_path = ROOT / plan['result']
    if result_path.exists():
        raise FileExistsError('D49 result already exists')
    output = ROOT / plan['artifact_dir']
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(id='D49', complete=False, plan_sha256=sha(args.plan),
        optimizer_updates=0, model_constructed=False, model_promotion=False,
        new_cloud_spend_usd=0, rows=[], failures=[], negative_controls=[])

    def save():
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    save()
    panel = json.loads((ROOT / plan['whole_song_panel']).read_text())
    cases = []
    for row in panel['rows']:
        chart = json.loads((ROOT / row['canonical']).read_text())
        events = [RhythmEvent(e['onset_us'], e['end_us'], e['type'],
            e['location']['lane'] if row['mode'] == 'mania_4k' else -1) for e in chart['events']]
        for view, shift in [('native', 0), ('frozen_model', row['frozen_model_shift_us'])]:
            shifted = [replace(e, onset_us=e.onset_us + shift, end_us=e.end_us + shift) for e in events]
            cases.append((shifted, row['mode'], row['chart_id'] + '/' + view))
    previous = json.loads((ROOT / plan['fixed_core_labels']).read_text())
    for row in previous['rows']:
        identity = row['identity']
        label = identity['panel_id'] + ('/translated' if identity['translated'] else '/original')
        cases.append(([RhythmEvent(**e) for e in row['target_events']], identity['mode'], label))
    for events, mode, label in cases:
        print(json.dumps(dict(phase='contract', label=label)), flush=True)
        try:
            report['rows'].append(inspect(events, mode, label, output))
        except Exception as exc:
            report['failures'].append(dict(label=label, events=len(events), error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc()))
        save()
    old = json.loads((ROOT / plan['fixed_bad_predictions']).read_text())
    for view in ('original', 'translated'):
        for core in old['generation'][view]['cores']:
            if core['identity']['mode'] != 'mania_4k':
                continue
            for row in (core['rounds'][0], core['rounds'][-1]):
                events = [RhythmEvent(**e) for e in row['events']]
                expected_rejection = row['mania_lane_conflicts'] > 0
                error = None
                try:
                    actions = encode(events, 'mania_4k')
                    assert replay(actions, 'mania_4k').events == tuple(sorted(events))
                except ValueError as exc:
                    error = str(exc)
                report['negative_controls'].append(dict(view=view, panel_id=core['identity']['panel_id'],
                    round=row['round'], notes=len(events), physical_peak=peak_occupancy(events),
                    prior_lane_conflicts=row['mania_lane_conflicts'], expected_rejection=expected_rejection,
                    rejected=error is not None, error=error, matched_prior_legality=(error is not None) == expected_rejection))
    rows = report['rows']
    report.update(complete=True, passed=not report['failures'] and len(rows) == len(cases)
        and all(r['matched_prior_legality'] for r in report['negative_controls'])
        and any(r['physical_peak'] == 8 and r['rejected'] for r in report['negative_controls']),
        wall_s=time.time() - started,
        summary=dict(expected_views=len(cases), completed_views=len(rows),
            event_views=sum(r['events'] for r in rows), action_boundaries=sum(r['actions'] for r in rows),
            independent_state_checks=sum(r['before_and_after_state_checks'] for r in rows),
            chunk_roundtrips=sum(r['chunk_roundtrips'] for r in rows),
            cuts_with_active_holds=sum(r['cuts_with_active_holds'] for r in rows),
            maximum_actions=max((r['actions'] for r in rows), default=0),
            maximum_phira_active_holds=max((r['maximum_active_holds'] for r in rows if r['mode'] == 'phira'), default=0),
            maximum_phira_hold_start_groups=max((r['maximum_active_hold_start_groups'] for r in rows if r['mode'] == 'phira'), default=0),
            maximum_phira_simultaneous_heads=max((r['maximum_heads_at_boundary'] for r in rows if r['mode'] == 'phira'), default=0)))
    save()
    print(json.dumps({k: report[k] for k in ('complete', 'passed', 'summary', 'wall_s')}))


if __name__ == '__main__':
    main()

"""Read-only witnesses for upstream rhythms that no four-lane decoder can place."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from beatfcn.model.group_canvas import decode_group_targets
from beatfcn.model.lane_draft import make_lane_draft, maximum_overlap
from scripts.acoustic_probe_metrics import physical_events
from scripts.matched_acoustic_data import make_example

ROOT = Path(__file__).resolve().parents[1]


def occupancy_profile(events):
    rows = []
    for head in sorted({e['onset_us'] for e in events}):
        active = [e for e in events if e['onset_us'] < head < e['end_us']]
        starts = [e for e in events if e['onset_us'] == head]
        rows.append(dict(time_us=head, required_lanes=len(active) + len(starts),
                         active_holds=active, starting_members=starts))
    return rows


def physical_hold_groups(events, identity):
    groups = {}
    for e in physical_events(events, identity):
        if e['type'] == 'hold':
            groups.setdefault(e['onset_us'], []).append(e['end_us'])
    return {head: sorted(tails) for head, tails in groups.items()}


def main():
    paths = ['reports/r13_acoustic_identifiability.json', 'reports/r14_health.json',
             'inputs/d32_acoustic_panel/manifest.json', 'bench/rhythm-v2/d32-covered-acoustic-panel-v1.json',
             'scripts/audit_lane_feasibility.py', 'beatfcn/model/lane_draft.py']
    prior = json.loads((ROOT / paths[0]).read_text())
    manifest = json.loads((ROOT / paths[2]).read_text())
    panel = json.loads((ROOT / paths[3]).read_text())
    output = dict(id='D33', complete=False, model_updates=0, new_cloud_compute_usd=0,
        hypothesis='The fixed R13 rhythm itself may exceed four simultaneous actions before lane assignment.',
        matching='Occupancy is lane-independent and exact. Tail set pairs use equal physical hold-head time; members with equal head are compared in sorted-tail order only when counts agree.',
        input_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        rows=[], shared_hold_tail_pairs=[])
    raw_by_identity = {}
    for item in manifest['items']:
        if item['row']['mode'] != 'mania_4k':
            continue
        originals = {}
        for translated in (False, True):
            variant = 'translated' if translated else 'original'
            _, labels, identity = make_example(item, panel, translated=translated)
            truth = [asdict(e) for e in decode_group_targets(labels)]
            saved = next(c for c in prior['generation'][variant]['cores'] if c['identity'] == identity)
            truth_profile = occupancy_profile(truth)
            for round_result in saved['rounds']:
                events = round_result['events']; profile = occupancy_profile(events)
                peak = max(profile, key=lambda row: row['required_lanes'])
                if peak['required_lanes'] != maximum_overlap(make_lane_draft(events)):
                    raise AssertionError('independent occupancy scans disagree')
                at = peak['time_us']
                truth_at = [e for e in truth if e['onset_us'] == at or e['onset_us'] < at < e['end_us']]
                witness = dict(peak, physical_time_us=at + identity['source_clip_start_us'],
                    truth_at_same_time=truth_at)
                comparisons = []
                for hold in peak['active_holds']:
                    candidates = [e for e in truth if e['type'] == 'hold' and abs(e['onset_us'] - hold['onset_us']) <= 20_000]
                    comparisons.append(dict(predicted_hold=hold, true_hold_candidates=candidates,
                        tail_errors_ms=[(hold['end_us'] - e['end_us']) / 1000 for e in candidates]))
                output['rows'].append(dict(identity=identity, round=round_result['round'],
                    notes=len(events), types=dict(Counter(e['type'] for e in events)),
                    true_peak=max(r['required_lanes'] for r in truth_profile),
                    predicted_peak=peak['required_lanes'], impossible=peak['required_lanes'] > 4,
                    overfull_onsets=sum(r['required_lanes'] > 4 for r in profile),
                    witness=witness, witness_hold_comparisons=comparisons))
            originals[variant] = dict(identity=identity, truth=truth, events=saved['rounds'][0]['events'])
        a, b = originals['original'], originals['translated']
        ag, bg = physical_hold_groups(a['events'], a['identity']), physical_hold_groups(b['events'], b['identity'])
        ta, tb = physical_hold_groups(a['truth'], a['identity']), physical_hold_groups(b['truth'], b['identity'])
        lo, hi = item['source_core_us'] + 2_000_000, item['source_core_us'] + 8_000_000
        for head in sorted(ag.keys() & bg.keys()):
            if not lo <= head < hi or len(ag[head]) != len(bg[head]):
                continue
            output['shared_hold_tail_pairs'].append(dict(panel_id=item['panel_id'], physical_head_us=head,
                original_physical_tails_us=ag[head], translated_physical_tails_us=bg[head],
                predicted_physical_tail_deltas_ms=[(y - x) / 1000 for x, y in zip(ag[head], bg[head])],
                original_truth_physical_tails_us=ta.get(head, []), translated_truth_physical_tails_us=tb.get(head, [])))
    deltas = [d for row in output['shared_hold_tail_pairs'] for d in row['predicted_physical_tail_deltas_ms']]
    output.update(complete=True, peak_across_all_rounds=max(r['predicted_peak'] for r in output['rows']),
        impossible_rows=sum(r['impossible'] for r in output['rows']),
        first_round_impossible=[dict(identity=r['identity'], peak=r['predicted_peak']) for r in output['rows'] if r['round'] == 1 and r['impossible']],
        matched_shared_holds=len(deltas), unchanged_physical_tails=sum(d == 0 for d in deltas),
        plus_two_seconds_physical_tails=sum(d == 2000 for d in deltas))
    (ROOT / 'reports/d33_lane_feasibility.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('complete','peak_across_all_rounds','impossible_rows','matched_shared_holds','unchanged_physical_tails','plus_two_seconds_physical_tails')}))


if __name__ == '__main__':
    main()

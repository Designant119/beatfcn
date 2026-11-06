"""Audit the prototype canvas on entire existing train charts, including long holds.

This is an exploratory capacity audit, separate from the frozen 290-window gate.
It does not drop targets, change the capacity after observing data, or train.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.canonical import build_model_view
from beatfcn.model.event_canvas import ClockLayout, RhythmEvent, encode_targets, decode_targets
from beatfcn.train.data import chart_from_canon, model_shift_us


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT/'bench/rhythm-v2/r09-positive-pairs-prepare-v1.json'
    plan = json.loads(path.read_text())
    results = []
    for row in plan['train_rows']:
        assert row['split'] == 'train'
        source = ROOT/plan['train_canonical_root']/row['source']/f'{row["native_id"]}.json'
        if sha(source) != plan['train_canonical_sha256'][row['chart_id']]:
            raise ValueError(f'canonical source changed: {source}')
        doc = json.loads(source.read_text())
        chart = chart_from_canon(doc, model_shift_us(row, True))
        view = build_model_view(row['mode'], (), chart.events)
        result = {'chart_id':row['chart_id'], 'mode':row['mode'], 'source_sha256':sha(source),
                  'audio_sha256':row['audio_sha256'], 'source_events':len(chart.events),
                  'existing_model_view_issues':[i.to_json() for i in view.quarantine], 'phases':[]}
        events = [RhythmEvent.from_event(e, row['mode']) for group in view.target_groups for e in group.events]
        result['quantized_events'] = len(events)
        result['holds_crossing_8s'] = sum(e.type == 'hold' and e.onset_us//8_000_000 < (e.end_us-1)//8_000_000 for e in events)
        # Whole audio interval, not the last true event, sets the canvas length.
        duration = round(row['audio_sec']*1_000_000)
        for phase in (0, 200_000):
            layout = ClockLayout(0, duration, phase_us=phase)
            inside = [e for e in events if 0 <= e.onset_us < duration]
            buckets = Counter(layout.block_for(e.onset_us) for e in inside)
            item = {'phase_us':phase, 'outside_audio_heads':len(events)-len(inside),
                    'max_notes_per_block':max(buckets.values(), default=0),
                    'overflow_blocks_at_32':sum(n>32 for n in buckets.values()),
                    'capacity_excess_notes_at_32':sum(max(0,n-32) for n in buckets.values()),
                    'exact_roundtrip':False}
            try:
                if view.quarantine:
                    raise ValueError('existing model-view quarantines; not an admitted target')
                encoded = encode_targets(events, layout, row['mode'], slots=32)
                restored = decode_targets(encoded)
                if Counter(restored) != Counter(events):
                    raise AssertionError('event multiset changed')
                item['exact_roundtrip'] = True
            except (ValueError, AssertionError) as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
            result['phases'].append(item)
        results.append(result)
    summary = {}
    for mode in ('mania_4k','phira'):
        rows = [r for r in results if r['mode'] == mode]
        phases = [p for r in rows for p in r['phases']]
        summary[mode] = {'charts':len(rows), 'events':sum(r['quantized_events'] for r in rows),
                         'holds_crossing_8s':sum(r['holds_crossing_8s'] for r in rows),
                         'max_notes_per_400ms':max(p['max_notes_per_block'] for p in phases),
                         'overflow_blocks_at_32_both_phases':sum(p['overflow_blocks_at_32'] for p in phases),
                         'exact_roundtrips':sum(p['exact_roundtrip'] for p in phases), 'attempts':len(phases),
                         'charts_with_existing_model_view_issues':sum(bool(r['existing_model_view_issues']) for r in rows)}
    result = {'schema':'beatfcn.canvas_full_training_capacity.v1', 'complete':True,
              'scope':'44 existing training charts, two phases, existing millisecond model-view policy; not full corpus admission',
              'plan_sha256':sha(path), 'script_sha256':sha(Path(__file__)), 'slots_per_block':32,
              'phases_us':[0,200000], 'summary':summary, 'charts':results,
              'new_paid_compute_usd':0, 'optimizer_updates':0, 'capacity_automatically_changed':False}
    (ROOT/'reports/p2_canvas_full_training_capacity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summary,'failures':[
        {'chart_id':r['chart_id'],'phases':[p for p in r['phases'] if not p['exact_roundtrip']]}
        for r in results if any(not p['exact_roundtrip'] for p in r['phases'])]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

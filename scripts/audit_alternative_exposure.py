"""What did training actually show through alternative_context? (CPU; no model, no audio features.)

Replays the exact sibling draws of CounterfactualExposureDataset._attach_references (seed, request index,
slot key, rate) over evenly spaced exposure steps of a frozen plan, on the tape panel, and reports per mode:
  - slot shares: density-edited (never get a reference), native, sibling drawn / applied, skip reasons;
  - the shown sibling: direction (denser / sparser), choice (nearest / uniform), events in the window;
  - how informative it is: share of the target's window onsets with a sibling onset within 10 / 50 ms
    (recall) and share of sibling window onsets within 10 ms of a target onset (precision), aligned vs the
    same sibling shifted by +1 s (chance level; what the spread probe's alignment control removes).

  python -m scripts.audit_alternative_exposure --plan /data/round3v2-l/plan-r2.json --root / \
      --steps 3000 --out reports/<run>/alternative-exposure-audit.json
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOLERANCES_US = (10_000, 50_000)
SHIFT_US = 1_000_000


def nearest_abs(a, b):
    """|a_i - nearest b| for sorted unique int arrays (inf when b is empty)."""
    if not len(b):
        return np.full(len(a), np.inf)
    pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    left, right = b[np.maximum(pos - 1, 0)], b[pos]
    return np.minimum(np.abs(a - left), np.abs(a - right)).astype(float)


def overlap(target, sibling, tolerances=TOLERANCES_US):
    """Recall of target onsets by sibling onsets and precision of sibling onsets (both sorted unique)."""
    out = {}
    if len(target):
        d = nearest_abs(target, sibling)
        out.update({f'recall_{t // 1000}ms': float(np.mean(d <= t)) for t in tolerances})
    if len(sibling) and len(target):
        out[f'precision_{tolerances[0] // 1000}ms'] = float(np.mean(nearest_abs(sibling, target) <= tolerances[0]))
    return out


def audit(panel, exposure, settings, *, step_indices, shift_us=SHIFT_US):
    from beatfcn.train.alternative_context import SiblingIndex, alternative_source, draw_alternative
    from beatfcn.train.history_context import chart_arrays
    from beatfcn.train.local_request_panel import event_request
    seed = int(exposure.get('seed', 0))
    index = SiblingIndex(panel)
    counts = defaultdict(Counter)
    shown = defaultdict(list)
    for step_index in step_indices:
        update = exposure['steps'][step_index]
        keys = update.get('occurrences', update['epochs'])
        for slot, request_index in enumerate(update['indices']):
            record = panel.records[request_index]
            c = counts[record['mode']]
            c['slots'] += 1
            if update['rates'][slot] is not None:
                c['edited_target'] += 1
                continue
            c['native'] += 1
            request = event_request(record['request'])
            draw = draw_alternative(settings, seed, request_index, keys[slot])
            source, info = alternative_source(index, request, record, draw, settings)
            c[info['reason'] or 'applied'] += 1
            if source is None:
                continue
            onsets = chart_arrays(panel, record['chart_index'])['onset_us']
            target = np.unique(onsets[(onsets >= request.start_us) & (onsets < request.end_us)]).astype(np.int64)
            sib = np.unique(np.asarray([e.onset_us for e in source.events], np.int64))
            sib_window = sib[(sib >= request.start_us) & (sib < request.end_us)]
            row = dict(direction=info['direction'], choice=info['choice'], window_events=info['window_events'],
                       target_window_onsets=int(len(target)), sibling_window_onsets=int(len(sib_window)))
            row.update({f'aligned_{k}': v for k, v in overlap(target, sib_window).items()})
            shifted = np.unique(sib + int(shift_us))
            shifted = shifted[(shifted >= request.start_us) & (shifted < request.end_us)]
            row.update({f'shifted_{k}': v for k, v in overlap(target, shifted).items()})
            shown[record['mode']].append(row)
    return counts, shown


def summarize(counts, shown):
    out = {}
    for mode, c in sorted(counts.items()):
        rows = shown.get(mode, [])
        slots, native = c['slots'], max(1, c['native'])
        block = dict(slots=c['slots'], counts=dict(c),
                     edited_share=c['edited_target'] / max(1, slots), native_share=c['native'] / max(1, slots),
                     applied_share_of_native=c['applied'] / native, applied_share_of_slots=c['applied'] / max(1, slots))
        if rows:
            def mean(key):
                values = [r[key] for r in rows if key in r]
                return float(np.mean(values)) if values else None
            block['shown'] = dict(
                n=len(rows), direction=dict(Counter(r['direction'] for r in rows)),
                choice=dict(Counter(r['choice'] for r in rows)),
                window_events_quantiles=[float(q) for q in np.quantile([r['window_events'] for r in rows], [.1, .5, .9])],
                empty_target_windows=sum(r['target_window_onsets'] == 0 for r in rows),
                **{k: mean(k) for k in ('aligned_recall_10ms', 'aligned_recall_50ms', 'aligned_precision_10ms',
                                        'shifted_recall_10ms', 'shifted_recall_50ms', 'shifted_precision_10ms')})
        out[mode] = block
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/'))
    p.add_argument('--steps', type=int, default=3000, help='evenly spaced exposure steps (4 slots each)')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    started = time.monotonic()
    from beatfcn.train.alternative_context import normalize_alternative_context
    from scripts.preflight_reference_paths import TapePanel
    plan = json.loads(a.plan.read_text())
    settings = normalize_alternative_context(plan.get('alternative_context'))
    if settings is None:
        raise SystemExit('plan has no alternative_context')
    exposure = json.loads((a.root / plan['exposure'].lstrip('/')).read_text())
    panel = TapePanel(a.root, plan['tape'])
    total = len(exposure['steps'])
    steps = sorted({int(i) for i in np.linspace(0, total - 1, min(a.steps, total))})
    counts, shown = audit(panel, exposure, settings, step_indices=steps)
    report = dict(schema='beatfcn.alternative-exposure-audit.v1', plan=str(a.plan), plan_id=plan.get('id'),
                  settings=settings, exposure_steps_total=total, exposure_steps_sampled=len(steps),
                  by_mode=summarize(counts, shown), wall_s=round(time.monotonic() - started, 1))
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')


if __name__ == '__main__':
    main()

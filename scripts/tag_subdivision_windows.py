"""Tag training requests by the subdivisions their human target uses (ROADMAP: Phira beat feel; CPU, no model).

For every TRAIN request of a tape, the target chart's onsets inside the window are placed on the chart's own
timing grid (chart_timing table; segments are on the targets' clock) and the window gets a group key:
  <mode>:triplet   >= MIN notes on 1/3 or 1/6 lines (and not on 1/4 / 1/8 lines in the same beats)
  <mode>:fine      >= MIN notes on 1/4 or 1/8 lines
  <mode>:plain     everything else (beats / half beats only, or no valid timing)
The keys feed counterfactual_exposure.build_exposure(mode_weights=..., mode_of=...) unchanged, e.g.
{'phira:triplet': 3, 'phira:fine': 1.5, 'phira:plain': 1, 'mania_4k:triplet': 1, ...}, so a later round can
oversample the beat feels the model gets wrong (Phira triplets: recall 0.26 on TEST) without touching the loader.

  python -m scripts.tag_subdivision_windows --tape /data/full-database-v1/native-v2-tape/tape.json \
      --timing /data/full-database-v1/chart-timing/chart_timing-v2.json --root / --out tags.json [--min-notes 2]
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL_US = 5_000


def subdivision(t_us, starts, beats):
    """'1/1' | '1/2' | '1/4' | '1/8' | 'triplet' | 'other' for one onset on a piecewise-constant grid."""
    k = max(0, int(np.searchsorted(starts, t_us, 'right')) - 1)
    b = beats[k]
    x = (t_us - starts[k]) / b
    for name, div in (('1/1', 1), ('1/2', 2), ('1/4', 4)):
        if abs(x * div - round(x * div)) * b / div <= TOL_US:
            return name
    for div in (3, 6):
        if abs(x * div - round(x * div)) * b / div <= TOL_US:
            return 'triplet'
    if abs(x * 8 - round(x * 8)) * b / 8 <= TOL_US:
        return '1/8'
    return 'other'


def window_key(mode, onsets, segments, min_notes=2):
    if not segments or not len(onsets):
        return f'{mode}:plain', Counter()
    starts = np.array([s[0] for s in segments], float)
    beats = np.array([s[1] for s in segments], float)
    c = Counter(subdivision(int(t), starts, beats) for t in np.unique(onsets))
    if c['triplet'] >= min_notes and c['triplet'] >= c['1/4'] + c['1/8']:
        return f'{mode}:triplet', c
    if c['1/4'] + c['1/8'] >= min_notes:
        return f'{mode}:fine', c
    return f'{mode}:plain', c


def tag(panel, timing, *, min_notes=2):
    """{request index: key} over TRAIN records of a TapePanel-like panel (records + _arrays)."""
    out, totals = {}, Counter()
    for i, r in enumerate(panel.records):
        if r['split'] != 'train':
            continue
        entry = timing.get(r['chart_id']) or {}
        segments = entry.get('segments') if entry.get('valid') else None
        arrays = panel._arrays(r['chart_index'])
        arrays = arrays[1] if isinstance(arrays, tuple) else arrays
        on = np.asarray(arrays['onset_us'])
        req = r['request']
        inside = on[(on >= req['start_us']) & (on < req['end_us'])]
        key, _ = window_key(r['mode'], inside, segments, min_notes=min_notes)
        out[i] = key
        totals[key] += 1
    return out, totals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tape', required=True)
    p.add_argument('--timing', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/'))
    p.add_argument('--min-notes', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from scripts.preflight_reference_paths import TapePanel
    panel = TapePanel(a.root, a.tape)
    timing = json.loads((a.root / str(a.timing).lstrip('/')).read_text())['charts']
    tags, totals = tag(panel, timing, min_notes=a.min_notes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(schema='beatfcn.subdivision-tags.v1', tape=a.tape, min_notes=a.min_notes,
                                     totals=dict(totals), tags={str(k): v for k, v in tags.items()})) + '\n')
    print(json.dumps(dict(totals), indent=1))


if __name__ == '__main__':
    main()

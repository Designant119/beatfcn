"""Where do drafts disagree with the best-matching human chart? (ROADMAP item 1; CPU, no model, no cloud.)

Inputs: a workflow-benchmark run (draft_rows -> best-matching reference chart per song), its saved consensus
drafts (drafts-consensus-*.json, model clock), the reference targets (<chart>-targets.npz from the tape), the
charter timing table and the 10 ms complete mel of each song. Distinct onsets are matched 1-1 within 20 ms
(greedy by distance). Every reference onset is a hit or a MISS; every draft onset a hit or an EXTRA. Rates are
broken down by: beat position on the reference chart's own grid (model clock), reference local density,
onset strength (mel spectral flux percentile within the song), distance to the 8 s core boundaries, chord size,
and hold vs tap.

  python -m scripts.selection_error_analysis --run reports/.../T2_product/workflow-benchmark.json \
      --drafts /tmp/w4/T2_product.json --refs /tmp/ea/refs --mel /tmp/ea/mel --timing /tmp/ct2.json
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

TOL_US = 20_000
CORE_US = 8_000_000


def match(a, b, tol=TOL_US):
    """Greedy 1-1 matching of sorted unique int arrays by |a-b| <= tol; returns boolean hit masks."""
    pairs = []
    j0 = 0
    for i, x in enumerate(a):
        while j0 < len(b) and b[j0] < x - tol:
            j0 += 1
        j = j0
        while j < len(b) and b[j] <= x + tol:
            pairs.append((abs(int(b[j]) - int(x)), i, j))
            j += 1
    pairs.sort()
    ha, hb = np.zeros(len(a), bool), np.zeros(len(b), bool)
    for _, i, j in pairs:
        if not ha[i] and not hb[j]:
            ha[i] = hb[j] = True
    return ha, hb


def beat_class(t_us, segments, shift_us):
    """Position of t on the chart grid: '1/1', '1/2', '1/4', '1/3|1/6', 'finer' (within 5 ms), else 'off'."""
    if not segments:
        return 'no_timing'
    starts = np.array([s[0] for s in segments], float) + shift_us
    k = max(0, np.searchsorted(starts, t_us, 'right') - 1)
    beat = float(segments[k][1])
    x = (t_us - starts[k]) / beat
    for name, div in (('1/1', 1), ('1/2', 2), ('1/4', 4), ('1/3|1/6', 6), ('1/8', 8), ('finer', 16)):
        if abs(x * div - round(x * div)) * beat / div <= 5_000:
            if name == '1/3|1/6' and abs(x * 3 - round(x * 3)) * beat / 3 > 5_000 and abs(x * 6 - round(x * 6)) * beat / 6 > 5_000:
                continue
            return name
    return 'off'


def flux(mel):
    m = mel.astype(np.float32)
    d = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0).sum(1)
    return d


def analyse(run, drafts, refs_dir, mel_dir, timing):
    rows = {r['song_id']: r for r in run['readout_rows'][readout_name(run)]['draft_rows']}
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (mode, axis) -> bucket -> [hits, total]
    for sid, d in drafts.items():
        mode = d['mode']
        ref_id = rows[sid]['min_ref']
        z = np.load(Path(refs_dir) / (ref_id.replace(':', '-') + '-targets.npz'))
        ref_on, ref_end, ref_kind = z['onset_us'].astype(np.int64), z['end_us'].astype(np.int64), z['kind']
        r_uni, r_idx = np.unique(ref_on, return_index=True)
        r_count = np.array([np.sum(ref_on == t) for t in r_uni])
        r_hold = np.array([bool(np.any((ref_on == t) & (ref_end > ref_on))) for t in r_uni])
        d_on = np.unique(np.array([e['onset_us'] for e in d['events']], np.int64))
        hr, hd = match(r_uni, d_on)
        entry = timing.get(ref_id) or {}
        segs = entry.get('segments') if entry.get('valid') else None
        shift = 0  # table segments are already on the targets' clock (verified: 100% of onsets on grid at shift 0)
        mel = np.load(Path(mel_dir) / f"{d['audio_sha256']}.npy")
        fl = flux(mel)
        rank = np.argsort(np.argsort(fl)) / max(1, len(fl) - 1)

        def strength(t):
            i = int(round(t / 10_000))
            lo, hi = max(0, i - 2), min(len(rank), i + 3)
            return float(rank[lo:hi].max()) if hi > lo else 0.

        def dens(t):
            return np.sum(np.abs(r_uni - t) <= 2_000_000) / 4.

        def add(axis, bucket, hit, side):
            stats[(mode, side + ':' + axis)][bucket][0] += int(hit)
            stats[(mode, side + ':' + axis)][bucket][1] += 1

        for t, h, c, hold in zip(r_uni, hr, r_count, r_hold):
            add('beat', beat_class(t, segs, shift), h, 'recall')
            add('strength', f'q{min(3, int(strength(t) * 4))}', h, 'recall')
            dn = dens(t)
            add('density', '<3/s' if dn < 3 else '3-6/s' if dn < 6 else '6-9/s' if dn < 9 else '>=9/s', h, 'recall')
            add('core_edge', 'near(<0.5s)' if min(t % CORE_US, CORE_US - t % CORE_US) < 500_000 else 'inside', h, 'recall')
            add('chord', 'single' if c == 1 else 'chord', h, 'recall')
            add('hold', 'hold' if hold else 'tap', h, 'recall')
        for t, h in zip(d_on, hd):
            add('beat', beat_class(t, segs, shift), h, 'precision')
            add('strength', f'q{min(3, int(strength(t) * 4))}', h, 'precision')
            add('core_edge', 'near(<0.5s)' if min(t % CORE_US, CORE_US - t % CORE_US) < 500_000 else 'inside', h, 'precision')
    out = {}
    for (mode, axis), buckets in sorted(stats.items()):
        out.setdefault(mode, {})[axis] = {b: dict(rate=round(h / n, 3), n=n) for b, (h, n) in sorted(buckets.items())}
    return out


def readout_name(run):
    names = [k for k in run['readout_rows'] if k.endswith('jackfree')] or list(run['readout_rows'])
    return names[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--drafts', type=Path, required=True)
    p.add_argument('--refs', type=Path, required=True)
    p.add_argument('--mel', type=Path, required=True)
    p.add_argument('--timing', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    report = analyse(json.loads(a.run.read_text()), json.loads(a.drafts.read_text()), a.refs, a.mel,
                     json.loads(a.timing.read_text())['charts'])
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

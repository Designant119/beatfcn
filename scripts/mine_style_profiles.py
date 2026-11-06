"""Track C1: mine charting STYLE from human charts (no model, no GPU).

Per chart: a style vector (density-free ratios) and a local pattern histogram over heads (onset groups):
  mania_4k: single / jump (2) / hand (3) / quad (4) heads; jack (a lane re-pressed by the next head); trill
            (single notes alternating on two lanes A B A B); roll (>= 3 consecutive singles on distinct lanes);
            LN share (hold heads) and LN occupancy (time with >= 1 key held); burst (IOI <= 1/2 of the song's
            median IOI).
  phira:    tap / hold / flick / drag shares of notes; doubles (2+ notes per head); drag runs (>= 3 consecutive
            drag heads); burst.
Train charts are clustered per mode (standardised features, numpy k-means++, best of 8 restarts) into K style
profiles; each profile reports size, centroid (raw units), density median and the features that set it apart
(z >= 0.5), which name it.

  python -m scripts.mine_style_profiles --tape-dir DIR --out reports/style/style-profiles-v1.json \
      [--k 6 --mania-sample 4000 --phira-sample 4000]
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAP, HOLD, FLICK, DRAG = 0, 1, 2, 3
MANIA_FEATURES = ('single', 'jump', 'hand', 'quad', 'jack', 'trill', 'roll', 'ln_heads', 'ln_occupancy', 'burst')
PHIRA_FEATURES = ('tap', 'hold', 'flick', 'drag', 'double', 'drag_run', 'hold_occupancy', 'burst')


def heads(onset, end, kind, lane):
    order = np.lexsort((lane, onset))
    onset, end, kind, lane = onset[order], end[order], kind[order], lane[order]
    groups, start = [], 0
    for i in range(1, len(onset) + 1):
        if i == len(onset) or onset[i] != onset[start]:
            groups.append((int(onset[start]), [(int(end[j]), int(kind[j]), int(lane[j])) for j in range(start, i)]))
            start = i
    return groups


def occupancy(onset, end, kind, duration_us):
    iv = sorted((int(a), int(b)) for a, b, k in zip(onset, end, kind) if k == HOLD and b > a)
    total, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / max(duration_us, 1)


def burst_share(groups):
    t = np.array([g[0] for g in groups], np.float64)
    if len(t) < 3:
        return 0.0
    ioi = np.diff(t)
    med = np.median(ioi[ioi > 0]) if np.any(ioi > 0) else 0
    return float(np.mean(ioi <= 0.5 * med)) if med > 0 else 0.0


def mania_vector(z, duration_us):
    onset, end, kind, lane = (np.asarray(z[k]) for k in ('onset_us', 'end_us', 'kind', 'lane'))
    g = heads(onset, end, kind, lane)
    n = len(g)
    if n < 20:
        return None
    sizes = [len(x[1]) for x in g]
    lanes = [frozenset(e[2] for e in x[1]) for x in g]
    jack = sum(bool(lanes[i] & lanes[i - 1]) for i in range(1, n)) / (n - 1)
    single = [i for i in range(n) if sizes[i] == 1]
    sl = {i: next(iter(lanes[i])) for i in single}
    trill = roll = 0
    for i in range(3, n):
        if all(sizes[j] == 1 for j in range(i - 3, i + 1)):
            a, b, c, d = (sl[j] for j in range(i - 3, i + 1))
            if a == c and b == d and a != b:
                trill += 1
    for i in range(2, n):
        if all(sizes[j] == 1 for j in range(i - 2, i + 1)) and len({sl[i - 2], sl[i - 1], sl[i]}) == 3:
            roll += 1
    c = Counter(min(s, 4) for s in sizes)
    notes = len(onset)
    return dict(single=c[1] / n, jump=c[2] / n, hand=c[3] / n, quad=c[4] / n, jack=jack, trill=trill / max(1, n - 3),
                roll=roll / max(1, n - 2), ln_heads=float(np.mean(kind == HOLD)),
                ln_occupancy=occupancy(onset, end, kind, duration_us), burst=burst_share(g),
                heads_per_s=n / (duration_us / 1e6), notes_per_s=notes / (duration_us / 1e6))


def phira_vector(z, duration_us):
    onset, end, kind = (np.asarray(z[k]) for k in ('onset_us', 'end_us', 'kind'))
    lane = np.zeros_like(kind)
    g = heads(onset, end, kind, lane)
    n = len(g)
    if n < 20:
        return None
    kinds = [set(e[1] for e in x[1]) for x in g]
    run = drag_run = 0
    for k in kinds:
        run = run + 1 if k == {DRAG} else 0
        drag_run += run >= 3
    notes = len(onset)
    return dict(tap=float(np.mean(kind == TAP)), hold=float(np.mean(kind == HOLD)), flick=float(np.mean(kind == FLICK)),
                drag=float(np.mean(kind == DRAG)), double=float(np.mean([len(x[1]) >= 2 for x in g])),
                drag_run=drag_run / n, hold_occupancy=occupancy(onset, end, kind, duration_us), burst=burst_share(g),
                heads_per_s=n / (duration_us / 1e6), notes_per_s=notes / (duration_us / 1e6))


def kmeans(x, k, *, restarts=8, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):  # k-means++
            d = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(x[rng.choice(len(x), p=d / d.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((x - c[lab]) ** 2).sum())
        if inertia < best[0]:
            best = (inertia, c, lab)
    return best


def profiles(vectors, names, k, seed):
    x = np.array([[v[f] for f in names] for v in vectors], np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    zx = (x - mu) / sd
    inertia, cz, lab = kmeans(zx, k, seed=seed)
    out = []
    for j in np.argsort(-np.bincount(lab, minlength=k)):
        m = lab == j
        z = cz[j]
        tags = [f"{names[i]}{'+' if z[i] > 0 else '-'}" for i in np.argsort(-np.abs(z)) if abs(z[i]) >= 0.5][:4]
        out.append(dict(size=int(m.sum()), share=round(float(m.mean()), 3), tags=tags,
                        centroid={f: round(float(x[m, i].mean()), 4) for i, f in enumerate(names)},
                        z={f: round(float(z[i]), 2) for i, f in enumerate(names)},
                        heads_per_s_median=round(float(np.median([vectors[t]['heads_per_s'] for t in np.where(m)[0]])), 2)))
    return dict(features=list(names), mean=dict(zip(names, np.round(mu, 4).tolist())),
                sd=dict(zip(names, np.round(sd, 4).tolist())), profiles=out, inertia=round(inertia, 1),
                charts=len(vectors), labels=lab.tolist())


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--tape-dir', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--k', type=int, default=6)
    a.add_argument('--mania-sample', type=int, default=4000)
    a.add_argument('--phira-sample', type=int, default=4000)
    a.add_argument('--seed', type=int, default=20261103)
    args = a.parse_args()
    charts = [json.loads(f.read_text()) for f in sorted(args.tape_dir.glob('*-chart.json'))]
    rng = random.Random(args.seed)
    result = dict(schema='beatfcn.style-profiles.v1', source='train split of the native tape', k=args.k)
    for mode, fn, names, cap in (('mania_4k', mania_vector, MANIA_FEATURES, args.mania_sample),
                                 ('phira', phira_vector, PHIRA_FEATURES, args.phira_sample)):
        pool = [c for c in charts if c['mode'] == mode and c['split'] == 'train']
        rng.shuffle(pool)
        vectors, ids = [], []
        for c in pool[:cap]:
            z = np.load(args.tape_dir / f"{c['chart_id'].replace(':', '-')}-targets.npz")
            v = fn(z, c['duration_us'])
            if v is not None:
                vectors.append(v); ids.append(c['chart_id'])
        prof = profiles(vectors, names, args.k, args.seed)
        prof['chart_ids'] = ids
        result[mode] = prof
        print(mode, len(vectors), 'charts')
        for p in prof['profiles']:
            print(f"  {p['share']:.0%}  hps {p['heads_per_s_median']:5.2f}  {' '.join(p['tags'])}  "
                  + ' '.join(f"{f}={p['centroid'][f]:.2f}" for f in names))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()

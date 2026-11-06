"""Build the difficulty preset table (Phira level / mania star -> median heads per second) from the database.

  python -m scripts.build_difficulty_presets --tape-dir DIR --phira-catalog phira_all_charts.jsonl \
      --out reports/presets/difficulty-presets-v1.json [--mania-sample 3000]
Train-split charts only. Phira: catalog 'difficulty' rounded to integer levels, weighted log-linear fit over 6..18; mania: approximate star (beatfcn.events.difficulty_presets.mania_star) in 0.5 bins 1.0..8.0.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from beatfcn.events.difficulty_presets import mania_star, monotone
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--tape-dir', type=Path, required=True)
    a.add_argument('--phira-catalog', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--mania-sample', type=int, default=3000)
    a.add_argument('--seed', type=int, default=20261103)
    args = a.parse_args()
    cat = {}
    for line in args.phira_catalog.read_text().splitlines():
        r = json.loads(line); cat[r['id']] = r
    # per-chart metadata files (the 600 MB tape.json is not needed)
    charts = [json.loads(f.read_text()) for f in sorted(args.tape_dir.glob('*-chart.json'))]
    phira, mania = defaultdict(list), defaultdict(list)
    mania_charts = [c for c in charts if c['mode'] == 'mania_4k' and c['split'] == 'train']
    random.Random(args.seed).shuffle(mania_charts)
    keep = {c['chart_id'] for c in mania_charts[:args.mania_sample]}
    stars = []
    for c in charts:
        if c['split'] != 'train':
            continue
        z = np.load(args.tape_dir / f"{c['chart_id'].replace(':', '-')}-targets.npz")
        hps = len(np.unique(z['onset_us'])) / max(c['duration_us'] / 1e6, 1e-6)
        if c['mode'] == 'phira':
            meta = cat.get(int(c['chart_id'].split(':')[1]))
            if meta:
                phira[int(round(float(meta['difficulty'])))].append(hps)
        elif c['chart_id'] in keep:
            notes = list(zip(z['onset_us'], z['end_us'], z['kind'], z['lane']))
            s = mania_star(notes)
            stars.append((s, hps))
            mania[round(s * 2) / 2].append(hps)
    def table(groups, lo, hi, min_n):
        xs = [k for k in sorted(groups) if lo <= k <= hi and len(groups[k]) >= min_n]
        med = [float(np.median(groups[k])) for k in xs]
        n = [len(groups[k]) for k in xs]
        fx, fy = monotone(xs, med, n)
        return dict(x=fx, heads_per_s=[round(v, 3) for v in fy], raw_median=[round(v, 3) for v in med], n=n)
    def loglinear(groups, lo, hi, min_n, grid):
        # Phira: few charts below level 13 and a noisy median there -> one weighted log-linear fit
        # (density grows by a constant factor per level), weights = sqrt(n)
        xs = [k for k in sorted(groups) if lo <= k <= hi and len(groups[k]) >= min_n]
        med = np.array([np.median(groups[k]) for k in xs]); n = np.array([len(groups[k]) for k in xs])
        b, a = np.polyfit(np.array(xs, float), np.log(med), 1, w=np.sqrt(n))
        return dict(x=list(grid), heads_per_s=[round(float(np.exp(a + b * x)), 3) for x in grid],
                    fit=dict(kind='log-linear', per_level_factor=round(float(np.exp(b)), 4)),
                    raw_x=xs, raw_median=[round(float(v), 3) for v in med], n=n.tolist())
    out = dict(schema='beatfcn.difficulty-presets.v1', unit='heads (distinct onsets) per second, song level',
               phira_level=loglinear(phira, 6, 18, 8, range(6, 19)), mania_star=table(mania, 1.0, 8.0, 15),
               mania_star_algorithm='approximate legacy osu!mania strain (beatfcn.events.difficulty_presets.mania_star)',
               mania_sample=len(stars), source='train split of the native tape')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k in ('phira_level', 'mania_star')}, indent=1))


if __name__ == '__main__':
    main()

"""Track C3: style / pattern distribution of model drafts vs the human charts of the same songs (CPU only).

  python -m scripts.style_gap --run RUN/workflow-benchmark.json --drafts RUN/drafts-consensus-8-jackfree.json \
      --tape-dir DIR [--out gap.json]
Per song: the draft's style vector (scripts.mine_style_profiles) vs the MEAN over the song's human reference
charts; per mode the mean model - human difference with a song-bootstrap 80% CI, next to the human-human spread
(mean over songs of the std across that song's human charts) as the natural scale.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mine_style_profiles import MANIA_FEATURES, PHIRA_FEATURES, mania_vector, phira_vector  # noqa: E402

TYPE_CODE = {'tap': 0, 'hold': 1, 'flick': 2, 'drag': 3}


def draft_arrays(events):
    return dict(onset_us=np.array([e['onset_us'] for e in events], np.int64),
                end_us=np.array([e['end_us'] for e in events], np.int64),
                kind=np.array([TYPE_CODE[e['type']] for e in events], np.int8),
                lane=np.array([int(e.get('lane', -1)) for e in events], np.int8))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--drafts', type=Path, required=True)
    a.add_argument('--tape-dir', type=Path, required=True)
    a.add_argument('--out', type=Path)
    args = a.parse_args()
    run = json.loads(args.run.read_text())
    drafts = json.loads(args.drafts.read_text())
    rng = np.random.default_rng(0)
    out = {}
    for mode, fn, names in (('mania_4k', mania_vector, MANIA_FEATURES), ('phira', phira_vector, PHIRA_FEATURES)):
        diffs, spreads = [], []
        for song in run['panel']['songs']:
            if song['mode'] != mode or song['song_id'] not in drafts:
                continue
            dur = int(song['duration_us'])
            dv = fn(draft_arrays(drafts[song['song_id']]['events']), dur)
            hv = [fn(np.load(args.tape_dir / f"{r['chart_id'].replace(':', '-')}-targets.npz"), dur)
                  for r in song['references']]
            hv = [v for v in hv if v is not None]
            if dv is None or not hv:
                continue
            h = np.array([[v[f] for f in names] for v in hv])
            diffs.append(np.array([dv[f] for f in names]) - h.mean(0))
            if len(hv) >= 2:
                spreads.append(h.std(0))
        d = np.array(diffs)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(4000)])
        lo, hi = np.percentile(boots, [10, 90], axis=0)
        sp = np.mean(spreads, 0) if spreads else np.full(len(names), np.nan)
        out[mode] = {f: dict(model_minus_human=round(float(d[:, i].mean()), 4), ci80=[round(float(lo[i]), 4), round(float(hi[i]), 4)],
                             human_spread=round(float(sp[i]), 4)) for i, f in enumerate(names)}
        out[mode]['_songs'] = len(d)
        print(mode, len(d), 'songs')
        for f in names:
            r = out[mode][f]
            flag = '  <-- outside human spread' if abs(r['model_minus_human']) > r['human_spread'] and (r['ci80'][0] > 0 or r['ci80'][1] < 0) else ''
            print(f"  {f:14s} model-human {r['model_minus_human']:+.3f} [{r['ci80'][0]:+.3f},{r['ci80'][1]:+.3f}]  human spread {r['human_spread']:.3f}{flag}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()

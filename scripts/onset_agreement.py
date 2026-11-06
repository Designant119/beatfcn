"""Whole-song timing agreement from workflow-benchmark edit operations (no model, no audio, no cloud).

The workflow benchmark aligns a draft to a reference chart (1-1 matching within 150 ms) and records
edit ops: matched = reference_notes - insert = draft_notes - delete; ``retime`` = matched notes whose
onset is off by more than the keep tolerance (20 ms); ``keep`` = matched with time, type, lane (and tail)
all within tolerance. From that:
  time F1@20  = 2 (matched - retime) / (draft_notes + reference_notes)   (per note; type/lane ignored)
  keep F1     = 2 keep / (draft_notes + reference_notes)                  (time + type + lane + tail)
Pooled over notes per mode, with a song-level bootstrap 80% CI.

  python -m scripts.onset_agreement --model ft/wf_base/workflow-benchmark.json [--readout consensus_8] \
      --human human-floor-full/workflow-benchmark.json
Human references: for each human chart as the "draft", its best-matching other chart of the song
(min CER; the same rule as the model's min-ref), plus all same-coarse-difficulty pairs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def pair_counts(row):
    ops = row['ops']
    matched = row['reference_notes'] - ops['insert']
    if matched != row['draft_notes'] - ops['delete']:
        raise ValueError(f"inconsistent ops for {row['song_id']}")
    return dict(draft=row['draft_notes'], ref=row['reference_notes'], time=matched - ops['retime'], keep=ops['keep'])


def min_ref_rows(rows):
    """Best-matching reference per (song, draft) by CER."""
    best = {}
    for r in rows:
        key = (r['song_id'], r['draft'])
        if key not in best or r['cer'] < best[key]['cer']:
            best[key] = r
    return list(best.values())


def summarize(rows, *, n_boot=2000, seed=0):
    by_mode = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mode[r['mode']][r['song_id']].append(pair_counts(r))
    out = {}
    rng = np.random.default_rng(seed)
    for mode, songs in sorted(by_mode.items()):
        ids = sorted(songs)
        tot = {s: {k: sum(c[k] for c in songs[s]) for k in ('draft', 'ref', 'time', 'keep')} for s in ids}

        def f1(sel, key):
            num = sum(tot[s][key] for s in sel)
            den = sum(tot[s]['draft'] + tot[s]['ref'] for s in sel)
            return 2 * num / den if den else float('nan')
        block = dict(songs=len(ids), pairs=sum(len(v) for v in songs.values()),
                     draft_notes=sum(t['draft'] for t in tot.values()), reference_notes=sum(t['ref'] for t in tot.values()))
        for key, name in (('time', 'time_f1_20ms'), ('keep', 'keep_f1')):
            boots = [f1(rng.choice(ids, len(ids)), key) for _ in range(n_boot)]
            block[name] = dict(value=round(f1(ids, key), 4), ci80=[round(float(x), 4) for x in np.percentile(boots, [10, 90])])
        d, r_, t = block['draft_notes'], block['reference_notes'], sum(x['time'] for x in tot.values())
        block['time_precision_20ms'] = round(t / d, 4) if d else None
        block['time_recall_20ms'] = round(t / r_, 4) if r_ else None
        out[mode] = block
    return out


def model_rows(path, readout=None):
    d = json.loads(Path(path).read_text())
    rows = d['rows'] if readout is None else d['readout_rows'][readout]['rows']
    return min_ref_rows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, action='append', default=[])
    p.add_argument('--readout')
    p.add_argument('--human', type=Path)
    a = p.parse_args()
    report = {}
    for path in a.model:
        report[f'model:{path}:{a.readout or "single"}'] = summarize(model_rows(path, a.readout))
    if a.human:
        rows = json.loads(a.human.read_text())['rows']
        report[f'human_min_ref:{a.human}'] = summarize(min_ref_rows(rows))
        report[f'human_same_coarse:{a.human}'] = summarize([r for r in rows if r['coarse_stratum'] == r['draft_coarse']])
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

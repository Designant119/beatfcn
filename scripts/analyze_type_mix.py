"""Where does the Phira type collapse (and the mania jack excess) come from? Local, light, sampled; no model is loaded.

Numbers: raw manifest vs tape mix (drag policy); exposure-weighted and loss-weighted mix the model trains on; a
first-order kind chain under decode temperature; real r5 samples at kind T .3 and 1.0 incl. kind confusion against
the human chart; mania same-key repeats in native vs density-edited training targets vs r5 samples.

  nice -n 10 python -m scripts.analyze_type_mix --out REPORT_DIR/type-mix.json
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.events.canonical import TYPES  # noqa: E402
from beatfcn.model.event_canvas import RhythmEvent  # noqa: E402
from beatfcn.model.full_event_codec import EventRequest, EventVocabulary, encode_events  # noqa: E402
from beatfcn.train.counterfactual_density import edit_events_to_multiplier  # noqa: E402

CORE_US = 8_000_000
FAST_US = 250_000  # same "fast" gap as the report D style feature p_jack
K = len(TYPES)


def shares(counts):
    total = float(np.sum(counts))
    return {t: round(float(c)/total, 4) if total else None for t, c in zip(TYPES, counts)}


def kind_counts(kinds):
    return np.bincount(np.asarray(kinds, np.int64), minlength=K)[:K]


def chart_table(meta_rows):
    """Tape order -> first request index of each chart (requests are consecutive 8 s cores)."""
    first, cursor = [], 0
    for row in meta_rows:
        first.append(cursor)
        cursor += math.ceil(row['duration_us']/CORE_US)
    return np.asarray(first), cursor


@lru_cache(maxsize=256)
def load_chart(tape, name):
    with np.load(Path(tape)/name, allow_pickle=False) as z:
        return tuple(z[k].astype(np.int64) for k in ('onset_us', 'end_us', 'kind', 'lane'))


def window(arrays, start, end, duration, mode):
    on, stop, kind, lane = arrays
    lo, hi = np.searchsorted(on, [start, end])
    events = [RhythmEvent(int(on[i]), int(stop[i]), TYPES[int(kind[i])], int(lane[i])) for i in range(lo, hi)]
    carry_index = np.flatnonzero((on < start) & (stop > start) & (kind == TYPES.index('hold')))
    carry = tuple(RhythmEvent(int(on[i]), int(stop[i]), 'hold', int(lane[i])) for i in carry_index)
    return events, EventRequest(mode, start, end, duration, known_carry=carry)


def same_key_stats(events):
    """Report D's p_jack and p_chordjack numerators/denominator over one list of mania events."""
    groups = defaultdict(int)
    for e in events:
        groups[e.onset_us] |= 1 << e.lane
    times = sorted(groups)
    jack = chord = fast = 0
    for a, b in zip(times, times[1:]):
        if b-a > FAST_US:
            continue
        fast += 1
        if groups[a] & groups[b]:
            if max(bin(groups[a]).count('1'), bin(groups[b]).count('1')) == 1:
                jack += 1
            else:
                chord += 1
    return np.asarray([jack, chord, fast], np.int64)


def raw_vs_tape(manifest_path, meta_rows, tape):
    """Manifest type counts are before the drag policy; the tape's are after."""
    rows = {r['chart_id']: r for r in meta_rows if r['mode'] == 'phira'}
    out = {split: dict(raw=np.zeros(K, np.int64), tape=np.zeros(K, np.int64), chart_raw=[], chart_tape=[], charts=0,
                       with_removal=0, other_type_diff=0) for split in ('train', 'val')}
    with open(manifest_path) as stream:
        for line in stream:
            if '"phira:' not in line[:40]:
                continue
            m = json.loads(line)
            row = rows.get(m['chart_id'])
            if row is None or row['split'] not in out:
                continue
            raw = np.asarray([m['types'].get(t, 0) for t in TYPES], np.int64)
            _, _, kind, _ = load_chart(tape, row['target'])
            kept = kind_counts(kind)
            entry = out[row['split']]
            entry['raw'] += raw
            entry['tape'] += kept
            entry['charts'] += 1
            entry['with_removal'] += int(raw[3] > kept[3])
            entry['other_type_diff'] += int((raw[:3] != kept[:3]).any())
            if raw.sum() and kept.sum():
                entry['chart_raw'].append(raw/raw.sum())
                entry['chart_tape'].append(kept/kept.sum())
    result = {}
    for split, e in out.items():
        chart_raw, chart_tape = np.asarray(e['chart_raw']), np.asarray(e['chart_tape'])
        result[split] = dict(
            charts=e['charts'], charts_with_drag_removed=e['with_removal'], charts_other_types_differ=e['other_type_diff'],
            raw_counts=dict(zip(TYPES, e['raw'].tolist())), tape_counts=dict(zip(TYPES, e['tape'].tolist())),
            drag_removed_fraction=round(float(1-e['tape'][3]/max(e['raw'][3], 1)), 4),
            event_share_raw=shares(e['raw']), event_share_tape=shares(e['tape']),
            chart_mean_raw=dict(zip(TYPES, chart_raw.mean(0).round(4).tolist())),
            chart_mean_tape=dict(zip(TYPES, chart_tape.mean(0).round(4).tolist())),
            chart_median_drag_raw=round(float(np.median(chart_raw[:, 3])), 4),
            chart_median_drag_tape=round(float(np.median(chart_tape[:, 3])), 4),
            charts_no_drag_tape=round(float((chart_tape[:, 3] == 0).mean()), 4),
            top10pct_charts_share_of_drags_tape=round(float(np.sort(chart_tape[:, 3])[-len(chart_tape)//10:].sum()
                                                            / max(chart_tape[:, 3].sum(), 1e-9)), 4))
    return result


def exposure_mix(exposure, meta_rows, firsts, tape, *, vocab, sample, seed, type_weight):
    """The Phira mix per exposed slot, after density edits, with the per-request loss normalisation applied.

    event_loss averages per request and type_ce per request's kind tokens, so one kind token's weight in the
    objective is 1/T + type_weight/K (T tokens, K kind tokens of its window)."""
    slots = [(step['indices'][s], step['rates'][s], step['edit_seeds'][s]) for step in exposure['steps']
             for s in range(4)]
    charts = np.searchsorted(firsts, [s[0] for s in slots], side='right')-1
    phira = [i for i, c in enumerate(charts) if meta_rows[c]['mode'] == 'phira']
    rng = np.random.default_rng(seed)
    chosen = rng.choice(phira, min(sample, len(phira)), replace=False)
    per = defaultdict(lambda: dict(events=np.zeros(K), loss=np.zeros(K), windows=0, drag_windows=0, empty=0,
                                   window_share=np.zeros(K)))
    for i in chosen:
        index, rate, edit_seed = slots[i]
        row = meta_rows[charts[i]]
        arrays = load_chart(tape, row['target'])
        start = int(index-firsts[charts[i]])*CORE_US
        events, request = window(arrays, start, min(start+CORE_US, row['duration_us']), row['duration_us'], 'phira')
        if rate is not None:
            events, _ = edit_events_to_multiplier(events, request.start_us, request.end_us, float(rate), 'phira',
                                                  request.duration_us, seed=int(edit_seed), carry=request.known_carry)
        for key in ('all', 'native' if rate is None else f'x{rate:g}'):
            entry = per[key]
            entry['windows'] += 1
            if not events:
                entry['empty'] += 1
                continue
            counts = kind_counts([TYPES.index(e.type) for e in events])
            tokens = len(encode_events(events, request, vocab=vocab))
            entry['events'] += counts
            entry['loss'] += counts*(1/tokens+type_weight/len(events))
            entry['window_share'] += counts/len(events)
            entry['drag_windows'] += int(counts[3] > 0)
    return {key: dict(windows=e['windows'], empty_windows=e['empty'], windows_with_drag=e['drag_windows'],
                      event_weighted=shares(e['events']), loss_weighted=shares(e['loss']),
                      window_mean=shares(e['window_share'])) for key, e in per.items()}


def kind_chain(meta_rows, tape, *, charts, seed, class_weights):
    """First-order P(next kind | previous kind) on train charts; its stationary mix under decode temperature, with and
    without a converged class-weight prior shift. The real model sees more context, so this bounds the shape only."""
    rows = [r for r in meta_rows if r['mode'] == 'phira' and r['split'] == 'train']
    picks = np.random.default_rng(seed).choice(len(rows), min(charts, len(rows)), replace=False)
    counts = np.zeros((K, K))
    for i in picks:
        _, _, kind, _ = load_chart(tape, rows[int(i)]['target'])
        np.add.at(counts, (kind[:-1], kind[1:]), 1)
    p = counts/counts.sum(1, keepdims=True)

    def stationary(t, w=np.ones(K)):
        q = (p*w)**(1/t)
        q = q/q.sum(1, keepdims=True)
        v = np.ones(K)/K
        for _ in range(5000):
            v = v@q
        return dict(zip(TYPES, v.round(4).tolist()))
    return dict(charts=len(picks), transition={a: dict(zip(TYPES, p[i].round(4).tolist())) for i, a in enumerate(TYPES)},
                stationary={str(t): stationary(t) for t in (1., .7, .5, .3)},
                class_weights=class_weights, stationary_weighted={str(t): stationary(t, np.asarray([class_weights[k] for k in TYPES]))
                                                                  for t in (1., .7, .5, .3)})


def human_of(meta_rows, tape, mode):
    by_audio = defaultdict(list)
    for r in meta_rows:
        if r['mode'] == mode and r['split'] == 'val':
            by_audio[r['audio']].append(r)
    return by_audio


def confusion(model_events, human, tol_us=50_000):
    """Model kind given the nearest human head within 50 ms (one-to-one by time order)."""
    matrix = np.zeros((K, K), np.int64)
    on, _, kind, _ = human
    used = np.zeros(len(on), bool)
    for e in model_events:
        i = np.searchsorted(on, e['onset_us'])
        best = None
        for j in range(max(i-3, 0), min(i+3, len(on))):
            if not used[j] and abs(on[j]-e['onset_us']) <= tol_us and (
                    best is None or abs(on[j]-e['onset_us']) < abs(on[best]-e['onset_us'])):
                best = j
        if best is not None:
            used[best] = True
            matrix[kind[best], TYPES.index(e['type'])] += 1
    return matrix


def sample_mix(path, meta_rows, tape, *, single_samples=True):
    """Chart-mean Phira mix, confusion and mania same-key stats of saved r5 drafts (seed rows or consensus)."""
    data = json.loads(Path(path).read_text())
    if single_samples:
        rows = [row for seed in data.values() for row in seed.values()]
    else:
        rows = [row for row in (data.values() if isinstance(data, dict) else data)]
    phira = human_of(meta_rows, tape, 'phira')
    mix, matrix, jack = [], np.zeros((K, K), np.int64), np.zeros(3, np.int64)
    jack_chart = []
    for row in rows:
        events = row['events']
        if row['mode'] == 'phira' and events:
            mix.append(kind_counts([TYPES.index(e['type']) for e in events])/len(events))
            refs = phira.get(row['audio_sha256'])
            if refs:
                matrix += confusion(events, load_chart(tape, refs[0]['target']))
        elif row['mode'] == 'mania_4k':
            stats = same_key_stats([RhythmEvent(e['onset_us'], e['end_us'], e['type'], e['lane']) for e in events])
            jack += stats
            jack_chart.append(stats[0]/max(stats[2], 1))
    recall = {t: (dict(zip(TYPES, (matrix[i]/max(matrix[i].sum(), 1)).round(3).tolist())) | {'n': int(matrix[i].sum())})
              for i, t in enumerate(TYPES)}
    return dict(drafts=len(rows), phira_chart_mean=dict(zip(TYPES, np.mean(mix, 0).round(4).tolist())),
                model_kind_given_human_kind=recall,
                mania_p_jack_chart_mean=round(float(np.mean(jack_chart)), 4) if jack_chart else None,
                mania_p_jack_pooled=round(float(jack[0]/max(jack[2], 1)), 4))


def human_panel_mix(sample_path, meta_rows, tape):
    """Tape mix (first val chart per audio) on the same songs as the saved samples."""
    seed = next(iter(json.loads(Path(sample_path).read_text()).values()))
    phira = human_of(meta_rows, tape, 'phira')
    mania = human_of(meta_rows, tape, 'mania_4k')
    mix, jack = [], []
    for row in seed.values():
        if row['mode'] == 'phira' and row['audio_sha256'] in phira:
            _, _, kind, _ = load_chart(tape, phira[row['audio_sha256']][0]['target'])
            mix.append(kind_counts(kind)/max(len(kind), 1))
        elif row['mode'] == 'mania_4k' and row['audio_sha256'] in mania:
            on, end, kind, lane = load_chart(tape, mania[row['audio_sha256']][0]['target'])
            stats = same_key_stats([RhythmEvent(int(a), int(b), TYPES[int(k)], int(c)) for a, b, k, c in zip(on, end, kind, lane)])
            jack.append(stats[0]/max(stats[2], 1))
    return dict(phira_chart_mean=dict(zip(TYPES, np.mean(mix, 0).round(4).tolist())), phira_songs=len(mix),
                mania_p_jack_chart_mean=round(float(np.mean(jack)), 4), mania_songs=len(jack))


def mania_edits(exposure, meta_rows, firsts, tape, *, sample, seed, avoid_same_lane=False):
    """Report D's p_jack on the windows the model is actually trained on, per density treatment."""
    slots = [(step['indices'][s], step['rates'][s], step['edit_seeds'][s]) for step in exposure['steps']
             for s in range(4)]
    charts = np.searchsorted(firsts, [s[0] for s in slots], side='right')-1
    mania = [i for i, c in enumerate(charts) if meta_rows[c]['mode'] == 'mania_4k']
    chosen = np.random.default_rng(seed).choice(mania, min(sample, len(mania)), replace=False)
    totals = defaultdict(lambda: np.zeros(3, np.int64))
    windows = Counter()
    for i in chosen:
        index, rate, edit_seed = slots[i]
        row = meta_rows[charts[i]]
        arrays = load_chart(tape, row['target'])
        start = int(index-firsts[charts[i]])*CORE_US
        events, request = window(arrays, start, min(start+CORE_US, row['duration_us']), row['duration_us'], 'mania_4k')
        native = same_key_stats(events)
        key = 'native' if rate is None else f'x{rate:g}'
        windows[key] += 1
        if rate is not None and rate > 0:
            edited, _ = edit_events_to_multiplier(events, request.start_us, request.end_us, float(rate), 'mania_4k',
                                                  request.duration_us, seed=int(edit_seed), carry=request.known_carry,
                                                  avoid_same_lane=avoid_same_lane)
            stats = same_key_stats(edited)
            totals[key] += stats
            totals['all_trained'] += stats
            totals[key+'_source'] += native
        elif rate is None:
            totals['native'] += native
            totals['all_trained'] += native
    return dict(windows=dict(windows), **{key: dict(p_jack=round(float(v[0]/max(v[2], 1)), 4),
        p_chordjack=round(float(v[1]/max(v[2], 1)), 4), fast_gaps=int(v[2])) for key, v in totals.items()})


def main():
    report = Path('/Users/yukoval/yukoval-projects/beatfcn/reports')
    goal = report/'goal-100usd-20261101'
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--tape', default='/Users/yukoval/beatfcn-backup/data/native-v2-tape')
    a.add_argument('--meta', default=str(report/'research-20261104/data/meta.json'))
    a.add_argument('--manifest', default='/Users/yukoval/yukoval-projects/beatfcn/inputs/full-database-cloud-v1/manifest/'
                                         'charts.offset-v2-3d99ff6bd5e7.jsonl')
    a.add_argument('--exposure', default='/Users/yukoval/beatfcn-backup/round4/round4-subdiv/exposure.json')
    a.add_argument('--samples-t03', default=str(goal/'r5/sdecode/P1_anchor/drafts-samples.json'))
    a.add_argument('--samples-t1', default=str(goal/'r5/sdecode/K1_kind1/drafts-samples.json'))
    a.add_argument('--consensus-t03', default=str(goal/'r5/D_cold_r5/drafts-consensus-8-jackfree.json'))
    a.add_argument('--consensus-t1', default=str(goal/'r5/sdecode/K1_kind1/drafts-consensus-8-jackfree.json'))
    a.add_argument('--r100-samples', default=str(goal/'random100/A_cold/drafts-samples.json'))
    a.add_argument('--r100-consensus', default=str(goal/'random100/A_cold/drafts-consensus-8-jackfree.json'))
    a.add_argument('--sample', type=int, default=4000)
    a.add_argument('--seed', type=int, default=20261104)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    meta_rows = json.loads(Path(args.meta).read_text())
    firsts, total = chart_table(meta_rows)
    tape, vocab = args.tape, EventVocabulary(relative_blocks=True, tail_query=True)
    exposure = json.loads(Path(args.exposure).read_text())
    out = dict(requests_from_meta=total, seed=args.seed)
    out['raw_vs_tape'] = raw_vs_tape(args.manifest, meta_rows, tape)
    out['kind_chain'] = kind_chain(meta_rows, tape, charts=800, seed=args.seed,
                                   class_weights=dict(tap=.68, hold=1.33, flick=1.87, drag=1.74))
    out['exposure_mix'] = exposure_mix(exposure, meta_rows, firsts, tape, vocab=vocab, sample=args.sample,
                                       seed=args.seed, type_weight=1.)
    out['mania_edits'] = mania_edits(exposure, meta_rows, firsts, tape, sample=args.sample, seed=args.seed)
    out['mania_edits_avoid_same_lane'] = mania_edits(exposure, meta_rows, firsts, tape, sample=args.sample, seed=args.seed,
                                                     avoid_same_lane=True)
    out['r5_samples_random100_cold_kind_t03'] = dict(
        human_same_songs=human_panel_mix(args.r100_samples, meta_rows, tape),
        single=sample_mix(args.r100_samples, meta_rows, tape),
        consensus=sample_mix(args.r100_consensus, meta_rows, tape, single_samples=False))
    out['r5_samples'] = dict(
        human_same_songs=human_panel_mix(args.samples_t1, meta_rows, tape),
        single_kind_t03=sample_mix(args.samples_t03, meta_rows, tape),
        single_kind_t1=sample_mix(args.samples_t1, meta_rows, tape),
        consensus_kind_t03=sample_mix(args.consensus_t03, meta_rows, tape, single_samples=False),
        consensus_kind_t1=sample_mix(args.consensus_t1, meta_rows, tape, single_samples=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1)+'\n')
    print(args.out)


if __name__ == '__main__':
    main()

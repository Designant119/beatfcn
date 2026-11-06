"""Human reference numbers for benchmark v2 (reads tape targets only; no model).

  .venv/bin/python -m scripts.benchmark_human_reference --tape inputs/native-v2-d94-580/tape.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.train.requirement_diagnostics import GAP_EDGES_MS, SHORT_GAP_MS

ROOT = Path(__file__).resolve().parents[1]


def chart_stats(root, chart):
    with np.load(root/chart['target_path'], allow_pickle=False) as archive:
        onset, kind = archive['onset_us'], archive['kind']
    distinct = np.unique(onset)
    gaps = np.diff(distinct)/1000.
    windows = [(r['request']['end_us']-r['request']['start_us'], r['event_count']) for r in chart['requests']]
    return gaps, Counter(TYPES[int(k)] for k in kind), windows, int((kind == TYPES.index('hold')).sum())


def reference(root, tape):
    manifest = json.loads((root/tape).read_text())
    out = {}
    for split in ('train', 'val', 'all'):
        for mode in ('mania_4k', 'phira'):
            charts = [c for c in manifest['charts'] if c['mode'] == mode and (split == 'all' or c['split'] == split)]
            if not charts:
                continue
            gaps, types, rates, silent, chart_short = [], Counter(), [], 0, []
            for chart in charts:
                g, t, windows, _ = chart_stats(root, chart)
                gaps.append(g)
                types.update(t)
                chart_short.append(float((g < SHORT_GAP_MS).mean()) if len(g) else 0.)
                for span, count in windows:
                    rates.append(count/(span/1e6))
                    silent += count == 0
            gaps = np.concatenate(gaps)
            hist = np.histogram(gaps, GAP_EDGES_MS)[0]
            rates = np.asarray(rates)
            total = sum(types.values())
            out[f'{split}/{mode}'] = dict(charts=len(charts), gaps=int(len(gaps)),
                short_gap_fraction_lt40ms=float((gaps < 40).mean()), lt30ms=float((gaps < 30).mean()),
                lt20ms=float((gaps < 20).mean()), lt10ms=float((gaps < 10).mean()),
                per_chart_short_gap_fraction_p50=float(np.median(chart_short)),
                per_chart_short_gap_fraction_p90=float(np.quantile(chart_short, .9)),
                gap_ms_p01=float(np.quantile(gaps, .01)), gap_ms_p05=float(np.quantile(gaps, .05)),
                gap_ms_median=float(np.median(gaps)),
                gap_histogram=dict(edges_ms=[e if np.isfinite(e) else 'inf' for e in GAP_EDGES_MS],
                                   fraction=(hist/hist.sum()).round(4).tolist()),
                type_mix={k: round(types[k]/total, 4) for k in TYPES if types[k]},
                window_events_per_s=dict(mean=float(rates.mean()), p10=float(np.quantile(rates, .1)),
                    p50=float(np.median(rates)), p90=float(np.quantile(rates, .9))),
                silent_window_fraction=silent/len(rates), windows=len(rates))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tape', type=Path, default=Path('inputs/native-v2-d94-580/tape.json'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    value = dict(tape=str(a.tape), statistics=reference(a.root, a.tape),
                 note='distinct-onset gaps (chord members are not gaps); windows = tape requests')
    text = json.dumps(value, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()

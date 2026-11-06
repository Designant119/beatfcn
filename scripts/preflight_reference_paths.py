"""Pre-flight for the reference-input data paths of a training plan (CPU, no model, no features).

Runs the EXACT training encoders over every train chart that the paths can touch, so a bad chart
fails here instead of killing a GPU run hours in:
  - alternative_context: every (chart, sibling) pair of SiblingIndex, windows at song start /
    middle / end, region [start - pad, end + pad), reference_inputs_with_alternative;
  - history_context: the longest prefix (and an infill suffix) of the same windows, check_no_leak +
    reference encoding;
  - charter_timing: segment_array over every valid table entry (the loader's validator).

  python -m scripts.preflight_reference_paths --plan /data/<run>/plan.json [--root /] [--out report.json]
Exit 1 with the first failures listed when anything fails.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TapePanel:
    """Charts / records / chart arrays of a request tape (targets only; no features)."""

    def __init__(self, root, tape):
        self.root = Path(root)
        manifest = json.loads((self.root / tape.lstrip('/')).read_text()) if str(tape).startswith('/') \
            else json.loads((self.root / tape).read_text())
        self.charts = manifest['charts']
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index, chart_id=chart['chart_id'],
                                         audio_sha256=chart['audio_sha256'], split=chart['split'], mode=chart['mode']))
        self._cache = {}

    def _arrays(self, chart_index):
        if chart_index not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            path = self.root / self.charts[chart_index]['target_path'].lstrip('/')
            with np.load(path, allow_pickle=False) as z:
                self._cache[chart_index] = {k: z[k].copy() for k in ('onset_us', 'end_us', 'kind', 'lane')}
        return self._cache[chart_index]


def preflight(panel, *, alternative=None, history=None, max_failures=20):
    from beatfcn.train.alternative_context import (SiblingIndex, alternative_source, normalize_alternative_context,
                                                   reference_inputs_with_alternative)
    from beatfcn.train.history_context import chart_arrays, history_sources, normalize_history_context
    from beatfcn.train.local_request_panel import event_request
    alternative = normalize_alternative_context(alternative) if alternative else None
    history = normalize_history_context(history) if history else None
    by_chart = {}
    for i, r in enumerate(panel.records):
        if r['split'] == 'train':
            by_chart.setdefault(r['chart_index'], []).append(i)
    index = SiblingIndex(panel)
    counts, failures = dict(windows=0, pairs=0, alternative_ok=0, history_ok=0), []
    charts = sorted(by_chart) if history else sorted({c for g in index.groups.values() for c in g})
    for chart_index in charts:
        indices = sorted(by_chart.get(chart_index, ()), key=lambda i: panel.records[i]['request']['start_us'])
        if not indices:
            continue
        picks = sorted({indices[0], indices[len(indices) // 2], indices[-1]})
        siblings = index.siblings(chart_index) if alternative else []
        for i in picks:
            record = panel.records[i]
            request = event_request(record['request'])
            counts['windows'] += 1
            hist = ()
            if history:
                draw = dict(applied=True, prefix_s=max(history['lengths_s']), role='human_input', infill=True,
                            suffix_s=max(history['suffix_lengths_s']))
                try:
                    hist, _ = history_sources(chart_arrays(panel, chart_index), request, draw, history,
                                              chart_id=record['chart_id'], audio_sha256=record['audio_sha256'])
                    reference_inputs_with_alternative(request, record, hist, ())
                    counts['history_ok'] += 1
                except Exception as exc:  # noqa: BLE001 - collected
                    failures.append(dict(path='history', chart=record['chart_id'], start_us=request.start_us,
                                         error=f'{type(exc).__name__}: {exc}'[:300]))
            for j in siblings:
                counts['pairs'] += 1
                near = dict(applied=True, denser=index.heads(j) > index.heads(chart_index), pick=0.)
                try:
                    # force this sibling: the same region/encoding as alternative_source, any draw
                    from beatfcn.train.alternative_context import ROLE
                    from beatfcn.train.history_context import visible_source
                    pad = int(round(alternative['pad_s'] * 1e6))
                    region = (max(0, request.start_us - pad), min(request.duration_us, request.end_us + pad))
                    source = visible_source(chart_arrays(panel, j), mode=request.mode, audio_sha256=record['audio_sha256'],
                                            role=ROLE, region=region, source_id=f"alternative:{panel.charts[j]['chart_id']}")
                    reference_inputs_with_alternative(request, record, hist, (source,),
                                                      max_rows=4096 + alternative['max_reference_rows'])
                    alternative_source(index, request, record, near, alternative)
                    counts['alternative_ok'] += 1
                except Exception as exc:  # noqa: BLE001 - collected
                    failures.append(dict(path='alternative', chart=record['chart_id'], sibling=panel.charts[j]['chart_id'],
                                         start_us=request.start_us, error=f'{type(exc).__name__}: {exc}'[:300]))
            if len(failures) >= max_failures:
                return counts, failures
    return counts, failures


def check_timing_table(path):
    from beatfcn.train.charter_timing import segment_array
    table = json.loads(Path(path).read_text())
    bad = []
    for chart_id, entry in table['charts'].items():
        if entry.get('valid'):
            try:
                segment_array(entry['segments'])
            except Exception as exc:  # noqa: BLE001 - collected
                bad.append(dict(chart=chart_id, error=str(exc)[:200]))
    return dict(valid=sum(bool(e.get('valid')) for e in table['charts'].values()), failures=bad)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/'))
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    started = time.monotonic()
    plan = json.loads(a.plan.read_text())
    report = dict(plan=str(a.plan), plan_id=plan.get('id'))
    if plan.get('charter_timing'):
        report['charter_timing'] = check_timing_table(a.root / plan['charter_timing']['path'].lstrip('/'))
    panel = TapePanel(a.root, plan['tape'])
    counts, failures = preflight(panel, alternative=plan.get('alternative_context'), history=plan.get('history_context'))
    report.update(counts=counts, failures=failures, wall_s=round(time.monotonic() - started, 1))
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')
    if failures or report.get('charter_timing', {}).get('failures'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

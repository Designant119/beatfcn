"""Modal app: data analyses that feed design decisions.

dup_onsets: do exact duplicate Phira notes (same line/side/x/type/time) sit on stronger audio
onsets than ordinary notes? Per chart, each head time gets the onset strength (max positive
spectral flux within +-20 ms, after the chart's own global offset delta) and is classed as
dup / chord (>=2 heads at the time, no exact duplicate) / single. Strength is compared as a
within-chart percentile so loud and quiet songs are comparable.
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA = '/data'
app = modal.App('beatfcn-analysis')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy==2.2.6', 'pyyaml==6.0.3')
         .add_local_python_source('beatfcn'))


def _classify(events):
    from collections import defaultdict
    by_t = defaultdict(list)
    for e in events:
        loc = e['location']
        by_t[e['onset_us']].append((e['type'], loc.get('line_id'), loc.get('side'), loc.get('native_x')))
    out = {}
    for t, sigs in by_t.items():
        dup = len(sigs) != len(set(sigs))
        out[t] = 'dup' if dup else ('chord' if len(sigs) >= 2 else 'single')
    return out, by_t


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=4096, timeout=1800)
def dup_chunk(rows: list) -> list:
    import numpy as np
    from beatfcn.ingest.offset import flux_envelope
    res = []
    for r in rows:
        try:
            doc = json.load(open(f"{DATA}/canon/phira/{r['native_id']}.json"))
            mel = np.load(f"{DATA}/mel10/{r['audio_sha256']}.npy").astype(np.float32)
        except Exception as exc:
            res.append({'chart_id': r['chart_id'], 'error': str(exc)[:120]})
            continue
        env = flux_envelope(mel)
        delta = ((r.get('offset') or {}).get('delta_ms') or 0) / 1000.0
        cls, by_t = _classify(doc['events'])
        times = np.array(sorted(cls), dtype=np.int64)
        fr = np.arange(len(env)) * 0.01

        def strength(t_sec):
            a = max(int((t_sec - 0.02) * 100), 0)
            b = min(int((t_sec + 0.02) * 100) + 1, len(env))
            return float(env[a:b].max()) if b > a else float('nan')
        s = np.array([strength(t / 1e6 + delta) for t in times])
        ok = np.isfinite(s)
        pct = np.full(len(s), np.nan)
        if ok.sum() >= 10:
            order = s[ok].argsort().argsort()
            pct[ok] = (order + 0.5) / ok.sum()
        kinds = np.array([cls[int(t)] for t in times])
        dup_sigs = []
        for t in times[kinds == 'dup']:
            sigs = by_t[int(t)]
            for sig in set(sigs):
                n = sigs.count(sig)
                if n > 1:
                    dup_sigs.append({'type': sig[0], 'mult': n})
        out = {'chart_id': r['chart_id'], 'name': r.get('title'), 'rating': r.get('rating'), 'n_heads': int(len(times)),
               'n_dup_times': int((kinds == 'dup').sum()), 'n_chord_times': int((kinds == 'chord').sum()),
               'delta_ms': r.get('offset', {}).get('delta_ms'), 'dup_types': dup_sigs}
        for k in ('dup', 'chord', 'single'):
            m = (kinds == k) & np.isfinite(pct)
            out[f'pct_{k}'] = float(np.median(pct[m])) if m.any() else None
            out[f'n_{k}'] = int(m.sum())
            out[f'top25_{k}'] = float((pct[m] >= 0.75).mean()) if m.any() else None
        # one 8 s example window centred on the densest duplicate cluster
        if (kinds == 'dup').any():
            dt = times[kinds == 'dup']
            centre = int(dt[np.argmax([(np.abs(dt - t) < 4_000_000).sum() for t in dt])])
            a, b = centre - 4_000_000, centre + 4_000_000
            lines = sorted({e['location'].get('line_id') for e in doc['events']}, key=lambda x: (len(str(x)), str(x)))
            li = {l: i for i, l in enumerate(lines)}
            notes = [{'t': round((e['onset_us'] - a) / 1e6, 3), 'line': li[e['location'].get('line_id')],
                      'x': e['location'].get('native_x'), 'type': e['type'], 'side': e['location'].get('side'),
                      'dup': cls[e['onset_us']] == 'dup' and by_t[e['onset_us']].count(
                          (e['type'], e['location'].get('line_id'), e['location'].get('side'), e['location'].get('native_x'))) > 1}
                     for e in doc['events'] if a <= e['onset_us'] < b]
            i0, i1 = max(int((a / 1e6 + delta) * 100), 0), min(int((b / 1e6 + delta) * 100), len(env))
            out['example'] = {'start_s': round(a / 1e6, 2), 'notes': notes,
                              'env': [round(float(v), 2) for v in env[i0:i1]], 'env_offset_s': round(max(-(a / 1e6 + delta), 0), 2)}
        res.append(out)
    return res


@app.local_entrypoint()
def dup_onsets(manifest: str, out: str = 'reports/phira_duplicate_onsets.json'):
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r['source'] == 'phira' and r.get('status') == 'parsed'
            and 'duplicate_object' in (r.get('quarantine_reasons') or [])]
    chunks = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    res = [x for part in dup_chunk.map(chunks) for x in part]
    Path(out).write_text(json.dumps(res, ensure_ascii=False))
    import numpy as np
    good = [r for r in res if 'error' not in r and r.get('pct_dup') is not None]
    print(json.dumps({'charts': len(rows), 'analysed': len(good),
                      'median_pct': {k: float(np.median([r[f'pct_{k}'] for r in good if r[f'pct_{k}'] is not None])) for k in ('dup', 'chord', 'single')}}))

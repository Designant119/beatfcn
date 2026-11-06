"""Read-only condition scope/coverage audit; no model inference or relabelling."""
from collections import Counter

from ..events import controls as C
from ..events.canonical import Event
from ..model import features as F


def inspect_sample(sample, c0, duration):
    request = sample.request
    mode = request['mode']
    regions = F.editable_runs(request['regions'])
    core = C.intersect(regions, [(c0, min(c0 + F.CORE_US, duration))])
    events = [Event.from_json(e) for e in request['observed_events']] + list(sample.hidden)
    request_raw = C.raw_controls(mode, events, regions)
    local_raw = C.raw_controls(mode, events, core)
    fields = {}
    for field in C.MODE_FIELDS[mode]:
        band = request['controls'].get(field)
        expected = C.band_of(field, request_raw[field])
        local_band = C.band_of(field, local_raw[field])
        fields[field] = dict(requested_band=band, request_truth_band=expected,
                             local_truth_band=local_band, request_value=request_raw[field],
                             local_value=local_raw[field],
                             label_inconsistent=band is not None and band != expected,
                             local_band_differs=(band != local_band if band is not None and local_band is not None else None))
    return dict(mode=mode, kind=sample.recipe['kind'], c0_us=int(c0),
                editable_request_us=C.length(regions), editable_core_us=C.length(core),
                exact_scope=C.union(regions) == C.union(core), fields=fields,
                local_heads=dict(Counter(e.type for e in events if C.contains(core, e.onset_us))))


def summarize(records):
    result = {}
    for r in records:
        key = f"{r['variant']}/{r['mode']}/{r['kind']}"
        cell = result.setdefault(key, dict(samples=0, exact_scope=0, local_heads=Counter(), fields={}))
        cell['samples'] += 1
        cell['exact_scope'] += int(r['exact_scope'])
        cell['local_heads'].update(r['local_heads'])
        for field, values in r['fields'].items():
            f = cell['fields'].setdefault(field, dict(present=0, missing=0, label_inconsistent=0,
                 comparable_local=0, local_band_differs=0, local_undefined_or_outside_table=0,
                 request_bands=Counter(), local_bands=Counter()))
            present = values['requested_band'] is not None
            f['present' if present else 'missing'] += 1
            f['label_inconsistent'] += int(values['label_inconsistent'])
            if values['local_band_differs'] is not None:
                f['comparable_local'] += 1
                f['local_band_differs'] += int(values['local_band_differs'])
            if values['local_truth_band'] is None:
                f['local_undefined_or_outside_table'] += 1
            for source, output in [('requested_band', 'request_bands'), ('local_truth_band', 'local_bands')]:
                if values[source] is not None:
                    f[output][str(values[source])] += 1
    return result

"""Lazy event supervision over admitted songs, with explicit reference permission."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, teacher_sequence
from beatfcn.train.event_reference import encode_reference
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.worker_cache import WorkerCacheMixin


def event_request(value):
    value = dict(value)
    value['known_carry'] = tuple(RhythmEvent(**e) for e in value.get('known_carry', ()))
    value['copy_times'] = tuple(value.get('copy_times', ()))
    if value.get('allowed_types') is not None:
        value['allowed_types'] = tuple(value['allowed_types'])
    return EventRequest(**value)


def request_inputs(features, audio, request, controls, *, sources=()):
    """Inputs for either teacher forcing or free generation; no target is read."""
    if request.duration_us != features.records[audio]['duration_us']:
        raise ValueError('Request duration differs from the admitted sample clock')
    reference, provenance = encode_reference(mode=request.mode, audio_sha256=audio,
        duration_us=request.duration_us, origin_us=request.start_us,
        unknown_regions=((request.start_us, request.end_us),), sources=sources)
    inputs = dict(features.window(audio, request.start_us, request.end_us), **reference,
                  controls=np.asarray(controls, np.float32),
                  mode=np.asarray(0 if request.mode == 'mania_4k' else 1, np.int64))
    return inputs, provenance


class LocalRequestPanel(WorkerCacheMixin):
    """Cache a few chart arrays and legal supports, not the full corpus in RAM."""
    worker_cache_specs = (('_arrays', '_read_arrays', 4), ('sequence', '_sequence', 4096))
    def __init__(self, root, tape, features, *, allow_partial_features=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root/tape).read_text())
        if not self.manifest['complete'] or self.manifest['failures']:
            raise ValueError('Request tape is incomplete or failed')
        self.features = LocalFeatureStore(root, features, allow_partial=allow_partial_features)
        self.charts = self.manifest['charts']
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index,
                    chart_id=chart['chart_id'], audio_sha256=chart['audio_sha256'],
                    component=chart['component'], split=chart['split'], mode=chart['mode'],
                    stratum=chart['stratum']))
        missing = {r['audio_sha256'] for r in self.records} - self.features.records.keys()
        if missing and not allow_partial_features:
            raise ValueError('Request tape includes unaudited audio')
        self._arrays = lru_cache(maxsize=4)(self._read_arrays)
        self.sequence = lru_cache(maxsize=4096)(self._sequence)

    def _read_arrays(self, index):
        chart = self.charts[index]
        tokens = np.load(self.root/chart['token_path'], mmap_mode='r', allow_pickle=False)
        with np.load(self.root/chart['target_path'], allow_pickle=False) as archive:
            targets = {k: archive[k].copy() for k in archive.files}
        return tokens, targets

    def _sequence(self, index):
        row = self.records[index]
        tokens, _ = self._arrays(row['chart_index'])
        values = tokens[row['token_start']:row['token_stop']].tolist()
        if len(values) != row['tokens']:
            raise ValueError('Prepared token slice changed')
        return teacher_sequence(values, event_request(row['request']))

    def targets(self, index):
        row = self.records[index]
        _, values = self._arrays(row['chart_index'])
        sl = slice(row['target_start'], row['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(*(values[key][sl] for key in ('onset_us', 'end_us', 'kind', 'lane')))]

    def row(self, index, *, controls=None):
        row = self.records[index]
        inputs, _ = request_inputs(self.features, row['audio_sha256'], event_request(row['request']),
                                   row['controls'] if controls is None else controls)
        return dict(inputs=inputs, sequence=self.sequence(index))

    def batch(self, indices, *, controls=None, device='cpu'):
        inputs, sequences = collate([self.row(i, controls=None if controls is None else controls[j])
                                    for j, i in enumerate(indices)])
        return ({k: v.to(device) for k, v in inputs.items()}, {k: v.to(device) for k, v in sequences.items()})

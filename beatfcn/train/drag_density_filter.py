"""Opt-in Owner data intervention: remove dense drag events, never whole charts.

Detect on the full original chart, so overlapping requests have identical labels.
Duplicates count as separate native objects. Only drag objects in qualifying
half-open windows are removed; every other event is preserved byte-for-byte.
"""
from collections import Counter
from functools import lru_cache

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.worker_cache import WorkerCacheMixin


RULE = dict(id='drag-density-1s-18-v1', window_us=1_000_000, minimum_drag_objects=18,
    interval='[first_drag_onset,first_drag_onset+1000000)',
    duplicate_policy='Count every original object, including exact simultaneous duplicates',
    removal='Union of drag objects inside every qualifying window;non-drag untouched',
    detection_scope='Entire original chart before request slicing;never iterative after removal')


def dense_drag_mask(onsets, kinds, *, window_us=1_000_000, minimum_drag_objects=18):
    onsets = np.asarray(onsets, np.int64)
    kinds = np.asarray(kinds)
    if onsets.shape != kinds.shape or window_us <= 0 or minimum_drag_objects < 2:
        raise ValueError('Invalid density filter contract')
    drag = np.flatnonzero(kinds == TYPES.index('drag'))
    order = np.argsort(onsets[drag], kind='stable')
    ids = drag[order]
    values = onsets[ids]
    delta = np.zeros(len(ids)+1, np.int64)
    right = 0
    windows = []
    for left in range(len(ids)):
        right = max(right, left)
        while right < len(ids) and int(values[right])-int(values[left]) < window_us:
            right += 1
        if right-left >= minimum_drag_objects:
            delta[left] += 1
            delta[right] -= 1
            windows.append((int(values[left]), int(values[left])+window_us, right-left))
    removed = np.zeros(len(onsets), bool)
    removed[ids[np.cumsum(delta[:-1]) > 0]] = True
    if np.any(removed & (kinds != TYPES.index('drag'))):
        raise AssertionError('Non-drag removal')
    return removed, windows


class DragFilteredPanel(WorkerCacheMixin):
    """Request-panel adapter used by both training and evaluation row factories."""
    worker_cache_specs = (('_arrays', '_read', 8),)
    def __init__(self, panel, *, window_us=1_000_000, minimum_drag_objects=18):
        self.base = panel
        self.window_us = window_us
        self.minimum_drag_objects = minimum_drag_objects
        self._arrays = lru_cache(maxsize=8)(self._read)
        self.records = [dict(v) for v in panel.records]

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _read(self, chart_index):
        chart = self.base.charts[chart_index]
        with np.load(self.base.root/chart['target_path'], allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        removed, windows = dense_drag_mask(arrays['onset_us'], arrays['kind'],
            window_us=self.window_us, minimum_drag_objects=self.minimum_drag_objects)
        return arrays, removed, windows

    def targets(self, index):
        row = self.records[index]
        values, removed, _ = self._arrays(row['chart_index'])
        ids = np.arange(row['target_start'], row['target_stop'])
        ids = ids[~removed[ids]]
        return [RhythmEvent(int(values['onset_us'][i]), int(values['end_us'][i]),
                            TYPES[int(values['kind'][i])], int(values['lane'][i])) for i in ids]

    def filter_events(self, index, events):
        values, removed, _ = self._arrays(self.records[index]['chart_index'])
        removed_times = set(map(int, values['onset_us'][removed]))
        return [e for e in events if e.type != 'drag' or e.onset_us not in removed_times]

    def removal_record(self, index):
        row = self.records[index]
        values, removed, _ = self._arrays(row['chart_index'])
        ids = np.arange(row['target_start'], row['target_stop'])
        removed_ids = ids[removed[ids]]
        return dict(index=index, chart_id=row['chart_id'], removed_absolute_array_indices=removed_ids.tolist(),
            original_events=len(ids), retained_events=len(ids)-len(removed_ids),
            types_before=dict(Counter(TYPES[int(k)] for k in values['kind'][ids])),
            types_after=dict(Counter(TYPES[int(k)] for k in values['kind'][ids[~removed[ids]]])))

    def sequence(self, index):
        from beatfcn.model.full_event_codec import encode_events, teacher_sequence
        from .local_request_panel import event_request
        request = event_request(self.records[index]['request'])
        return teacher_sequence(encode_events(self.targets(index), request), request)

    def batch(self, indices, *, controls=None, device='cpu'):
        from beatfcn.train.local_panel import collate
        inputs, sequences = collate([self.row(i, controls=None if controls is None
            else controls[j]) for j, i in enumerate(indices)])
        return ({k: v.to(device) for k, v in inputs.items()},
                {k: v.to(device) for k, v in sequences.items()})

    def row(self, index, *, controls=None):
        from .local_request_panel import event_request, request_inputs
        record = self.records[index]
        # Only explicit caller controls can be inherited; data-derived train
        # controls are recomputed by make_row from this adapter's targets.
        controls = record['controls'] if controls is None else controls
        inputs = self.base.row(index, controls=controls)['inputs']
        return dict(inputs=inputs, sequence=self.sequence(index))


def filtered_request_row(filtered_panel, raw_row):
    """Filter after the ORIGINAL public partition, preserving all other labels.

    Use this entry point for completion/training views, rather than repartitioning
    an already filtered chart. Derived training controls are refreshed, but an
    inference caller's control or exact copy anchors are never inferred from GT.
    """
    from .local_request_panel import event_request
    from .reference_completion import partition
    from .typed_inventory import carry_reference_row
    from .event_conditions import training_labels, encode_controls
    from beatfcn.model.local_song_rollout import canonical
    panel=filtered_panel.base
    index=raw_row['index'];record=panel.records[index];regions=()
    if raw_row['view'] in ('middle','continuation','random_groups'):
        original=event_request(record['request'])
        request,visible,_,regions=partition(panel.targets(index),original,raw_row['view'],
            f"{record['chart_id']}:{original.start_us}",20261027)
        if request != raw_row['request'] or list(visible) != list(raw_row['visible']):
            raise ValueError('Original public partition changed')
    targets=filtered_panel.filter_events(index,raw_row['targets'])
    visible=filtered_panel.filter_events(index,raw_row['visible'])
    row={**raw_row,'targets':targets,'visible':visible}
    if raw_row['view'] in ('single_condition','all_conditions'):
        if record['split']!='train':raise ValueError('No hidden validation controls')
        r=raw_row['request']
        bands,_=training_labels('phira',canonical(filtered_panel.targets(index)+list(r.known_carry),'phira'),(r.start_us,r.end_us))
        # Keep which control field was requested, even if its value changes.
        from .event_conditions import CONTROL_FIELDS
        if raw_row['view']=='single_condition':
            names=[CONTROL_FIELDS[j] for j in range(len(CONTROL_FIELDS)) if raw_row['inputs']['controls'][j,2]>.5]
            bands={k:v for k,v in bands.items() if k in names}
        controls=encode_controls('phira',bands,(r.start_us,r.end_us),origin_us=r.start_us)
        row['inputs']={**raw_row['inputs'],'controls':controls}
    return carry_reference_row(row,record['audio_sha256'],regions=regions)

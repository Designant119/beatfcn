import json
"""Fork-friendly exposure dataset for a multiprocessing training loader.

The dataset holds the panel and the frozen exposure; DataLoader worker processes
fork it and prepare requests without touching CUDA.  Create the loader iterator
before the model is moved to CUDA so forked workers never inherit a CUDA context.

Each item is one exposure update: a small list of physical microbatches with CPU
tensors already padded to the graph shapes.  Pinning is left to DataLoader.
"""
import multiprocessing
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from beatfcn.events.canonical import TYPES
from beatfcn.train.local_shapes import pad_input_shapes, pad_sequence_shapes
from beatfcn.model.full_event_codec import EventVocabulary, encode_events, teacher_sequence
from beatfcn.train.counterfactual_density import edit_events_to_multiplier
from types import SimpleNamespace

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.counterfactual_exposure import FORMAT, condition_inputs, sample_conditions, target_conditions
from beatfcn.train.charter_timing import (ChartTimingTable, attach_window_beat, normalize_charter_timing,
                                          timing_dropped)
LEVEL_CURVE_TAG = 909


SONG_STYLE_TAG = 0x53545945
SONG_STYLE_FIELDS = ('hold_head_ratio', 'hold_occupancy_ratio', 'chord_ratio', 'jack_ratio', 'flick_ratio', 'drag_ratio')


def normalize_level_curve(settings):
    """None/False -> None (OFF); {'p': float in [0, 1]} -> validated settings."""
    if settings is None or settings is False:
        return None
    if not isinstance(settings, dict) or set(settings) - {'p'} or not 0 <= float(settings.get('p', 0)) <= 1:
        raise ValueError("level_curve must be {'p': probability}")
    return dict(p=float(settings['p']))


from beatfcn.train.alternative_context import (SiblingIndex, alternative_source, draw_alternative,
                                               normalize_alternative_context, reference_inputs_with_alternative)
from beatfcn.train.history_context import (chart_arrays, draw_history, history_sources,
                                           normalize_history_context, reference_inputs)
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request, request_inputs
from beatfcn.train.local_panel import collate


class ExposureDataset(Dataset):
    def __init__(self, panel, exposure, *, microbatch, input_shapes=None,
                 sequence_buckets=None, start_update=0, vocabulary=None,
                 onset=None, mel_bins=0):
        if microbatch not in (1, 2, 4):
            raise ValueError('microbatch must be one, two, or four')
        self.panel = panel
        self.steps = exposure['steps']
        self.microbatch = int(microbatch)
        self.input_shapes = dict(input_shapes) if input_shapes else None
        self.sequence_buckets = list(sequence_buckets) if sequence_buckets else None
        if self.sequence_buckets and (self.sequence_buckets != sorted(set(self.sequence_buckets))
                or self.sequence_buckets[0] < 1):
            raise ValueError('Sequence buckets must be positive and strictly increasing')
        if int(start_update) != start_update or not 0 <= start_update <= len(self.steps):
            raise ValueError('Invalid exposure resume position')
        self.start_update = int(start_update)
        if self.panel.records is None:
            raise ValueError('Panel records are required')
        self.vocab = make_vocabulary(vocabulary)
        self.onset = dict(onset) if onset else None
        self.mel_bins = int(mel_bins)

    def __len__(self):
        return len(self.steps) - self.start_update

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        update = self.steps[index + self.start_update]
        if (len(update['indices']) != 4 or len(update['specified_condition_fields']) != 4
                or update['effective_update'] != index + self.start_update + 1):
            raise ValueError('Malformed or discontinuous exposure update')
        prepared = []
        for offset in range(0, 4, self.microbatch):
            ids = update['indices'][offset:offset+self.microbatch]
            controls = []
            for request_index, specified in zip(
                    ids, update['specified_condition_fields'][offset:offset+self.microbatch]):
                if not 0 <= request_index < len(self.panel.records):
                    raise ValueError('Exposure request index is out of range')
                row = self.panel.records[request_index]
                if row['split'] != 'train':
                    raise ValueError('Training exposure touches validation')
                control = np.asarray(row['controls'], np.float32).copy()
                hide = [i for i in range(len(control)) if i not in specified]
                control[hide, :3] = 0
                controls.append(control)
            if callable(getattr(self.panel, 'row', None)):
                rows = [self.panel.row(i, controls=c) for i, c in zip(ids, controls)]
                if not is_default_vocabulary(self.vocab) or self.onset is not None:
                    rows = [self._native_row(i, row) for i, row in zip(ids, rows)]
                inputs, sequence = collate(rows, input_shapes=self.input_shapes,
                    sequence_buckets=self.sequence_buckets)
            else:
                if not is_default_vocabulary(self.vocab) or self.onset is not None:
                    raise ValueError('Vocabulary/onset supervision needs a panel with row()/targets()')
                inputs, sequence = self.panel.batch(ids, controls=controls, device='cpu')
                if self.input_shapes:
                    inputs = pad_input_shapes(inputs, self.input_shapes)
                if self.sequence_buckets:
                    tokens = sequence['targets'].shape[1]
                    bucket = next((v for v in self.sequence_buckets if v >= tokens), None)
                    if bucket is None:
                        raise ValueError('Sequence exceeds the largest frozen bucket')
                    sequence = pad_sequence_shapes(sequence, bucket)
            prepared.append((ids, inputs, sequence))
        return dict(step=update['effective_update'], indices=update['indices'],
                    batches=prepared)

    def _native_row(self, index, row):
        record = self.panel.records[index]
        request = event_request(record['request'])
        events = self.panel.targets(index)
        sequence = request_sequence(self.panel, index, request, self.vocab, events=events)
        if self.onset is not None:
            sequence = dict(sequence, **onset_supervision(self.panel, record, request, events,
                row['inputs'], mel_bins=self.mel_bins, scope=self.onset.get('scope', 'window')))
        return dict(row, sequence=sequence)


def make_vocabulary(settings=None):
    """The run's token vocabulary; {} or None is the historical default layout."""
    settings = dict(settings or {})
    try:
        return EventVocabulary(**settings)
    except TypeError as exc:
        raise ValueError(f'Unsupported vocabulary settings {sorted(settings)}: {exc}') from None


def is_default_vocabulary(vocab):
    return vocab is None or vocab == EventVocabulary()


def request_sequence(panel, index, request, vocab, events=None):
    """Teacher sequence in the run's vocabulary.

    Tape token streams were encoded with the default vocabulary; any other
    layout (e.g. relative_blocks) re-encodes the request's target events.
    """
    if events is None and is_default_vocabulary(vocab) and callable(getattr(panel, 'sequence', None)):
        return panel.sequence(index)
    if events is None:
        events = panel.targets(index)
    return teacher_sequence(encode_events(events, request, vocab=vocab), request, vocab=vocab)


def _base_inputs(panel, index, request, controls):
    """Audio/reference inputs without building the (discarded) tape sequence."""
    if isinstance(panel, LocalRequestPanel):
        inputs, _ = request_inputs(panel.features, panel.records[index]['audio_sha256'],
                                   request, controls)
        return inputs
    return dict(panel.row(index)['inputs'])


def _song_onsets(panel, record):
    """All native head onsets/types of the record's chart, for onset-head context."""
    arrays = getattr(panel, '_arrays', None)
    if arrays is None or 'chart_index' not in record:
        return None
    _, values = arrays(record['chart_index'])
    return (np.asarray(values['onset_us'], np.int64),
            [TYPES[int(k)] for k in values['kind']])


def onset_supervision(panel, record, request, events, inputs, *, mel_bins, scope='window'):
    """Onset-head targets for one (possibly edited) request.

    Core frames use the request's ACTUAL (edited) target; with scope='window'
    the audio context outside the core uses the native chart's onsets (the
    context is never edited), so context frames are not falsely negative.
    scope='core' marks context frames invalid instead. Frames follow the
    stream the head reads: mel rows when mel_bins, else local MERT frames.
    """
    from beatfcn.train.training_objective import onset_frame_targets
    stream = 'mel' if mel_bins else 'local'
    times = np.asarray(inputs[stream+'_times'], np.float64)
    valid = np.asarray(inputs[stream+'_valid'], bool).copy()
    onsets = [e.onset_us for e in events]
    kinds = [e.type for e in events]
    if scope == 'window':
        native = _song_onsets(panel, record)
        if native is not None:
            outside = (native[0] < request.start_us) | (native[0] >= request.end_us)
            onsets += native[0][outside].tolist()
            kinds += [kind for kind, keep in zip(native[1], outside) if keep]
    elif scope == 'core':
        absolute = times*1e6 + request.start_us
        valid &= (absolute >= request.start_us) & (absolute < request.end_us)
    else:
        raise ValueError('Onset supervision scope must be window or core')
    return onset_frame_targets(onsets, times, request.start_us, onset_types=kinds,
                               frame_valid=valid)


class CounterfactualExposureDataset(ExposureDataset):
    """Build density-edited targets and their own conditions at load time.

    Requires a ``counterfactual-exposure-v2`` exposure (see
    counterfactual_exposure.build_exposure): each slot carries its treatment
    (rates: None=native, else a density multiplier), edit seed and frozen
    condition dropout. Every sample is conditioned on its ACTUAL target: the
    coarse achieved-density curve and the window controls recomputed from the
    edited target (plus incoming holds), each subject to the frozen dropout.
    Incoming holds (known_carry) stay in the request, and mania insertions
    avoid held lanes.
    """

    def __init__(self, panel, exposure, *, microbatch, input_shapes=None,
                 sequence_buckets=None, cell_us=100_000, curve_cell_us=2_000_000,
                 start_update=0, vocabulary=None, onset=None, mel_bins=0,
                 density_scale=16., history_context=None, charter_timing=None, alternative_context=None,
                 level_curve=None, song_style=None):
        if exposure.get('format') != FORMAT:
            raise ValueError('Counterfactual training requires a counterfactual-exposure-v2 '
                             'exposure; rebuild it with scripts/prepare_native_counterfactual_training.py')
        super().__init__(panel, exposure, microbatch=microbatch,
            input_shapes=input_shapes, sequence_buckets=sequence_buckets,
            start_update=start_update, vocabulary=vocabulary, onset=onset, mel_bins=mel_bins)
        self.cell_us = int(cell_us)
        self.curve_cell_us = int(curve_cell_us)
        self.density_scale = float(density_scale)
        # Optional visible native context (beatfcn.train.history_context); None
        # keeps the historical blank-reference contract byte for byte.
        self.history = normalize_history_context(history_context)
        self.history_seed = int(exposure.get('seed', 0))
        # Optional charter timing (beatfcn.train.charter_timing): each chart's own
        # beat grid as inputs['beat'], dropped per (request, epoch/occurrence)
        # with the frozen rate. None keeps the historical inputs byte for byte.
        self.timing = normalize_charter_timing(charter_timing)
        self.timing_table = None
        if self.timing is not None:
            if not mel_bins:
                raise ValueError('charter_timing features ride on the 10ms mel frames; mel_bins is 0')
            self.timing_table = ChartTimingTable.from_settings(self.timing, root=getattr(panel, 'root', None))
        # Optional sibling chart (beatfcn.train.alternative_context): another human
        # difficulty of the same song as a human_alternative reference. None keeps
        # the history-only / blank-reference paths byte for byte.
        self.alternative = normalize_alternative_context(alternative_context)
        self.siblings = SiblingIndex(panel) if self.alternative is not None else None
        # Optional level-only density request (round 5, owner: conditions must never degrade a chart): with
        # probability p a curve-specified sample's density curve is replaced by ONE constant = the target
        # chart's song-level mean onset rate (x the edit rate), the format of a difficulty request at inference,
        # so a flat request means "overall level", not "flatten the song". None keeps the inputs byte for byte.
        self.level_curve = normalize_level_curve(level_curve)
        # Products ask for one style per song while windows train on their own bands: with probability p a native
        # window sees its whole chart's bands, so a song-level request means what it says.
        self.song_style = None if not song_style else float(song_style['p'])
        self._song_bands_cache = {}
        self._song_rate = {}

    def sample(self, request_index, rate, *, edit_seed, curve_specified, specified_fields,
               history_key=None, timing_key=None, alternative_key=None):
        """One prepared row plus an audit record (used by the data probe).

        ``history_key`` (the slot's exposure epoch/occurrence) is required when
        history_context is on; its draw is recorded in ``audit['history']``.
        ``timing_key`` (same slot key) is required when charter_timing is on;
        the chart's timing / dropout draw is recorded in ``audit['charter_timing']``.
        ``alternative_key`` (same slot key) is required when alternative_context is on;
        its draw is recorded in ``audit['alternative']``.
        """
        record = self.panel.records[request_index]
        if record['split'] != 'train':
            raise ValueError('Training exposure touches validation')
        request = event_request(record['request'])
        native = self.panel.targets(request_index)
        events = native
        if rate is not None:
            events, _ = edit_events_to_multiplier(native, request.start_us, request.end_us,
                float(rate), request.mode, request.duration_us, seed=int(edit_seed))
        sequence = request_sequence(self.panel, request_index, request, self.vocab,
                                    events=None if rate is None else events)
        conditions, audit = sample_conditions(request.mode, events, request,
            curve_specified=curve_specified, specified_fields=specified_fields,
            render_cell_us=self.cell_us, curve_cell_us=self.curve_cell_us,
            scale=self.density_scale)
        if self.level_curve is not None and curve_specified:
            level = self._level_request(request_index, record, request, rate, edit_seed)
            if level is not None:
                conditions = dict(conditions, **{k: v for k, v in level.items() if k != 'controls'})
                audit['level_curve'] = True
        if self.song_style is not None and rate is None:
            shown = self._song_style_shown(request_index, record, request, edit_seed, audit)
            if shown is not None:
                conditions = dict(conditions, controls=condition_inputs(request.mode, request, bands=shown, segments=(),
                                  render_cell_us=self.cell_us, scale=self.density_scale)['controls'])
                audit['song_style'] = True
        inputs = _base_inputs(self.panel, request_index, request, conditions['controls'])
        inputs.update(conditions)
        if self.alternative is not None:
            history, audit['alternative'] = self._attach_references(inputs, request_index, record, request, rate,
                                                                    history_key, alternative_key)
            if self.history is not None:
                audit['history'] = history
        elif self.history is not None:
            audit['history'] = self._attach_history(inputs, request_index, record, request, rate, history_key)
        if self.timing is not None:
            audit['charter_timing'] = self._attach_timing(inputs, request_index, record, request, rate, timing_key)
        if self.onset is not None:
            sequence = dict(sequence, **onset_supervision(self.panel, record, request, events,
                inputs, mel_bins=self.mel_bins, scope=self.onset.get('scope', 'window')))
        audit.update(request=request, events=events, native=native, rate=rate)
        return dict(inputs=inputs, sequence=sequence), audit

    def _song_style_shown(self, request_index, record, request, edit_seed, audit):
        u = np.random.default_rng([int(self.history_seed) & 0xFFFFFFFF, SONG_STYLE_TAG, int(request_index) & 0xFFFFFFFF,
                                   int(edit_seed) & 0xFFFFFFFF]).random()
        if u >= self.song_style or not audit['shown_fields']:
            return None
        chart = record['chart_index']
        if chart not in self._song_bands_cache:
            a = chart_arrays(self.panel, chart)
            events = [RhythmEvent(int(x), int(y), TYPES[int(k)], int(l)) for x, y, k, l in zip(a['onset_us'], a['end_us'], a['kind'], a['lane'])]
            whole = SimpleNamespace(start_us=0, end_us=int(request.duration_us))
            self._song_bands_cache[chart], _ = target_conditions(request.mode, events, whole, carry=())
        song = self._song_bands_cache[chart]
        return {f: song.get(f, audit['bands'].get(f)) if f in SONG_STYLE_FIELDS else audit['bands'].get(f) for f in audit['shown_fields']}

    def _level_request(self, request_index, record, request, rate, edit_seed):
        u = np.random.default_rng([int(self.history_seed) & 0xFFFFFFFF, LEVEL_CURVE_TAG, int(request_index) & 0xFFFFFFFF,
                                   int(edit_seed) & 0xFFFFFFFF]).random()
        if u >= self.level_curve['p']:
            return None
        chart = record['chart_index']
        if chart not in self._song_rate:
            onsets = np.unique(np.asarray(chart_arrays(self.panel, chart)['onset_us']))
            self._song_rate[chart] = len(onsets) / max(request.duration_us / 1e6, 1e-6)
        level = self._song_rate[chart] * (float(rate) if rate is not None else 1.0)
        return condition_inputs(request.mode, request, segments=[dict(start_us=int(request.start_us),
                                end_us=int(request.end_us), density=float(level))], render_cell_us=self.cell_us,
                                scale=self.density_scale)

    def _attach_timing(self, inputs, request_index, record, request, rate, timing_key):
        if timing_key is None:
            raise ValueError('charter_timing needs the slot draw key (exposure epoch/occurrence)')
        dropped = timing_dropped(self.history_seed, request_index, timing_key, self.timing['dropout'])
        info = attach_window_beat(inputs, self.timing_table, record['chart_id'], request.start_us, dropped=dropped)
        info.update(dropped=bool(dropped), dropout=self.timing['dropout'], key=timing_key,
                    edited_target=rate is not None)
        return info

    def _attach_references(self, inputs, request_index, record, request, rate, history_key, alternative_key):
        """History (same chart, outside the window) + sibling (another chart) encoded as ONE reference."""
        if rate is not None:
            skipped = dict(applied=False, reason='edited_target_native_only')
            return (dict(skipped) if self.history is not None else None), dict(skipped)
        history_sources_, history_info = (), None
        if self.history is not None:
            if history_key is None:
                raise ValueError('history_context needs the slot draw key (exposure epoch/occurrence)')
            draw = draw_history(self.history, self.history_seed, request_index, history_key)
            history_sources_, history_info = history_sources(chart_arrays(self.panel, record['chart_index']), request,
                draw, self.history, chart_id=record.get('chart_id', str(record['chart_index'])),
                audio_sha256=record['audio_sha256'])
        if alternative_key is None:
            raise ValueError('alternative_context needs the slot draw key (exposure epoch/occurrence)')
        draw = draw_alternative(self.alternative, self.history_seed, request_index, alternative_key)
        source, alternative_info = alternative_source(self.siblings, request, record, draw, self.alternative)
        alternative_sources = (source,) if source is not None else ()
        if history_sources_ or alternative_sources:
            reference, _ = reference_inputs_with_alternative(request, record, history_sources_, alternative_sources,
                max_rows=4096 + self.alternative['max_reference_rows'])
            inputs.update(reference)
            rows = int(len(reference['reference']))
            if history_sources_:
                history_info['reference_rows'] = rows
            if alternative_sources:
                alternative_info['reference_rows'] = rows
        return history_info, alternative_info

    def _attach_history(self, inputs, request_index, record, request, rate, history_key):
        if rate is not None:
            return dict(applied=False, reason='edited_target_native_only')
        if history_key is None:
            raise ValueError('history_context needs the slot draw key (exposure epoch/occurrence)')
        draw = draw_history(self.history, self.history_seed, request_index, history_key)
        sources, info = history_sources(chart_arrays(self.panel, record['chart_index']), request, draw,
            self.history, chart_id=record.get('chart_id', str(record['chart_index'])),
            audio_sha256=record['audio_sha256'])
        if sources:
            reference, _ = reference_inputs(request, record['audio_sha256'], sources)
            inputs.update(reference)
            info['reference_rows'] = int(len(reference['reference']))
        return info

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        update = self.steps[index + self.start_update]
        if update['effective_update'] != index + self.start_update + 1 or len(update['indices']) != 4:
            raise ValueError('Counterfactual exposure updates are discontinuous or malformed')
        prepared = []
        for offset in range(0, 4, self.microbatch):
            rows = []
            ids = update['indices'][offset:offset+self.microbatch]
            for slot in range(offset, offset+len(ids)):
                history_key = timing_key = alternative_key = None
                if self.history is not None or self.timing is not None or self.alternative is not None:
                    key = update.get('occurrences', update['epochs'])[slot]
                    history_key = key if self.history is not None else None
                    timing_key = key if self.timing is not None else None
                    alternative_key = key if self.alternative is not None else None
                row, _ = self.sample(update['indices'][slot], update['rates'][slot],
                    edit_seed=update['edit_seeds'][slot],
                    curve_specified=update['curve_specified'][slot],
                    specified_fields=update['specified_condition_fields'][slot],
                    history_key=history_key, timing_key=timing_key, alternative_key=alternative_key)
                rows.append(row)
            inputs, sequences = collate(rows, input_shapes=self.input_shapes,
                                        sequence_buckets=self.sequence_buckets)
            prepared.append((ids, inputs, sequences))
        return dict(step=update['effective_update'], indices=update['indices'],
                    batches=prepared)


def _identity(value):
    return value


class GroupedExposureDataset(Dataset):
    """Pack adjacent four-request exposure groups without changing their order."""
    def __init__(self, base, groups, max_tokens=None):
        if groups < 1 or len(base) % groups:
            raise ValueError('Grouped exposure must end on a complete group')
        self.base, self.groups, self.max_tokens = base, int(groups), max_tokens

    def __len__(self):
        return len(self.base)//self.groups

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows, indices, original_batches = [], [], []
        for offset in range(self.groups):
            item = self.base[index*self.groups+offset]
            for ids, inputs, sequence in item['batches']:
                original_batches.append((ids, inputs, sequence))
                indices.extend(ids)
                for row in range(len(ids)):
                    rows.append({section: {key: value[row].numpy() for key, value in tensors.items()}
                                 for section, tensors in [('inputs', inputs), ('sequence', sequence)]})
        if self.max_tokens and max(b[2]['targets'].shape[1] for b in original_batches) > self.max_tokens:
            return dict(step=item['step'], indices=indices, batches=original_batches)
        inputs, sequence = collate(rows)
        return dict(step=item['step'], indices=indices, batches=[(indices, inputs, sequence)])


def _worker_init(worker_id):
    torch.set_num_threads(1)
    if sys.platform == 'darwin':
        # macOS torch uses file_system sharing. Its exec'd shm manager must not
        # keep Python's tracker pipe alive: the tracker waits for EOF while the
        # manager waits for this worker's parent to exit (Python 3.12 teardown).
        from multiprocessing import resource_tracker
        os.set_inheritable(resource_tracker.getfd(), False)


def plan_onset_settings(plan):
    """Onset supervision settings of a plan, or None when the onset head is off."""
    if not plan['config'].get('onset_head'):
        return None
    return dict(plan.get('onset_supervision') or {})


def plan_loader_settings(plan):
    """build_exposure_loader keyword settings that a plan freezes (shared by the
    runner and scripts/bench_exposure_loader.py so both build identical items).

    ``history_context`` / ``charter_timing`` / ``alternative_context`` are only
    emitted when the plan sets them, so older plans produce the identical settings dict."""
    settings = dict(
        counterfactual_rates=plan.get('counterfactual_rates'),
        counterfactual_relative=bool(plan.get('counterfactual_relative', True)),
        cell_us=int(plan.get('counterfactual_cell_us', 100_000)),
        curve_cell_us=int(plan.get('counterfactual_curve_cell_us', 2_000_000)),
        grouped_max_tokens=plan.get('grouped_max_tokens'),
        vocabulary=plan.get('vocabulary'), onset=plan_onset_settings(plan),
        mel_bins=int(plan['config'].get('mel_bins', 0)))
    if plan.get('history_context'):
        settings['history_context'] = normalize_history_context(plan['history_context'])
    if plan.get('alternative_context'):
        settings['alternative_context'] = normalize_alternative_context(plan['alternative_context'])
    if plan.get('level_curve'):
        settings['level_curve'] = normalize_level_curve(plan['level_curve'])
    if plan.get('song_style'):
        settings['song_style'] = plan['song_style']
    if plan.get('charter_timing'):
        if not plan['config'].get('beat_features'):
            raise ValueError('charter_timing needs config.beat_features (else the model ignores the input)')
        settings['charter_timing'] = normalize_charter_timing(plan['charter_timing'])
    return settings


def build_exposure_loader(panel, exposure, *, microbatch, input_shapes=None,
                          sequence_buckets=None, num_workers=8, pin_memory=True,
                          prefetch_factor=2, persistent_workers=True,
                          start_update=0, multiprocessing_context=None, seed=0,
                          counterfactual_rates=None, cell_us=100_000,
                          counterfactual_relative=True, grouped_max_tokens=None,
                          vocabulary=None, curve_cell_us=2_000_000, onset=None,
                          mel_bins=0, density_scale=16., history_context=None, charter_timing=None,
                          alternative_context=None, level_curve=None, song_style=None):
    """Counterfactual mode follows the exposure format, not a loose plan flag.

    ``counterfactual_rates``/``counterfactual_relative`` are accepted for plan
    compatibility only: the treatments are frozen in the v2 exposure, and the
    retired v1 (rates-only / absolute nps) exposure is rejected.
    """
    counterfactual = exposure.get('format') == FORMAT
    if not counterfactual and (counterfactual_rates or exposure['steps'] and 'rates' in exposure['steps'][0]):
        raise ValueError('Legacy counterfactual exposure is retired; rebuild a counterfactual-exposure-v2')
    if counterfactual and not counterfactual_relative:
        raise ValueError('Absolute-density counterfactuals are retired; use relative multipliers')
    if history_context and not counterfactual:
        raise ValueError('history_context needs a counterfactual-exposure-v2 exposure')
    if charter_timing and not counterfactual:
        raise ValueError('charter_timing needs a counterfactual-exposure-v2 exposure')
    if alternative_context and not counterfactual:
        raise ValueError('alternative_context needs a counterfactual-exposure-v2 exposure')
    num_workers = int(num_workers)
    # Execution-only cap (does not change data, order or plan identity): each
    # forked worker gradually copies the ~3 GB request panel, so 28 workers
    # grew host memory past 96 GiB on the full database.
    cap = os.environ.get('BEATFCN_MAX_LOADER_WORKERS')
    if cap and num_workers > int(cap):
        print(json.dumps(dict(phase='loader_worker_cap', planned=num_workers, used=int(cap))), flush=True)
        num_workers = int(cap)
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError('Invalid loader worker count or prefetch factor')
    if multiprocessing_context is None:
        multiprocessing_context = 'fork' if sys.platform == 'linux' else 'spawn'
    if num_workers and multiprocessing_context == 'fork' and torch.cuda.is_initialized():
        raise ValueError('Start fork workers before initializing CUDA')
    if multiprocessing_context not in multiprocessing.get_all_start_methods():
        raise ValueError('Unsupported multiprocessing context')
    dataset = (CounterfactualExposureDataset(panel, exposure, microbatch=min(microbatch, 4),
        input_shapes=input_shapes, sequence_buckets=sequence_buckets,
        cell_us=cell_us, curve_cell_us=curve_cell_us, start_update=start_update,
        vocabulary=vocabulary, onset=onset, mel_bins=mel_bins, density_scale=density_scale,
        history_context=history_context, charter_timing=charter_timing,
        alternative_context=alternative_context, level_curve=level_curve,
        song_style=song_style)
        if counterfactual else
        ExposureDataset(panel, exposure, microbatch=min(microbatch, 4),
            input_shapes=input_shapes, sequence_buckets=sequence_buckets,
            start_update=start_update, vocabulary=vocabulary, onset=onset, mel_bins=mel_bins))
    if microbatch > 4:
        if microbatch not in (8, 16):
            raise ValueError('Grouped physical microbatch must be eight or sixteen')
        dataset = GroupedExposureDataset(dataset, microbatch//4, max_tokens=grouped_max_tokens)
    return DataLoader(dataset, batch_size=None, shuffle=False,
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and int(num_workers) > 0),
        prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_identity)

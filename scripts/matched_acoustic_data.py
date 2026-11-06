"""Train-only diagnostic crops with identical non-acoustic inputs per mode."""
from collections import Counter
from dataclasses import dataclass, asdict, fields, replace
import hashlib
import json

import numpy as np
import torch

from beatfcn.events.canonical import MODE_TYPES, Event, build_model_view
from beatfcn.events.controls import MODE_FIELDS
from beatfcn.model.event_canvas import ClockLayout, RhythmEvent, new_canvas
from beatfcn.model.features import CONTROL_FIELDS
from beatfcn.model.group_canvas import encode_group_targets, decode_group_targets
from beatfcn.train.data import chart_from_canon, model_shift_us
from scripts.r09_local_data import ROOT, file_sha


@dataclass
class MatchedInputs:
    blank: object
    local: torch.Tensor
    local_times: torch.Tensor
    local_valid: torch.Tensor
    song: torch.Tensor
    song_times: torch.Tensor
    duration_ms: int
    condition_fields: torch.Tensor
    conditions: torch.Tensor
    known: torch.Tensor
    reference: torch.Tensor
    allowed_heads: torch.Tensor
    allowed_types: torch.Tensor


def target_events(chart, c0, plan):
    end = c0 + plan['core_us']
    clip_start = c0 - plan['core_origin_in_clip_us']
    clip_end = clip_start + plan['clip_us']
    if any(e.is_hold and e.onset_us < c0 < e.end_us for e in chart.events):
        raise ValueError('incoming_hold_requires_observed_context')
    raw = [e for e in chart.events if c0 <= e.onset_us < end]
    view = build_model_view(chart.mode, [], raw)
    if not view.ok:
        raise ValueError('canonical_model_view_quarantined')
    quantized = [e for group in view.target_groups for e in group.events]
    if len(quantized) != len(raw):
        raise ValueError('model_view_changed_multiplicity')
    if any(not c0 <= e.onset_us < end or e.end_us > clip_end for e in quantized):
        raise ValueError('quantized_target_outside_core_or_full_tail_support')
    events = tuple(sorted(RhythmEvent(e.onset_us - clip_start, e.end_us - clip_start,
        e.type, e.location.lane if chart.mode == 'mania_4k' else -1) for e in quantized))
    layout = ClockLayout(plan['core_origin_in_clip_us'], plan['core_origin_in_clip_us'] + plan['core_us'])
    labels = encode_group_targets(events, layout, chart.mode, groups=32, members=8)
    if decode_group_targets(labels) != events:
        raise ValueError('target_roundtrip_changed_events')
    return labels, events


def read_chart(item):
    doc = json.loads((ROOT / item['canonical']).read_text())
    row = item['row']
    if row['split'] != 'train' or any(doc[k] != row[k] for k in ('chart_id', 'mode', 'audio_sha256')):
        raise ValueError('source identity or train split mismatch')
    return chart_from_canon(doc, model_shift_us(row, True))


def make_example(item, plan, *, translated=False):
    delta = plan['untrained_translation_us'] if translated else 0
    c0 = item['source_core_us'] + delta
    clip_start = c0 - plan['core_origin_in_clip_us']
    if clip_start % 2_000_000 or c0 % 40_000:
        raise ValueError('feature clocks not aligned')
    chart = read_chart(item)
    labels, events = target_events(chart, c0, plan)
    path = ROOT / plan['feature_root'] / item['row']['audio_sha256']
    scales = np.load(ROOT / plan['scale_file'])
    begin = (c0 - 4_000_000) // 40_000
    layers = []
    for layer in (9, 16, 24):
        raw = np.load(path / f'L{layer}.npy', mmap_mode='r')
        scale = scales[f'scale_{layer}']
        if raw.dtype != np.int8 or raw.shape[1:] != (8, 1024):
            raise ValueError('feature representation mismatch')
        if begin < 0 or begin + 400 > len(raw):
            raise ValueError('local audio unsupported')
        layers.append(raw[begin:begin + 400].astype(np.float32) * scale)
    summary = np.load(path / 'S20.npy', mmap_mode='r')
    start = clip_start // 2_000_000
    count = plan['clip_us'] // 2_000_000
    song = np.array(summary[start:start + count], dtype=np.float32, copy=True)
    if song.shape != (16, 8, 1024):
        raise ValueError('summary shape or support mismatch')
    local = torch.from_numpy(np.stack(layers, 1))
    local_times = torch.from_numpy(((np.arange(400, dtype=np.float64) + .5) * .04 - 4).astype(np.float32))
    song_times = torch.from_numpy(((np.arange(count, dtype=np.float64) + .5) * 2 -
                                  plan['core_origin_in_clip_us'] / 1e6).astype(np.float32))
    conditions = torch.tensor([[0., float(name in MODE_FIELDS[chart.mode]), 0., 0., 0.,
        plan['core_us'] / 1e6, 1., -1.] for name in CONTROL_FIELDS], dtype=torch.float32)
    inputs = MatchedInputs(new_canvas(labels.layout, chart.mode, slots=32), local, local_times,
        torch.ones(400, dtype=torch.bool), torch.from_numpy(song), song_times,
        plan['clip_us'] // 1000, torch.arange(len(CONTROL_FIELDS)), conditions,
        torch.empty(0, 5), torch.empty(0, 5), torch.ones(20, 400, dtype=torch.bool),
        torch.tensor([name in MODE_TYPES[chart.mode] for name in ('tap', 'hold', 'flick', 'drag')]))
    identity = dict(chart_id=item['row']['chart_id'], mode=chart.mode,
        audio_sha256=item['row']['audio_sha256'], source_core_us=c0, source_clip_start_us=clip_start,
        c0_us=plan['core_origin_in_clip_us'], translated=translated, panel_id=item['panel_id'],
        target_sha256=hashlib.sha256(json.dumps([asdict(e) for e in events]).encode()).hexdigest())
    return inputs, labels, identity


def metadata_sha(inputs):
    h = hashlib.sha256()
    blank = inputs.blank
    h.update(json.dumps(dict(layout=asdict(blank.layout), mode=blank.mode,
                            copy_times=blank.copy_times, slots=blank.slots), sort_keys=True).encode())
    h.update(blank.values.tobytes())
    for field in fields(inputs):
        if field.name in ('blank', 'local', 'song'):
            continue
        value = getattr(inputs, field.name)
        h.update(field.name.encode())
        if torch.is_tensor(value):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.numpy().tobytes())
        else:
            h.update(json.dumps(value).encode())
    return h.hexdigest()


def swap_audio(inputs, other):
    if metadata_sha(inputs) != metadata_sha(other):
        raise ValueError('cannot swap audio across different non-acoustic metadata')
    return replace(inputs, local=other.local, song=other.song)


def common_physical_targets(normal, translated, normal_identity, shifted_identity, plan):
    a = normal_identity['source_core_us']
    b = shifted_identity['source_core_us']
    lo, hi = max(a, b), min(a, b) + plan['core_us']
    def physical(labels, identity):
        offset = identity['source_clip_start_us']
        return Counter((e.onset_us + offset, e.end_us + offset, e.type, e.lane)
                       for e in decode_group_targets(labels) if lo <= e.onset_us + offset < hi)
    return physical(normal, normal_identity) == physical(translated, shifted_identity), [lo, hi]

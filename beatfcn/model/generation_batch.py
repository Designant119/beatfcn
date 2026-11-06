"""Append-only inference metadata; training labels are not rebuilt after each note.

The first batch comes from the normal collator. Static request/audio tensors stay
on device. Only the new completed group and its next Y context are updated, using
the same padding and numeric conversions as collate_meta. This batch must never
be passed to a training loss: hold labels and other training targets remain stale.
"""
import numpy as np
import torch

from . import features as F
from .batch import BUCKET, LANE_HISTORY, MAX_NOTES, NO_ONSET, PREV_ONSETS, dequant


GROUP_KEYS = ('grp_valid', 'grp_t', 'grp_n', 'note_type', 'note_lane', 'note_line',
              'note_side', 'note_x', 'note_dur')
ROW_KEYS = ('right_rows', 'prev_onsets', 'lane_history')
INFERENCE_PREFIX_KEYS = GROUP_KEYS + ROW_KEYS


class GenerationBatch:
    def __init__(self, raw, example, device):
        if example.groups:
            raise ValueError('generation metadata must start with an empty prefix')
        self.cpu = {key: raw[key] for key in INFERENCE_PREFIX_KEYS}
        self.bt = dequant({k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()})
        self.count = 0
        self.mode = example.mode
        self.history = sorted({float(t) for t in example.known[:, 0] if t < 0})
        self.known_lanes = [(round(float(row[0]) * F.SEC), int(row[2]))
                            for row in example.known if int(row[2]) in range(4)]
        self.completed = {}

    def _grow(self):
        if self.count < self.cpu['grp_t'].shape[1]:
            return
        for key in INFERENCE_PREFIX_KEYS:
            old = self.cpu[key]
            shape = list(old.shape)
            shape[1] += BUCKET['G']
            fill = NO_ONSET if key == 'prev_onsets' else -1 if key in ('note_lane', 'note_line', 'note_side') else 0
            new = torch.full(shape, fill, dtype=old.dtype)
            new[:, :old.shape[1]] = old
            self.cpu[key] = new
            self.bt[key] = new.to(self.bt[key].device)

    def append(self, group, next_right):
        if len(group['notes']) > MAX_NOTES:
            raise F.SkipExample('group has more notes than MAX_NOTES')
        self._grow()
        i = self.count
        arrays = {k: v.numpy() for k, v in self.cpu.items()}
        arrays['grp_valid'][0, i] = True
        arrays['grp_t'][0, i] = group['t_rel_us'] / F.SEC
        arrays['grp_n'][0, i] = len(group['notes'])
        for j, note in enumerate(group['notes']):
            for field in ('type', 'lane', 'line', 'side', 'x'):
                arrays['note_' + field][0, i, j] = note[field]
            arrays['note_dur'][0, i, j] = note.get('end_rel_us', 0) / F.SEC
        row = i + 1
        arrays['right_rows'][0, row] = next_right
        self.history.append(group['t_rel_us'] / F.SEC)
        recent = self.history[-PREV_ONSETS:][::-1]
        arrays['prev_onsets'][0, row, :len(recent)] = recent
        if self.mode == 0:
            cursor = group['t_rel_us']
            bits = self.completed.setdefault(cursor, np.zeros(4, np.float32))
            for note in group['notes']:
                if note['lane'] in range(4):
                    bits[note['lane']] = 1
            for onset, lane in self.known_lanes:
                if onset <= cursor:
                    self.completed.setdefault(onset, np.zeros(4, np.float32))[lane] = 1
            for j, onset in enumerate(sorted(self.completed, reverse=True)[:LANE_HISTORY]):
                arrays['lane_history'][0, row, j, :4] = self.completed[onset]
                arrays['lane_history'][0, row, j, 4:] = (onset / F.SEC, 1.)
        for key in INFERENCE_PREFIX_KEYS:
            if self.bt[key].device.type != 'cpu':
                index = row if key in ROW_KEYS else i
                self.bt[key][:, index:index + 1].copy_(self.cpu[key][:, index:index + 1])
        self.count += 1

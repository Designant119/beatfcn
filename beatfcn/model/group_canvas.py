"""Fixed-size onset-group canvas with explicit learned cardinalities.

Counts in completed labels are supervision, not a first-pass input. Group heads
are shared by every member, so simultaneous notes cannot drift independently.
All capacities are explicit; there is no truncation or deduplication fallback.
"""
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from .event_canvas import EMPTY, MASK, CanvasOverflow, ClockLayout, RhythmEvent
from beatfcn.events.canonical import TYPES


@dataclass
class GroupCanvas:
    layout: ClockLayout
    mode: str
    copy_times: tuple[int, ...]
    counts: np.ndarray  # [blocks]: number of groups; -1 unknown
    heads: np.ndarray  # [blocks, group slots, 2]: block ms offset, COPY index
    multiplicity: np.ndarray  # [blocks, group slots]: 0 empty, -1 unknown, >=1 members
    notes: np.ndarray  # [blocks, group slots, members, 3]: kind, absolute tail ms, lane

    @property
    def groups(self):
        return self.heads.shape[1]

    @property
    def members(self):
        return self.notes.shape[2]


def new_group_canvas(layout, mode, *, groups=32, members=8, copy_times=(), unknown=True):
    if mode not in ('mania_4k', 'phira'):
        raise ValueError('unsupported mode')
    if type(groups) is not int or type(members) is not int or min(groups, members) <= 0:
        raise ValueError('invalid group/member capacity')
    times = tuple(copy_times)
    if any(type(t) is not int for t in times) or len(times) != len(set(times)):
        raise ValueError('copy times must be unique integer microseconds')
    blocks = len(layout.blocks)
    notes = np.full((blocks, groups, members, 3), -1, np.int64)
    notes[..., 0] = MASK if unknown else EMPTY
    return GroupCanvas(layout, mode, times,
                       np.full(blocks, -1 if unknown else 0, np.int64),
                       np.full((blocks, groups, 2), -1, np.int64),
                       np.full((blocks, groups), -1 if unknown else 0, np.int64), notes)


def encode_group_targets(events, layout, mode, *, groups=32, members=8, copy_times=()):
    result = new_group_canvas(layout, mode, groups=groups, members=members,
                              copy_times=copy_times, unknown=False)
    buckets = [defaultdict(list) for _ in layout.blocks]
    for event in events:
        if event.type not in TYPES or mode == 'mania_4k' and event.type not in ('tap', 'hold'):
            raise ValueError('unsupported event type')
        if (mode == 'mania_4k' and event.lane not in range(4)) or (mode == 'phira' and event.lane != -1):
            raise ValueError('invalid location')
        if event.type == 'hold':
            if event.end_us <= event.onset_us or event.end_us % 1000:
                raise ValueError('hold requires positive millisecond tail')
        elif event.end_us != event.onset_us:
            raise ValueError('non-hold must end at its head')
        buckets[layout.block_for(event.onset_us)][event.onset_us].append(event)
    copy_lookup = {t: i for i, t in enumerate(result.copy_times)}
    for block, times in enumerate(buckets):
        if len(times) > groups:
            raise CanvasOverflow(f'block {block} has {len(times)} groups, capacity {groups}')
        result.counts[block] = len(times)
        base = layout.blocks[block][0]
        for group, onset in enumerate(sorted(times)):
            notes = sorted(times[onset], key=lambda e: (e.lane if mode == 'mania_4k' else TYPES.index(e.type),
                                                       e.end_us, TYPES.index(e.type)))
            if len(notes) > members:
                raise CanvasOverflow(f'block {block} group {group} has {len(notes)} notes, capacity {members}')
            result.multiplicity[block, group] = len(notes)
            if onset in copy_lookup:
                result.heads[block, group, 1] = copy_lookup[onset]
            elif onset % 1000:
                raise ValueError('sub-millisecond head requires an authorized COPY_TIME')
            else:
                result.heads[block, group, 0] = (onset - base) // 1000
            for index, event in enumerate(notes):
                result.notes[block, group, index] = (TYPES.index(event.type),
                    event.end_us // 1000 if event.type == 'hold' else -1, event.lane)
    return result


def decode_group_targets(canvas):
    b, g, m = len(canvas.layout.blocks), canvas.groups, canvas.members
    shapes = ((b,), (b, g, 2), (b, g), (b, g, m, 3))
    arrays = (canvas.counts, canvas.heads, canvas.multiplicity, canvas.notes)
    if any(a.shape != shape or a.dtype != np.int64 for a, shape in zip(arrays, shapes)):
        raise ValueError('malformed group arrays')
    if canvas.mode not in ('mania_4k', 'phira'):
        raise ValueError('unsupported mode')
    if any((a < -1).any() for a in arrays) or (canvas.counts > g).any() or (canvas.counts < 0).any():
        raise ValueError('unknown or out-of-range group count/field')
    result = []
    for block, (base, start, stop) in enumerate(canvas.layout.blocks):
        count = int(canvas.counts[block])
        for group in range(g):
            n = int(canvas.multiplicity[block, group])
            head, copy = map(int, canvas.heads[block, group])
            notes = canvas.notes[block, group]
            if group >= count:
                if n != 0 or head != -1 or copy != -1 or (notes[:, 0] != EMPTY).any() or (notes[:, 1:] != -1).any():
                    raise ValueError('inactive group contains payload')
                continue
            if not 1 <= n <= m or (head >= 0) == (copy >= 0):
                raise ValueError('active group requires members and exactly one shared head')
            if copy >= 0:
                if copy >= len(canvas.copy_times):
                    raise ValueError('unknown COPY_TIME index')
                onset = canvas.copy_times[copy]
            else:
                onset = base + head * 1000
            if not start <= onset < stop:
                raise ValueError('group head crosses owning block')
            if (notes[n:, 0] != EMPTY).any() or (notes[n:, 1:] != -1).any():
                raise ValueError('inactive member contains payload')
            for kind, tail, lane in notes[:n]:
                kind, tail, lane = int(kind), int(tail), int(lane)
                if kind not in range(4) or canvas.mode == 'mania_4k' and kind > 1:
                    raise ValueError('invalid active member type')
                if (canvas.mode == 'mania_4k' and lane not in range(4)) or (canvas.mode == 'phira' and lane != -1):
                    raise ValueError('invalid member location')
                if kind == 1:
                    if tail * 1000 <= onset:
                        raise ValueError('hold tail must follow shared head')
                    end = tail * 1000
                else:
                    if tail != -1:
                        raise ValueError('non-hold has tail')
                    end = onset
                result.append(RhythmEvent(onset, end, TYPES[kind], lane))
    # Preserve multiplicity. Prediction order/conflicts are audited separately.
    return tuple(sorted(result))


def masked_group_view(labels, visible):
    visible = np.asarray(visible)
    if visible.shape != labels.multiplicity.shape or visible.dtype != np.bool_:
        raise ValueError('visible mask must have the fixed group shape')
    result = new_group_canvas(labels.layout, labels.mode, groups=labels.groups,
                              members=labels.members, copy_times=labels.copy_times)
    result.heads[visible] = labels.heads[visible]
    result.multiplicity[visible] = labels.multiplicity[visible]
    result.notes[visible] = labels.notes[visible]
    # Never copy the total target count alongside a partially hidden block.
    # Count supervision is a separate argument to the model's loss.
    return result

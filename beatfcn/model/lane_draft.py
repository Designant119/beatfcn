"""Lane-free rhythm drafts and prefix-only occupancy for component diagnostics.

The rhythm is an explicitly supplied draft, not a hidden generation target.
There is deliberately no lane-label field on LaneDraft.
"""
from collections import Counter
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LaneDraft:
    # Each member retains its complete tail; equal members retain multiplicity.
    events: tuple[tuple[int, int, str], ...]
    features: torch.Tensor


def time_basis(seconds):
    values = torch.as_tensor(seconds, dtype=torch.float32)
    frequency = values.new_tensor([.25, .5, 1., 2., 4., 8., 16.])
    phase = values[..., None] * frequency * (2 * math.pi)
    return torch.cat((values[..., None] / 32., phase.sin(), phase.cos()), -1)


def make_lane_draft(events, *, origin_us=8_000_000):
    triples = tuple(sorted((int(e['onset_us']), int(e['end_us']), e['type']) for e in events))
    # The contract orders by onset/type/end, independent of supplied lane values.
    triples = tuple(sorted(triples, key=lambda e: (e[0], e[2], e[1])))
    if not triples or any(kind not in ('tap', 'hold') or tail < head or
                          (kind == 'tap' and tail != head) or
                          (kind == 'hold' and tail == head) for head, tail, kind in triples):
        raise ValueError('nonempty mania rhythm with complete valid tails required')
    heads = torch.tensor([e[0] for e in triples], dtype=torch.float32)
    tails = torch.tensor([e[1] for e in triples], dtype=torch.float32)
    previous = torch.cat((heads[:1], heads[:-1]))
    following = torch.cat((heads[1:], heads[-1:]))
    counts, seen = Counter(e[0] for e in triples), Counter()
    kinds, ranks, sizes = [], [], []
    for head, _, kind in triples:
        kinds.append([float(kind == 'tap'), float(kind == 'hold')])
        ranks.append(seen[head] / 4.)
        sizes.append(counts[head] / 4.)
        seen[head] += 1
    features = torch.cat((time_basis((heads - origin_us) / 1e6),
        time_basis((tails - heads) / 1e6), time_basis((heads - previous) / 1e6),
        time_basis((following - heads) / 1e6), torch.tensor(kinds),
        torch.tensor(ranks)[:, None], torch.tensor(sizes)[:, None]), -1)
    return LaneDraft(triples, features)


def ordered_lane_labels(events):
    """Canonicalize labels only; tied lane-free members have identical payloads."""
    ordered = sorted(events, key=lambda e: (e['onset_us'], e['type'], e['end_us'], e['lane']))
    if any(e['lane'] not in range(4) for e in ordered):
        raise ValueError('mania lane label outside 0..3')
    return torch.tensor([e['lane'] for e in ordered], dtype=torch.long)


class LanePrefix:
    def __init__(self):
        self.release = [-1] * 4
        self.last_head = [-1] * 4
        self.last_lane = 4  # unknown / start of sequence

    def features(self, head):
        return torch.tensor([max(0, tail - head) / 32e6 for tail in self.release] +
                            [float(t == head) for t in self.last_head], dtype=torch.float32)

    def allowed(self, head):
        # A hold ending exactly at this head releases its lane.
        return torch.tensor([tail <= head and last != head
                             for tail, last in zip(self.release, self.last_head)])

    def advance(self, event, lane):
        if lane not in range(4):
            raise ValueError('selected lane outside 0..3')
        head, tail, _ = event
        self.release[lane] = max(self.release[lane], tail)
        self.last_head[lane], self.last_lane = head, lane


def prefix_inputs(draft, labels, *, conditioned):
    if len(labels) != len(draft.events):
        raise ValueError('label count differs from rhythm members')
    if not conditioned:
        return torch.full((len(labels),), 4, dtype=torch.long), torch.zeros(len(labels), 8)
    state = LanePrefix()
    previous, occupancy = [], []
    for event, lane in zip(draft.events, labels.tolist()):
        previous.append(state.last_lane)
        occupancy.append(state.features(event[0]))
        state.advance(event, int(lane))
    return torch.tensor(previous), torch.stack(occupancy)


def assign_lanes(draft, lanes):
    if len(lanes) != len(draft.events):
        raise ValueError('incomplete lane assignment cannot become a chart')
    return [dict(onset_us=h, end_us=t, type=k, lane=int(lane))
            for (h, t, k), lane in zip(draft.events, lanes)]


def rhythm_signature(events):
    return Counter((e['onset_us'], e['end_us'], e['type']) for e in events)


def maximum_overlap(draft):
    """Minimum lanes for half-open holds and same-time tap/head multiplicity."""
    groups = {}
    for event in draft.events:
        groups.setdefault(event[0], []).append(event)
    releases, maximum = [], 0
    for head, events in sorted(groups.items()):
        releases = [tail for tail in releases if tail > head]
        maximum = max(maximum, len(releases) + len(events))
        releases.extend(tail for _, tail, kind in events if kind == 'hold')
    return maximum

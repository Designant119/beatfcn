"""Parallel geometry packets and small encoder for Phira context-only Stage A."""
import numpy as np
import torch
from torch import nn
from ..spatial.context import Packet, POINTS


def attach_context(ex, context, observed, generated, evidence):
    def pack(packets):
        n = len(packets)
        return {'points': np.stack([p.relative(ex.c0_us) for p in packets]) if n else np.zeros((0, POINTS, 7), np.float32),
                'valid': np.stack([p.valid for p in packets]) if n else np.zeros((0, POINTS), bool),
                'status': np.asarray([p.status for p in packets], np.int64),
                'source': np.asarray([p.source_kind for p in packets], np.int64)}
    known, refs = [], []
    for event in observed:
        known.append(context.event_packet(event) if context is not None else Packet())
    known += [Packet() for _ in generated]
    from ..events.canonical import Event
    for ref in evidence:
        for raw in ref['selected']:
            refs.append(context.event_packet(Event.from_json(raw), reference_id=ref['reference_id'])
                        if context is not None else Packet())
        refs += [Packet() for _ in ref['unselected_us']]
    ex.space = {'known': pack(known), 'refs': pack(refs),
                'memory': pack(context.memory if context is not None else [])}
    if context is not None:
        ex.provenance['spatial_projection'] = {'cache_key': context.cache_key, **context.audit}
    # Canonical observable order cannot depend on native ID spelling. Keep every
    # event, including identical multiplicities. Generated spatial packets are unknown.
    if ex.mode == 1 and ex.stage_a_arm in ('removed', 'operation'):
        for name, rows in (('known', ex.known), ('refs', ex.refs)):
            packet = ex.space[name]
            order = sorted(range(len(rows)), key=lambda i: (tuple(rows[i]),
                           tuple(packet['points'][i].ravel()) if ex.stage_a_arm == 'operation' else ()))
            rows[:] = rows[order]
            ex.space[name] = {k: v[order] for k, v in packet.items()}


def collate_space(examples, out, K, R):
    if not any(ex.stage_a_arm is not None for ex in examples):
        return
    arms = {ex.stage_a_arm for ex in examples}
    if len(arms) != 1:
        raise ValueError('one Stage A arm per batch; arms must not be mixed')
    M = max([len(ex.space['memory']['status']) for ex in examples] + [1])
    out['stage_a_arm'] = next(iter(arms))
    for name, size in (('known', K), ('refs', R), ('memory', M)):
        shapes = {'points': (POINTS, 7), 'valid': (POINTS,), 'status': (), 'source': ()}
        for key, suffix in shapes.items():
            dtype = torch.float32 if key == 'points' else torch.bool if key == 'valid' else torch.long
            arr = torch.zeros((len(examples), size, *suffix), dtype=dtype)
            for b, ex in enumerate(examples):
                value = ex.space[name][key]
                arr[b, :len(value)] = torch.from_numpy(value)
            out[f'space_{name}_{key}'] = arr
    out['space_memory_point_valid'] = out['space_memory_valid']
    out['space_memory_valid'] = out['space_memory_valid'].any(-1)


class SpaceEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, width))
        self.status = nn.Embedding(5, width)
        self.source = nn.Embedding(6, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, points, valid, status, source):
        t, nx, ny, d, w = (points[..., i] for i in range(5))
        # (n,d) and (-n,-d) define the same strip, without a sign-boundary jump.
        feats = torch.stack([torch.sign(t) * torch.log1p(t.abs()), nx * nx - ny * ny,
                             2 * nx * ny, nx * d, ny * d, w], -1)
        enc = self.point(feats)
        pooled = (enc * valid[..., None]).sum(-2) / valid.sum(-1).clamp(min=1)[..., None]
        present = valid.any(-1) | (status != 0) | (source != 0)
        return self.norm(pooled + self.status(status) + self.source(source)) * present[..., None]


def encode_packets(encoder, bt, name):
    mask = bt['space_memory_point_valid'] if name == 'memory' else bt[f'space_{name}_valid']
    return encoder(bt[f'space_{name}_points'], mask, bt[f'space_{name}_status'], bt[f'space_{name}_source'])

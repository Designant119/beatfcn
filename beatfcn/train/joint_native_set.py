"""Input/target separation, multiset matching and unfiltered set decoding."""
from dataclasses import asdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.joint_native_set import relative_tail_nll
from beatfcn.model.local_typed_canvas import best_tail
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.joint_reference import joint_reference_inputs
from beatfcn.train.local_bundle import batch_rows
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.native_joint_support import NativeJointSupport, KINDS
from beatfcn.train.reference_completion import partition


SUPPORT = NativeJointSupport()


def injective_assignment(costs):
    """Rectangular Hungarian potentials, one distinct query per target column.

    Numpy implementation avoids adding a binary dependency to the MPS probe.
    Identical target columns are deliberately retained. Ties are deterministic.
    """
    costs = np.asarray(costs, np.float64)
    q, n = costs.shape
    if n > q or not np.isfinite(costs).all():
        raise ValueError('Invalid rectangular assignment')
    if not n:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a = costs.T
    u, v = np.zeros(n + 1), np.zeros(q + 1)
    assigned, previous = np.zeros(q + 1, np.int64), np.zeros(q + 1, np.int64)
    for i in range(1, n + 1):
        assigned[0] = i
        j0 = 0
        distances, used = np.full(q + 1, np.inf), np.zeros(q + 1, bool)
        while True:
            used[j0] = True
            current = a[assigned[j0] - 1] - u[assigned[j0]] - v[1:]
            improve = (~used[1:]) & (current < distances[1:])
            distances[1:][improve] = current[improve]
            previous[1:][improve] = j0
            candidates = np.where(used[1:], np.inf, distances[1:])
            j1 = int(candidates.argmin()) + 1
            delta = distances[j1]
            u[assigned[used]] += delta
            v[used] -= delta
            distances[~used] -= delta
            j0 = j1
            if not assigned[j0]:
                break
        while j0:
            j1 = previous[j0]
            assigned[j0] = assigned[j1]
            j0 = j1
    rows = np.flatnonzero(assigned[1:])
    return rows, assigned[rows + 1] - 1


def adapted_row(panel, row, *, mask_reference=False):
    regions = ()
    if row['view'] in ('middle', 'continuation', 'random_groups'):
        record = panel.records[row['index']]
        original = event_request(record['request'])
        request, visible, _, regions = partition(panel.targets(row['index']), original, row['view'],
            f"{record['chart_id']}:{original.start_us}", 20261027)
        if request.start_us != row['request'].start_us or list(visible) != list(row['visible']):
            raise ValueError('Original visible request identity mismatch')
    visible = () if mask_reference else row['visible']
    regions = () if mask_reference else regions
    reference, provenance = joint_reference_inputs(row['request'], panel.records[row['index']]['audio_sha256'],
        visible, regions, source_id=f"joint-set:{row['index']}:{row['view']}", role=row['source_role'])
    return explicit_row({**row, 'inputs': {**row['inputs'], **reference}, 'provenance': provenance})


def set_batch(row, device='cpu'):
    request = row['request']
    inputs, _ = batch_rows([explicit_row(row)])
    heads, valid = SUPPORT.addresses(request)
    relative_times = (heads.reshape(-1) - request.start_us).astype(np.float64) / 1e6
    inputs.update(joint_frame_times=torch.arange(len(heads), dtype=torch.float32)[None] * .04,
        joint_head_times=torch.tensor(relative_times, dtype=torch.float32)[None],
        joint_head_valid=torch.from_numpy(valid.reshape(1, -1)),
        joint_kind_allowed=torch.tensor([[True] + [request.allowed_types is None or k in request.allowed_types for k in KINDS]]))
    encoded = SUPPORT.encode(row['targets'], request)
    target = dict(address=torch.tensor([m.cell * heads.shape[1] + m.address for m in encoded.members], dtype=torch.long),
        kind=torch.tensor([m.kind + 1 for m in encoded.members], dtype=torch.long),
        head_us=np.asarray([heads[m.cell, m.address] for m in encoded.members], np.int64),
        delta_ms=torch.tensor([m.delta_ms for m in encoded.members], dtype=torch.long))
    return {k: v.to(device) for k, v in inputs.items()}, target


def tail_for_heads(model, queries, heads_us, request, inputs):
    # Integer operations stay on CPU. MPS integer divisions of absolute us have
    # previously rounded via float32. Only bounded relative seconds cross over.
    heads_us = np.asarray(heads_us, dtype=np.int64)
    relative = torch.tensor((heads_us - request.start_us) / 1e6, dtype=torch.float32, device=queries.device)
    high = torch.tensor(request.duration_us // 1000 - heads_us // 1000, dtype=torch.long, device=queries.device)
    return (*model.tail(queries, relative, inputs), high)


def match_targets(model, output, target, request, inputs):
    n = len(target['kind'])
    q = output['kind'].shape[1]
    if n > q:
        raise ValueError(f'Event set capacity overflow: {n}>{q}')
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    device = output['kind'].device
    with torch.no_grad():
        kind = output['kind'][0].log_softmax(-1)
        head = output['head'][0].log_softmax(-1)
        cost = -kind[:, target['kind'].to(device)] - head[:, target['address'].to(device)]
        holds = torch.where(target['kind'] == 2)[0].tolist()
        # Full hold cost, in bounded chunks: no Q*N*song-ms tensor.
        for lo in range(0, len(holds), 4):
            ids = holds[lo:lo + 4]
            queries = output['query'][0].repeat_interleave(len(ids), 0)
            heads = np.tile(target['head_us'][ids], q)
            b, c, m, hi = tail_for_heads(model, queries, heads, request, inputs)
            delta = target['delta_ms'][ids].to(device).repeat(q)
            cost[:, ids] += relative_tail_nll(b, c, m, hi, delta).reshape(q, len(ids))
        if not bool(torch.isfinite(cost).all()):
            raise ValueError('Nonfinite assignment cost')
        rows, cols = injective_assignment(cost.cpu().double().numpy())
    return rows, cols


def set_loss(model, output, target, request, inputs):
    rows, cols = match_targets(model, output, target, request, inputs)
    device = output['kind'].device
    classes = torch.zeros(model.queries, dtype=torch.long, device=device)
    classes[rows] = target['kind'][cols].to(device)
    type_nll = F.cross_entropy(output['kind'][0], classes, reduction='none')
    unmatched = classes == 0
    negative = type_nll[unmatched].mean() if bool(unmatched.any()) else type_nll.sum() * 0
    if len(rows) == 0:
        return negative, dict(positive=negative * 0, absent=negative, head=negative * 0, tail=negative * 0)
    head_nll = F.cross_entropy(output['head'][0, rows], target['address'][cols].to(device), reduction='none')
    tail_nll = head_nll * 0
    hold = np.where(target['kind'][cols].numpy() == 2)[0]
    if len(hold):
        chosen = rows[hold]
        b, c, m, hi = tail_for_heads(model, output['query'][0, chosen], target['head_us'][cols[hold]], request, inputs)
        loss_tail = relative_tail_nll(b, c, m, hi, target['delta_ms'][cols[hold]].to(device))
        # aten::index_copy.out is absent on this machine's MPS backend. This
        # incidence-matrix product is the same sparse insertion and derivative;
        # it neither falls back to CPU nor changes the target/loss definition.
        incidence = (torch.arange(len(rows), device=device)[:, None]
                     == torch.tensor(hold, device=device)[None, :]).to(loss_tail.dtype)
        tail_nll = tail_nll + incidence @ loss_tail
    positive = type_nll[rows] + head_nll + tail_nll
    # C10-G's explicit balanced matching surrogate. It is intentionally not
    # calibrated natural event probability. Empty/false-positive checks are kept.
    groups = [positive[classes[rows] == k].mean() for k in range(1, 5) if bool((classes[rows] == k).any())]
    pos = torch.stack(groups).mean()
    loss = .5 * pos + .5 * negative
    return loss, dict(positive=pos, absent=negative, head=head_nll.mean(), tail=tail_nll.sum() / max(len(hold), 1))


@torch.no_grad()
def decode_set(model, inputs, request, *, deadline=None, output=None):
    started = time.monotonic()
    model.eval()
    output = model(inputs) if output is None else output
    classes = output['kind'][0].argmax(-1).cpu().numpy()
    addresses = output['head'][0].argmax(-1).cpu().numpy()
    heads, valid = SUPPORT.addresses(request)
    events, tokens, failure = [], [], None
    try:
        active = np.where(classes > 0)[0]
        if len(active) > request.max_events:
            raise ValueError('Predicted event resource overflow')
        predicted_heads = heads.reshape(-1)[addresses[active]]
        hold = np.where(classes[active] == 2)[0]
        tails = {}
        if len(hold):
            b, c, m, _ = tail_for_heads(model, output['query'][0, active[hold]], predicted_heads[hold], request, inputs)
            for j, slot in enumerate(hold):
                h = int(predicted_heads[slot])
                delta = best_tail(b[j].cpu().numpy(), c[j].cpu().numpy(), m[j].cpu().numpy(),
                    h % 1000, request.duration_us - (h // 1000) * 1000) // 1000
                tails[int(slot)] = (h // 1000 + delta) * 1000
        for j, slot in enumerate(active):
            address = int(addresses[slot])
            if not valid.reshape(-1)[address]:
                raise ValueError('Invalid native/COPY head decision')
            h = int(predicted_heads[j])
            events.append(RhythmEvent(h, tails[j] if j in tails else h, KINDS[classes[slot] - 1]))
        events.sort()
        tokens = encode_events(events, request)
        if len(tokens) > request.max_tokens:
            raise ValueError('Set native serialization resource overflow')
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Set decode deadline')
    except (ValueError, TimeoutError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
    return dict(complete=failure is None, failure=failure, events=[asdict(e) for e in events],
        tokens=tokens, query_classes=classes.tolist(), selected_addresses=addresses.tolist(), wall_s=time.monotonic() - started)

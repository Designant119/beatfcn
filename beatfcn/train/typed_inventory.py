"""Public-carry preserving inputs and exact within-type member likelihood."""
from collections import Counter
from dataclasses import asdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.typed_inventory import count_nll, count_modes
from .exact_cell_support import ExactCellSupport
from .joint_reference import joint_reference_inputs
from .local_bundle import annotated_sequence, batch_rows
from .local_request_panel import event_request
from .native_cell_chain import previous_vectors, canonical_head_mask, endpoint_decision
from .native_joint_support import KINDS
from .reference_completion import partition, sparse_reference


SUPPORT = ExactCellSupport()


def signature(event):
    return (event.onset_us, event.end_us, event.type, event.lane)


def inventory_row(panel, row, *, mask_reference=False):
    """No target is used to construct references or incoming-carry content."""
    regions = ()
    if row['view'] in ('middle', 'continuation', 'random_groups'):
        record = panel.records[row['index']]
        original = event_request(record['request'])
        request, visible, _, regions = partition(panel.targets(row['index']), original, row['view'],
            f"{record['chart_id']}:{original.start_us}", 20261027)
        if request.start_us != row['request'].start_us or list(visible) != list(row['visible']):
            raise ValueError('Original public partition identity')
    return carry_reference_row(row, panel.records[row['index']]['audio_sha256'],
                               regions=regions, mask_reference=mask_reference)


def carry_reference_row(row, audio_sha256, *, regions=(), mask_reference=False):
    request = row['request']
    visible = () if mask_reference else tuple(row['visible'])
    regions = () if mask_reference else tuple(regions)
    reference, provenance = joint_reference_inputs(request, audio_sha256, visible, regions,
        source_id=f"typed-inventory:{row['index']}:{row['view']}", role=row['source_role'])
    already = Counter(map(signature, visible))
    needed = Counter(map(signature, request.known_carry)) - already
    extra = []
    for event in request.known_carry:
        key = signature(event)
        if needed[key] > 0:
            extra.append(event)
            needed[key] -= 1
    if extra:
        carry = sparse_reference(extra, request)
        for key in ('reference', 'reference_times', 'reference_valid'):
            reference[key] = np.concatenate((reference[key], carry[key][1:]), axis=0)
        provenance = [*provenance, *[dict(role='authorized_incoming_hold', positive_only=True,
            source_id='public-request-carry', event=asdict(e)) for e in extra]]
    # Keep the caller's original multiset; sequence is an unused legal collator stub.
    return {**row, 'inputs':{**row['inputs'], **reference}, 'provenance':provenance,
            'known_regions':tuple(row.get('known_regions', ())),
            'sequence':annotated_sequence([], request), 'carry_reference_additions':[asdict(e) for e in extra]}


def inventory_batch(row, device='cpu'):
    inputs, _ = batch_rows([row])
    request = row['request']
    heads, allowed = SUPPORT.addresses(request)
    for lo, hi in row.get('known_regions', ()):
        allowed &= ~((heads >= lo) & (heads < hi))
    _, encoded = SUPPORT.encode(row['targets'], request)
    counts = np.zeros((len(heads), 4), np.int64)
    members = [[[] for _ in range(4)] for _ in heads]
    for member in encoded:
        if not allowed[member.cell, member.address]:
            raise ValueError('Target in caller-protected known interval')
        counts[member.cell, member.kind] += 1
        members[member.cell][member.kind].append(member)
    for c in range(len(heads)):
        for k in range(4):
            members[c][k].sort(key=lambda m:(heads[c, m.address], m.end_us))
    permitted = np.zeros((len(heads), 4), bool)
    for k, name in enumerate(KINDS):
        if request.allowed_types is not None and name not in request.allowed_types:
            continue
        valid = allowed.copy()
        if name == 'hold':
            has_end = (heads//1000+1)*1000 <= request.duration_us
            for copy in request.copy_times:
                has_end |= (heads < copy) & (copy <= request.duration_us)
            valid &= has_end
        permitted[:, k] = valid.any(-1)
    inputs.update(joint_frame_times=torch.arange(len(heads), dtype=torch.float32)[None]*.04,
                  cell_has_address=torch.from_numpy(allowed.any(-1))[None],
                  typed_allowed=torch.from_numpy(permitted)[None])
    if np.any((counts > 0) & ~permitted) or np.any(counts.sum(-1) > 64):
        raise ValueError('Typed target has no legal count support')
    target = dict(counts=counts, members=members, heads=heads, allowed=allowed)
    return {k:v.to(device) for k,v in inputs.items()}, target


def typed_head_mask(heads, allowed, previous, request, kind):
    valid = canonical_head_mask(heads, allowed, previous, request)
    if kind == 1:
        possible = (heads//1000+1)*1000 <= request.duration_us
        for copy in request.copy_times:
            possible |= (heads < copy) & (copy <= request.duration_us)
        valid &= possible
    return valid


def run_typed_members(model, output, inputs, request, target, counts, *, teacher, trace=False, deadline=None):
    if counts.shape != (len(target['heads']), 4) or np.any(counts < 0) or np.any(counts.sum(-1) > 64):
        raise ValueError('Typed member count capacity')
    total = int(counts.sum())
    if total > min(256, request.max_events):
        raise ValueError('Predicted total event OVERFLOW')
    active = np.argwhere(counts > 0)
    zero = output['typed_scores'].sum()*0
    parts = dict(head=zero, tail=zero)
    steps, events = [], []
    if not len(active):
        return events, parts, steps, 0
    device = output['x'].device
    cells = torch.tensor(active[:, 0], device=device)
    kinds = torch.tensor(active[:, 1], device=device)
    vec = torch.tensor(counts[active[:, 0]], device=device)
    state, embedding = model.initial_typed_members(output, cells, vec, kinds)
    previous = [None]*len(active)
    holds = 0
    for rank in range(int(counts.max())):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Typed member deadline')
        state, query = model.member_step(output, cells, embedding, state,
                                         previous_vectors(model, previous, request.start_us, inputs))
        addresses = target['heads'][active[:, 0]]
        features = model.address_features(output, inputs, addresses, cells, request.start_us)
        masks = np.stack([typed_head_mask(addresses[j], target['allowed'][c], previous[j], request, k)
                          for j,(c,k) in enumerate(active)])
        if not masks.any(-1).all():
            raise ValueError('Typed member has no reachable head')
        logits = model.head_logits(query, features).masked_fill(~torch.tensor(masks, device=device), -torch.inf)
        if teacher:
            chosen = np.array([target['members'][c][k][rank].address for c,k in active], np.int64)
            if not masks[np.arange(len(active)), chosen].all():
                raise ValueError('Unreachable true typed member')
        else:
            chosen = logits.argmax(-1).detach().cpu().numpy()
        indices = torch.tensor(chosen, device=device)
        selected = features[torch.arange(len(active), device=device), indices]
        if teacher:
            parts['head'] = parts['head'] + F.cross_entropy(logits, indices, reduction='sum')
        current = []
        for j,(cell,kind) in enumerate(active):
            head = int(addresses[j, chosen[j]])
            end = head
            tail = []
            if kind == 1:
                value = target['members'][cell][kind][rank].end_us if teacher else None
                end, nll, tail = endpoint_decision(model, output, inputs, state[j], selected[j],
                    request, head, previous[j], target=value, trace=trace)
                parts['tail'] = parts['tail'] + nll
                holds += 1
            event = RhythmEvent(head, int(end), KINDS[int(kind)])
            current.append(event)
            events.append(event)
            if trace:
                steps.append(dict(cell=int(cell), kind=int(kind), rank=rank, event=asdict(event),
                                  head_probability=logits[j].softmax(-1).detach().cpu().tolist(), tail=tail))
        keep = np.array([j for j,(c,k) in enumerate(active) if counts[c,k] > rank+1], np.int64)
        if not len(keep):
            break
        previous = [current[j] for j in keep]
        active = active[keep]
        cells, kinds, state, embedding = cells[keep], kinds[keep], state[keep], embedding[keep]
    return events, parts, steps, holds


def inventory_loss(model, output, inputs, request, target):
    counts = torch.tensor(target['counts'][None], device=output['x'].device)
    count = count_nll(output['typed_scores'], output['overflow'], inputs['typed_allowed'], counts).sum()
    _, parts, _, holds = run_typed_members(model, output, inputs, request, target, target['counts'], teacher=True)
    denominator = len(target['counts'])
    loss = (count+parts['head']+parts['tail'])/denominator
    return loss, dict(count=count/denominator, head=parts['head']/denominator, tail=parts['tail']/denominator,
                      events=int(target['counts'].sum()), holds=holds)


@torch.no_grad()
def decode_inventory(model, inputs, request, layout, *, output=None, deadline=None, trace=False):
    if set(layout) != {'heads', 'allowed'}:
        raise ValueError('Decode accepts public addresses, never labels')
    model.eval()
    output = model(inputs) if output is None else output
    counts, overflow = count_modes(output['typed_scores'], output['overflow'], inputs['typed_allowed'])
    events, steps, failure, legacy = [], [], None, None
    try:
        if overflow:
            raise ValueError('Explicit inventory OVERFLOW')
        events, _, steps, _ = run_typed_members(model, output, inputs, request, layout, counts,
                                               teacher=False, trace=trace, deadline=deadline)
        SUPPORT.encode(events, request)
        if 1+3*len(events) > request.max_tokens:
            raise ValueError('Exact field tape overflow')
        if all(e.type != 'hold' or e.end_us % 1000 == 0 for e in events):
            legacy = encode_events(events, request)
            if len(legacy) > request.max_tokens:
                raise ValueError('Native grammar token overflow')
    except (ValueError, TimeoutError) as exc:
        failure = f'{type(exc).__name__}: {exc}'
    return dict(complete=failure is None, failure=failure, events=[asdict(e) for e in sorted(events)],
                count_decision=counts.tolist(), overflow_cells=overflow, field_tape_version='typed-exact-us-events-v1',
                legacy_tokens=legacy, steps=steps)

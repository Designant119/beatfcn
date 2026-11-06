"""C16 count law; unchanged C15 public support and within-type event members."""
import torch
from dataclasses import asdict

from beatfcn.model.full_event_codec import encode_events
from beatfcn.model.shared_total_inventory import count_nll, count_modes
from .typed_inventory import run_typed_members, SUPPORT


def apply_update(optimizer, report):
    """Record an applied step before a later finite-state check can raise."""
    report['accumulation_phase'] = 'optimizer_step_started'
    optimizer.step()
    report['updates'] += 1
    report['optimizer_updates'] = report['updates']
    report['accumulation_phase'] = 'applied'


def inventory_loss(model, output, inputs, request, target):
    counts = torch.tensor(target['counts'][None], device=output['x'].device)
    count = count_nll(output['typed_scores'], output['total_scores'], inputs['typed_allowed'], counts).sum()
    _, parts, _, holds = run_typed_members(model, output, inputs, request, target, target['counts'], teacher=True)
    denominator = len(target['counts'])
    loss = (count + parts['head'] + parts['tail']) / denominator
    return loss, dict(count=count/denominator, head=parts['head']/denominator,
                      tail=parts['tail']/denominator, events=int(target['counts'].sum()), holds=holds)


@torch.no_grad()
def decode_inventory(model, inputs, request, layout, *, output=None, deadline=None, trace=False):
    # Same native serializer and public support; a distinct count decision contract.
    if set(layout) != {'heads', 'allowed'}:
        raise ValueError('Decode accepts public addresses, never labels')
    model.eval()
    output = model(inputs) if output is None else output
    counts, overflow = count_modes(output['typed_scores'], output['total_scores'], inputs['typed_allowed'])
    events, steps, failure, legacy = [], [], None, None
    try:
        if overflow:
            raise ValueError('Explicit shared-total OVERFLOW')
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
                count_decision=counts.tolist(), overflow_cells=overflow,
                field_tape_version='shared-total-exact-us-events-v1', legacy_tokens=legacy, steps=steps)

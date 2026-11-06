"""Visibility and unique-address adapter for new package experiments only."""
from dataclasses import replace

from .local_bundle import annotated_sequence
from .local_typed_canvas import canvas_batch


def explicit_reference_row(row):
    """Only visible input events can supply incoming holds in this local panel."""
    request = row['request']
    visible = row.get('visible', ())
    carry = tuple(e for e in visible if e.type == 'hold'
                  and e.onset_us < request.start_us < e.end_us)
    clean = replace(request, known_carry=carry)
    return {**row, 'request': clean, 'sequence': annotated_sequence(row['targets'], clean),
            'dropped_unprovided_carry': len(set(request.known_carry) - set(carry))}


def canonical_batch(rows, device='cpu', *, mask_reference=False):
    inputs, target = canvas_batch(rows, device, mask_reference=mask_reference)
    # Merge address aliases, never event multiplicity. encode_canvas uses the
    # first exact address, so retaining the first alias preserves its labels.
    for b in range(len(rows)):
        for frame in range(inputs['canvas_head_us'].shape[1]):
            seen = set()
            for address in inputs['canvas_head_allowed'][b, frame].nonzero().flatten().tolist():
                value = int(inputs['canvas_head_us'][b, frame, address])
                if value in seen:
                    inputs['canvas_head_allowed'][b, frame, address] = False
                else:
                    seen.add(value)
    return inputs, target

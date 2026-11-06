"""Numerical equivalence on supervised queries, excluding padded queries."""
import torch


@torch.no_grad()
def compare_valid_logits(reference, actual, valid, *, allowed=None, tolerance=1e-3):
    if (reference.ndim != 3 or actual.shape != reference.shape or valid.shape != reference.shape[:2]
            or valid.dtype != torch.bool or not bool(valid.any())
            or (allowed is not None and (allowed.shape != reference.shape or allowed.dtype != torch.bool))):
        raise ValueError('Logit comparison requires equal shapes and explicit valid query mask')
    a, b = reference[valid], actual[valid]
    nonfinite = int((~torch.isfinite(a)).sum()) + int((~torch.isfinite(b)).sum())
    difference = None if nonfinite else float((a - b).abs().max())
    if allowed is not None:
        legal = allowed[valid]
        if not bool(legal.any(-1).all()):
            raise ValueError('Every valid query needs legal target support')
        a, b = a.masked_fill(~legal, -1e9), b.masked_fill(~legal, -1e9)
    mismatches = int((a.argmax(-1) != b.argmax(-1)).sum())
    return dict(valid_queries=int(valid.sum()), excluded_padding_queries=int((~valid).sum()),
                maximum_logit_difference=difference, nonfinite=nonfinite,
                argmax_disagreements=mismatches,
                passed=not nonfinite and difference < tolerance and mismatches == 0)

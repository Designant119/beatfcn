"""Explicit request-boundary MPS cache cleanup; never clear accumulated gradients."""
import gc

import torch

from .process_memory import physical_footprint


def memory_sample():
    torch.mps.synchronize()
    return dict(**physical_footprint(),
                driver_bytes=torch.mps.driver_allocated_memory(),
                allocated_bytes=torch.mps.current_allocated_memory(),
                recommended_bytes=torch.mps.recommended_max_memory())


def clear_request_cache():
    # The caller drops forward/loss references after backward. Parameter .grad
    # remains live until the optimizer boundary; empty_cache must not erase it.
    gc.collect()
    torch.mps.synchronize()
    before = memory_sample()
    torch.mps.empty_cache()
    return dict(before=before, after=memory_sample())

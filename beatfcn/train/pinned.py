"""Pinned staging ring for DataLoader batches.

The main process copies every packed batch (batch.pack_batch: one buffer per dtype) into a preallocated pinned
slot and hands out views into it, so host->device copies stay asynchronous without PyTorch's pin-memory thread.
That thread allocated pinned memory per batch (buffer sizes differ from batch to batch, so its size-keyed cache
rarely hit), and cudaHostAlloc waits for the busy device: delivery was capped near one batch per step while the
workers sat idle. Slots are reused once the copies that read them have completed (a CUDA event per slot).
"""
from __future__ import annotations
import time
import numpy as np
import torch


class PinnedRing:
    def __init__(self, depth: int = 64, pin: bool | None = None, copy_method: str = 'torch'):
        if copy_method not in ('torch', 'bytes'):
            raise ValueError('unsupported staging copy method')
        self.depth = depth
        self.copy_method = copy_method
        self.pin = torch.cuda.is_available() if pin is None else pin
        self.slots = [{} for _ in range(depth)]
        self.events = [None] * depth
        self.i = 0
        self.stats = {'grown': 0, 'wait_s': 0., 'allocate_s': 0., 'copy_s': 0., 'copy_bytes': 0}

    def preallocate(self, largest: dict):
        """Reserve exact tape maxima before compute starts, one allocation per dtype."""
        if any(self.slots) or any(event is not None for event in self.events):
            raise ValueError('preallocation requires a fresh staging ring')
        total = self.depth * sum(buf.numel() * buf.element_size() for buf in largest.values())
        if total > 512 * 2**20:
            raise ValueError('metadata staging exceeds the 512 MiB host-memory bound')
        started = time.perf_counter()
        for key, buf in largest.items():
            bank = torch.empty((self.depth, buf.numel()), dtype=buf.dtype, pin_memory=self.pin)
            for index, slot in enumerate(self.slots):
                slot[key] = bank[index]
        self.stats['allocate_s'] += time.perf_counter() - started
        self.stats['grown'] += self.depth * len(largest)
        return total

    def take(self, packed: dict) -> tuple[dict, int]:
        """Copy {dtype name: 1-D tensor} into the next slot; returns (views into the slot, slot index)."""
        i = self.i
        self.i = (self.i + 1) % self.depth
        ev = self.events[i]
        if ev is not None:
            started = time.perf_counter()
            ev.synchronize()  # the device copies that read this slot (a step long finished) must be complete
            self.stats['wait_s'] += time.perf_counter() - started
            self.events[i] = None
        slot = self.slots[i]
        out = {}
        for key, buf in packed.items():
            n = buf.numel()
            cur = slot.get(key)
            if cur is None or cur.numel() < n:
                started = time.perf_counter()
                cur = torch.empty(max(n, int(1.5 * n)), dtype=buf.dtype)
                if self.pin:
                    cur = cur.pin_memory()
                self.stats['allocate_s'] += time.perf_counter() - started
                slot[key] = cur
                self.stats['grown'] += 1
            started = time.perf_counter()
            if self.copy_method == 'bytes':
                if buf.device.type != 'cpu' or buf.ndim != 1 or not buf.is_contiguous() or buf.requires_grad:
                    raise ValueError('byte staging requires contiguous 1-D CPU metadata without gradients')
                # This is transport, not a tensor operation: preserve every bit,
                # including bfloat16 and NaNs, without entering the ATen copy kernel.
                np.copyto(cur[:n].view(torch.uint8).numpy(), buf.view(torch.uint8).numpy())
            else:
                cur[:n].copy_(buf)
            self.stats['copy_s'] += time.perf_counter() - started
            self.stats['copy_bytes'] += n * buf.element_size()
            out[key] = cur[:n]
        return out, i

    def done(self, i: int, stream=None):
        """The slot's views were consumed by copies enqueued on `stream`: release it once they complete."""
        if stream is None or not self.pin:
            return
        ev = torch.cuda.Event()
        ev.record(stream)
        self.events[i] = ev

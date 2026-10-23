"""Throughput-oriented cache encoder: frozen teacher -> int8 taps + S20 summaries, songs streamed in groups.

Design (GPU occupancy first, measured in beatfcn/cloud/encode_app.py):
  * one fixed batch shape; the last partial batch of a group is padded with silence windows that already
    live on the device (a pageable host->device copy may synchronise the stream);
  * the forward *and* the int8 quantisation run as one unit (TapRunner). In 'graph' modes that unit is a
    captured CUDA graph, so the host issues ~6 calls per batch instead of hundreds of kernel launches
    (compiled Inductor code launches each kernel from Python and loses the GIL race to reader/writer
    threads: 68 windows/s on an H100 that computes 207 windows/s in isolation);
  * windows are built and converted to bf16 on a reader thread (pinned), results return by async
    device->host copies that writer threads wait on; the main thread never commits or syncs per group.
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from . import memory as M

CENTRE_CELLS = M.HOP // M.FRAMES_PER_CELL  # 128 kept cells per window


def tap_forward(enc, want):
    """Teacher forward returning the centre cells of the wanted taps (same math as TapEncoder)."""
    from .pupu import make_rope
    teacher = enc.teacher
    grid = (M.WINDOW // M.FRAMES_PER_CELL, M.BANDS)
    rope = make_rope(grid, teacher.head_dim, device=teacher.patch_embed.proj.weight.device)
    keep = slice(M.EDGE // M.FRAMES_PER_CELL, (M.EDGE + M.HOP) // M.FRAMES_PER_CELL)
    top = max(want)

    def fwd(w):
        x = teacher.patch_embed(w)
        out = []
        for l, block in enumerate(teacher.teacher.blocks, 1):
            x = block(x, rope)
            if l in want:
                out.append(x.reshape(x.shape[0], *grid, -1)[:, keep])
            if l >= top:
                break
        return out
    return fwd


def quantiser(layers, mean=None, scale=None, comp=None):
    """Per tap: centre (mean), optional projection (comp), int8 with per-dim scale; None -> float16."""
    def q(l, t):
        x = t.float()
        if mean and l in mean:
            x = x - mean[l]
        if comp and l in comp:
            x = x @ comp[l]
        if scale and l in scale:
            return (x / scale[l]).round().clamp(-127, 127).to(torch.int8)
        return x.half()
    return q


class TapRunner:
    """Fixed-batch forward + quantisation. Returns per wanted layer a device tensor [batch, 128, 8, D]:
    int8 (or fp16) for cache layers, the raw tap (bf16) for summary-only layers.

    mode: 'eager' | 'compile' | 'graph' (eager captured as a CUDA graph) | 'compile-graph'.
    In graph modes the returned tensors are the graph's static outputs: consume them (copy) before the
    next call; stream order makes that safe without host syncs."""

    def __init__(self, fwd, want, layers, quant, batch, window_shape, device, mode='eager', dtype=torch.bfloat16):
        self.want, self.layers, self.batch, self.mode = list(want), set(layers), batch, mode
        f = torch.compile(fwd) if 'compile' in mode else fwd

        def unit(w):
            return [quant(l, t) if l in self.layers else t for l, t in zip(self.want, f(w))]
        self.unit = unit
        self.static_in = torch.zeros((batch, *window_shape), dtype=dtype, device=device)
        self.graph = None
        if 'graph' in mode:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm-up (and compile) outside capture
                    unit(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = unit(self.static_in)

    def __call__(self, w, pad_rows):
        """w: [n<=batch, ...] host (pinned) or device windows in self.static_in's dtype."""
        n = w.shape[0]
        self.static_in[:n].copy_(w, non_blocking=True)
        if n < self.batch:
            self.static_in[n:].copy_(pad_rows[:self.batch - n])
        if self.graph is not None:
            self.graph.replay()
            return self.static_out
        return self.unit(self.static_in)


def summary_half(tap):
    """== memory.summary_means on any device, float16 result."""
    n = -(-tap.shape[0] // M.SUMMARY_CELLS)
    x = tap.new_zeros((n * M.SUMMARY_CELLS, *tap.shape[1:]), dtype=torch.float32)
    x[:tap.shape[0]] = tap.float()
    cnt = torch.full((n,), float(M.SUMMARY_CELLS), device=tap.device)
    cnt[-1] = tap.shape[0] - (n - 1) * M.SUMMARY_CELLS
    return (x.reshape(n, M.SUMMARY_CELLS, *tap.shape[1:]).sum(1) / cnt[:, None, None]).half()


class _NoEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def run(work, load_mel, save, runner, layers, *, group=4, writers=6, device='cuda', summary_layer=20,
        on_first_group=None, on_steady_end=None, max_pending=None):
    """Encode songs `work` (ids). load_mel(id) -> float32 [frames, 128]; save(id, [(name, ndarray)]) writes
    L{l} for each cache layer and S{summary_layer} last. Returns throughput / host-time statistics."""
    cuda = str(device).startswith('cuda')
    pad = torch.from_numpy(M.silence_row()).to(device, runner.static_in.dtype)
    pad_rows = pad.view(1, 1, 1, -1).expand(runner.batch, 1, M.WINDOW, pad.shape[0])
    groups = [work[i:i + group] for i in range(0, len(work), group)]
    max_pending = max_pending or 3 * group
    stats = {'songs': 0, 'windows': 0, 'bytes': 0, 'write_s': 0.0}
    host_t = {'wait_reader': 0.0, 'launch': 0.0, 'post': 0.0, 'wait_writer': 0.0}
    want = runner.want

    def prep(g):  # reader thread: mels -> bf16 window batch for the whole group (pinned on CUDA)
        mels = [load_mel(sha) for sha in g]
        plans = [M.plan(m.shape[0]) for m in mels]
        win = torch.cat([M.window_batch(m, pl.starts, M.silence_row()) for m, pl in zip(mels, plans)])
        win = win.to(runner.static_in.dtype)
        return g, plans, (win.pin_memory() if cuda else win)

    def write_after(ev, sha, host):
        ev.synchronize()
        t = time.time()
        arrays = [(name, h.numpy()) for name, h in host]
        save(sha, arrays)
        return sum(a.nbytes for _, a in arrays), time.time() - t

    reader, writer, pending = ThreadPoolExecutor(1), ThreadPoolExecutor(writers), []

    def drain(limit):
        while len(pending) > limit:
            tw = time.time()
            b, ws = pending.pop(0).result()
            host_t['wait_writer'] += time.time() - tw
            stats['bytes'] += b
            stats['write_s'] += ws
    nxt = reader.submit(prep, groups[0]) if groups else None
    t_steady = w_steady = b_steady = None
    with torch.inference_mode():
        for gi in range(len(groups)):
            tw = time.time()
            g, plans, win = nxt.result()
            host_t['wait_reader'] += time.time() - tw
            if gi + 1 < len(groups):
                nxt = reader.submit(prep, groups[gi + 1])
            tl = time.time()
            total = win.shape[0]
            acc = {}
            for i in range(0, total, runner.batch):
                w = win[i:i + runner.batch]
                n = w.shape[0]
                outs = runner(w, pad_rows)
                for l, t in zip(want, outs):
                    if l not in acc:
                        acc[l] = t.new_empty((total * CENTRE_CELLS, *t.shape[2:]))
                    acc[l][i * CENTRE_CELLS:(i + n) * CENTRE_CELLS].copy_(t[:n].reshape(-1, *t.shape[2:]))
            host_t['launch'] += time.time() - tl
            tp = time.time()
            off = 0
            for sha, pl in zip(g, plans):
                arrays = [(f'L{l}', acc[l][off:off + pl.cells]) for l in layers]
                arrays.append((f'S{summary_layer}', summary_half(acc[summary_layer][off:off + pl.cells])))
                off += len(pl.starts) * CENTRE_CELLS
                host = []
                for name, t in arrays:
                    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=cuda)
                    h.copy_(t, non_blocking=cuda)
                    host.append((name, h))
                ev = torch.cuda.Event() if cuda else _NoEvent()
                ev.record()
                pending.append(writer.submit(write_after, ev, sha, host))
                stats['songs'] += 1
                stats['windows'] += len(pl.starts)
            host_t['post'] += time.time() - tp
            if gi == 0:  # first group carries one-off costs (compile/capture warm-up): steady stats start here
                if cuda:
                    torch.cuda.synchronize()
                if on_first_group:
                    on_first_group()
                t_steady, w_steady = time.time(), stats['windows']
                for k in host_t:
                    host_t[k] = 0.0
            drain(max_pending)
    if t_steady is not None and len(groups) > 1:  # steady phase ends with the last enqueue, before the flush
        if cuda:
            torch.cuda.synchronize()
        wall = time.time() - t_steady
        stats['steady_windows_per_s'] = round((stats['windows'] - w_steady) / wall, 1)
        stats['steady_host_s'] = {k: round(v, 1) for k, v in host_t.items()} | {'wall': round(wall, 1)}
    if on_steady_end:
        on_steady_end()
    tf = time.time()
    drain(0)
    stats['flush_s'] = round(time.time() - tf, 1)
    stats['write_MBps_per_thread'] = round(stats['bytes'] / 1e6 / max(stats['write_s'], 1e-9), 1)
    reader.shutdown()
    writer.shutdown()
    return stats

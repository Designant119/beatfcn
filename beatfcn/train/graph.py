"""One optimiser step as a replayed CUDA graph (forward + backward + clipping + AdamW + zero_grad).

Requirements met elsewhere: fixed batch shapes (batch.STATIC), no host syncs in the model, capturable fused
AdamW with tensor LR. The step reads static input buffers that the loop fills before each replay; the
counterfactual loss (every few steps, variable shapes) runs eagerly just before the replay and its grads
accumulate into the same static .grad tensors (zero_grad(set_to_none=False) runs at the end of the step).
GPU only for capture; `capture=False` runs the same step function eagerly (tests, CPU smoke).
"""
from __future__ import annotations
import torch


class GraphedStep:
    def __init__(self, runs, clip: float, device, capture=True, warmup=3, before_capture=None):
        """runs: objects with .model and .opt (AdamW(capturable=True, fused=True), tensor lr).
        before_capture(): called once right before the capture (e.g. SongPool.wait_idle)."""
        self.runs, self.clip, self.dev = runs, clip, torch.device(device)
        self.before_capture = before_capture
        self.capture = capture and self.dev.type == 'cuda'
        self.warmup, self.static, self.graph, self.out = warmup, None, None, None
        self.microbatches = None
        self.seen = 0

    # ------------------------------------------------------------ static inputs
    def fill(self, bt, w_select: float):
        """Copy a (device-resident audio, host metadata) batch into the static buffers."""
        self.fill_many([bt], w_select)

    def fill_many(self, batches, w_select: float):
        """Fixed accumulation slots; shapes may differ between slots, never across replays.

        No sample is cropped/resampled and no summary is pooled differently to fit a graph.
        Loss scaling matches ordinary gradient accumulation.
        """
        if not batches:
            raise ValueError('at least one microbatch is required')
        if not any(getattr(getattr(r.model, 'c', None), 'compact_inner', False) for r in self.runs):
            # Dynamic optional compaction metadata must not invalidate the existing padded graph path.
            batches = [{k:v for k,v in b.items() if k not in ('inner_index','hold_inner_index')} for b in batches]
        if self.microbatches is None:
            self.microbatches = []
            for bt in batches:
                buf = {k: (torch.empty_like(v, device=self.dev) if torch.is_tensor(v) else v) for k, v in bt.items()}
                buf['w_select'] = torch.zeros((), device=self.dev)
                self.microbatches.append(buf)
            self.static = self.microbatches[0]
        if len(batches) != len(self.microbatches):
            raise ValueError('accumulation count cannot change after buffers are allocated')
        for bt, buf in zip(batches, self.microbatches):
            if set(bt) - {'w_select'} != set(buf) - {'w_select'}:
                raise ValueError('input keys changed after buffers were allocated')
            for k, v in bt.items():
                if torch.is_tensor(v):
                    s = buf[k]
                    if s.shape != v.shape or s.dtype != v.dtype:
                        raise ValueError(f'static shape/dtype mismatch for {k}: {tuple(v.shape)} {v.dtype} vs {tuple(s.shape)} {s.dtype}')
                    s.copy_(v, non_blocking=True)
                else:
                    buf[k] = v
            buf['w_select'].fill_(w_select)

    # ------------------------------------------------------------ the step
    def _body(self):
        cuda = self.dev.type == 'cuda'
        mets = []
        for r in self.runs:
            acc = {}
            for bt in self.microbatches:
                b = dict(bt)  # the model pops/replaces entries (dequant): never on shared buffers
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=cuda, cache_enabled=False):
                    loss, met = r.model(b)
                (loss / len(self.microbatches)).backward()
                for k, value in met.items():
                    if k == '_illegal':
                        acc[k] = acc.get(k, 0) + value
                    else:
                        l, n, c = value
                        a = acc.setdefault(k, [0., 0, 0])
                        a[0], a[1], a[2] = a[0] + l * n, a[1] + n, a[2] + c
            # Public format remains (mean loss, count, correct), without host synchronization.
            mets.append({k: (value if k == '_illegal' else
                            (value[0] / value[1].clamp(min=1), value[1], value[2])) for k, value in acc.items()})
        norms = []
        for r in self.runs:
            norms.append(torch.nn.utils.clip_grad_norm_(r.model.parameters(), self.clip))
            r.opt.step()
            r.opt.zero_grad(set_to_none=False)  # grads stay allocated: static tensors the graph writes
        return mets, norms

    def step(self):
        """Run one step on the current static inputs; returns ([metrics per run], [grad norms])."""
        if not self.capture:
            return self._body()
        if self.graph is None:
            # parameters that take no gradient (e.g. the disabled Phira placement heads) keep grad=None, exactly
            # as in the plain loop: AdamW skips them; the warm-up step allocates every grad that is used
            if self.seen < self.warmup:  # eager warm-up on a side stream (allocator / cuBLAS / RNG state)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    out = self._body()
                torch.cuda.current_stream().wait_stream(s)
                self.seen += 1
                return out
            if self.before_capture:
                self.before_capture()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._body()
            # the capture itself did not run: replay once for this batch
        self.graph.replay()
        return self.out

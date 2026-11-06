"""CUDA Graph capture for one physical microbatch of the event model.

The graph replays forward + backward with static input buffers, so kernel launch
overhead stops scaling with model depth.  Fixed shapes come from the runner's
input padding plus a bounded set of sequence buckets; buckets outside that set
fall back to eager so a rare long window can never truncate supervision.

The optimizer step stays outside the graph.  fp16 loss scaling is applied with a
static device scalar (``scale_buf``) so GradScaler semantics remain intact:
the caller copies ``scaler.get_scale()`` into the graph, then calls
``scaler.unscale_(optimizer)`` before clipping and stepping.
"""
import threading

import torch

from beatfcn.model.full_event_model import event_loss
from beatfcn.train.training_objective import training_loss


# cudaHostAlloc (pin_memory) from a prefetch worker invalidates an in-flight
# graph capture.  Capture and pinned allocations take this lock.
CAPTURE_LOCK = threading.RLock()


class GraphStep:
    def __init__(self, model, optimizer, *, autocast_dtype=None, warmup=3, loss_settings=None):
        if warmup < 1:
            raise ValueError('CUDA graph warmup must be at least one iteration')
        self.model = model
        self.loss_settings = loss_settings
        self.optimizer = optimizer
        self.autocast_dtype = autocast_dtype
        self.warmup = int(warmup)
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('CUDA graph capture requires a CUDA model')
        self.weight_buf = torch.ones((), device=self.device)
        self.scale_buf = torch.ones((), device=self.device)
        self.entries = {}
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    def set_scale(self, scale):
        self.scale_buf.fill_(float(scale))

    @staticmethod
    def signature(inputs, sequence):
        return (tuple(sorted((k, tuple(v.shape), str(v.dtype)) for k, v in inputs.items())),
                tuple(sorted((k, tuple(v.shape), str(v.dtype)) for k, v in sequence.items())))

    def _forward_backward(self, inputs, sequence):
        if self.autocast_dtype is not None:
            with torch.autocast('cuda', dtype=self.autocast_dtype):
                logits = self.model(inputs, sequence)
        else:
            logits = self.model(inputs, sequence)
        loss = training_loss(logits, sequence, self.loss_settings) * self.weight_buf * self.scale_buf
        loss.backward()
        return loss

    def _capture(self, inputs, sequence):
        with CAPTURE_LOCK:
            static_inputs = {k: v.clone() for k, v in inputs.items()}
            static_sequence = {k: v.clone() for k, v in sequence.items()}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmup):
                    self._forward_backward(static_inputs, static_sequence)
            torch.cuda.current_stream().wait_stream(stream)
            # Warmup gradients are not a real microbatch.
            self.optimizer.zero_grad(set_to_none=False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                loss = self._forward_backward(static_inputs, static_sequence)
            # Capture itself ran one backward; clear it before the caller replays.
            self.optimizer.zero_grad(set_to_none=False)
            # Keep only the storage view: retaining the autograd graph would pin
            # the AccumulateGrad nodes to the capture stream and warn on replays.
            loss_out = loss.detach()
            del loss
            self.captures += 1
            entry = dict(graph=graph, inputs=static_inputs, sequence=static_sequence, loss=loss_out)
            self.entries[self.signature(inputs, sequence)] = entry
            return entry

    def capture(self, inputs, sequence):
        """Capture a bucket without replaying it (pre-training warmup)."""
        key = self.signature(inputs, sequence)
        if key in self.entries:
            return
        self.weight_buf.fill_(1.0)
        self._capture(inputs, sequence)

    def run(self, inputs, sequence, loss_weight, *, allow_capture=True):
        """Replay one microbatch; returns the scaled loss tensor or None to fall back."""
        key = self.signature(inputs, sequence)
        entry = self.entries.get(key)
        if entry is None:
            if not allow_capture:
                self.fallbacks += 1
                return None
            entry = self._capture(inputs, sequence)
        for name, value in inputs.items():
            entry['inputs'][name].copy_(value, non_blocking=True)
        for name, value in sequence.items():
            entry['sequence'][name].copy_(value, non_blocking=True)
        self.weight_buf.fill_(float(loss_weight))
        entry['graph'].replay()
        self.replays += 1
        return entry['loss']

    def summary(self):
        return dict(captures=self.captures, replays=self.replays,
                    fallbacks=self.fallbacks, buckets=len(self.entries))

"""Two separate LFM3-derived research hypotheses, with no text weights.

Budget allocation changes only the temporal trunk's width/depth. Context
matching changes only a training objective on a fixed mel-only network.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F

from .local_access_canvas import AccessBlock
from .mel_only_canvas import MelOnlyCanvas


class ContextMelCanvas(MelOnlyCanvas):
    """Optional audio-only coarse counterfactual; ordinary forward unchanged."""

    def encode(self, inputs):
        if 'coarse_mel' not in inputs:
            return super().encode(inputs)
        # Keep the fine path unchanged. The coarse projection sees the donor
        # payload while the native target clock/masks remain fixed.
        original_projection = self.mel_context_projection
        donor = inputs['coarse_mel']
        original = inputs['mel']
        valid, times = inputs['mel_valid'][0], inputs['mel_times'][0]
        bins = torch.floor(times[valid] / .5).long()
        bins -= bins.min()
        groups = int(bins.max()) + 1
        weights = F.one_hot(bins, groups).float()
        counts = weights.sum(0).clamp_min(1)
        delta = (weights.T @ (donor[0, valid] - original[0, valid])) / counts[:, None]
        # A local forward hook adds exactly W*(donor-original) to coarse
        # projected bins, preserving the common encoder implementation.
        hook = original_projection.register_forward_hook(
            lambda module, args, output: output + F.linear(delta, module.weight, None))
        try:
            return super().encode(inputs)
        finally:
            hook.remove()


class BudgetTemporal(nn.Module):
    def __init__(self, config, width, depth):
        super().__init__()
        if (width, depth) not in ((192, 8), (144, 16)):
            raise ValueError('Only the two frozen budget allocations')
        #136/4=34 is incompatible with the existing four-way rotary clock.
        #144/4=36 is native; FFN384 keeps the complete model within5%.
        trunk = replace(config, d=width, heads=4, kv_heads=2,
                        ffn=576 if depth == 8 else 384)
        self.enter = nn.Linear(config.d, width, bias=False)
        self.leave = nn.Linear(width, config.d, bias=False)
        schedule = 'CCWCGCWC' * (depth // 8)
        self.blocks = nn.ModuleList(AccessBlock(trunk, op, radius=16) for op in schedule)

    def forward(self, x, valid):
        x = torch.where(valid[..., None], x, 0.)
        z = self.enter(x)
        for layer in self.blocks:
            z = layer(z, valid)
        return torch.where(valid[..., None], x + self.leave(z), 0.)


class BudgetAllocationCanvas(ContextMelCanvas):
    def __init__(self, config, *, density_reference, width=192, depth=8):
        super().__init__(config, density_reference=density_reference)
        self.temporal = nn.ModuleList([BudgetTemporal(config, width, depth)])


class ContextMatchingCanvas(ContextMelCanvas):
    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        self.local_match = nn.Sequential(nn.Linear(config.d, 64), nn.SiLU(), nn.Linear(64, 64))
        self.context_match = nn.Sequential(nn.Linear(config.d, 64), nn.SiLU(), nn.Linear(64, 64))

    def matching_views(self, inputs):
        # Only observed audio/clocks. No target, control or reference enters.
        center = float(inputs['canvas_width_s'][0]) / 2
        times = inputs['mel_times'][0]
        valid = inputs['mel_valid'][0]
        local = valid & (times >= center - 1.) & (times < center + 1.)
        context = valid & (times >= center - 6.) & (times < center + 6.) & ~local
        if int(local.sum()) < 50 or int(context.sum()) < 50:
            raise ValueError('Insufficient disjoint observed audio for matching')
        return local, context

    def local_embedding(self, inputs):
        keep, _ = self.matching_views(inputs)
        times = inputs['mel_times'][:, keep] - float(inputs['canvas_width_s'][0]) / 2
        x = self.mel_projection(inputs['mel'][:, keep]) + self.mel_source + self.source_ids.weight[0]
        for layer in self.encoders[0]:
            x = layer(x, times, torch.ones_like(times, dtype=torch.bool))
        return F.normalize(self.local_match(x.mean(1)), dim=-1)

    def context_embedding(self, inputs):
        _, keep = self.matching_views(inputs)
        mel = inputs['mel'][0, keep]
        times = inputs['mel_times'][0, keep] - float(inputs['canvas_width_s'][0]) / 2
        bins = torch.floor(times / .5).long()
        bins -= bins.min()
        weights = F.one_hot(bins, int(bins.max()) + 1).float()
        counts = weights.sum(0)
        active = counts > 0
        weights = weights[:, active]
        counts = counts[active]
        pooled = (weights.T @ mel) / counts[:, None]
        clock = ((weights.T @ times) / counts)[None]
        x = self.mel_context_projection(pooled)[None] + self.mel_context_source + self.source_ids.weight[1]
        for layer in self.encoders[1]:
            x = layer(x, clock, torch.ones_like(clock, dtype=torch.bool))
        return F.normalize(self.context_match(x.mean(1)), dim=-1)

    def matching_loss(self, inputs, negative_inputs):
        if len(negative_inputs) != 3:
            raise ValueError('Exactly three frozen audio-only hard negatives')
        query = self.local_embedding(inputs)
        candidates = torch.cat([self.context_embedding(inputs)] +
                               [self.context_embedding(v) for v in negative_inputs], 0)
        logits = query @ candidates.T / .1
        return F.cross_entropy(logits, torch.zeros(1, device=logits.device, dtype=torch.long)), logits

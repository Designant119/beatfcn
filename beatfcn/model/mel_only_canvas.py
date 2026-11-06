"""Fresh trainable mel-only typed canvas; no pretrained local/song stream."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.model.local_bundle import read_grid
from beatfcn.model.local_typed_canvas import FRAMES, SLOTS, LocalTypedCanvas


class MelOnlyCanvas(LocalTypedCanvas):
    """Use fine 10ms mel and 0.5s pooled mel, never MERT/Pupu tensors."""

    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        for name in ('local_projection', 'song_projection', 'band_queries', 'band_embedding', 'layer_mix'):
            delattr(self, name)
        self.mel_context_projection = nn.Linear(config.mel_bins, config.d)
        self.mel_context_source = nn.Parameter(torch.randn(config.d) * .02)

    def encode(self, inputs):
        if len(inputs['mode']) != 1:
            raise ValueError('MelOnlyCanvas uses real microbatch one for explicit context pooling')
        mel = inputs['mel']
        if mel.ndim != 3 or mel.shape[-1] != self.c.mel_bins:
            raise ValueError('10ms mel tensor contract mismatch')
        valid = inputs['mel_valid']
        times = inputs['mel_times']
        fine = self.mel_projection(mel) + self.mel_source + self.source_ids.weight[0]
        for layer in self.encoders[0]:
            fine = layer(fine, times, valid)

        keep = valid[0]
        compact = mel[:, keep]
        compact_times = times[:, keep]
        if not compact.shape[1]:
            raise ValueError('MelOnlyCanvas requires at least one valid mel frame')
        bins = torch.floor(compact_times[0] / .5).long()
        bins = bins - bins.min()
        groups = int(bins.max()) + 1
        weights = F.one_hot(bins, groups).float()
        counts = weights.sum(0).clamp_min(1)
        pooled = (weights.transpose(0, 1) @ compact[0]) / counts[:, None]
        pooled_times = (weights.transpose(0, 1) @ compact_times[0]) / counts
        pooled = (self.mel_context_projection(pooled)[None] + self.mel_context_source
                  + self.source_ids.weight[1])
        pooled_times = pooled_times[None]
        for layer in self.encoders[1]:
            pooled = layer(pooled, pooled_times, torch.ones_like(pooled_times, dtype=torch.bool))

        reference = self.reference_projection(inputs['reference'])
        reference = torch.cat((self.empty_reference, reference), 1)
        reference_times = torch.cat((torch.zeros_like(times[:, :1]), inputs['reference_times']), 1)
        reference_valid = torch.cat((torch.ones(1, 1, dtype=torch.bool, device=mel.device),
            inputs['reference_valid']), 1)
        reference = reference + self.source_ids.weight[2]
        for layer in self.encoders[2]:
            reference = layer(reference, reference_times, reference_valid)
        controls = self.project_controls(inputs['controls'])
        control_times = torch.zeros_like(inputs['controls'][..., 0])
        control_valid = torch.ones_like(control_times, dtype=torch.bool)
        values = torch.cat((fine, pooled, reference, controls), 1)
        reference_timed = reference_valid.clone()
        reference_timed[:, 0] = False
        return dict(values=values,
            times=torch.cat((times, pooled_times, reference_times, control_times), 1),
            valid=torch.cat((valid, torch.ones_like(pooled_times, dtype=torch.bool),
                reference_valid, control_valid), 1),
            timed=torch.cat((valid, torch.ones_like(pooled_times, dtype=torch.bool),
                reference_timed, torch.zeros_like(control_valid)), 1), mode=inputs['mode'])

    def forward(self, inputs):
        controls = inputs['controls'].clone()
        density = controls[:, 0].clone()
        controls[:, 0] = 0
        memory = self.encode({**inputs, 'controls': controls})
        b = len(controls)
        m = inputs['mel'].shape[1]
        query = torch.arange(FRAMES, device=controls.device).float()[None] * .04
        query = query.expand(b, -1)
        four = query[..., None] + torch.tensor([.005, .015, .025, .035], device=query.device)
        mel = read_grid(memory['values'][:, :m], inputs['mel_times'], inputs['mel_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((query / 8, (query / 8).square(), torch.sin(query * math.pi),
            torch.cos(query * math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        style = (inputs['reference'] * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(controls.flatten(1)).chunk(2, -1)
        x = x * (1 + .1 * scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        for layer in self.address:
            x = layer(x, query + .02, memory)
        for layer in self.temporal:
            x = layer(x, inputs['canvas_valid'])
        x = self.norm(x)
        base_count = self.quantity(x).view(b, FRAMES, 4, SLOTS + 1)
        rho = density[:, :2].mean(-1) * 64
        tilt = torch.where(density[:, 2] > .5,
            F.softplus(self.density_alpha) * (rho.clamp_min(1e-4) / self.density_reference).log(), 0.)
        count = base_count + tilt[:, None, None, None] * torch.arange(SLOTS + 1, device=x.device)
        slot = x[:, :, None, None] + self.kind_slot[None, None]
        slot = slot + self.slot_read(slot)
        offset = self.offset(slot).masked_fill(~inputs['canvas_head_allowed'][:, :, None, None], -1e9)
        hold = slot[:, :, 1]
        return dict(count=count, base_count=base_count, existence=self.existence(x), offset=offset,
            tail_block=self.tail_block(hold), tail_cell=self.tail_cell(hold), tail_ms=self.tail_ms(hold))

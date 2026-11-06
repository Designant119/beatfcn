"""Attention-free non-autoregressive typed canvas with Mamba-2/SSD mixing."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.fullstack_mamba import FullStackMambaConfig, MambaBlock
from beatfcn.model.local_bundle import read_grid
from beatfcn.model.local_typed_canvas import (
    COPIES, FRAMES, SLOTS, LocalTypedCanvas, canvas_loss,
)


class MambaCanvas(LocalTypedCanvas):
    """LocalTypedCanvas input/output contract with bidirectional SSD mixing."""

    def __init__(self, config=None, *, density_reference):
        config = config or FullEventConfig(
            d=256, heads=8, kv_heads=4, ffn=768, encoder_layers=1,
            decoder_layers=0, dropout=0., mel_bins=128,
            ssm_state=128, ssm_headdim=64)
        super().__init__(config, density_reference=density_reference)
        c = config
        mixer = FullStackMambaConfig(
            d=c.d, ffn=c.ffn, encoder_layers=c.encoder_layers,
            taps=c.taps, bands=c.bands, band_slots=c.band_slots,
            tap_dim=c.tap_dim, reference_dim=c.reference_dim,
            controls=c.controls, control_dim=c.control_dim, mel_bins=c.mel_bins,
            d_state=c.ssm_state, headdim=c.ssm_headdim, dropout=c.dropout,
        )
        # Rebinding these containers removes every Attention instance created by
        # the reusable LocalTypedCanvas projections and output-head initializer.
        self.encoders = nn.ModuleList(
            nn.ModuleList(MambaBlock(mixer, bidirectional=True)
                          for _ in range(c.encoder_layers))
            for _ in range(3))
        self.address = nn.ModuleList(MambaBlock(mixer, bidirectional=True)
                                     for _ in range(2))
        self.temporal = nn.ModuleList(MambaBlock(mixer, bidirectional=True)
                                      for _ in range(2))
        self.time_projection = nn.Linear(33, c.d, bias=False)
        self.register_buffer('time_periods', torch.logspace(math.log10(.02), math.log10(2400), 16))

    def clock(self, times):
        angles = times[..., None] * (2 * math.pi / self.time_periods)
        features = torch.cat((times[..., None] / 1200, angles.sin(), angles.cos()), -1)
        return self.time_projection(features)

    def encode(self, inputs):
        """Encode only real rows; no right padding advances an SSD state."""
        if len(inputs['mode']) != 1:
            raise ValueError('MambaCanvas uses real microbatch one')
        c = self.c
        local = inputs['local']
        if (local.ndim != 5 or local.shape[1] != c.taps
                or local.shape[3:] != (c.bands, c.tap_dim)):
            raise ValueError('Local frontend tensor contract mismatch')

        # Filter the original local frames before either projection or scanning.
        keep = inputs['local_valid'][0]
        local = local[:, :, keep]
        local_times = inputs['local_times'][:, keep]
        n = local.shape[2]
        mixed = (F.layer_norm(local.float(), (c.tap_dim,))
                 * self.layer_mix.softmax(0)[None, :, None, None, None]).sum(1)
        band = self.local_projection(mixed) + self.band_embedding
        weights = (torch.einsum('btfd,qd->btqf', band, self.band_queries).float()
                   .div(math.sqrt(c.d)).softmax(-1).to(band.dtype))
        fine = torch.einsum('btqf,btfd->btqd', weights, band).reshape(1, n * c.band_slots, c.d)
        fine_times = local_times.repeat_interleave(c.band_slots, 1)
        fine_is_mel = torch.zeros(1, n * c.band_slots, dtype=torch.bool, device=local.device)

        # The compact encoded mel rows, rather than original n:m offsets, remain
        # available for the four read_grid queries on each physical canvas cell.
        mk = inputs['mel_valid'][0]
        mel = inputs['mel'][:, mk]
        mel_times = inputs['mel_times'][:, mk]
        fine = torch.cat((fine, self.mel_projection(mel) + self.mel_source), 1)
        fine_times = torch.cat((fine_times, mel_times), 1)
        fine_is_mel = torch.cat((fine_is_mel, torch.ones_like(mel_times, dtype=torch.bool)), 1)
        if not fine.shape[1]:
            raise ValueError('MambaCanvas requires nonempty local or mel audio')
        order = fine_times[0].argsort(stable=True)
        fine, fine_times = fine[:, order], fine_times[:, order]
        fine_is_mel = fine_is_mel[:, order]

        sk, rk = inputs['song_valid'][0], inputs['reference_valid'][0]
        song = self.song_projection(F.layer_norm(inputs['song'][:, sk].float(), (c.tap_dim,)))
        song_times = inputs['song_times'][:, sk]
        reference = torch.cat((self.empty_reference,
                               self.reference_projection(inputs['reference'][:, rk])), 1)
        reference_times = torch.cat((torch.zeros_like(inputs['reference_times'][:, :1]),
                                     inputs['reference_times'][:, rk]), 1)
        streams = ((fine, fine_times), (song, song_times), (reference, reference_times))
        encoded, times, timed = [], [], []
        for i, (x, t) in enumerate(streams):
            x = x + self.clock(t) + self.source_ids.weight[i] + self.mode_ids(inputs['mode'])[:, None]
            if x.shape[1]:
                for layer in self.encoders[i]:
                    x = layer(x)
            encoded.append(x)
            times.append(t)
            source_timed = torch.ones_like(t, dtype=torch.bool)
            if i == 2:
                source_timed[:, 0] = False
            timed.append(source_timed)
        audio = encoded[0][:, fine_is_mel[0]]
        audio_times = fine_times[:, fine_is_mel[0]]

        controls = self.project_controls(inputs['controls']) + self.mode_ids(inputs['mode'])[:, None]
        encoded.append(controls)
        times.append(torch.zeros_like(inputs['controls'][..., 0]))
        timed.append(torch.zeros_like(times[-1], dtype=torch.bool))
        values = torch.cat(encoded, 1)
        return dict(values=values, times=torch.cat(times, 1),
                    valid=torch.ones_like(values[..., 0], dtype=torch.bool),
                    timed=torch.cat(timed, 1), mode=inputs['mode'],
                    audio=audio, audio_times=audio_times,
                    audio_valid=torch.ones_like(audio_times, dtype=torch.bool))

    def forward(self, inputs):
        controls = inputs['controls'].clone()
        density = controls[:, 0].clone()
        controls[:, 0] = 0  # density has exactly one route into count logits
        memory = self.encode({**inputs, 'controls': controls})
        b = len(controls)
        frame_valid = inputs['canvas_valid'][0]
        if not bool(frame_valid.any()):
            raise ValueError('MambaCanvas requires at least one valid canvas frame')
        query = torch.arange(FRAMES, device=controls.device).float()[None] * .04
        query = query[:, frame_valid]
        four = query[..., None] + torch.tensor([.005, .015, .025, .035], device=query.device)
        mel = read_grid(memory['audio'], memory['audio_times'], memory['audio_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((query / 8, (query / 8).square(),
                             torch.sin(query * math.pi), torch.cos(query * math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        visible_reference = inputs['reference'].masked_fill(~visible[..., None], 0)
        style = visible_reference.sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(controls.flatten(1)).chunk(2, -1)
        x = x * (1 + .1 * scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]

        # Queries and compact memory form one sequence in every fusion layer;
        # both directions are true SSD scans, never attention over memory.
        context = memory['values']
        frame_count = x.shape[1]
        for layer in self.address:
            joint = layer(torch.cat((x, context), 1))
            x, context = joint[:, :frame_count], joint[:, frame_count:]
        for layer in self.temporal:
            x = layer(x)
        compact = self.norm(x)
        x = compact.new_zeros(b, FRAMES, self.c.d)
        x[:, frame_valid] = compact

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
                    tail_block=self.tail_block(hold), tail_cell=self.tail_cell(hold),
                    tail_ms=self.tail_ms(hold))

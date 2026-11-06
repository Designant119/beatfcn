"""Matched temporal-access experiment, inspired by LFM3 operator allocation.

This is an offline music canvas, not a port of the causal text model. Both
variants have the same five short-convolution and three attention blocks.
Only two attention masks differ; all trainable tensors can be shared exactly.
The common upstream audio/address encoders already have global access. The
experiment concerns *additional canvas mixing*, not the whole receptive field.
"""
import torch
from torch import nn

from .full_event_model import Attention, FeedForward, RMSNorm
from .mel_only_canvas import MelOnlyCanvas


ACCESS_SCHEDULES = {
    'global': 'CCGCGCGC',
    'mixed': 'CCWCGCWC',
}


class AccessBlock(nn.Module):
    def __init__(self, config, operator, *, radius=16):
        super().__init__()
        if operator not in 'CWG' or len(operator) != 1 or radius < 0:
            raise ValueError('Invalid temporal operator or radius')
        self.operator = operator
        self.radius = radius
        self.pre_mix = RMSNorm(config.d)
        self.post_mix = RMSNorm(config.d)
        self.pre_ffn = RMSNorm(config.d)
        self.post_ffn = RMSNorm(config.d)
        if operator == 'C':
            self.in_projection = nn.Linear(config.d, 3 * config.d, bias=False)
            self.depthwise = nn.Conv1d(config.d, config.d, 7, padding=3,
                                      groups=config.d, bias=False)
            self.out_projection = nn.Linear(config.d, config.d, bias=False)
        else:
            self.attention = Attention(config)
        self.ffn = FeedForward(config)

    def attention_mask(self, valid):
        if self.operator == 'C':
            raise ValueError('Convolution has no attention mask')
        positions = torch.arange(valid.shape[1], device=valid.device)
        if self.operator == 'W':
            allowed = (positions[:, None] - positions[None]).abs() <= self.radius
        else:
            allowed = torch.ones(valid.shape[1], valid.shape[1],
                                 device=valid.device, dtype=torch.bool)
        return (allowed[None, None] & valid[:, None, :, None]
                & valid[:, None, None, :])

    def forward(self, x, valid):
        # Multiplying NaN padding by zero is insufficient: remove it first.
        x = torch.where(valid[..., None], x, 0.)
        z = self.pre_mix(x)
        if self.operator == 'C':
            left, right, value = self.in_projection(z).chunk(3, -1)
            z = self.depthwise((left * value).transpose(1, 2)).transpose(1, 2)
            z = self.out_projection(right * z)
        else:
            positions = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
            times = positions.float() * .04 + .02
            z = self.attention(z, times, positions,
                               self.attention.project_kv(z, times, positions),
                               self.attention_mask(valid))
        x = x + self.post_mix(z)
        x = x + self.post_ffn(self.ffn(self.pre_ffn(x)))
        return torch.where(valid[..., None], x, 0.)


class AccessCanvas(MelOnlyCanvas):
    def __init__(self, config, *, density_reference, access='global', radius=16):
        if access not in ACCESS_SCHEDULES:
            raise ValueError('Unknown access schedule')
        super().__init__(config, density_reference=density_reference)
        self.access = access
        self.temporal = nn.ModuleList(
            AccessBlock(config, operator, radius=radius)
            for operator in ACCESS_SCHEDULES[access])

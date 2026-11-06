# Copyright (c) 2024, Albert Gu and Tri Dao.
# Adapted from state-spaces/mamba v2.2.6.post3 under Apache-2.0.
# Full upstream sources and license: inputs/fullstack_mamba_v1/upstream/.
"""Explicit Mamba-2/SSD reference backend, without CUDA kernels or fallbacks.

The equations follow Mamba2Simple: input-dependent dt/B/C, scalar per-head A,
causal depthwise convolution, D skip, and gate-before-RMSNorm. This backend is
for bounded CPU/MPS architecture experiments, not fused-kernel speed claims.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def segment_sums(x):
    length = x.shape[-1]
    expanded = x[..., :, None].expand(*x.shape, length)
    lower = torch.arange(length, device=x.device)[:, None] > torch.arange(length, device=x.device)[None]
    sums = expanded.masked_fill(~lower, 0).cumsum(-2)
    inclusive = torch.arange(length, device=x.device)[:, None] >= torch.arange(length, device=x.device)[None]
    return sums.masked_fill(~inclusive, -torch.inf)


def ssd_scan(x, a, b, c, chunk_size, initial_state=None):
    """Exact discrete SSD; internal tail padding has identity state transitions.

    x [B,L,H,P] already includes dt, a [B,L,H] is dt*A; b/c [B,L,H,N].
    The matrix products are the factored form of the upstream einsums.
    """
    batch, length, heads, width = x.shape
    pad = (-length) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
        c = F.pad(c, (0, 0, 0, 0, 0, pad))
    chunks = x.shape[1] // chunk_size
    # All block tensors use B,chunk,head,time,feature for batched matmul.
    xx = x.reshape(batch, chunks, chunk_size, heads, width).permute(0, 1, 3, 2, 4)
    bb = b.reshape(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)
    cc = c.reshape(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)
    aa = a.reshape(batch, chunks, chunk_size, heads).permute(0, 1, 3, 2)
    cumulative = aa.cumsum(-1)
    decay = segment_sums(aa).exp()
    within = ((cc @ bb.transpose(-2, -1)) * decay) @ xx
    decay_to_end = (cumulative[..., -1:] - cumulative).exp()
    block_states = xx.transpose(-2, -1) @ (bb * decay_to_end[..., None])
    if initial_state is None:
        initial_state = torch.zeros_like(block_states[:, 0])
    states = torch.cat((initial_state[:, None], block_states), 1)
    chunk_decay = segment_sums(F.pad(cumulative[..., -1].transpose(1, 2), (1, 0))).exp()
    all_states = torch.einsum('bhzc,bchpn->bzhpn', chunk_decay, states)
    boundary, final = all_states[:, :-1], all_states[:, -1]
    outside = (cc @ boundary.transpose(-2, -1)) * cumulative.exp()[..., None]
    output = (within + outside).permute(0, 1, 3, 2, 4).reshape(batch, -1, heads, width)
    return output[:, :length], final


class GatedRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        gated = x * F.silu(z)
        return gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + 1e-5) * self.weight


class Mamba2Reference(nn.Module):
    """Mamba2Simple equations, ngroups=1, with explicit cached prefix state."""
    def __init__(self, dim, *, d_state=128, headdim=64, chunk_size=64):
        super().__init__()
        if (2 * dim) % headdim or min(dim, d_state, headdim, chunk_size) < 1:
            raise ValueError('Invalid Mamba-2 dimensions')
        self.dim, self.inner = dim, 2 * dim
        self.d_state, self.headdim = d_state, headdim
        self.heads, self.chunk_size = self.inner // headdim, chunk_size
        conv_dim = self.inner + 2 * d_state
        self.in_proj = nn.Linear(dim, 2 * self.inner + 2 * d_state + self.heads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, 4, padding=3, groups=conv_dim, bias=True)
        dt = torch.exp(torch.rand(self.heads) * math.log(.1 / .001) + math.log(.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = GatedRMSNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def _parts(self, u):
        return self.in_proj(u).split((self.inner, self.inner + 2 * self.d_state, self.heads), -1)

    def _ssm_parts(self, conv):
        x, b, c = conv.split((self.inner, self.d_state, self.d_state), -1)
        x = x.reshape(*x.shape[:-1], self.heads, self.headdim)
        b = b[..., None, :].expand(*b.shape[:-1], self.heads, self.d_state)
        c = c[..., None, :].expand(*c.shape[:-1], self.heads, self.d_state)
        return x, b, c

    def forward(self, u, *, return_cache=False):
        if u.ndim != 3 or not u.shape[1]:
            raise ValueError('A nonempty independent sequence is required')
        z, raw, dt = self._parts(u)
        conv = F.silu(self.conv1d(raw.transpose(1, 2)).transpose(1, 2)[:, :u.shape[1]])
        x, b, c = self._ssm_parts(conv)
        dt = F.softplus(dt + self.dt_bias)
        y, state = ssd_scan(x * dt[..., None], dt * -self.A_log.exp(), b, c, self.chunk_size)
        y = y + x * self.D[None, None, :, None]
        out = self.out_proj(self.norm(y.flatten(-2), z))
        if return_cache:
            # Internal SSD padding never advances this real convolution history.
            history = F.pad(raw.transpose(1, 2), (max(0, 3 - raw.shape[1]), 0))[..., -3:]
            return out, dict(conv=history, state=state)
        return out

    def step(self, u, cache):
        if u.ndim != 3 or u.shape[1] != 1:
            raise ValueError('Cached decoding takes exactly one real token')
        z, raw, dt = self._parts(u)
        window = torch.cat((cache['conv'], raw.transpose(1, 2)), -1)
        conv = F.silu((window * self.conv1d.weight[:, 0][None]).sum(-1) + self.conv1d.bias)
        x, b, c = self._ssm_parts(conv[:, None])
        x, b, c = x[:, 0], b[:, 0], c[:, 0]
        dt = F.softplus(dt[:, 0] + self.dt_bias)
        state = (cache['state'] * (dt * -self.A_log.exp()).exp()[..., None, None]
                 + (x * dt[..., None])[..., None] * b[..., None, :])
        y = (state * c[..., None, :]).sum(-1) + x * self.D[None, :, None]
        out = self.out_proj(self.norm(y.flatten(-2)[:, None], z))
        return out, dict(conv=window[..., -3:], state=state)

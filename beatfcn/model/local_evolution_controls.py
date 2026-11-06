"""Isolated single-intervention successors; historical model defaults stay intact."""
import torch
from torch.nn import functional as F

from .grouped_chart_process import GroupedChartProcess
from .mamba2_reference import Mamba2Reference, segment_sums


class AudioInteractionGroupedProcess(GroupedChartProcess):
    """Keep the existing projections, add their candidate-specific interaction."""

    def context(self, inputs, request):
        context = super().context(inputs, request)
        audio = self.address_key(context['x'][context['frame']])
        phase = context['keys'] - audio
        context['keys'] = context['keys'] + audio * phase
        return context


def reset_boundary_scan(x, a, b, c, chunk_size):
    """Same within-chunk SSD equation, with zero state at each chunk boundary."""
    batch, length, heads, width = x.shape
    pad = (-length) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
        c = F.pad(c, (0, 0, 0, 0, 0, pad))
    chunks = x.shape[1] // chunk_size
    xx = x.reshape(batch, chunks, chunk_size, heads, width).permute(0, 1, 3, 2, 4)
    bb = b.reshape(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)
    cc = c.reshape(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)
    aa = a.reshape(batch, chunks, chunk_size, heads).permute(0, 1, 3, 2)
    within = ((cc @ bb.transpose(-2, -1)) * segment_sums(aa).exp()) @ xx
    return within.permute(0, 1, 3, 2, 4).reshape(batch, -1, heads, width)[:, :length]


class BoundaryResetMamba2(Mamba2Reference):
    """State-reset control; convolution/gates/parameters/position inputs unchanged."""

    def forward(self, u, *, return_cache=False):
        if return_cache:
            raise ValueError('This bidirectional experimental control has no cache API')
        if u.ndim != 3 or not u.shape[1]:
            raise ValueError('A nonempty independent sequence is required')
        z, raw, dt = self._parts(u)
        conv = F.silu(self.conv1d(raw.transpose(1, 2)).transpose(1, 2)[:, :u.shape[1]])
        x, b, c = self._ssm_parts(conv)
        dt = F.softplus(dt + self.dt_bias)
        y = reset_boundary_scan(x * dt[..., None], dt * -self.A_log.exp(), b, c, self.chunk_size)
        y = y + x * self.D[None, None, :, None]
        return self.out_proj(self.norm(y.flatten(-2), z))


def reset_mamba_boundaries(model):
    """Replace only each SSD core, preserving every parameter and the RNG state."""
    rng = torch.get_rng_state()
    replacements = []
    for module in model.modules():
        if isinstance(getattr(module, 'mixer', None), Mamba2Reference):
            old = module.mixer
            new = BoundaryResetMamba2(old.dim, d_state=old.d_state,
                                     headdim=old.headdim, chunk_size=old.chunk_size)
            new.load_state_dict(old.state_dict())
            replacements.append((module, new.to(old.in_proj.weight.device)))
    for module, new in replacements:
        module.mixer = new
    torch.set_rng_state(rng)
    return len(replacements)

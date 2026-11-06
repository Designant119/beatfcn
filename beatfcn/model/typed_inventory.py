"""Exact typed inventory law; the capacity constraint is part of its support."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .local_bundle import read_grid
from .native_cell_chain import NativeCellChain


def safe_logsumexp(values, dim):
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, values, torch.zeros_like(values))
    result = safe.logsumexp(dim)
    return result.masked_fill(~reachable.squeeze(dim), -torch.inf)


def masked_scores(scores, allowed):
    count = torch.arange(scores.shape[-1], device=scores.device)
    permitted = allowed[..., None] | (count == 0)
    return scores.masked_fill(~permitted, -torch.inf)


def count_totals(scores):
    """Log mass by total, including zero, for any number of types."""
    size = scores.shape[-1]
    total = torch.arange(size, device=scores.device)[:, None]
    number = torch.arange(size, device=scores.device)[None, :]
    indices = (total - number).clamp_min(0)
    triangle = number <= total
    state = scores.new_full((*scores.shape[:-2], size), -torch.inf)
    state[..., 0] = 0
    for kind in range(scores.shape[-2]):
        terms = state[..., indices] + scores[..., kind, None, :]
        state = safe_logsumexp(terms.masked_fill(~triangle, -torch.inf), -1)
    return state


def count_logz(scores, allowed):
    return safe_logsumexp(count_totals(masked_scores(scores, allowed)), -1)


def count_nll(scores, overflow, allowed, counts):
    cap = scores.shape[-1] - 1
    if counts.shape != scores.shape[:-1]:
        raise ValueError('Typed count shape')
    if bool(((counts < 0) | (counts > cap)).any()) or bool((counts.sum(-1) > cap).any()):
        raise ValueError('Target outside typed inventory capacity')
    if bool(((counts > 0) & ~allowed).any()):
        raise ValueError('Positive target has no allowed typed address')
    potential = scores.gather(-1, counts[..., None]).squeeze(-1).sum(-1)
    editable = allowed.any(-1)
    overflow_loss = torch.where(editable, F.softplus(overflow), torch.zeros_like(overflow))
    return count_logz(scores, allowed) - potential + overflow_loss


def count_log_marginals(scores, overflow, allowed):
    """Each type has native classes plus the shared explicit failure atom."""
    scores = masked_scores(scores, allowed)
    z = safe_logsumexp(count_totals(scores), -1)
    size = scores.shape[-1]
    n = torch.arange(size, device=scores.device)[:, None]
    t = torch.arange(size, device=scores.device)[None, :]
    legal_total = n + t < size
    editable = allowed.any(-1)
    nonoverflow = torch.where(editable, F.logsigmoid(-overflow), torch.zeros_like(overflow))
    failure = F.logsigmoid(overflow).masked_fill(~editable, -torch.inf)
    result = []
    for k in range(4):
        other = count_totals(torch.cat((scores[..., :k, :], scores[..., k+1:, :]), -2))
        mass = safe_logsumexp(other[..., None, :].expand(*other.shape[:-1], size, size)
                             .masked_fill(~legal_total, -torch.inf), -1)
        p = scores[..., k, :] + mass - z[..., None] + nonoverflow[..., None]
        result.append(torch.cat((p, failure[..., None]), -1))
    return torch.stack(result, -2)


def _constrained_mode(scores):
    """Rare capacity-binding path, deterministic lexicographic ties."""
    cap = scores.shape[-1] - 1
    dp = [(0., ())] + [(-math.inf, ())] * cap
    for row in scores:
        after = []
        for t in range(cap + 1):
            candidates = [(dp[t-n][0]+float(row[n]), dp[t-n][1]+(n,)) for n in range(t+1)
                          if math.isfinite(dp[t-n][0]) and math.isfinite(float(row[n]))]
            after.append(min(candidates, key=lambda v: (-v[0], v[1])) if candidates else (-math.inf, ()))
        dp = after
    return min(dp, key=lambda v: (-v[0], v[1]))


@torch.no_grad()
def count_modes(scores, overflow, allowed):
    if scores.shape[0] != 1:
        raise ValueError('Request-level inventory decode requires batch one')
    values = masked_scores(scores, allowed)
    z = count_logz(scores, allowed)[0].cpu().double().numpy()
    array = values[0].cpu().double().numpy()
    omega = overflow[0].cpu().double().numpy()
    editable = allowed[0].any(-1).cpu().numpy()
    counts = array.argmax(-1).astype(np.int64)
    failures = []
    for c, row in enumerate(array):
        if counts[c].sum() <= row.shape[-1]-1:
            best = float(sum(row[k, counts[c, k]] for k in range(4)))
        else:
            best, mode = _constrained_mode(row)
            counts[c] = mode
        # log(sigmoid(o))-log(sigmoid(-o)) == o, without saturation.
        if editable[c] and omega[c] > best-z[c]:
            failures.append(c)
    return counts, failures


class TypedInventory(NativeCellChain):
    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        del self.cell_count, self.kind_read
        self.typed_count = nn.Linear(config.d, 261)
        self.type_embedding = nn.Embedding(4, 32)
        self.member_initial = nn.Linear(config.d+128+32, 192)
        self.member_query = nn.Sequential(nn.Linear(192+config.d+128+32, config.d), nn.SiLU())
        with torch.no_grad():
            self.typed_count.weight[-1].zero_()
            self.typed_count.bias[-1] = -8

    def forward(self, inputs):
        # Same acoustic computation as the frozen native chain, new count law.
        memory = self.encode(inputs)
        times = inputs['joint_frame_times']
        four = times[..., None] + times.new_tensor([.005, .015, .025, .035])
        mel = read_grid(memory['audio'], memory['audio_times'], memory['audio_valid'], four, .01)
        x = self.frame_read(mel.flatten(-2))
        clock = torch.stack((times/8, (times/8).square(), torch.sin(times*math.pi), torch.cos(times*math.pi)), -1)
        x = x + self.clock_read(clock) + self.mode_ids(inputs['mode'])[:, None]
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        ref = inputs['reference'].masked_fill(~visible[..., None], 0)
        style = ref.sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        scale, bias = self.direct(inputs['controls'].flatten(1)).chunk(2, -1)
        x = x*(1+.1*scale[:, None].tanh()) + bias[:, None] + self.style(style)[:, None]
        context = memory['values']
        frames = x.shape[1]
        for layer in self.address:
            joined = layer(torch.cat((x, context), 1))
            x, context = joined[:, :frames], joined[:, frames:]
        for layer in self.temporal:
            x = layer(x)
        x = self.norm(x)
        fine = int(inputs['local_valid'][0].sum())*self.c.band_slots + int(inputs['mel_valid'][0].sum())
        songs = int(inputs['song_valid'][0].sum())
        if not songs:
            raise ValueError('No ordered whole-song states')
        song_values = memory['values'][:, fine:fine+songs]
        song_times = inputs['song_times'][:, inputs['song_valid'][0]]
        song_valid = torch.ones_like(song_times, dtype=torch.bool)
        direct_song = read_grid(song_values, song_times, song_valid, times+.02, .5)
        contextual = {k:memory[k][:, fine:] for k in ('values', 'times', 'valid', 'timed')}
        x = self.context_read(x+direct_song, times+.02, contextual)
        values = self.typed_count(x)
        return dict(typed_scores=values[..., :260].reshape(*values.shape[:-1], 4, 65),
                    overflow=values[..., 260], x=x, song=direct_song, reference=x,
                    song_values=song_values, song_times=song_times, song_valid=song_valid)

    def initial_typed_members(self, output, cells, counts, types):
        embedding = torch.cat((self.count_embedding(counts).flatten(-2), self.type_embedding(types)), -1)
        state = self.member_initial(torch.cat((output['x'][0, cells], embedding), -1)).tanh()
        return state, embedding

"""C16: normalized total count and conditional typed multiset composition."""
import math

import numpy as np
import torch
from torch import nn

from .local_bundle import read_grid
from .typed_inventory import TypedInventory, count_totals, masked_scores, safe_logsumexp


def total_log_probs(total_scores, allowed):
    if total_scores.shape[-1] < 3 or total_scores.shape[:-1] != allowed.shape[:-1]:
        raise ValueError('Total-count shape')
    classes = torch.arange(total_scores.shape[-1], device=total_scores.device)
    legal = allowed.any(-1)[..., None] | (classes == 0)
    return total_scores.masked_fill(~legal, -torch.inf).log_softmax(-1)


def count_nll(scores, total_scores, allowed, counts):
    cap = scores.shape[-1] - 1
    if total_scores.shape != (*scores.shape[:-2], cap + 2) or counts.shape != scores.shape[:-1]:
        raise ValueError('Shared total count shape')
    if bool(((counts < 0) | (counts > cap)).any()) or bool((counts.sum(-1) > cap).any()):
        raise ValueError('Target outside shared-total capacity')
    if bool(((counts > 0) & ~allowed).any()):
        raise ValueError('Positive target has no legal type')
    totals = counts.sum(-1)
    z = count_totals(masked_scores(scores, allowed))
    log_q = total_log_probs(total_scores, allowed)
    potential = scores.gather(-1, counts[..., None]).squeeze(-1).sum(-1)
    return -log_q.gather(-1, totals[..., None]).squeeze(-1) + z.gather(-1, totals[..., None]).squeeze(-1) - potential


def count_log_marginals(scores, total_scores, allowed):
    """Unconditional per-type native counts plus the explicit overflow atom."""
    scores = masked_scores(scores, allowed)
    size = scores.shape[-1]
    q = total_log_probs(total_scores, allowed)
    z = count_totals(scores)
    n = torch.arange(size, device=scores.device)[:, None]
    t = torch.arange(size, device=scores.device)[None, :]
    total = n + t
    legal = total < size
    index = total.clamp_max(size - 1)
    reachable = torch.isfinite(z)
    by_total = (q[..., :size] - torch.where(reachable, z, torch.zeros_like(z))).masked_fill(~reachable, -torch.inf)
    results = []
    for k in range(4):
        other = count_totals(torch.cat((scores[..., :k, :], scores[..., k+1:, :]), -2))
        terms = other[..., None, :] + by_total[..., index]
        mass = safe_logsumexp(terms.masked_fill(~legal, -torch.inf), -1)
        native = scores[..., k, :] + mass
        results.append(torch.cat((native, q[..., -1:]), -1))
    return torch.stack(results, -2)


def composition_mode(scores, total):
    """Exact conditional MAP; deterministic lexicographic tie rule."""
    dp = [(0., ())] + [(-math.inf, ())] * total
    for row in scores:
        after = []
        for value in range(total + 1):
            choices = [(dp[value-n][0] + float(row[n]), dp[value-n][1] + (n,))
                       for n in range(value+1)
                       if math.isfinite(dp[value-n][0]) and math.isfinite(float(row[n]))]
            after.append(min(choices, key=lambda v: (-v[0], v[1])) if choices else (-math.inf, ()))
        dp = after
    if not math.isfinite(dp[total][0]):
        raise ValueError('No legal type composition for chosen total')
    return dp[total][1]


@torch.no_grad()
def count_modes(scores, total_scores, allowed):
    if scores.shape[0] != 1:
        raise ValueError('Request-level decode requires batch one')
    totals = total_log_probs(total_scores, allowed)[0].argmax(-1).cpu().numpy()
    values = masked_scores(scores, allowed)[0].cpu().double().numpy()
    cap = scores.shape[-1] - 1
    counts = np.zeros((len(values), 4), np.int64)
    failures = []
    for c, total in enumerate(totals):
        if total > cap:
            failures.append(c)
        elif total:
            counts[c] = composition_mode(values[c], int(total))
    return counts, failures


class SharedTotalInventory(TypedInventory):
    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        # Replace the old scalar overflow and typed count head as one count-family change.
        self.typed_count = nn.Linear(config.d, 260)
        self.total_count = nn.Linear(config.d, 66)
        with torch.no_grad():
            self.total_count.weight[-1].zero_()
            self.total_count.bias[-1] = -8
        self.query_coverage = None

    def _record_query_range(self, role, values, inputs, output, origin_us):
        if self.query_coverage is None:
            return
        times = (np.asarray(values).reshape(-1) - origin_us) / 1e6
        sources = [('mel', inputs['mel_times'], inputs['mel_valid'], .005)]
        if role == 'endpoint':
            sources.append(('song', output['song_times'], output['song_valid'], .25))
        for name, clock, valid, padding in sources:
            actual = clock[valid].detach().cpu().numpy()
            low, high = float(actual.min())-padding, float(actual.max())+padding
            key = role+'_'+name
            row = self.query_coverage.setdefault(key, dict(queries=0, in_range=0,
                query_min_s=math.inf, query_max_s=-math.inf, source_min_s=low, source_max_s=high))
            row['queries'] += len(times)
            row['in_range'] += int(((times >= low) & (times <= high)).sum())
            row['query_min_s'] = min(row['query_min_s'], float(times.min()))
            row['query_max_s'] = max(row['query_max_s'], float(times.max()))

    def address_features(self, output, inputs, heads_us, cells, origin_us):
        self._record_query_range('head_candidates_including_masked', heads_us, inputs, output, origin_us)
        return super().address_features(output, inputs, heads_us, cells, origin_us)

    def endpoint_keys(self, output, inputs, absolute_us, origin_us):
        self._record_query_range('endpoint', absolute_us, inputs, output, origin_us)
        return super().endpoint_keys(output, inputs, absolute_us, origin_us)

    def forward(self, inputs):
        # Frozen C15 acoustic computation; only count-family output changes.
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
        return dict(typed_scores=self.typed_count(x).reshape(*x.shape[:-1], 4, 65),
                    total_scores=self.total_count(x), x=x, song=direct_song, reference=x,
                    song_values=song_values, song_times=song_times, song_valid=song_valid)

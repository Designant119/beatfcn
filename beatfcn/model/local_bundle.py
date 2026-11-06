"""Opt-in local combination experiment; production defaults remain separate.

Candidate-local audio scoring, direct condition/style conditioning, gesture
readout, and a learned onset-group cardinality share the production decoder.
Targets never enter encode/decode. Cardinality is supervised separately and
used only at the first member decision of each generated onset group.
"""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .full_event_codec import BOS, END, NEW, FIELDS, EventState
from .full_event_model import FullEventModel


TIME_FIELDS = ('head_block', 'head_cell', 'head_ms', 'tail_block', 'tail_cell', 'tail_ms')


def read_grid(values, times, valid, query, spacing):
    """Linear reads on the actual feature clock; out-of-coverage reads are zero.

    spacing only finds the two neighboring nominal frame indices. Interpolation
    uses their recorded times, including a partial final mel frame.
    """
    b, n, d = values.shape
    shape = query.shape
    flat = query.reshape(b, -1)
    position = (flat - times[:, :1]) / spacing
    lo = position.floor().long().clamp(0, n - 1)
    hi = (lo + 1).clamp(max=n - 1)
    gather = lambda x, i: x.gather(1, i[..., None].expand(-1, -1, d))
    left, right = gather(values, lo), gather(values, hi)
    t0, t1 = times.gather(1, lo), times.gather(1, hi)
    fraction = ((flat - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0, 1)
    vl, vr = valid.gather(1, lo), valid.gather(1, hi)
    # A padded right neighbor cannot dilute a valid final frame.
    fraction = fraction * vr
    result = left + fraction[..., None] * (right - left)
    last = times.masked_fill(~valid, -1e9).amax(1, keepdim=True)
    covered = vl & (flat >= times[:, :1] - spacing / 2) & (flat <= last + spacing / 2)
    return (result * covered[..., None]).reshape(*shape, d)


class LocalBundleModel(FullEventModel):
    def __init__(self, config, vocab=None):
        super().__init__(config, vocab)
        d, small = config.d, 32
        self.audio_read = nn.Linear(d, small)
        self.song_read = nn.Linear(d, small)
        self.context_read = nn.Linear(d, small)
        self.condition_direct = nn.Sequential(nn.Linear(config.controls * config.control_dim, d),
                                              nn.SiLU(), nn.Linear(d, d))
        self.style_direct = nn.Linear(config.reference_dim, d, bias=False)
        self.time_fields = nn.Embedding(len(TIME_FIELDS), small)
        self.time_value = nn.Linear(1, small, bias=False)
        self.time_score = nn.Linear(small, 1, bias=False)
        self.action_read = nn.Sequential(nn.Linear(d + small, d), nn.SiLU())
        self.action_heads = nn.ModuleDict({name: nn.Linear(d, self.vocab.sizes[FIELDS.index(name)])
                                          for name in ('kind', 'group', 'member', 'mania')})
        self.group_size = nn.Linear(d, 1)
        self.last_group_log_count = None
        # Start as a bounded residual; auxiliary count starts at two members.
        nn.init.normal_(self.time_score.weight, std=.01)
        for head in self.action_heads.values():
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.group_size.weight)
        nn.init.constant_(self.group_size.bias, math.log(math.expm1(math.log(3))))

    def encode(self, inputs):
        memory = super().encode(inputs)
        n = inputs['local'].shape[2] * self.c.band_slots
        m, s = inputs['mel'].shape[1], inputs['song'].shape[1]
        memory.update(audio=self.audio_read(memory['values'][:, n:n + m]),
                      audio_times=inputs['mel_times'], audio_valid=inputs['mel_valid'],
                      song_audio=self.song_read(memory['values'][:, n + m:n + m + s]),
                      song_times=inputs['song_times'], song_valid=inputs['song_valid'],
                      origin_s=inputs['origin_s'])
        visible = inputs['reference_valid'] & (inputs['reference'][..., 10] > 0)
        style = (inputs['reference'] * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        memory['direct_condition'] = self.condition_direct(inputs['controls'].flatten(1)) + self.style_direct(style)
        memory['head_range'] = (int(inputs['origin_s'].min().item() // 2),
                                int(inputs['end_s'].max().item() // 2) + 1)
        memory['tail_blocks'] = min(self.vocab.blocks, int(inputs['duration_s'].max().item() // 2) + 1)
        return memory

    def audio_at(self, memory, query):
        local = read_grid(memory['audio'], memory['audio_times'], memory['audio_valid'], query, .01)
        song = read_grid(memory['song_audio'], memory['song_times'], memory['song_valid'], query, .5)
        return local + song

    def decode(self, tokens, fields, times, valid, memory, *, cache=None):
        # Same causal/current-shifted-input convention and layer implementation
        # as FullEventModel.decode; full and cached calls take this one path.
        b, n = tokens.shape
        if cache is not None and (self.training or n != 1 or b != cache['batch_size']):
            raise ValueError('Cached decoding requires eval, one step, fixed batch')
        offset = cache['position'] if cache is not None else 0
        positions = torch.arange(offset, offset + n, device=tokens.device)[None].expand(b, -1)
        x = self.token_embedding(tokens) + self.field_embedding(fields) + self.mode_ids(memory['mode'])[:, None]
        x = x + memory['direct_condition'][:, None]
        kv = cache['memory_kv'] if cache is not None else self.memory_kv(memory)
        for i, layer in enumerate(self.decoder):
            x = layer(x, times, positions, kv[i], memory['valid'], valid,
                      cache['layers'][i] if cache is not None else None)
        h = self.final_norm(x)
        logits = self.output(h)
        if cache is not None:
            cache['position'] += n
            all_tokens = torch.cat((cache.get('bundle_tokens', tokens[:, :0]), tokens), 1)
            cache['bundle_tokens'] = all_tokens
        else:
            all_tokens = tokens
        block_start = self.vocab.token('tail_block', 0)
        # Grammar tail_block -> tail_cell -> tail_ms makes the block the
        # immediately preceding shifted input at each tail_ms position.
        # Avoid cummax, which Torch 2.6 does not implement on native MPS.
        previous = torch.cat((torch.full_like(all_tokens[:, :1], BOS), all_tokens[:, :-1]), 1)
        block_values = (previous - block_start)[:, -n:]
        context = self.context_read(h)
        # Chunk by field to avoid a dense token x full-vocabulary x audio tensor.
        addition = torch.zeros_like(logits)
        for fi, name in enumerate(TIME_FIELDS):
            row, column = ((fields == FIELDS.index(name)) & valid).nonzero(as_tuple=True)
            if row.numel() == 0:
                continue
            real_rows = row.numel()
            # Bound compiled MPS shapes without changing valid queries. Padded
            # duplicates are discarded before scattering and carry no loss.
            if cache is None:
                pad = (-real_rows) % 16
                row = torch.cat((row, row[:1].expand(pad)))
                column = torch.cat((column, column[:1].expand(pad)))
            if name == 'head_block':
                lo, hi = memory['head_range']
                values = torch.arange(lo, hi, device=tokens.device, dtype=torch.float32)
            else:
                size = memory['tail_blocks'] if name.endswith('block') else (50 if name.endswith('cell') else 40)
                values = torch.arange(size, device=tokens.device, dtype=torch.float32)
            real_values = len(values)
            pad = (-real_values) % 32
            values = torch.cat((values, values[:1].expand(pad)))
            if name.endswith('block'):
                q = values[None] * 2 + 1 - memory['origin_s'][row, None]
                width = 2.
            else:
                base = times[row, column]
                if name == 'tail_cell':
                    base = (tokens[row, column] - block_start).float() * 2 - memory['origin_s'][row]
                elif name == 'tail_ms':
                    cell = tokens[row, column] - self.vocab.token('tail_cell', 0)
                    base = block_values[row, column].float() * 2 + cell.float() * .04 - memory['origin_s'][row]
                scale = .04 if name.endswith('cell') else .001
                q = base[:, None] + values[None] * scale + (scale / 2 if name.endswith('cell') else 0)
                width = .04 if name.endswith('cell') else .01
            selected = {k: memory[k].index_select(0, row) for k in
                        ('audio', 'audio_times', 'audio_valid', 'song_audio', 'song_times', 'song_valid')}
            audio = sum(self.audio_at(selected, q + shift * width) for shift in (-.25, 0., .25)) / 3
            relative = (q - times[row, column, None]) / 8
            z = context[row, column, None] + audio + self.time_fields.weight[fi]
            z = z + self.time_value(relative[..., None])
            scores = self.time_score(torch.tanh(z)).squeeze(-1)
            start = self.vocab.token(name, 0)
            addition[row[:real_rows, None], column[:real_rows, None], start + values[:real_values].long()] = scores[:real_rows, :real_values]
        logits = logits + addition
        action = self.action_read(torch.cat((h, self.audio_at(memory, times)), -1))
        for name, head in self.action_heads.items():
            start = self.vocab.token(name, 0)
            mask = (fields == FIELDS.index(name)) & valid
            logits[:, :, start:start + head.out_features] = (
                logits[:, :, start:start + head.out_features] + head(action) * mask[..., None])
        self.last_group_log_count = F.softplus(self.group_size(action).squeeze(-1))
        return logits


def bundle_loss(model, logits, sequence, *, action_weight=1.):
    """Equal field means retain complete event supervision and rare tail fields."""
    raw = F.cross_entropy(logits.float().masked_fill(~sequence['allowed'], -1e9).flatten(0, 1),
                          sequence['targets'].flatten(), reduction='none').view_as(sequence['targets'])
    loss, mass = raw.new_zeros(()), raw.new_zeros(())
    for i, name in enumerate(FIELDS):
        mask = (sequence['fields'] == i) & sequence['valid']
        count = mask.sum()
        weight = action_weight if name in ('kind', 'tail_block', 'tail_cell', 'tail_ms') else 1.
        loss = loss + weight * (raw * mask).sum() / count.clamp_min(1)
        mass = mass + weight * (count > 0)
    count_mask = sequence['count_valid'] & sequence['valid']
    size_loss = F.smooth_l1_loss(model.last_group_log_count, sequence['group_log_count'], reduction='none')
    count_loss = (size_loss * count_mask).sum() / count_mask.sum().clamp_min(1)
    return loss / mass.clamp_min(1) + .2 * count_loss


@torch.no_grad()
def generate_bundle(model, inputs, requests, max_tokens, *, deadline=None):
    if model.training or len(requests) != len(inputs['mode']):
        raise ValueError('Generation requires eval mode and a fixed independent batch')
    states = [EventState(r, model.vocab) for r in requests]
    desired = [None] * len(states)
    memory = model.encode(inputs); cache = model.start_cache(memory)
    device = inputs['local'].device
    resource_failure = [None] * len(states)
    deadline_hit = False
    for _ in range(max_tokens):
        if deadline is not None and time.monotonic() >= deadline:
            deadline_hit = True; break
        active = [not s.done and not failure for s, failure in zip(states, resource_failure)]
        if not any(active):
            break
        query = [s.query() if a else (0, 0.) for s, a in zip(states, active)]
        prev = [s.tokens[-1] if s.tokens else BOS for s in states]
        logits = model.decode(torch.tensor(prev, device=device)[:, None],
            torch.tensor([q[0] for q in query], device=device)[:, None],
            torch.tensor([q[1] for q in query], device=device, dtype=torch.float32)[:, None],
            torch.ones(len(states), 1, device=device, dtype=torch.bool), memory, cache=cache)[:, 0]
        counts = model.last_group_log_count[:, 0].cpu().tolist()
        legal = []
        for i, (state, enabled) in enumerate(zip(states, active)):
            choices = list(state.allowed_tokens()) if enabled else [BOS]
            if enabled and state.phase == 'member':
                if state.group_members == 0:
                    if not math.isfinite(counts[i]) or counts[i] > math.log1p(state.request.max_events):
                        resource_failure[i] = 'predicted_group_exceeds_event_resource'; active[i] = False
                        choices = [BOS]
                    else:
                        desired[i] = max(1, round(math.expm1(counts[i])))
                if active[i]:
                    value = NEW if state.group_members < desired[i] else END
                    token = model.vocab.token('member', value)
                    if token not in choices:
                        raise ValueError('Learned cardinality conflicts with grammar')
                    choices = [token]
            legal.append(choices)
        longest = max(map(len, legal))
        ids = torch.tensor([x + [x[0]] * (longest - len(x)) for x in legal], device=device)
        selected = ids.gather(1, logits.gather(1, ids).argmax(-1, keepdim=True)).squeeze(1).tolist()
        for state, token, enabled in zip(states, selected, active):
            if enabled:
                state.apply(token)
    from dataclasses import asdict
    return [dict(complete=s.done, events=[asdict(e) for e in sorted(s.events)], tokens=s.tokens,
                 deadline_reached=deadline_hit and not s.done,
                 token_limit_reached=not s.done and not deadline_hit and failure is None,
                 resource_failure=failure) for s, failure in zip(states, resource_failure)]

"""Batched cached free generation; one device/host decision transfer per step."""
import contextlib
from dataclasses import asdict
import os
import time
import warnings
import weakref

import numpy as np
import torch

from . import full_event_codec as _codec
from beatfcn.events.canonical import TYPES
from .full_event_codec import BOS, EventState, FastEventState

# Fast decoding (static KV cache, grouped-query attention without K/V repeat,
# O(1) grammar supports, batched per-row sampling) is the default; set
# BEATFCN_FAST_DECODE=0 or pass fast=False for the historical reference loop.
# On CUDA the fast decode step is replayed as a CUDA graph; each captured graph
# is checked once against the eager step and any capture error or mismatch
# falls back to eager (same buffers, same results). BEATFCN_DECODE_CUDA_GRAPH=0
# or cuda_graph=False disables it. Verify a new GPU image with
# scripts/bench_generate_speed.py --check.
FAST_DEFAULT = os.environ.get('BEATFCN_FAST_DECODE', '1') != '0'
CUDA_GRAPH_DEFAULT = os.environ.get('BEATFCN_DECODE_CUDA_GRAPH', '1') != '0'


def _autocast_context(device, autocast):
    """``autocast`` None keeps the historical fp32 decode; 'bf16' is an explicit opt-in (not bit-identical)."""
    if autocast is None:
        return contextlib.nullcontext()
    if autocast != 'bf16':
        raise ValueError(f'Unknown autocast {autocast!r}')
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)


def _decoding_rule(temperature, top_p):
    if temperature is None:
        if top_p is not None:
            raise ValueError('top_p requires a sampling temperature')
        return dict(rule='greedy_argmax_legal')
    if not temperature > 0 or (top_p is not None and not 0 < top_p <= 1):
        raise ValueError('Sampling requires temperature > 0 and 0 < top_p <= 1')
    return dict(rule='ancestral_legal', temperature=float(temperature),
                top_p=None if top_p is None else float(top_p))


# D3 onset shallow fusion: head-time decisions only (tails, kinds, lanes untouched).
ONSET_FUSION_FIELDS = ('head_block', 'head_cell', 'head_ms')
ONSET_FUSION_FLOOR = 1e-4


def _onset_fusion_rule(onset_fusion):
    """Validated ``onset_fusion`` options -> the rule recorded in ``decoding``."""
    if not isinstance(onset_fusion, dict) or 'weight' not in onset_fusion or set(onset_fusion) - {'weight', 'window_ms'}:
        raise ValueError("onset_fusion must be dict(weight=w[, window_ms=20])")
    weight, window = float(onset_fusion['weight']), onset_fusion.get('window_ms', 20)
    if not weight > 0 or type(window) is not int or window < 0:
        raise ValueError('onset_fusion needs weight > 0 and an integer window_ms >= 0')
    return dict(weight=weight, window_ms=window, floor=ONSET_FUSION_FLOOR, fields=list(ONSET_FUSION_FIELDS),
                term='weight * log(p_onset + floor) added to each legal head-time logit (after guidance, before '
                     'temperature); p_onset = max sigmoid(onset head channel 0) over valid frames within '
                     '+-window_ms of the candidate ms (head_ms, copy times) or of any ms of its legal span '
                     '(head_cell, head_block)')


def _onset_tables(memory, requests, window_ms):
    """Per row ``(lo_ms, table)``: ``table[i]`` = max onset probability over the row's valid
    onset-head frames within +-window_ms of absolute ms ``lo_ms + i`` (0 where none), for every
    ms a head can take in the request ([start, end)). Frame times are request-relative seconds."""
    if 'onset_logits' not in memory:
        raise ValueError('onset_fusion needs a model with an onset head (config.onset_head)')
    logits = memory['onset_logits'].float()
    if logits.ndim == 3:
        logits = logits[..., 0]
    probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    times = memory['onset_times'].double().cpu().numpy()
    valid = memory['onset_valid'].bool().cpu().numpy()
    tables = []
    for row, request in enumerate(requests):
        lo, hi = request.start_us // 1000, (request.end_us - 1) // 1000
        table = np.zeros(hi - lo + 1)
        keep = valid[row]
        # Frame time in ms from lo, rounded to 1 us (float32 seconds carry ~1e-4 ms noise).
        at = np.round((request.start_us / 1000 + times[row][keep] * 1000 - lo) * 1000) / 1000
        p = probs[row][keep]
        first = np.ceil(at - window_ms).astype(np.int64)
        last = np.floor(at + window_ms).astype(np.int64)
        for k in range(2 * window_ms + 2):
            index = first + k
            ok = (index <= last) & (index >= 0) & (index < len(table))
            if ok.any():
                np.maximum.at(table, index[ok], p[ok])
        tables.append((lo, table))
    return tables


def _onset_bias(state, table, weight):
    """``weight*log(p+floor)`` per legal value of a head-time phase (same order as
    ``allowed_values``), or None for any other phase."""
    phase = state.phase
    if state.done or phase not in ONSET_FUSION_FIELDS:
        return None
    origin, values_table = table
    values = np.asarray(state.allowed_values(), dtype=np.int64)
    if not len(values):
        return None
    lo, hi = state._heads_ms()

    def span_max(starts):  # contiguous adjacent spans covering [starts[0], hi]
        return np.maximum.reduceat(values_table[starts[0] - origin:hi - origin + 1], starts - starts[0])

    if phase == 'head_ms':
        p = values_table[state.head_block * 2000 + state.head_cell * 40 + values - origin]
    elif phase == 'head_cell':
        p = span_max(np.maximum(lo, state.head_block * 2000 + values * 40))
    else:
        p = np.empty(len(values))
        blocks = values < state.vocab.blocks
        if blocks.any():
            p[blocks] = span_max(np.maximum(lo, (values[blocks] + state.block_base) * 2000))
        if (~blocks).any():
            copies = np.asarray([state.request.copy_times[v - state.vocab.blocks] for v in values[~blocks]])
            p[~blocks] = values_table[np.clip((copies + 500) // 1000 - origin, 0, len(values_table) - 1)]
    return (weight * np.log(p + ONSET_FUSION_FLOOR)).astype(np.float32)


def _fusion_matrix(states, active, tables, weight, longest):
    """[B, longest] head-time bias (padding repeats the first legal entry, like the index
    padding), or None when no active row is choosing a head time."""
    bias = None
    for row, (state, enabled) in enumerate(zip(states, active)):
        if not enabled:
            continue
        values = _onset_bias(state, tables[row], weight)
        if values is None:
            continue
        if bias is None:
            bias = np.zeros((len(states), longest), np.float32)
        bias[row, :len(values)] = values
        bias[row, len(values):] = values[0]
    return bias


def _sample(scores, support, temperature, top_p, generator):
    """Ancestral draw over the legal support only. ``scores`` [B,L] are the
    gathered legal logits; ``support`` marks real (non-padding) entries."""
    if generator is not None and scores.device != generator.device:
        scores, support = scores.to(generator.device), support.to(generator.device)
    scores = (scores.float() / temperature).masked_fill(~support, float('-inf'))
    probs = scores.softmax(-1)
    if top_p is not None and top_p < 1:
        ordered, order = probs.sort(-1, descending=True)
        # Keep the smallest prefix whose mass reaches top_p (always >= 1 token).
        keep_sorted = (ordered.cumsum(-1) - ordered) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        probs = probs.masked_fill(~keep, 0.)
    return torch.multinomial(probs, 1, generator=generator)


def _row_generators(generator, rows):
    """None/one shared ``torch.Generator`` -> None; a sequence -> per-row list.

    Per-row generators make each row's draws a function of that row's own
    seed only (a row consumes its generator only while it is active), so a
    case's sample does not depend on which companions share its batch.
    """
    if generator is None or isinstance(generator, torch.Generator):
        return None
    generators = list(generator)
    if len(generators) != rows or not all(isinstance(g, torch.Generator) for g in generators):
        raise ValueError('generator list must hold one torch.Generator per request row')
    return generators


ENCODE_CHUNK_ROWS = 64


def _encode(model, inputs):
    """model.encode, in row chunks above ENCODE_CHUNK_ROWS (the encoder treats rows independently; one
    full-batch encode of ~300 rows needs ~9 GB of FFN activations). Batches of <= 64 rows call
    model.encode exactly as before (bitwise-identical historical runs)."""
    rows = int(inputs['mode'].shape[0])
    if rows <= ENCODE_CHUNK_ROWS:
        return model.encode(inputs)
    parts = []
    for lo in range(0, rows, ENCODE_CHUNK_ROWS):
        chunk = {k: (v[lo:lo + ENCODE_CHUNK_ROWS] if torch.is_tensor(v) and v.ndim and v.shape[0] == rows else v)
                 for k, v in inputs.items()}
        parts.append(model.encode(chunk))
    return {k: (torch.cat([part[k] for part in parts], 0) if torch.is_tensor(parts[0][k]) else parts[0][k])
            for k in parts[0]}


MEMORY_LENGTH_STEP = 256


def _pad_memory(memory, length):
    """Tail-pad encoded rows with masked slots: same attended rows, one geometry for many batches."""
    missing = length - memory['values'].shape[1]
    if missing <= 0:
        return memory
    padded = dict(memory)
    for key in ('values', 'times', 'valid', 'timed'):
        value = memory[key]
        padded[key] = torch.cat((value, value.new_zeros(value.shape[0], missing, *value.shape[2:])), 1)
    return padded


def _rounded_memory_length(length):
    return -(-length // MEMORY_LENGTH_STEP) * MEMORY_LENGTH_STEP


def _encode_shared(model, inputs, memory_index):
    """``_encode``; with ``memory_index`` the inputs hold unique rows and each request row picks one."""
    memory = _encode(model, inputs)
    if memory_index is None:
        return memory
    rows = len(inputs['mode'])
    return {k: (v[memory_index] if torch.is_tensor(v) and v.ndim and v.shape[0] == rows else v)
            for k, v in memory.items()}


def _pattern_rule(constraints):
    """Validated decode-time mania pattern constraints (``None`` = off).

    ``max_chord``: at most this many keys per onset group (hold heads count).
    ``min_same_lane_ms``: a key pressed less than this long ago (within the current core)
    cannot be pressed again (anti-jack). When a group would have no legal pattern left the
    unconstrained support is kept and counted as ``relaxed`` (never a grammar dead end)."""
    if not constraints:
        return None
    unknown = set(constraints) - {'max_chord', 'min_same_lane_ms'}
    if unknown:
        raise ValueError(f'Unknown pattern constraints {sorted(unknown)}')
    rule = {}
    if constraints.get('max_chord') is not None:
        if not (int(constraints['max_chord']) == constraints['max_chord'] and 1 <= constraints['max_chord'] <= 4):
            raise ValueError('max_chord must be an integer in [1, 4]')
        rule['max_chord'] = int(constraints['max_chord'])
    if constraints.get('min_same_lane_ms') is not None:
        if not float(constraints['min_same_lane_ms']) > 0:
            raise ValueError('min_same_lane_ms must be positive')
        rule['min_same_lane_ms'] = float(constraints['min_same_lane_ms'])
    return rule or None


def _pattern_filter(state, rule, counters):
    """Filtered legal 'mania' tokens under ``rule`` (same order as allowed_tokens), or None = unchanged."""
    if rule is None or state.done or state.phase != 'mania':
        return None
    values = state.allowed_values()
    recent = set()
    if 'min_same_lane_ms' in rule:
        limit = state.onset - int(round(rule['min_same_lane_ms'] * 1000))
        recent = {e.lane for e in state.events if e.onset_us > limit and e.onset_us < state.onset}
    cap = rule.get('max_chord', 4)
    keep = []
    for value in values:
        digits = [(value // 3 ** lane) % 3 for lane in range(4)]
        if sum(d > 0 for d in digits) > cap or any(d and lane in recent for lane, d in enumerate(digits)):
            continue
        keep.append(value)
    if len(keep) == len(values):
        return None
    if not keep:
        counters['relaxed'] = counters.get('relaxed', 0) + 1
        return None
    counters['filtered'] = counters.get('filtered', 0) + 1
    return tuple(state.vocab.token('mania', v) for v in keep)


def _sample_rows(scores, legal, active, temperature, top_p, generators):
    """Per-row ancestral draws; inactive rows take index 0 without RNG use.
    ``temperature`` may be a scalar or one value per row."""
    picks = []
    for row, support in enumerate(legal):
        if not active[row]:
            picks.append(scores.new_zeros((1, 1), dtype=torch.long)); continue
        row_scores = scores[row:row+1, :len(support)]
        mask = torch.ones_like(row_scores, dtype=torch.bool)
        t = temperature[row] if isinstance(temperature, (list, tuple)) else temperature
        picks.append(_sample(row_scores, mask, t, top_p, generators[row]).to(scores.device))
    return torch.cat(picks, 0)


@torch.no_grad()
def generate_batch(model, inputs, requests, max_tokens, *, deadline=None,
                   temperature=None, top_p=None, generator=None,
                   guidance=None, unconditional_inputs=None, field_temperature=None,
                   onset_fusion=None, fast=None, cuda_graph=None, stats=None, pattern_constraints=None,
                   memory_index=None, autocast=None, compile_decode=False, on_step=None):
    """Free generation over the legal grammar.

    temperature=None (default) is the historical greedy argmax over legal
    tokens, unchanged. temperature>0 draws ancestrally from the legal softmax
    (optionally nucleus top_p), reproducible with ``generator``: either one
    shared ``torch.Generator`` (draws then depend on batch composition) or a
    list with one generator per row (each row's sample depends only on its
    own generator, model inputs and request). A row that
    reaches a grammar dead end is marked ``failure='grammar_dead_end'`` and
    stopped; other rows continue.

    ``fast`` (default ``FAST_DEFAULT``) selects the fixed-shape decode loop
    (``_generate_fast``) for attention-mixer FullEventModels; ``fast=False``
    is the historical reference loop. Both make the same grammar decisions and
    the same RNG draws; logits agree to fp32 round-off (see
    tests/test_fast_generate.py). ``cuda_graph`` (CUDA only) replays the decode
    step as a captured graph. ``stats``: optional dict filled with loop timings.

    ``onset_fusion`` (default None = off): ``dict(weight=w[, window_ms=20])``
    D3 shallow fusion with the model's own onset head -- every legal
    head_block/head_cell/head_ms logit gets ``w*log(p_onset+1e-4)`` (see
    ``_onset_bias``); recorded as ``decoding['onset_fusion']``.

    ``pattern_constraints`` (default None = off): ``dict(max_chord=k, min_same_lane_ms=g)`` charter
    playability limits on mania lane patterns (see ``_pattern_rule``); recorded with filtered /
    relaxed group counts as ``decoding['pattern_constraints']``.

    ``memory_index`` (default None): ``inputs`` hold unique encoder rows and request row i uses
    row ``memory_index[i]`` (encoder dedupe across seeds of one core); on this path the memory is
    tail-padded to a stable length so CUDA graphs survive batch changes. ``autocast``: None (fp32,
    default) or 'bf16'. ``compile_decode``: torch.compile the eager decode step (no CUDA graphs).
    ``on_step``: called once per fast-loop step (progress heartbeat).
    """
    expected_rows = len(inputs['mode']) if memory_index is None else len(memory_index)
    if model.training or len(requests) != expected_rows or not requests or max_tokens < 1:
        raise ValueError('Generation requires eval mode and one request per input row')
    decoding = _decoding_options(temperature, top_p, field_temperature)
    if autocast is not None:
        decoding['autocast'] = autocast
    if guidance is not None:
        if unconditional_inputs is None or not guidance > 0:
            raise ValueError('Guidance needs unconditional_inputs and a positive weight')
        decoding['guidance'] = float(guidance)
    fusion = None
    if onset_fusion is not None:
        fusion = _onset_fusion_rule(onset_fusion)
        decoding['onset_fusion'] = dict(fusion)
    patterns = _pattern_rule(pattern_constraints)
    if patterns is not None:
        decoding['pattern_constraints'] = dict(patterns, counts={})
    row_generators = _row_generators(generator, len(requests))
    rng_mode = (None if temperature is None else 'per_row' if row_generators is not None
                else 'shared_batch_generator' if generator is not None else 'global_torch_rng')
    if fast is None:
        fast = FAST_DEFAULT
    options = dict(deadline=deadline, temperature=temperature, top_p=top_p, generator=generator,
                   guidance=guidance, unconditional_inputs=unconditional_inputs,
                   field_temperature=field_temperature, decoding=decoding,
                   row_generators=row_generators, rng_mode=rng_mode, fusion=fusion, patterns=patterns,
                   memory_index=memory_index)
    with _autocast_context(inputs['local'].device, autocast):
        if fast and fast_supported(model):
            if cuda_graph is None:
                cuda_graph = CUDA_GRAPH_DEFAULT
            return _generate_fast(model, inputs, requests, max_tokens, cuda_graph=bool(cuda_graph) and not compile_decode,
                                  stats=stats, compile_decode=compile_decode, on_step=on_step, **options)
        return _generate_reference(model, inputs, requests, max_tokens, **options)


def _decoding_options(temperature, top_p, field_temperature):
    decoding = _decoding_rule(temperature, top_p)
    if field_temperature:
        # Per-grammar-field temperature, e.g. {'kind': .3, 'mania': .5}: keep the
        # timing / continue decisions calibrated (T) while making categorical
        # choices more decisive. A decoding readout, recorded in ``decoding``.
        if temperature is None or not all(v > 0 for v in field_temperature.values()):
            raise ValueError('field_temperature needs sampling and positive values')
        decoding['field_temperature'] = {k: float(v) for k, v in sorted(field_temperature.items())}
    return decoding


def fast_supported(model):
    """Fast loop needs the static-cache API and an all-attention decoder."""
    config = getattr(model, 'c', None)
    return (hasattr(model, 'decode_static') and hasattr(model, 'start_static_cache')
            and getattr(config, 'history_mixer', None) == 'attention')


def _generate_reference(model, inputs, requests, max_tokens, *, deadline, temperature, top_p, generator,
                        guidance, unconditional_inputs, field_temperature, decoding, row_generators, rng_mode,
                        fusion=None, patterns=None, memory_index=None):
    """Historical per-token loop (the equivalence reference for the fast path)."""
    states = [EventState(request, model.vocab) for request in requests]
    failures = [None] * len(states)
    device = inputs['local'].device
    memory = _encode_shared(model, inputs, memory_index); cache = model.start_cache(memory)
    tables = _onset_tables(memory, requests, fusion['window_ms']) if fusion else None
    if guidance is not None:
        memory_u = _encode_shared(model, unconditional_inputs, memory_index); cache_u = model.start_cache(memory_u)
    deadline_reached = False
    for _ in range(max_tokens):
        if deadline is not None and time.monotonic() >= deadline:
            deadline_reached = True; break
        active = [not state.done and failure is None for state, failure in zip(states, failures)]
        if not any(active): break
        # Completed/failed rows stay in the batch, advancing an ignored private
        # cache. No new request is inserted into a row without a fresh cache.
        queries = [state.query() if enabled else (0, 0.) for state, enabled in zip(states, active)]
        previous = [state.tokens[-1] if state.tokens else BOS for state in states]
        logits = model.decode(torch.tensor(previous, device=device)[:, None],
            torch.tensor([q[0] for q in queries], device=device)[:, None],
            torch.tensor([q[1] for q in queries], device=device, dtype=torch.float32)[:, None],
            torch.ones(len(states), 1, device=device, dtype=torch.bool), memory, cache=cache)[:, 0]
        if guidance is not None:
            logits_u = model.decode(torch.tensor(previous, device=device)[:, None],
                torch.tensor([q[0] for q in queries], device=device)[:, None],
                torch.tensor([q[1] for q in queries], device=device, dtype=torch.float32)[:, None],
                torch.ones(len(states), 1, device=device, dtype=torch.bool), memory_u, cache=cache_u)[:, 0]
            logits = logits_u.float() + float(guidance) * (logits.float() - logits_u.float())
        counts = decoding['pattern_constraints']['counts'] if patterns is not None else None
        legal = [((_pattern_filter(state, patterns, counts) if patterns is not None else None) or state.allowed_tokens())
                 if enabled else (BOS,) for state, enabled in zip(states, active)]
        for row, support in enumerate(legal):
            if not len(support):
                failures[row], active[row], legal[row] = 'grammar_dead_end', False, (BOS,)
        longest = max(map(len, legal))
        indices = torch.tensor([list(s) + [s[0]] * (longest - len(s)) for s in legal], device=device)
        scores = logits.gather(1, indices)
        if fusion:
            scores = scores.float()
            bias = _fusion_matrix(states, active, tables, fusion['weight'], longest)
            if bias is not None:
                scores = scores + torch.from_numpy(bias).to(scores.device)
        if temperature is None:
            # Repeating the first legal token for support padding cannot create
            # an illegal winner; ties keep the first occurrence (scalar argmax).
            choice = scores.argmax(-1, keepdim=True)
        else:
            row_temperature = temperature
            if field_temperature:
                row_temperature = [float(field_temperature.get(state.phase, temperature)) if enabled else temperature
                                   for state, enabled in zip(states, active)]
            if row_generators is not None:
                choice = _sample_rows(scores, legal, active, row_temperature, top_p, row_generators)
            else:
                support = torch.tensor([[i < len(s) for i in range(longest)] for s in legal], device=device)
                if isinstance(row_temperature, list):
                    scaled = scores.float() / torch.tensor(row_temperature, device=scores.device)[:, None]
                    choice = _sample(scaled, support, 1.0, top_p, generator).to(device)
                else:
                    choice = _sample(scores, support, temperature, top_p, generator).to(device)
        selected = indices.gather(1, choice).squeeze(1).tolist()
        for state, token, enabled in zip(states, selected, active):
            if enabled: state.apply(token)
    return [dict(complete=state.done, events=[asdict(e) for e in sorted(state.events)], tokens=state.tokens,
                 deadline_reached=deadline_reached and not state.done and failure is None,
                 token_limit_reached=not state.done and not deadline_reached and failure is None,
                 failure=failure, decoding=dict(decoding), rng_mode=rng_mode)
            for state, failure in zip(states, failures)]


# --------------------------------------------------------------------------
# Fast path: fixed-shape decode step + O(1) grammar supports + batched draws.
# --------------------------------------------------------------------------

def _length_bucket(needed, capacity):
    """Attended K/V length rounded up to a power of two (>= 64): a few static shapes per run."""
    length = 64
    while length < needed:
        length *= 2
    return min(length, capacity)


def _compiled_decode(model):
    if not hasattr(model, '_compiled_decode_static'):
        model._compiled_decode_static = torch.compile(model.decode_static)
    return model._compiled_decode_static


class _EagerStep:
    """One static-cache decode step per call (CPU / MPS / CUDA without graphs).

    ``compiled``: torch.compile'd decode_static; lengths are then bucketed so it sees few shapes.
    The first step stays eager because it allocates the lazy K/V buffers. ``cache`` (seat tables)
    is a prebuilt per-row cache; ``step`` then takes one position per row."""

    def __init__(self, model, memory, capacity, *, compiled=None, cache=None):
        self.model, self.position, self.compiled = model, 0, compiled
        self.cache = cache if cache is not None else model.start_static_cache(memory, capacity)
        self.device = (memory['values'] if cache is None else self.cache['slots']).device
        self.pos = torch.zeros(1 if cache is None else self.cache['batch_size'], dtype=torch.long, device=self.device)

    def step(self, ints, times, positions=None):
        ints = torch.from_numpy(ints).to(self.device)
        times = torch.from_numpy(times).to(self.device)[:, None]
        if positions is None:
            self.pos.fill_(self.position)
            needed = self.position + 1
        else:
            self.pos.copy_(torch.from_numpy(positions).to(self.device))
            needed = int(positions.max()) + 1
        decode = self.model.decode_static
        if self.compiled is not None and self.cache['keys'] is not None:
            decode, needed = self.compiled, _length_bucket(needed, self.cache['capacity'])
        logits = decode(ints[0][:, None], ints[1][:, None], times, self.cache, self.pos, needed)
        self.position += 1
        return logits[:, 0]


# model -> {role: _GraphStep}; one geometry per role is kept (memory bounded).
_GRAPH_RUNNERS = weakref.WeakKeyDictionary()
_GRAPH_FAILURES = []  # capture/self-check failures in this process (eager fallback taken)


def _graph_key(model, memory, capacity):
    first = next(model.parameters())
    return (len(memory['mode']), int(capacity), str(memory['values'].device), tuple(memory['values'].shape),
            tuple(memory['valid'].shape), str(memory['values'].dtype), torch.is_autocast_enabled(),
            first.data_ptr(), model.output.weight.data_ptr())


class _GraphStep:
    """CUDA-graph replay of ``decode_static`` with one graph per K/V-length bucket.

    Static input buffers are refilled from pinned host staging each step; the
    attended length is rounded up to a power-of-two bucket (>= 64), so a call
    captures at most log2(capacity/64)+1 graphs and later calls with the same
    geometry reuse them after reloading memory K/V in place. With a prebuilt per-row ``cache`` (seat
    tables) ``step`` takes one position per row and the graphs read them from a static buffer.
    """

    def __init__(self, model, memory, capacity, *, cache=None):
        self.model, self.capacity = model, int(capacity)
        self.per_row = cache is not None
        self.cache = cache if self.per_row else model.start_static_cache(memory, capacity)
        self.device = (self.cache['slots'] if self.per_row else memory['values']).device
        b = self.cache['batch_size']
        self.ints = torch.zeros(2, b, dtype=torch.long, device=self.device)
        self.times = torch.zeros(b, 1, dtype=torch.float32, device=self.device)
        self.pos = torch.zeros(b if self.per_row else 1, dtype=torch.long, device=self.device)
        self.host_pos = torch.zeros(b, dtype=torch.long).pin_memory() if self.per_row else None
        self.host_ints = torch.zeros(2, b, dtype=torch.long).pin_memory()
        self.host_times = torch.zeros(b, 1, dtype=torch.float32).pin_memory()
        self.graphs, self.pool, self.position = {}, torch.cuda.graph_pool_handle(), 0
        self.captures = self.replays = 0
        self.failure = None

    def load(self, memory):
        """Reuse captured graphs for a new batch with identical geometry."""
        for (k, v), (k2, v2) in zip(self.cache['memory_kv'], self.model.memory_kv(memory)):
            k.copy_(k2); v.copy_(v2)
        self.cache['memory_mask'].copy_(memory['valid'][:, None, None, :])
        self.cache['mode_embedding'].copy_(self.model.mode_ids(memory['mode'])[:, None])
        if self.cache['keys'] is not None:
            self.cache['keys'].zero_(); self.cache['values'].zero_()
        self.position = 0

    def _run(self, length):
        return self.model.decode_static(self.ints[0][:, None], self.ints[1][:, None], self.times,
                                        self.cache, self.pos, length)

    def _bucket(self, needed):
        return _length_bucket(needed, self.capacity)

    def _capture(self, length):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # allocate lazy K/V buffers + library workspaces outside capture
                expected = self._run(length)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = self._run(length)
        graph.replay()
        # Self-check once per captured bucket: the replay must reproduce the
        # eager step on the same inputs, or this runner stays eager.
        if not torch.allclose(output, expected, rtol=1e-4, atol=1e-4):
            raise RuntimeError('CUDA-graph replay differs from the eager decode step')
        self.graphs[length] = (graph, output)
        self.captures += 1
        return output

    def step(self, ints, times, positions=None):
        # Staging reuse is safe: the previous step ended with a device->host sync.
        self.host_ints.numpy()[:] = ints
        self.host_times.numpy()[:, 0] = times
        self.ints.copy_(self.host_ints, non_blocking=True)
        self.times.copy_(self.host_times, non_blocking=True)
        if positions is None:
            self.pos.fill_(self.position)
            needed = self.position + 1
        else:
            self.host_pos.numpy()[:] = positions
            self.pos.copy_(self.host_pos, non_blocking=True)
            needed = int(positions.max()) + 1
        output = None
        if self.failure is None:
            length = self._bucket(needed)
            entry = self.graphs.get(length)
            if entry is not None:
                entry[0].replay()
                self.replays += 1
                output = entry[1]
            else:
                try:
                    output = self._capture(length)
                except Exception as error:  # capture unsupported here: stay eager, same buffers
                    self.failure = f'{type(error).__name__}: {error}'
                    _GRAPH_FAILURES.append(self.failure)
                    warnings.warn(f'CUDA-graph decode disabled, eager fallback: {self.failure}')
                    self.graphs.clear()
        if output is None:
            output = self._run(needed)
        self.position += 1
        return output[:, 0]


def _step_decoder(model, memory, capacity, use_graph, role, *, compile_decode=False, stable_memory=False):
    if stable_memory:
        # Grouped scheduler: batches differ in padded length, so grow one rounded length instead of re-capturing.
        current = _GRAPH_RUNNERS.get(model, {}).get(role) if use_graph else None
        floor = current.key[3][1] if current is not None else 0
        memory = _pad_memory(memory, max(floor, _rounded_memory_length(memory['values'].shape[1])))
    if not use_graph:
        return _EagerStep(model, memory, capacity, compiled=_compiled_decode(model) if compile_decode else None)
    runners = _GRAPH_RUNNERS.setdefault(model, {})
    key = _graph_key(model, memory, capacity)
    runner = runners.get(role)
    if runner is not None and runner.key == key and runner.failure is None:
        runner.load(memory)
        return runner
    runners.pop(role, None)
    runner = _GraphStep(model, memory, capacity)
    runner.key = key
    runners[role] = runner
    return runner


_BATCHED_SAMPLING = {}


def _check_multinomial_emulation(device):
    """Does ``torch.multinomial(p, 1, generator=g)`` equal argmax(p / Exp(1)) with
    the Exp(1) noise drawn from ``g`` over the same row, consuming the same RNG
    state? (True for CPU/MPS torch 2.6; verified once per device/torch build.)"""
    try:
        for trial in range(27):
            length = (1, 2, 3, 7, 40, 81, 257, 601, 1500)[trial % 9]
            probs = torch.rand(1, length, generator=torch.Generator().manual_seed(trial)) + 1e-3
            if length > 2:
                probs[0, ::3] = 0.
            probs = (probs / probs.sum()).to(device)
            a = torch.Generator(device=device).manual_seed(1000 + trial)
            b = torch.Generator(device=device).manual_seed(1000 + trial)
            reference = torch.multinomial(probs, 1, generator=a)
            noise = torch.ones(3, length + 5, device=device)
            noise[1, :length].exponential_(1, generator=b)
            emulated = (probs / noise[1:2, :length]).argmax(-1, keepdim=True)
            after_a = torch.rand(3, generator=a, device=device)
            after_b = torch.rand(3, generator=b, device=device)
            if not torch.equal(reference.cpu(), emulated.cpu()) or not torch.equal(after_a.cpu(), after_b.cpu()):
                return False
        return True
    except (RuntimeError, TypeError, ValueError):
        return False


def _batched_sampling_ok(generators):
    device = generators[0].device
    if any(g.device != device for g in generators):
        return False
    key = (str(device), torch.__version__)
    if key not in _BATCHED_SAMPLING:
        _BATCHED_SAMPLING[key] = _check_multinomial_emulation(device)
    return _BATCHED_SAMPLING[key]


def _draw_noise(counts, active, generators, longest):
    """The Exp(1) draw ``torch.multinomial(p, 1, generator=g)`` makes over one
    legal row of length ``counts[row]``, per active row, padded with ones.
    Drawn before the logits are needed (it depends only on the supports)."""
    noise = torch.ones(len(counts), longest, device=generators[0].device)
    for row, (generator, enabled) in enumerate(zip(generators, active)):
        if enabled:
            noise[row, :int(counts[row])].exponential_(1, generator=generator)
    return noise


def _sample_rows_batched(scores, counts, temperature, top_p, noise):
    """``_sample_rows`` as one batched softmax/top-p/argmax over the padded rows
    using the per-row multinomial noise. Returns [2,B]: choice, finite flag."""
    device = noise.device
    x = scores.to(device).float()
    rows, longest = x.shape
    if isinstance(temperature, (list, tuple)):
        scaled = None
        for value in sorted(set(temperature)):
            part = x / value  # the reference divides each row by its python float
            if scaled is None:
                scaled = part
            else:
                pick = torch.tensor([t == value for t in temperature], device=device)[:, None]
                scaled = torch.where(pick, part, scaled)
    else:
        scaled = x / temperature
    padding = torch.from_numpy(np.arange(longest)[None] >= counts[:, None]).to(device)
    probs = scaled.masked_fill(padding, float('-inf')).softmax(-1)
    if top_p is not None and top_p < 1:
        ordered, order = probs.sort(-1, descending=True)
        keep_sorted = (ordered.cumsum(-1) - ordered) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        probs = probs.masked_fill(~keep, 0.)
    choice = (probs / noise).argmax(-1)
    valid = torch.isfinite(probs).all(-1) & (probs >= 0).all(-1)
    return torch.stack((choice, valid.long()))


def _support_spec(tokens):
    if not tokens:
        return BOS, 0, ()
    return tokens[0], len(tokens), tuple(tokens)


def _generate_fast(model, inputs, requests, max_tokens, *, deadline, temperature, top_p, generator,
                   guidance, unconditional_inputs, field_temperature, decoding, row_generators, rng_mode,
                   cuda_graph=False, stats=None, fusion=None, patterns=None, memory_index=None, compile_decode=False,
                   on_step=None):
    """Same decisions as ``_generate_reference`` with fixed-shape device work.

    Per step: one host->device copy of (prev token, field, time), one decode
    step (eager or CUDA-graph replay), one host->device copy of the padded
    legal-token index matrix (built in numpy from O(1) supports while the
    device computes), one device->host copy of the chosen column per row.
    On CUDA the decode step is asynchronous: ``decode_s`` is the launch and
    ``select_s`` includes waiting for it.
    """
    clock = time.perf_counter
    timing = dict(decode_s=0., grammar_s=0., select_s=0., encoder_s=0., steps=0)
    started = clock()
    # A substituted EventState (tests, experiments) keeps its own supports.
    fast_grammar = EventState is _codec.EventState
    state_class = FastEventState if fast_grammar else EventState
    states = [state_class(request, model.vocab) for request in requests]
    rows = len(states)
    failures = [None] * rows
    device = inputs['local'].device
    # One capture/self-check failure in this process keeps every later call eager.
    use_graph = bool(cuda_graph) and device.type == 'cuda' and not _GRAPH_FAILURES
    stable = memory_index is not None
    memory = _encode_shared(model, inputs, memory_index)
    tables = _onset_tables(memory, requests, fusion['window_ms']) if fusion else None
    decoders = [_step_decoder(model, memory, max_tokens, use_graph, 'conditional', compile_decode=compile_decode,
                              stable_memory=stable)]
    del memory
    if guidance is not None:
        decoders.append(_step_decoder(model, _encode_shared(model, unconditional_inputs, memory_index), max_tokens,
                                      use_graph, 'unconditional', compile_decode=compile_decode, stable_memory=stable))
    if stats is not None:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timing['encoder_s'] = clock() - started
    batched = row_generators is not None and _batched_sampling_ok(row_generators)
    ints = np.zeros((2, rows), np.int64)
    times = np.zeros(rows, np.float32)
    firsts = np.zeros(rows, np.int64)
    counts = np.zeros(rows, np.int64)
    every = np.arange(rows)
    staging = (torch.empty(rows * model.vocab.size, dtype=torch.long).pin_memory()
               if device.type == 'cuda' else None)
    deadline_reached = False
    for _ in range(max_tokens):
        if deadline is not None and time.monotonic() >= deadline:
            deadline_reached = True; break
        active = [not state.done and failure is None for state, failure in zip(states, failures)]
        if not any(active): break
        tick = clock()
        for row, (state, enabled) in enumerate(zip(states, active)):
            field, t = state.query() if enabled else (0, 0.)
            ints[0, row] = state.tokens[-1] if state.tokens else BOS
            ints[1, row] = field
            times[row] = t
        tock = clock()
        timing['grammar_s'] += tock - tick
        logits = [decoder.step(ints, times) for decoder in decoders]
        tick = clock()
        timing['decode_s'] += tick - tock
        # Legal supports are prefix-only, so they are built while the device decodes.
        explicit = []
        for row, (state, enabled) in enumerate(zip(states, active)):
            if enabled:
                filtered = (_pattern_filter(state, patterns, decoding['pattern_constraints']['counts'])
                            if patterns is not None else None)
                first, count, tokens = (_support_spec(filtered) if filtered is not None else
                                        state.allowed_spec() if fast_grammar
                                        else _support_spec(state.allowed_tokens()))
                if not count:
                    failures[row], active[row] = 'grammar_dead_end', False
            if not active[row]:
                first, count, tokens = BOS, 1, None
            firsts[row], counts[row] = first, count
            if tokens is not None:
                explicit.append((row, tokens))
        longest = int(counts.max())
        span = np.arange(longest)
        # Pad with the first legal token, exactly as the reference does.
        index = firsts[:, None] + span[None] * (span[None] < counts[:, None])
        for row, tokens in explicit:
            index[row, :len(tokens)] = tokens
            index[row, len(tokens):] = tokens[0]
        noise = (_draw_noise(counts, active, row_generators, longest)
                 if temperature is not None and batched else None)
        bias = _fusion_matrix(states, active, tables, fusion['weight'], longest) if fusion else None
        tock = clock()
        timing['grammar_s'] += tock - tick
        if staging is not None:  # pinned + async on CUDA; reuse is safe after last step's sync
            view = staging[:index.size].view(rows, longest)
            view.numpy()[:] = index
            indices = view.to(device, non_blocking=True)
        else:
            indices = torch.from_numpy(index).to(device)
        if guidance is not None:
            logits_u = logits[1]
            combined = logits_u.float() + float(guidance) * (logits[0].float() - logits_u.float())
        else:
            combined = logits[0]
        scores = combined.gather(1, indices)
        if fusion:
            scores = scores.float()
            if bias is not None:
                scores = scores + torch.from_numpy(bias).to(scores.device)
        if temperature is None:
            choice = scores.argmax(-1).cpu().numpy()
        else:
            row_temperature = temperature
            if field_temperature:
                row_temperature = [float(field_temperature.get(state.phase, temperature)) if enabled else temperature
                                   for state, enabled in zip(states, active)]
            if row_generators is not None and batched:
                drawn = _sample_rows_batched(scores, counts, row_temperature, top_p, noise).cpu()
                if not bool(drawn[1][torch.tensor(active)].all()):
                    raise RuntimeError('probability tensor contains either `inf`, `nan` or element < 0')
                choice = drawn[0].numpy()
            elif row_generators is not None:
                legal = [range(int(count)) for count in counts]
                choice = _sample_rows(scores, legal, active, row_temperature, top_p,
                                      row_generators).view(-1).cpu().numpy()
            else:
                support = torch.from_numpy(span[None] < counts[:, None]).to(device)
                if isinstance(row_temperature, list):
                    scaled = scores.float() / torch.tensor(row_temperature, device=scores.device)[:, None]
                    choice = _sample(scaled, support, 1.0, top_p, generator)
                else:
                    choice = _sample(scores, support, temperature, top_p, generator)
                choice = choice.view(-1).cpu().numpy()
        selected = index[every, choice].tolist()
        tick = clock()
        timing['select_s'] += tick - tock
        for state, token, enabled in zip(states, selected, active):
            if enabled: state.apply(token)
        timing['grammar_s'] += clock() - tick
        timing['steps'] += 1
        if on_step is not None:
            on_step()
    if stats is not None:
        sampling = None if row_generators is None else 'batched' if batched else 'per_row_reference'
        timing.update(wall_s=clock() - started, rows=rows, sampling=sampling, billed_row_steps=rows * timing['steps'],
                      encoded_rows=len(inputs['mode']),
                      tokens=sum(len(state.tokens) for state in states), cuda_graph=use_graph,
                      graph_captures=sum(getattr(d, 'captures', 0) for d in decoders))
        stats.update(timing)
    return [dict(complete=state.done, events=[asdict(e) for e in sorted(state.events)], tokens=state.tokens,
                 deadline_reached=deadline_reached and not state.done and failure is None,
                 token_limit_reached=not state.done and not deadline_reached and failure is None,
                 failure=failure, decoding=dict(decoding), rng_mode=rng_mode)
            for state, failure in zip(states, failures)]


# --------------------------------------------------------------------------
# Continuous batching: one fixed seat table, requests enter and leave per step.
# --------------------------------------------------------------------------

def merge_stats(total, part):
    """Accumulate decode timings/counters; ``occupancy`` = real tokens / billed row-steps."""
    for key in ('decode_s', 'grammar_s', 'select_s', 'encoder_s', 'wall_s', 'steps', 'billed_row_steps', 'tokens',
                'encoded_rows', 'graph_captures'):
        total[key] = total.get(key, 0) + part.get(key, 0)
    total['occupancy'] = total['tokens'] / total['billed_row_steps'] if total['billed_row_steps'] else None
    for key in ('sampling', 'cuda_graph'):
        if key in part:
            total[key] = part[key]
    return total


class ContinuousDecoder:
    """``_generate_fast`` with iteration-level refill.

    Each seat has its own position, memory mask, K/V write slot and generator, so a finished row
    gives up its seat without waiting for the longest row. A row's draws depend only on its own
    generator and its own decoded prefix, exactly as in the batch loop; masked K/V and memory slots
    get zero weight, so only fp32 reduction order can differ from a batch decode.

    Flow: ``encode`` -> ``seat`` -> ``step`` (returns finished seats) -> ``release``. ``compact``
    drops seats that can never be used again (the end of a panel).
    """

    def __init__(self, model, seats, max_tokens, *, temperature=None, top_p=None, field_temperature=None,
                 guidance=None, onset_fusion=None, pattern_constraints=None, cuda_graph=None, compile_decode=False,
                 autocast=None, on_step=None, stats=None, kind_bias=None):
        if model.training or not fast_supported(model) or seats < 1 or max_tokens < 1:
            raise ValueError('Continuous decoding needs an eval-mode attention decoder, seats >= 1 and max_tokens >= 1')
        self.model, self.seats, self.max_tokens = model, seats, max_tokens
        # Log-probability added to a Phira note type when it is drawn: {type: bias}. A preference knob, not a model output.
        self.kind_bias = {model.vocab.token('kind', TYPES.index(k)): float(v) for k, v in (kind_bias or {}).items()}
        self.temperature, self.top_p, self.field_temperature = temperature, top_p, field_temperature
        self.guidance = None if guidance is None else float(guidance)
        self.decoding = _decoding_options(temperature, top_p, field_temperature)
        if autocast is not None:
            self.decoding['autocast'] = autocast
        if guidance is not None:
            self.decoding['guidance'] = self.guidance
        self.fusion = None if onset_fusion is None else _onset_fusion_rule(onset_fusion)
        if self.fusion:
            self.decoding['onset_fusion'] = dict(self.fusion)
        self.patterns = _pattern_rule(pattern_constraints)
        if self.patterns is not None:
            self.decoding['pattern_constraints'] = dict(self.patterns, counts={})
        self.rng_mode = None if temperature is None else 'per_row'
        self.device = next(model.parameters()).device
        self.use_graph = (CUDA_GRAPH_DEFAULT if cuda_graph is None else bool(cuda_graph)) and not compile_decode \
            and self.device.type == 'cuda' and not _GRAPH_FAILURES
        self.compiled = _compiled_decode(model) if compile_decode else None
        self.autocast_context = _autocast_context(self.device, autocast)
        self.on_step, self.stats = on_step, stats
        self.timing = dict(decode_s=0., grammar_s=0., select_s=0., encoder_s=0., steps=0, billed_row_steps=0, tokens=0,
                           encoded_rows=0, graph_captures=0, wall_s=0.)
        self.started = time.perf_counter()
        self.fast_grammar = EventState is _codec.EventState
        self.state_class = FastEventState if self.fast_grammar else EventState
        self.batched = None
        self.idle_generator = (None if temperature is None else torch.Generator(
            device=self.device.type if self.device.type in ('cuda', 'mps') else 'cpu'))
        self._allocate_rows(seats)
        self.memory_length, self.decoders = 0, []

    def _allocate_rows(self, seats):
        self.states = [None] * seats
        self.failures = [None] * seats
        self.tables = [None] * seats
        self.generators = [self.idle_generator] * seats
        self.positions = np.zeros(seats, np.int64)
        self.ints = np.zeros((2, seats), np.int64)
        self.times = np.zeros(seats, np.float32)
        self.firsts = np.zeros(seats, np.int64)
        self.counts = np.zeros(seats, np.int64)
        self.every = np.arange(seats)
        self.staging = (torch.empty(seats * self.model.vocab.size, dtype=torch.long).pin_memory()
                        if self.device.type == 'cuda' else None)

    def free_seats(self):
        return [row for row, state in enumerate(self.states) if state is None]

    def _stacked_kv(self, memory):
        pairs = self.model.memory_kv(memory)
        return torch.stack([k for k, _ in pairs]), torch.stack([v for _, v in pairs])

    def encode(self, inputs, requests, unconditional_inputs=None):
        """Encoded rows ready to seat, one per input row (several seats may share one)."""
        tick = time.perf_counter()
        with self.autocast_context:
            memory = _encode(self.model, inputs)
            tables = _onset_tables(memory, requests, self.fusion['window_ms']) if self.fusion else None
            parts = [(*self._stacked_kv(memory), memory['valid'], self.model.mode_ids(memory['mode'])[:, None])]
            if self.guidance is not None:
                bare = _encode(self.model, unconditional_inputs)
                parts.append((*self._stacked_kv(bare), bare['valid'], self.model.mode_ids(bare['mode'])[:, None]))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.timing['encoder_s'] += time.perf_counter() - tick
        self.timing['encoded_rows'] += len(requests)
        return [dict(table=None if tables is None else tables[row],
                     parts=[(keys[:, row], values[:, row], valid[row], mode[row]) for keys, values, valid, mode in parts])
                for row in range(len(requests))]

    def _make_step(self, cache):
        if self.use_graph and not _GRAPH_FAILURES:
            return _GraphStep(self.model, None, self.max_tokens, cache=cache)
        return _EagerStep(self.model, None, self.max_tokens, compiled=self.compiled, cache=cache)

    def _new_cache(self, keys, values, mode, length):
        """Blank seat cache shaped like start_static_cache. Empty seats attend one zero memory slot (never NaN)."""
        layers, seats = keys.shape[0], self.seats
        memory_keys = keys.new_zeros(layers, seats, keys.shape[1], length, keys.shape[-1])
        memory_values = values.new_zeros(layers, seats, values.shape[1], length, values.shape[-1])
        mask = torch.zeros(seats, 1, 1, length, dtype=torch.bool, device=keys.device)
        mask[..., 0] = True
        return dict(batch_size=seats, capacity=self.max_tokens, memory_keys=memory_keys, memory_values=memory_values,
                    memory_kv=[(memory_keys[i], memory_values[i]) for i in range(layers)], memory_mask=mask,
                    mode_embedding=mode.new_zeros(seats, 1, mode.shape[-1]), keys=None, values=None,
                    slots=torch.arange(self.max_tokens, device=keys.device))

    def _ensure_memory(self, entries):
        """Allocate on first use; a longer row grows the table to a rounded length (graphs are re-captured)."""
        needed = max(entry['parts'][0][0].shape[2] for entry in entries)
        if needed <= self.memory_length:
            return
        length = _rounded_memory_length(needed)
        sample = entries[0]['parts']
        previous = [d.cache for d in self.decoders]
        caches = [self._new_cache(keys, values, mode, length) for keys, values, _, mode in sample]
        for cache, old in zip(caches, previous):
            cache['memory_keys'][:, :, :, :self.memory_length] = old['memory_keys']
            cache['memory_values'][:, :, :, :self.memory_length] = old['memory_values']
            cache['memory_mask'][..., :self.memory_length] = old['memory_mask']
            cache['mode_embedding'].copy_(old['mode_embedding'])
            cache['keys'], cache['values'] = old['keys'], old['values']
        self._retire_steps()
        self.decoders = [self._make_step(cache) for cache in caches]
        self.memory_length = length

    def _retire_steps(self):
        self.timing['graph_captures'] += sum(getattr(d, 'captures', 0) for d in self.decoders)

    def seat(self, row, request, entry, generator=None):
        """Put one request on a free seat; its K/V history starts empty at position 0."""
        self._ensure_memory([entry])
        for decoder, (keys, values, valid, mode) in zip(self.decoders, entry['parts']):
            cache, length = decoder.cache, keys.shape[2]
            cache['memory_keys'][:, row, :, :length] = keys
            cache['memory_values'][:, row, :, :length] = values
            cache['memory_mask'][row, ..., :length] = valid
            cache['memory_mask'][row, ..., length:] = False
            cache['mode_embedding'][row] = mode
        self.states[row] = self.state_class(request, self.model.vocab)
        self.failures[row], self.tables[row], self.positions[row] = None, entry['table'], 0
        if generator is not None:
            self.generators[row] = generator
            if self.batched is None:
                self.batched = _batched_sampling_ok([generator])

    def release(self, row, *, deadline_reached=False):
        """Free a seat and return the prediction dict ``generate_batch`` makes for such a row."""
        state, failure = self.states[row], self.failures[row]
        self.states[row], self.failures[row], self.tables[row] = None, None, None
        self.generators[row], self.positions[row] = self.idle_generator, 0
        self.timing['tokens'] += len(state.tokens)
        return dict(complete=state.done, events=[asdict(e) for e in sorted(state.events)], tokens=state.tokens,
                    deadline_reached=deadline_reached and not state.done and failure is None,
                    token_limit_reached=not state.done and not deadline_reached and failure is None,
                    failure=failure, decoding=dict(self.decoding), rng_mode=self.rng_mode)

    def compact(self, seats):
        """Shrink the table to ``seats`` rows (>= occupied). Returns {old row: new row} for occupied rows."""
        occupied = [row for row, state in enumerate(self.states) if state is not None]
        keep = (occupied + self.free_seats())[:seats]
        index = torch.tensor(keep, device=self.device)
        moved = dict(states=self.states, failures=self.failures, tables=self.tables, generators=self.generators,
                     positions=self.positions.copy())
        self._retire_steps()
        self.seats = seats
        self._allocate_rows(seats)
        for new, old in enumerate(keep):
            self.states[new], self.failures[new], self.tables[new] = (moved['states'][old], moved['failures'][old],
                                                                      moved['tables'][old])
            self.generators[new], self.positions[new] = moved['generators'][old], moved['positions'][old]
        decoders, self.decoders = self.decoders, []
        for decoder in decoders:
            old = decoder.cache
            cache = dict(old, batch_size=seats, memory_keys=old['memory_keys'][:, index],
                         memory_values=old['memory_values'][:, index], memory_mask=old['memory_mask'][index],
                         mode_embedding=old['mode_embedding'][index])
            cache['memory_kv'] = [(cache['memory_keys'][i], cache['memory_values'][i])
                                  for i in range(len(self.model.decoder))]
            if old['keys'] is not None:
                cache['keys'], cache['values'] = old['keys'][:, index], old['values'][:, index]
            self.decoders.append(self._make_step(cache))
        return {old: new for new, old in enumerate(keep) if old in occupied}

    def step(self):
        """One decode step over the table; returns the seats whose request just ended."""
        clock, timing = time.perf_counter, self.timing
        states, failures, patterns = self.states, self.failures, self.patterns
        active = [state is not None for state in states]
        ints, times, firsts, counts = self.ints, self.times, self.firsts, self.counts
        with self.autocast_context:
            tick = clock()
            for row, (state, enabled) in enumerate(zip(states, active)):
                if enabled:
                    field, t = state.query()
                    ints[0, row] = state.tokens[-1] if state.tokens else BOS
                    ints[1, row], times[row] = field, t
                else:
                    ints[0, row], ints[1, row], times[row] = BOS, 0, 0.
            tock = clock()
            timing['grammar_s'] += tock - tick
            logits = [decoder.step(ints, times, self.positions) for decoder in self.decoders]
            tick = clock()
            timing['decode_s'] += tick - tock
            explicit = []
            for row, (state, enabled) in enumerate(zip(states, active)):
                first, count, tokens = BOS, 1, None
                if enabled:
                    filtered = (_pattern_filter(state, patterns, self.decoding['pattern_constraints']['counts'])
                                if patterns is not None else None)
                    first, count, tokens = (_support_spec(filtered) if filtered is not None else
                                            state.allowed_spec() if self.fast_grammar
                                            else _support_spec(state.allowed_tokens()))
                    if not count:
                        failures[row], active[row] = 'grammar_dead_end', False
                        first, count, tokens = BOS, 1, None
                firsts[row], counts[row] = first, count
                if tokens is not None:
                    explicit.append((row, tokens))
            longest = int(counts.max())
            span = np.arange(longest)
            index = firsts[:, None] + span[None] * (span[None] < counts[:, None])
            for row, tokens in explicit:
                index[row, :len(tokens)] = tokens
                index[row, len(tokens):] = tokens[0]
            sampling = self.temperature is not None
            noise = _draw_noise(counts, active, self.generators, longest) if sampling and self.batched else None
            bias = (_fusion_matrix(states, active, self.tables, self.fusion['weight'], longest)
                    if self.fusion else None)
            tock = clock()
            timing['grammar_s'] += tock - tick
            if self.staging is not None:
                view = self.staging[:index.size].view(self.seats, longest)
                view.numpy()[:] = index
                indices = view.to(self.device, non_blocking=True)
            else:
                indices = torch.from_numpy(index).to(self.device)
            combined = logits[0]
            if self.guidance is not None:
                combined = logits[1].float() + self.guidance * (logits[0].float() - logits[1].float())
            scores = combined.gather(1, indices)
            if self.fusion:
                scores = scores.float()
                if bias is not None:
                    scores = scores + torch.from_numpy(bias).to(scores.device)
            if self.kind_bias:
                scores = scores.float()
                for row, (state, enabled) in enumerate(zip(states, active)):
                    if enabled and state.phase == 'kind':
                        for j in range(int(counts[row])):
                            add = self.kind_bias.get(int(index[row, j]))
                            if add:
                                scores[row, j] += add
            if not sampling:
                choice = scores.argmax(-1).cpu().numpy()
            else:
                row_temperature = self.temperature
                if self.field_temperature:
                    row_temperature = [float(self.field_temperature.get(state.phase, self.temperature)) if enabled
                                       else self.temperature for state, enabled in zip(states, active)]
                if self.batched:
                    drawn = _sample_rows_batched(scores, counts, row_temperature, self.top_p, noise).cpu()
                    if not bool(drawn[1][torch.tensor(active)].all()):
                        raise RuntimeError('probability tensor contains either `inf`, `nan` or element < 0')
                    choice = drawn[0].numpy()
                else:
                    legal = [range(int(count)) for count in counts]
                    choice = _sample_rows(scores, legal, active, row_temperature, self.top_p,
                                          self.generators).view(-1).cpu().numpy()
            selected = index[self.every, choice].tolist()
            tick = clock()
            timing['select_s'] += tick - tock
            finished = []
            for row, (state, token, enabled) in enumerate(zip(states, selected, active)):
                if enabled:
                    state.apply(token)
                    self.positions[row] += 1
                if state is not None and (state.done or failures[row] is not None
                                          or self.positions[row] >= self.max_tokens):
                    finished.append(row)
            timing['grammar_s'] += clock() - tick
            timing['steps'] += 1
            timing['billed_row_steps'] += self.seats
        if self.on_step is not None:
            self.on_step()
        return finished

    def finish(self):
        """Publish the timings to ``stats``."""
        self._retire_steps()
        self.timing['wall_s'] = time.perf_counter() - self.started
        if self.stats is not None:
            merge_stats(self.stats, dict(self.timing, cuda_graph=self.use_graph,
                                         sampling=None if self.temperature is None else
                                         'batched' if self.batched else 'per_row_reference'))

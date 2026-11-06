"""One normalized canonical event-package likelihood and declared decisions.

No target enters forward/decoding. Native-ms and authorized microsecond COPY
tails share one partition. Ordered heads/tails retain exact duplicates.
"""
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .canvas_diffusion import _legal_partition, _sample_tail
from .cardinality_dp import cardinality_counts
from .event_canvas import RhythmEvent
from .local_event_package import EventPackageCanvas, GRU_WIDTH, _allowed_kinds, _hurdle_count_log_probabilities
from .local_typed_canvas import FRAMES, SLOTS, KINDS, best_tail


def package_support(inputs, request):
    if len(inputs['mode']) != 1:
        raise ValueError('Canonical package microbatch must be one')
    kinds = torch.tensor([k in _allowed_kinds(request) for k in range(4)], device=inputs['mode'].device)
    last = max([request.duration_us // 1000 * 1000] +
        [t for t in request.copy_times if 0 <= t <= request.duration_us])
    hold = (inputs['canvas_head_allowed'] & (inputs['canvas_head_us'] < last)).any(-1)
    yes = torch.ones_like(hold)
    return inputs['canvas_valid'][..., None] & kinds[None, None] & torch.stack((yes, hold, yes, yes), -1)


def time_features(times, base_us, *, like):
    v = torch.as_tensor(times, device=like.device, dtype=like.dtype)
    d = (v - base_us) / 1e6
    return torch.stack((d / 1200, torch.log1p(d.clamp_min(0)),
        torch.sin(d * math.pi), torch.cos(d * math.pi)), -1)


class CanonicalPackageCanvas(EventPackageCanvas):
    def __init__(self, config, *, density_reference):
        super().__init__(config, density_reference=density_reference)
        self.copy_key = nn.Sequential(nn.Linear(4, config.d), nn.SiLU(), nn.Linear(config.d, config.d))
        self.copy_query = nn.Linear(config.d, config.d, bias=False)
        self.copy_feedback = nn.Linear(4, config.d)

    def forward(self, inputs):
        x = EventPackageCanvas.forward(self, inputs)['x']
        gate = self.count_gate(x)
        positive = self.count_positive(x).view(len(x), FRAMES, 4, SLOTS)
        logp = _hurdle_count_log_probabilities(gate, positive)
        density = inputs['controls'][:, 0]
        rho = density[:, :2].mean(-1) * 64
        tilt = torch.where(density[:, 2] > .5, F.softplus(self.density_alpha)
            * (rho.clamp_min(1e-4) / self.density_reference).log(), 0.)
        logp = logp + tilt[:, None, None, None] * torch.arange(SLOTS + 1, device=x.device)
        logp = logp - logp.logsumexp(-1, keepdim=True)
        return dict(x=x, count=logp.exp(), count_log_probs=logp)

    def tail_law(self, condition, request, base_us, minimum_us, *, target=None, rng=None, greedy=False):
        """Union of unique native-ms endpoints and explicit non-ms anchors."""
        if sum((target is not None, rng is not None, greedy)) != 1:
            raise ValueError('Choose exactly teacher, T1 or greedy tail decision')
        fields = self.tail_block(condition), self.tail_cell(condition), self.tail_ms(condition)
        lo = max(0, (minimum_us - base_us + 999) // 1000)
        hi = (request.duration_us - base_us) // 1000
        if hi >= len(fields[0]) * 2000:
            raise ValueError('Song exceeds represented native tail domain')
        copies = sorted({int(t) for t in request.copy_times
            if t % 1000 and minimum_us <= t <= request.duration_us})
        logs = []
        native = lo <= hi
        if native:
            logs.append(_legal_partition(*fields, lo, hi, block_size=2000, cell_size=40))
        features = time_features(copies, base_us, like=condition)
        copy_scores = (self.copy_key(features) * self.copy_query(condition)).sum(-1) / math.sqrt(len(condition))
        if copies:
            logs.append(copy_scores.logsumexp(0))
        if not logs:
            raise ValueError('No legal native/COPY hold endpoint')
        logz = torch.logsumexp(torch.stack(logs), 0)
        if target is not None:
            if target % 1000 == 0:
                delta = (target - base_us) // 1000
                if not native or not lo <= delta <= hi:
                    raise ValueError('Teacher native endpoint outside legal support')
                score = fields[0][delta // 2000] + fields[1][delta % 2000 // 40] + fields[2][delta % 40]
            else:
                if target not in copies:
                    raise ValueError('Teacher sub-ms endpoint lacks explicit COPY authorization')
                score = copy_scores[copies.index(target)]
            end = int(target)
            nll = logz - score
        else:
            arrays = [f.detach().float().cpu().double().numpy() for f in fields]
            cs = copy_scores.detach().float().cpu().double().numpy()
            if greedy:
                candidates = []
                if native:
                    delta = best_tail(*arrays, (lo - 1) * 1000, hi * 1000) // 1000
                    value = arrays[0][delta // 2000] + arrays[1][delta % 2000 // 40] + arrays[2][delta % 40]
                    candidates.append((float(value), base_us + delta * 1000))
                candidates.extend((float(s), t) for s, t in zip(cs, copies))
                end = max(candidates, key=lambda p: (p[0], -p[1]))[1]
            else:
                branch = np.asarray([float(v.detach().cpu()) for v in logs])
                weight = np.exp(branch - branch.max());weight /= weight.sum()
                choose = int(rng.choice(len(weight), p=weight))
                if native and choose == 0:
                    end = base_us + _sample_tail(*arrays, lo, hi, rng) * 1000
                else:
                    weight = np.exp(cs - cs.max());weight /= weight.sum()
                    end = copies[int(rng.choice(len(weight), p=weight))]
            nll = condition.sum() * 0
        if end % 1000:
            embedding = self.copy_feedback(time_features(end, base_us, like=condition))
        else:
            delta = (end - base_us) // 1000
            embedding = self.tail_block_ids.weight[delta // 2000] + self.tail_cell_ids.weight[delta % 2000 // 40] + self.tail_ms_ids.weight[delta % 40]
        return int(end), embedding, nll

    def ordered_members(self, x, frame, kind, count, inputs, request, *, teacher=None, rng=None, greedy=False):
        if not 1 <= count <= SLOTS or sum((teacher is not None, rng is not None, greedy)) != 1:
            raise ValueError('Invalid canonical member decision')
        allowed = inputs['canvas_head_allowed'][0, frame]
        heads = inputs['canvas_head_us'][0, frame]
        last_end = max([request.duration_us // 1000 * 1000] +
            [t for t in request.copy_times if 0 <= t <= request.duration_us])
        static = x[frame] + self.kind_ids.weight[kind] + self.count_ids.weight[count]
        hidden, previous = x.new_zeros(GRU_WIDTH), self.member_start
        previous_head = previous_end = None
        onset_nll = tail_nll = x.sum() * 0
        members = []
        for slot in range(count):
            hidden = self.member_gru(torch.cat((static, previous), -1)[None], hidden[None])[0]
            support = allowed if previous_head is None else allowed & (heads >= previous_head)
            if kind == 1:
                support = support & (heads < last_end)
            if not bool(support.any()):raise ValueError('No legal canonical head')
            logits = self.onset_read(hidden).masked_fill(~support, -1e9)
            if teacher is not None:
                offset = int(teacher['offset'][slot])
                if not bool(support[offset]):raise ValueError('Teacher outside canonical head support')
                onset_nll = onset_nll - logits.log_softmax(-1)[offset]
            elif greedy:offset = int(logits.argmax())
            else:
                p = logits.detach().float().cpu().double().softmax(-1).numpy()
                offset = int(rng.choice(len(p), p=p / p.sum()))
            head = int(heads[offset]);end = head
            onset = self.onset_ids.weight[offset]
            tail = self.no_tail
            if kind == 1:
                condition = self.tail_condition(torch.cat((hidden, onset), -1))
                minimum = max(head + 1, previous_end if previous_head == head else head + 1)
                end, tail, nll = self.tail_law(condition, request, request.start_us + frame * 40_000,
                    minimum, target=int(teacher['tail_us'][slot]) if teacher is not None else None,
                    rng=rng, greedy=greedy)
                tail_nll = tail_nll + nll
            previous = self.member_read(torch.cat((onset, tail), -1))
            previous_head, previous_end = head, end
            members.append(RhythmEvent(head, end, KINDS[kind]))
        return members, onset_nll, tail_nll


def canonical_package_loss(model, inputs, target, request):
    output = model(inputs)
    groups = package_support(inputs, request)
    count = target['count']
    if bool(((count > 0) & ~groups).any()) or bool(((count < 0) | (count > SLOTS)).any()):
        raise ValueError('Target outside complete package support')
    count_sum = (-output['count_log_probs'].gather(-1, count[..., None]).squeeze(-1) * groups).sum()
    onset_sum = tail_sum = output['x'].sum() * 0
    for frame, kind in ((count[0] > 0) & groups[0]).nonzero().tolist():
        n = int(count[0, frame, kind])
        _, onset, tail = model.ordered_members(output['x'][0], frame, kind, n, inputs, request,
            teacher={k:target[k][0, frame, kind, :n] for k in ('offset','tail_us')})
        onset_sum = onset_sum + onset;tail_sum = tail_sum + tail
    denominator = groups.sum().clamp_min(1).float()
    parts = {k:v / denominator for k,v in [('count',count_sum),('onset',onset_sum),('tail',tail_sum)]}
    return sum(parts.values()), dict(**parts, valid_groups=denominator)


@torch.no_grad()
def generate_canonical(model, inputs, request, *, decision='cardinality', seed=20261061, deadline=None):
    model.eval();started=time.monotonic()
    output = model(inputs)
    groups = package_support(inputs, request)[0]
    addresses = groups.nonzero().tolist()
    p = output['count'][0][groups].detach().float().cpu().double().numpy()
    rng = np.random.default_rng(seed)
    info = dict(decision=decision)
    if decision == 'cardinality':
        counts, decision_info = cardinality_counts(p, max_events=request.max_events)
        info.update(decision_info)
    elif decision == 'ancestral':
        counts = np.asarray([rng.choice(len(row),p=row/row.sum()) for row in p])
    elif decision == 'individual_mode':counts=p.argmax(-1)
    else:raise ValueError('Unknown predeclared package decision')
    if int(counts.sum()) > request.max_events:raise ValueError('Canonical package resource overflow')
    events=[]
    for (frame,kind),n in zip(addresses,counts):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Canonical generation deadline')
        if n:
            members,_,_=model.ordered_members(output['x'][0],frame,kind,int(n),inputs,request,
                rng=rng if decision=='ancestral' else None,greedy=decision!='ancestral')
            events.extend(members)
    info.update(counts=counts,wall_s=time.monotonic()-started)
    return sorted(events),info

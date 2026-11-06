"""Vectorized public support, exact duration prior and HN2 two-pass law.

No event objects, target-dependent masks, scalar device reads or Python loops
over events occur in the model/loss/sampler. CPU packing/export is separate.
"""
import math

import torch
from torch.nn import functional as F

from .native_diagnostic_priors import DURATION_UPPER_US


KINDS = ('tap', 'hold', 'flick', 'drag')
SLOTS = 512


def prepare_public_support(requests, known_regions=None, *, max_duration_us=512_000_000):
    """CPU only: support depends exclusively on public request/known regions."""
    regions = known_regions or [()] * len(requests)
    layouts, copies = [], []
    for request, protected in zip(requests, regions):
        if request.duration_us > max_duration_us:
            raise ValueError('Public duration exceeds HN2 endpoint block capacity')
        copy = sorted(set(int(t) for t in request.copy_times))
        values = set(range((request.start_us+999)//1000*1000, request.end_us, 1000))
        values.update(t for t in copy if request.start_us <= t < request.end_us)
        heads, allowed = [], []
        for h in sorted(values):
            if any(lo <= h < hi for lo, hi in protected):
                continue
            legal = [request.allowed_types is None or k in request.allowed_types for k in KINDS]
            legal[1] &= ((h//1000+1)*1000 <= request.duration_us or any(h < t <= request.duration_us for t in copy))
            if any(legal):
                heads.append(h)
                allowed.append(legal)
        layouts.append((heads, allowed))
        copies.append(copy)
    b = len(requests)
    hsize = max(1, max(len(h) for h, _ in layouts))
    csize = max(1, max(map(len, copies)))
    result = dict(head_us=torch.zeros(b, hsize, dtype=torch.int64), head_valid=torch.zeros(b, hsize, dtype=torch.bool),
                  type_allowed=torch.zeros(b, hsize, 4, dtype=torch.bool), copy_us=torch.zeros(b, csize, dtype=torch.int64),
                  copy_valid=torch.zeros(b, csize, dtype=torch.bool),
                  request_start_us=torch.tensor([r.start_us for r in requests], dtype=torch.int64),
                  request_end_us=torch.tensor([r.end_us for r in requests], dtype=torch.int64),
                  duration_us=torch.tensor([r.duration_us for r in requests], dtype=torch.int64))
    for i, ((heads, allowed), copy) in enumerate(zip(layouts, copies)):
        if heads:
            result['head_us'][i, :len(heads)] = torch.tensor(heads)
            result['head_valid'][i, :len(heads)] = True
            result['type_allowed'][i, :len(heads)] = torch.tensor(allowed)
        if copy:
            result['copy_us'][i, :len(copy)] = torch.tensor(copy)
            result['copy_valid'][i, :len(copy)] = True
    return result


def validate_targets(inputs, targets, valid):
    """Explicit CPU admission, not part of the CUDA hot path."""
    if targets.device.type != 'cpu' or valid.device.type != 'cpu':
        raise ValueError('Validate target admission before device transfer')
    if targets.shape[1:] != (512, 4) or valid.shape != targets.shape[:2]:
        raise ValueError('HN2 target shape/OVERFLOW')
    for b in range(targets.shape[0]):
        heads = {int(h): j for j, h in enumerate(inputs['head_us'][b].tolist()) if inputs['head_valid'][b, j]}
        copy = {int(v) for v, good in zip(inputs['copy_us'][b], inputs['copy_valid'][b]) if good}
        for h, e, k, lane in targets[b, valid[b]].tolist():
            if h not in heads or not 0 <= k < 4 or lane != -1 or not inputs['type_allowed'][b, heads[h], k]:
                raise ValueError('Target outside public head/type support')
            if k == 1:
                if not h < e <= int(inputs['duration_us'][b]) or (e % 1000 and e not in copy):
                    raise ValueError('Target outside complete native/COPY endpoint support')
            elif e != h:
                raise ValueError('Nonhold has duration')
    return True


def _draw(logits, generator=None):
    return torch.multinomial(logits.float().softmax(-1).reshape(-1, logits.shape[-1]), 1,
                             generator=generator).reshape(logits.shape[:-1])


def canonical_targets(targets, valid):
    """Canonical label order only. This function never feeds model inputs."""
    order = torch.arange(targets.shape[1], device=targets.device)[None].expand(targets.shape[0], -1)
    for column in (1, 2, 0):
        values = targets[..., column].gather(1, order)
        order = order.gather(1, values.argsort(dim=1, stable=True))
    active = valid.gather(1, order)
    order = order.gather(1, (~active).to(torch.int64).argsort(dim=1, stable=True))
    return targets.gather(1, order[..., None].expand(-1, -1, 4)), valid.gather(1, order)


def _finite_logits(logits, valid):
    # Inactive tuples may have no endpoint. Supply a computational dummy only;
    # its valid flag remains false and it never becomes a legal event.
    dummy = torch.zeros_like(valid)
    dummy[..., 0] = ~valid.any(-1)
    return logits.masked_fill(~(valid | dummy), -torch.inf).masked_fill(dummy, 0)


class EndpointDistribution:
    """B,S parallel relative 2s/40ms/exact-us tree and exact leaf prior."""
    def __init__(self, inputs, heads, prior, max_duration_us=512_000_000):
        self.inputs, self.heads = inputs, heads
        self.base = heads//1000*1000
        self.end = inputs['duration_us'][:, None]
        device = heads.device
        self.upper = torch.tensor(prior['duration_upper_us'], dtype=torch.int64, device=device)
        if tuple(prior['duration_upper_us']) != tuple(DURATION_UPPER_US):
            raise ValueError('HN2 requires the frozen exact diagnostic duration bins')
        self.weights = torch.tensor(prior['duration_counts'], dtype=torch.float32, device=device)
        self.lower = F.pad(self.upper[:-1], (1, 0))
        copy = inputs['copy_us']
        good = inputs['copy_valid'] & (copy % 1000 != 0)
        self.copies = torch.where(good, copy, torch.full_like(copy, torch.iinfo(torch.int64).max)).sort(1).values.contiguous()
        # COPY is a set of authorized coordinates, distinct from event-object
        # multiplicity. Keep static tensor shape while removing duplicate keys.
        if self.copies.shape[1] > 1:
            duplicate = F.pad(self.copies[:, 1:] == self.copies[:, :-1], (1, 0))
            self.copies = self.copies.masked_fill(duplicate, torch.iinfo(torch.int64).max).sort(1).values.contiguous()
        self.copy_good = self.copies != torch.iinfo(torch.int64).max
        lo = heads[..., None] + self.lower + 1
        hi = torch.minimum(heads[..., None] + self.upper, self.end[..., None])
        self.numbers = self.count(lo, hi)
        self.normalizer = ((self.numbers > 0)*self.weights).sum(-1).clamp_min(1e-30)
        self.max_blocks = math.ceil(max_duration_us/2_000_000) + 1

    def count(self, lo, hi):
        native = (hi//1000 - (lo+999)//1000 + 1).clamp_min(0)
        shape = lo.shape
        low = torch.searchsorted(self.copies, lo.reshape(lo.shape[0], -1).contiguous(), right=False)
        high = torch.searchsorted(self.copies, hi.reshape(hi.shape[0], -1).contiguous(), right=True)
        return (native + (high-low).reshape(shape)).masked_fill(lo > hi, 0)

    def interval_mass(self, lo, hi):
        lower = torch.maximum(lo[..., None], self.heads[..., None, None] + self.lower + 1)
        upper = torch.minimum(hi[..., None], torch.minimum(self.heads[..., None, None] + self.upper, self.end[..., None, None]))
        counts = self.count(lower, upper).float()
        per_leaf = self.weights / self.numbers.float().clamp_min(1) / self.normalizer[..., None]
        return (counts * per_leaf[..., None, :]).sum(-1)

    def nodes(self, level, choices):
        device = self.heads.device
        if level == 0:
            ids = torch.arange(self.max_blocks, device=device)
            start = self.base[..., None] + ids*2_000_000
            stop = start + 1_999_999
            values = ids.expand(*self.heads.shape, -1)
        elif level == 1:
            ids = torch.arange(50, device=device)
            start = self.base[..., None] + choices[0][..., None]*2_000_000 + ids*40_000
            stop = start + 39_999
            values = ids.expand(*self.heads.shape, -1)
        else:
            cell_start = self.base + choices[0]*2_000_000 + choices[1]*40_000
            native = cell_start[..., None] + torch.arange(40, device=device)*1000
            copies = self.copies[:, None, :].expand(*self.heads.shape, -1)
            values = torch.cat((native, copies), -1)
            in_cell = (values >= cell_start[..., None]) & (values < cell_start[..., None]+40_000)
            valid = in_cell & (values > self.heads[..., None]) & (values <= self.end[..., None])
            delta = (values-self.heads[..., None]).clamp(min=1, max=DURATION_UPPER_US[-1])
            bucket = torch.searchsorted(self.upper, delta.contiguous()).clamp(max=len(DURATION_UPPER_US)-1)
            numbers = self.numbers.gather(2, bucket).float().clamp_min(1)
            mass = self.weights[bucket]/self.normalizer[..., None]/numbers
            return values, values, mass.masked_fill(~valid, 0), valid
        lo = torch.maximum(start, self.heads[..., None]+1)
        hi = torch.minimum(stop, self.end[..., None])
        mass = self.interval_mass(lo, hi)
        valid = (lo <= hi) & (mass > 0)
        centers = ((lo+hi)//2).clamp_min(0)
        return values, centers, mass.masked_fill(~valid, 0), valid

    def factors(self, model, output, *, target=None, generator=None):
        choices, records = [], []
        nll = output['states'].sum(-1)*0
        supported = torch.ones_like(self.heads, dtype=torch.bool)
        end = self.heads
        for level in range(3):
            values, centers, mass, valid = self.nodes(level, choices)
            residual = model.tail_scores(output, self.heads, centers, level)
            logits = _finite_logits(residual.float()+mass.clamp_min(1e-38).log(), valid)
            if target is None:
                index = _draw(logits, generator)
            else:
                label = ((target-self.base)//2_000_000 if level == 0 else
                         (target-self.base) % 2_000_000//40_000 if level == 1 else target)
                matches = (values == label[..., None]) & valid
                supported &= matches.any(-1)
                index = matches.to(torch.int64).argmax(-1)
            logp = logits.log_softmax(-1).gather(-1, index[..., None]).squeeze(-1)
            nll = nll-logp
            chosen = values.gather(-1, index[..., None]).squeeze(-1)
            choices.append(chosen)
            records.append(dict(logits=logits, values=values, valid=valid, index=index, log_probability=logp))
            if level == 2:
                end = chosen
        return end, nll, supported, records


def sample_pass(model, output, prior, *, generator=None):
    """One normalized tuple product; returned discrete state is detached."""
    with torch.no_grad():
        memory = output['memory']
        probability = output['active_logits'].float().sigmoid()*memory['head_valid'].any(1)[:, None]
        active = torch.rand(probability.shape, device=probability.device, generator=generator) < probability
        addresses = _draw(model.head_logits(output), generator)
        heads = memory['head_us'].gather(1, addresses)
        heads = torch.where(memory['head_valid'].any(1)[:, None], heads, memory['inputs']['request_start_us'][:, None])
        kinds = _draw(model.type_logits(output, addresses), generator)
        tree = EndpointDistribution(memory['inputs'], heads, prior, model.c.max_duration_us)
        ends, _, _, _ = tree.factors(model, output, generator=generator)
        ends = torch.where(kinds == 1, ends, heads)
        events = torch.stack((heads, ends, kinds, torch.full_like(heads, -1)), -1)
        events = torch.where(active[..., None], events, torch.zeros_like(events))
        return dict(events=events.detach(), active=active.detach())


def loss_pass(model, output, targets, target_valid, prior, *, details=False):
    targets, target_valid = canonical_targets(targets, target_valid)
    memory = output['memory']
    b, s = target_valid.shape
    labels = target_valid.float()
    possible = memory['head_valid'].any(1)
    occupancy = F.binary_cross_entropy_with_logits(output['active_logits'].float(), labels, reduction='none')
    occupancy = (occupancy * possible[:, None]).mean(1)
    heads = torch.where(target_valid, targets[..., 0], memory['inputs']['request_start_us'][:, None])
    addresses = torch.searchsorted(memory['head_us'].contiguous(), heads.contiguous()).clamp(max=memory['head_us'].shape[1]-1)
    matched_head = memory['head_us'].gather(1, addresses) == heads
    head_nll = -model.head_logits(output).float().log_softmax(-1).gather(2, addresses[..., None]).squeeze(-1)
    kinds = targets[..., 2].clamp(0, 3)
    kind_logits = model.type_logits(output, addresses).float()
    kind_nll = -kind_logits.log_softmax(-1).gather(2, kinds[..., None]).squeeze(-1)
    hold = target_valid & (kinds == 1)
    tree = EndpointDistribution(memory['inputs'], heads, prior, model.c.max_duration_us)
    _, tail_nll, supported_tail, factors = tree.factors(model, output, target=targets[..., 1])
    counts = target_valid.sum(1).clamp_min(1)
    hold_counts = hold.sum(1).clamp_min(1)
    # where avoids 0*inf for padded/forced fields. Invalid REAL targets remain inf.
    head_loss = torch.where(target_valid, head_nll, 0).sum(1)/counts
    kind_loss = torch.where(target_valid, kind_nll, 0).sum(1)/counts
    tail_loss = torch.where(hold, tail_nll, 0).sum(1)/hold_counts
    allowed = memory['type_allowed'].gather(1, addresses[..., None].expand(-1, -1, 4)).gather(2, kinds[..., None]).squeeze(-1)
    supported = (~target_valid | (matched_head & allowed)) & (~hold | supported_tail)
    support_ok = supported.all()
    loss = (occupancy+head_loss+kind_loss+tail_loss).mean()
    loss = torch.where(support_ok, loss, loss*0+torch.inf)
    metrics = dict(occupancy=occupancy.mean().detach(), head=head_loss.mean().detach(),
                   kind=kind_loss.mean().detach(), tail=tail_loss.mean().detach(),
                   supported=support_ok.detach(), members=target_valid.sum().detach(), holds=hold.sum().detach())
    type_support = memory['type_allowed'].gather(1, addresses[..., None].expand(-1, -1, 4))
    metrics['forced_types'] = (target_valid & (type_support.sum(-1) == 1)).sum().detach()
    endpoint_times = (targets[..., 1]-memory['inputs']['request_start_us'][:, None]).float()/1e6
    if 'whole_bounds' in memory:
        bounds = memory['whole_bounds']
        covered = (endpoint_times >= bounds[:, :1]) & (endpoint_times <= bounds[:, 1:])
        metrics['hold_whole_fine_covered'] = (hold & covered).sum().detach()
    if details:
        metrics.update(type_logits=kind_logits.detach(), type_support=type_support.detach(),
                       targets=targets.detach(), target_valid=target_valid.detach(),
                       head_nll=head_nll.detach(), tail_nll=tail_nll.detach(),
                       tail_factors=[{k:v.detach() for k,v in r.items()} for r in factors], supported_members=supported.detach())
    return loss, metrics


def two_pass(model, inputs, targets, target_valid, prior, *, generator=None, sample_final=False):
    first = model(inputs, pass_index=0)
    z = sample_pass(model, first, prior, generator=generator)
    loss0, metrics0 = loss_pass(model, first, targets, target_valid, prior)
    second = model(inputs, state=z, memory=first['memory'], pass_index=1)
    loss1, metrics1 = loss_pass(model, second, targets, target_valid, prior)
    y = sample_pass(model, second, prior, generator=generator) if sample_final else None
    return loss0+loss1, dict(p0=metrics0, p1=metrics1), z, y


def training_step(model, batch, prior, *, generator=None):
    return two_pass(model, batch['inputs'], batch['targets'], batch['target_valid'], prior, generator=generator)


@torch.no_grad()
def generate(model, inputs, prior, *, generator=None):
    first = model(inputs, pass_index=0)
    z = sample_pass(model, first, prior, generator=generator)
    second = model(inputs, state=z, memory=first['memory'], pass_index=1)
    return z, sample_pass(model, second, prior, generator=generator)

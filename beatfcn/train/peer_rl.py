"""Per-action reinforcement on one's own samples, with a reward that does not look at any human chart.

The same window is sampled K times. An action gets reward -1 when
  * its onset has no support among the other K-1 samples (the noise that voting removes), or
  * it breaks a Phira playability rule that a charter would have to fix (close onsets, third finger,
    mixed special notes in a chord, a hold started on top of other holds).
With reward_majority set, the time action of an onset most other samples also chose gets a positive reward (the group
comparison of GRPO, applied per action because whole-window credit is too noisy at this budget; types are excluded).
All other actions get 0, and only the action that caused the fault is credited. The gradient is REINFORCE on that
action: minimising +log p of a punished draw lowers the expected fault rate. Rewards never see a reference chart, so
they cannot learn one charter's habits (reports/goal-100usd-20261101/r5/rl-headroom: picks on chart A do not transfer).
The supervised loss stays on: it is what keeps the notes from simply disappearing. With anchor_weight set, actions that
are neither punished nor rewarded are held near the frozen start, so the note types and style do not drift to the average.
"""
from bisect import bisect_left
from collections import Counter

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventState, NEW
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.condition_scst import prefix_sequence

SUPPORT_TOLERANCE_US = 20_000
MIN_GAP_US = 60_000
MIN_HOLD_GAP_US = 100_000
MAX_HOLDS = 2
MAX_FINGERS = 2
DEFAULT_FIELD_TEMPERATURE = {'kind': 1.0, 'head_block': .7, 'head_cell': .7, 'head_ms': .7}


def walk_groups(tokens, request, vocab):
    """Token positions of each onset group: the action that chose the time, and each member's actions."""
    state, groups, current, head = EventState(request, vocab), [], None, None
    for position, token in enumerate(tokens):
        field, value = vocab.decision(token)
        if field == 'head_block':
            head = position
        elif field == 'head_cell':
            head = position
        state.apply(token)
        if state.onset >= 0 and (current is None or current['onset_us'] != state.onset):
            current = dict(onset_us=state.onset, head=head, members=[])
            groups.append(current)
        if current is None:
            continue
        if field == 'member' and value == NEW:
            current['members'].append(dict(new=position, kind=None, type=None, end_us=None))
        elif field == 'kind':
            current['members'][-1].update(kind=position, type=state.kind)
        elif field == 'tail_ms' and request.mode == 'phira':
            current['members'][-1]['end_us'] = state.events[-1].end_us
    return groups


def rule_faults(groups):
    """(position, rule) for each Phira action that breaks a rule. Each fault is credited to the action that made it."""
    faults, last_onset, holds, last_hold = [], None, [], None
    for group in groups:
        t = group['onset_us']
        if last_onset is not None and t-last_onset < MIN_GAP_US:
            faults.append((group['head'], 'close_onset'))
        last_onset = t
        holds = [end for end in holds if end > t]
        special = None
        for order, member in enumerate(group['members']):
            if order >= MAX_FINGERS:
                faults.append((member['new'], 'third_finger'))
            kind = member['type']
            if kind is None or kind == 'tap':
                continue
            if special is not None and kind != special:
                faults.append((member['kind'], 'mixed_special'))
            special = special or kind
            if kind == 'hold':
                if len(holds) >= MAX_HOLDS or (last_hold is not None and 0 <= t-last_hold < MIN_HOLD_GAP_US):
                    faults.append((member['kind'], 'stacked_hold'))
                last_hold = t
                if member['end_us'] is not None:
                    holds.append(member['end_us'])
    return faults


def supported(onset, others, tolerance=SUPPORT_TOLERANCE_US):
    for times in others:
        k = bisect_left(times, onset-tolerance)
        if k < len(times) and times[k] <= onset+tolerance:
            return True
    return False


def peer_votes(onset, others):
    return sum(supported(onset, [times]) for times in others)


def peer_faults(sampled, min_support=1):
    """sampled: the group lists of the K samples of one window. Returns, per sample, the onset actions backed by fewer
    than min_support other samples. The product keeps an onset at 3 votes of 8, its own plus 2 others: min_support=2."""
    times = [sorted(g['onset_us'] for g in groups) for groups in sampled]
    out = []
    for i, groups in enumerate(sampled):
        others = [t for j, t in enumerate(times) if j != i]
        out.append([(g['head'], 'no_support') for g in groups if peer_votes(g['onset_us'], others) < min_support])
    return out


def peer_rewarded(sampled, majority=.5):
    """Group-relative half (GRPO): an onset most of the other samples also chose gets its time action reinforced.
    Only the time of a note, never its type: this must not push note types toward the average."""
    times = [sorted(g['onset_us'] for g in groups) for groups in sampled]
    out = []
    for i, groups in enumerate(sampled):
        others = [t for j, t in enumerate(times) if j != i]
        rewarded = []
        for g in groups:
            if others and peer_votes(g['onset_us'], others)/len(others) >= majority:
                rewarded.append(g['head'])
        out.append(rewarded)
    return out


def score_window(token_lists, request, vocab, rules=True, reward_majority=None, min_support=1):
    """Punished and rewarded positions per sample, and a count of why. A punished action is never also rewarded."""
    sampled = [walk_groups(tokens, request, vocab) for tokens in token_lists]
    peers = peer_faults(sampled, min_support)
    rewarded = peer_rewarded(sampled, reward_majority) if reward_majority is not None else [[] for _ in sampled]
    punished, kept, counts = [], [], Counter()
    for groups, faults, good in zip(sampled, peers, rewarded):
        all_faults = list(faults)
        if rules and request.mode == 'phira':
            all_faults += rule_faults(groups)
        counts.update(rule for _, rule in all_faults)
        counts['onsets'] += len(groups)
        bad = sorted({position for position, _ in all_faults})
        good = sorted(set(good)-set(bad))
        counts['majority'] += len(good)
        punished.append(bad)
        kept.append(good)
    return (punished, counts) if reward_majority is None else (punished, kept, counts)


def collate_prefixes(items, vocab, device):
    length = max(len(arrays['targets']) for arrays in items)
    rows = len(items)
    out = dict(tokens=np.zeros((rows, length), np.int64), fields=np.zeros((rows, length), np.int64),
               times=np.zeros((rows, length), np.float32), targets=np.zeros((rows, length), np.int64),
               allowed=np.zeros((rows, length, vocab.size), bool), valid=np.zeros((rows, length), bool))
    out['allowed'][..., 0] = True  # padding keeps a finite softmax; it is never gathered
    for row, arrays in enumerate(items):
        n = len(arrays['targets'])
        for key in out:
            out[key][row, :n] = arrays[key]
    return {key: torch.from_numpy(value).to(device) for key, value in out.items()}


def anchor_kl(log_probs, teacher_log_probs, mask):
    """Reverse KL(student || frozen start) summed over the masked actions, on the student's own rollout (MOPD's
    objective, arXiv 2606.30406, in its exact form). Illegal tokens carry -inf log p and zero probability."""
    probs = log_probs.exp()
    gap = (log_probs-teacher_log_probs).masked_fill(~torch.isfinite(log_probs), 0.)
    return ((probs*gap).sum(-1)*mask).sum()


def replay_punished(model, inputs, rows, sequence, punished, amp, rewarded=None, reward_weight=0., teacher=None, anchor_weight=0.):
    """Differentiable objective of one chunk of replays: sum of log p of the punished draws, minus the weighted
    log p of the rewarded draws. Minimising it lowers the first and raises the second. With a frozen teacher, every
    other action is also held near where it started."""
    picked = {key: value[rows] if torch.is_tensor(value) else value for key, value in inputs.items()}
    with amp():
        logits = model(picked, sequence)
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(picked, sequence)
    log_probs = logits.float().masked_fill(~sequence['allowed'], -torch.inf).log_softmax(-1)
    total = log_probs.new_zeros(())
    if teacher_logits is not None:
        teacher_log_probs = teacher_logits.float().masked_fill(~sequence['allowed'], -torch.inf).log_softmax(-1)
        mask = sequence['valid'].clone()
        for row, positions in enumerate(punished):
            mask[row, torch.tensor(list(positions)+list(rewarded[row] if rewarded else []), dtype=torch.long, device=mask.device)] = False
        total = total+anchor_weight*anchor_kl(log_probs, teacher_log_probs, mask)
    for row, positions in enumerate(punished):
        index = torch.tensor(positions, dtype=torch.long, device=log_probs.device)
        total = total+log_probs[row, index, sequence['targets'][row, index]].sum()
        if rewarded is not None and rewarded[row]:
            index = torch.tensor(rewarded[row], device=log_probs.device)
            total = total-reward_weight*log_probs[row, index, sequence['targets'][row, index]].sum()
    return total


class PeerRL:
    """One sampling round per call: adds the gradient of the penalty, returns counts. The caller owns the optimizer step."""

    def __init__(self, settings, vocab):
        self.vocab = vocab
        self.samples = int(settings.get('samples', 8))
        self.weight = float(settings.get('weight', 1.))
        self.chunk = int(settings.get('replay_chunk', 16))
        self.max_replay = int(settings.get('max_replay_rows', 48))
        self.rules = bool(settings.get('rules', True))
        self.field_temperature = dict(settings.get('field_temperature', DEFAULT_FIELD_TEMPERATURE))
        self.max_tokens = int(settings.get('max_tokens', 2048))
        self.reward_majority = settings.get('reward_majority')
        self.reward_weight = float(settings.get('reward_weight', 0.))
        self.min_support = int(settings.get('min_support', 1))
        # Sampling is most of the cost. At this learning rate a few steps barely move the policy, so one round is
        # replayed for `reuse` updates, each on a fresh random subset; no importance ratio is applied.
        self.reuse = int(settings.get('reuse', 1))
        self.held, self.held_left = None, 0
        self.anchor_weight = float(settings.get('anchor_weight', 0.))
        self.teacher = None  # frozen copy of the start, set by the caller when anchor_weight > 0

    @torch.no_grad()
    def sample(self, model, inputs, requests, valid_lengths, seed):
        device = next(model.parameters()).device
        rows = len(requests)
        wanted = [r for r in range(rows) for _ in range(self.samples)]
        limit = min(self.max_tokens, max(256, 2*int(max(valid_lengths))+64))
        generators = []
        for k in range(len(wanted)):
            generator = torch.Generator(device=device.type if device.type in ('cuda', 'mps') else 'cpu')
            generators.append(generator.manual_seed(int(seed)*1009+k))
        was_training = model.training
        model.eval()
        try:
            outputs = generate_batch(model, inputs, [requests[r] for r in wanted], limit, temperature=1.,
                                     field_temperature=self.field_temperature, generator=generators,
                                     memory_index=wanted, cuda_graph=False)
        finally:
            model.train(was_training)
        return [outputs[r*self.samples:(r+1)*self.samples] for r in range(rows)]

    def accumulate(self, model, inputs, requests, valid_lengths, seed, amp, restore_train=None):
        counts, replay, tokens = Counter(), [], 0
        if self.held_left > 0:
            inputs, requests, drawn = self.held
            self.held_left -= 1
        else:
            drawn = self.sample(model, inputs, requests, valid_lengths, seed)
            if restore_train is not None:
                restore_train()
            self.held, self.held_left = (inputs, requests, drawn), self.reuse-1
            counts['rounds'] += 1
        for row, (request, outputs) in enumerate(zip(requests, drawn)):
            token_lists = [list(o['tokens']) if not o['failure'] else [] for o in outputs]
            if self.reward_majority is None:
                punished, window = score_window(token_lists, request, self.vocab, rules=self.rules, min_support=self.min_support)
                rewarded = [[] for _ in punished]
            else:
                punished, rewarded, window = score_window(token_lists, request, self.vocab, rules=self.rules,
                                                          reward_majority=float(self.reward_majority),
                                                          min_support=self.min_support)
            counts.update(window)
            counts['windows'] += 1
            counts['failures'] += sum(bool(o['failure']) for o in outputs)
            for tokens_i, positions, good in zip(token_lists, punished, rewarded):
                tokens += len(tokens_i)
                if positions or good:
                    replay.append((row, tokens_i, request, positions, good))
        counts['tokens'] += tokens
        if not replay or not tokens:
            return counts
        pool = len(replay)
        order = np.random.default_rng(int(seed)).permutation(pool)[:self.max_replay]
        replay = [replay[k] for k in order]
        counts['replayed'] += len(replay)
        last = lambda item: max(item[3]+item[4])  # noqa: E731
        replay.sort(key=last)
        device = next(model.parameters()).device
        # A cut to max_replay keeps the expected gradient: the rows that remain stand for the whole pool.
        scale = self.weight/tokens*pool/len(replay)
        loss_value = 0.
        for start in range(0, len(replay), self.chunk):
            part = replay[start:start+self.chunk]
            arrays = []
            for item in part:
                prefix, _ = prefix_sequence(item[1][:last(item)+1], item[2], self.vocab)
                arrays.append(prefix)
            sequence = collate_prefixes(arrays, self.vocab, device)
            rows = torch.tensor([item[0] for item in part], device=device)
            total = replay_punished(model, inputs, rows, sequence, [item[3] for item in part], amp,
                                    [item[4] for item in part], self.reward_weight, self.teacher, self.anchor_weight)
            (total*scale).backward()
            loss_value += float(total.detach())*scale
            del total, sequence
        counts['loss_milli'] += round(loss_value*1000)
        return counts

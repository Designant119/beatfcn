"""Frozen-model probes of the path from prior lane identity to the next group.

Targets, audio, times, reference evidence and physical legality constraints stay
fixed. Interventions are deliberate missing-information inputs, not playable
counterfactual charts. Sensitivity establishes dependence, not training benefit.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch

from . import features as F


@contextmanager
def history_intervention(model, batch, group_index, condition):
    allowed = ('baseline', 'k8_y_off', 'k8_gru_off', 'k8_both_off',
               'outer_group_lanes_unknown', 'past_lane_memory_unknown')
    if condition not in allowed:
        raise ValueError('unknown history probe condition')
    if batch['mode'].numel() != 1 or int(batch['mode'][0]) != 0:
        raise ValueError('history probe requires one mania core')
    if not 0 <= group_index < int(batch['grp_valid'][0].sum()):
        raise ValueError('history probe group outside valid targets')
    bt = dict(batch)
    cursor = batch['grp_t'][0, group_index - 1] if group_index else batch['grp_t'].new_tensor(0.)
    with ExitStack() as stack:
        names = {'k8_y_off': ('lane_history_y',), 'k8_gru_off': ('lane_history_gru',),
                 'k8_both_off': ('lane_history_y', 'lane_history_gru')}.get(condition, ())
        if condition == 'past_lane_memory_unknown':
            names = ('lane_history_y', 'lane_history_gru')
            bt['known'] = batch['known'].clone()
            past = batch['known_valid'] & (batch['known'][..., 0] <= cursor)
            bt['known'][..., 2] = torch.where(past, -1., batch['known'][..., 2])
            grid = batch['blk_grid'].float()
            # Preserve occupancy/counts while removing its lane identity in completed cells.
            cell_times = (batch['blk_t'][..., None] - F.BLOCK_US / F.SEC / 2 +
                          (torch.arange(F.BLOCK_CELLS, device=grid.device) + .5) * F.CELL_US / F.SEC)
            bt['blk_grid'] = torch.where((cell_times <= cursor)[..., None, None],
                                          grid.mean(-2, keepdim=True).expand_as(grid), grid)
        for name in names:
            module = getattr(model, name, None)
            if module is not None:
                handle = module.register_forward_hook(lambda _m, _args, output: torch.zeros_like(output))
                stack.callback(handle.remove)
        if condition in ('outer_group_lanes_unknown', 'past_lane_memory_unknown'):
            original = model.group_embedding
            def group_embedding(view):
                shadow = dict(view)
                shadow['note_lane'] = view['note_lane'].clone()
                shadow['note_lane'][:, :group_index] = -1
                return original(shadow)
            stack.enter_context(patch.object(model, 'group_embedding', group_embedding))
        yield bt


def history_probe(model, batch, group_index):
    """Compare fixed teacher-forced targets; return small auditable CPU summaries."""
    if model.training:
        raise ValueError('history probe needs eval mode and frozen dropout')
    G = batch['grp_valid'].shape[1]
    n = int(batch['grp_n'][0, group_index])
    if n < 1:
        raise ValueError('probe requires a nonempty target group')
    row = group_index
    if model.c.compact_inner:
        selected = torch.where(batch['inner_index'] == group_index)[0]
        if len(selected) != 1:
            raise ValueError('compacted target row missing')
        row = int(selected[0])
    used = batch['busy'][0, group_index].bool().clone()
    masks, targets = [], []
    for j in range(n):
        target = int(batch['note_lane'][0, group_index, j])
        masks.append(~used.clone())
        targets.append(target)
        used[target] = True
    masks = torch.stack(masks)
    targets = torch.tensor(targets, device=masks.device)
    if not bool(masks.gather(1, targets[:, None]).all()):
        raise ValueError('probe target is illegal under fixed constraints')
    choice = masks.sum(-1) > 1
    baseline = None
    results = {}
    conditions = ['baseline', 'outer_group_lanes_unknown', 'past_lane_memory_unknown']
    if model.c.lane_history_groups:
        conditions += ['k8_y_off', 'k8_gru_off', 'k8_both_off']
    for condition in conditions:
        captured = {'lane': []}
        original_encode, original_init = model.encode, model.gru_init_bg
        def encode(*args, **kwargs):
            output = original_encode(*args, **kwargs)
            captured['outer'] = output[0][0, group_index].detach().float()
            return output
        def init(*args, **kwargs):
            output = original_init(*args, **kwargs)
            captured['gru'] = output[0, group_index].detach().float()
            return output
        with torch.no_grad(), ExitStack() as stack:
            view = stack.enter_context(history_intervention(model, batch, group_index, condition))
            stack.enter_context(patch.object(model, 'encode', encode))
            stack.enter_context(patch.object(model, 'gru_init_bg', init))
            hook = model.lane_h.register_forward_hook(
                lambda _m, _a, output: captured['lane'].append(output[row].detach().float()))
            stack.callback(hook.remove)
            loss, _ = model(view)
        logits = torch.stack(captured['lane'][:n]).masked_fill(~masks, -1e9)
        probabilities = logits.softmax(-1)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(probabilities).all()):
            raise ValueError('nonfinite diagnostic forward')
        target_logp = logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]
        if baseline is None:
            baseline = {**captured, 'probabilities': probabilities}
        report = {'targets': n, 'choice_targets': int(choice.sum()),
                  'target_nll': float(-target_logp.mean()),
                  'choice_nll': float(-target_logp[choice].mean()) if bool(choice.any()) else None,
                  'choice_correct': int(((logits.argmax(-1) == targets) & choice).sum()),
                  'first_free_correct': int(((masks.long().argmax(-1) == targets) & choice).sum()),
                  'lane_probability_total_variation': float((probabilities - baseline['probabilities']).abs().sum(-1).mean() / 2),
                  'outer_state_rms_change': float((captured['outer'] - baseline['outer']).square().mean().sqrt()),
                  'gru_state_rms_change': float((captured['gru'] - baseline['gru']).square().mean().sqrt())}
        results[condition] = report
    return {'schema': 'beatfcn.history_path_probe.v1', 'group_index': group_index, 'conditions': results,
            'target_onset_s': float(batch['grp_t'][0, group_index]),
            'target_types': batch['note_type'][0, group_index, :n].tolist(),
            'busy_lanes': batch['busy'][0, group_index].tolist(),
            'scope': 'Prior lane identity in outer Y, known K and past song-grid cells; K8 split by adapter. '
                     'Reference, within-group teacher forcing, current busy mask and future observed context remain. '
                     'This is dependence, not a from-training ablation or proof of information decodability.',
            'training_updates': 0}

"""Read-only reconstruction of a plain AdamW step, without applying it."""
import torch


@torch.no_grad()
def adamw_proposal(parameter, gradient, state, group):
    if group.get('amsgrad') or group.get('maximize'):
        raise ValueError('Audit only supports ordinary minimizing AdamW')
    if gradient.shape != parameter.shape or state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape:
        raise ValueError('Optimizer parameter identity mismatch')
    if any(not bool(torch.isfinite(t).all()) for t in [parameter, gradient, state['exp_avg'], state['exp_avg_sq']]):
        raise ValueError('Nonfinite model, gradient or optimizer state')
    b1, b2 = group['betas']; step = float(state['step']) + 1
    mean = torch.lerp(state['exp_avg'], gradient, 1-b1)
    second = state['exp_avg_sq'] * b2 + gradient.square() * (1-b2)
    root = (second / (1-b2**step)).sqrt()
    delta = -group['lr'] * mean / (1-b1**step) / (root + group['eps'])
    delta = delta - group['lr'] * group['weight_decay'] * parameter
    history_fraction = torch.where(second > 0, state['exp_avg_sq'] * b2 / second, torch.zeros_like(second))
    return dict(delta=delta, epsilon_fraction=group['eps']/(root+group['eps']), history_second_moment_fraction=history_fraction)

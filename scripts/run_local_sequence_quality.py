"""Closed D87 experiment; preserve objective helper for regression evidence only.

The immutable original runner is in inputs/code_baselines/d87-closed-v1/.
D87 failed whole-song, condition and rest gates. Default execution is disabled.
"""
from beatfcn.model.full_event_model import event_loss


def training_objective(loss_mode, fixed_objective, logits, sequence, modes):
    if loss_mode == 'request_mean':
        return event_loss(logits, sequence)
    if loss_mode == 'fixed_mode_sequence':
        return fixed_objective(logits, sequence, modes)
    raise ValueError('Unknown objective; no implicit fallback')


def run(*args, **kwargs):
    raise RuntimeError(
        'D87 is closed after quality failure; training/resume is disabled. '
        'See reports/d87_candidate_disposition_v1.json and the immutable '
        'inputs/code_baselines/d87-closed-v1 reproduction source.'
    )


if __name__ == '__main__':
    run()

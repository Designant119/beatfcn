"""Observe Adam's lazy state without creating state or fabricating updates."""


def parameter_adam_step(optimizer, parameter):
    state = optimizer.state.get(parameter)
    if not state or 'step' not in state:
        return 0, False
    return int(state['step']), True


def optimizer_inventory(model, optimizer):
    rows = []
    for name, parameter in model.named_parameters():
        step, initialized = parameter_adam_step(optimizer, parameter)
        rows.append(dict(name=name, adam_step=step, adam_state_initialized=initialized))
    return rows

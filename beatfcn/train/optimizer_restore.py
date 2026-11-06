"""Explicit name-based optimizer continuation when adding input parameters."""
from copy import deepcopy


def restore_named_optimizer_state(optimizer, saved, original_names, current_names):
    """Preserve old moments/options, give new parameters fresh state, own copies.

    This deliberately supports the experiment's single-group optimizer only.
    Numeric parameter IDs are positional, so adding a parameter cannot safely
    be handled by appending an ID without checking parameter names.
    """
    if len(saved['param_groups']) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('Single optimizer group required')
    old_ids = saved['param_groups'][0]['params']
    new_ids = optimizer.state_dict()['param_groups'][0]['params']
    if (len(old_ids) != len(original_names) or len(new_ids) != len(current_names)
            or len(set(original_names)) != len(original_names) or len(set(current_names)) != len(current_names)
            or not set(original_names) <= set(current_names)):
        raise ValueError('Parameter identity mismatch')
    mapping = dict(zip(current_names, new_ids))
    state = {mapping[name]: deepcopy(saved['state'][old_id])
             for name, old_id in zip(original_names, old_ids) if old_id in saved['state']}
    group = deepcopy(saved['param_groups'][0]); group['params'] = new_ids
    if 'param_names' in group:
        group['param_names'] = list(current_names)
    optimizer.load_state_dict(dict(state=state, param_groups=[group]))
    return dict(restored_parameters=len(state), new_parameters=sorted(set(current_names)-set(original_names)))

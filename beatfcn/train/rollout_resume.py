"""Restore a validated completed-core prefix after a wall-time interruption."""
from copy import deepcopy


def resume_deadline_prefix(state):
    if state['complete'] or state['failure'] not in (
        'generation_deadline', 'evaluation_wall_limit',
    ):
        return None
    if state.get('legality'):
        raise ValueError('Cannot resume a prefix with legality failures')
    result = deepcopy(state)
    cores = result['cores']
    end, kept, offset = 0, [], 0
    for index, core in enumerate(cores):
        if core['start_us'] != end or not end < core['end_us'] <= state['duration_us']:
            raise ValueError('Noncontiguous or invalid core boundary')
        count = core['event_count']
        if not isinstance(count, int) or count < 0:
            raise ValueError('Invalid event count')
        events = result['events'][offset:offset + count]
        if len(events) != count or any(
            not core['start_us'] <= e['onset_us'] < core['end_us'] for e in events
        ):
            raise ValueError('Events do not match their recorded core')
        offset += count
        if not core['complete']:
            if index != len(cores) - 1 or not core.get('deadline_reached'):
                raise ValueError('Only a final deadline core can be discarded')
        else:
            kept.append(core)
            end = core['end_us']
    if offset != len(result['events']) or end != state['next_start_us']:
        raise ValueError('Prefix counters disagree with recorded events or cursor')
    if end >= state['duration_us']:
        raise ValueError('Incomplete state has no remaining core')
    result['cores'] = kept
    # Slice by multiplicity, preserving duplicate notes and cross-boundary holds.
    result['events'] = result['events'][:sum(c['event_count'] for c in kept)]
    result['failure'] = None
    return result

"""CPU-only preparation of real, contradictory same-prefix condition pairs."""
from collections import defaultdict
import json

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import CONTROL_FIELDS, paired_control_views, training_labels


def common_fork(a, b):
    stop = min(len(a['targets']), len(b['targets']))
    differing = np.flatnonzero(a['targets'][:stop] != b['targets'][:stop])
    if not len(differing):
        return None
    position = int(differing[0])
    for key in ('tokens', 'fields', 'times', 'valid', 'allowed'):
        if not np.array_equal(a[key][:position+1], b[key][:position+1]):
            raise ValueError('Pair differs in visible prefix, time, validity or legal support')
    targets = [int(x['targets'][position]) for x in (a, b)]
    if not all(bool(x['allowed'][position, t]) for x in (a, b) for t in targets):
        raise ValueError('Both real next targets must be legal under either condition')
    return position, targets


def build_pair_catalog(panel):
    groups = defaultdict(list)
    for index, row in enumerate(panel.records):
        if row['split'] == 'train':
            groups[(row['mode'], row['audio_sha256'], json.dumps(row['request'], sort_keys=True))].append(index)
    pairs, checked = [], {}

    def events(index):
        if index not in checked:
            row = panel.records[index]
            incoming = [RhythmEvent(**v) for v in row['request']['known_carry']]
            result = canonical(incoming + panel.targets(index), row['mode'])
            scope = (row['request']['start_us'], row['request']['end_us'])
            bands, _ = training_labels(row['mode'], result, scope)
            if {k:list(v) for k,v in bands.items()} != row['bands']:
                raise ValueError('Recomputed scoped labels differ from frozen tape: ' + row['chart_id'])
            checked[index] = result
        return checked[index]

    for indices in groups.values():
        for offset, a in enumerate(indices):
            ra = panel.records[a]
            for b in indices[offset+1:]:
                rb = panel.records[b]
                if ra['chart_id'] == rb['chart_id']:
                    continue
                fields = [f for f in CONTROL_FIELDS if f in ra['bands'] and f in rb['bands'] and ra['bands'][f] != rb['bands'][f]]
                if not fields:
                    continue
                branch = common_fork(panel.sequence(a), panel.sequence(b))
                if branch is None:
                    continue
                fork, targets = branch
                scope = (ra['request']['start_us'], ra['request']['end_us'])
                for field in fields:
                    # This also recomputes occupancy from incoming visible holds.
                    view = paired_control_views(ra['mode'], events(a), events(b), scope, field, origin_us=scope[0])
                    pairs.append(dict(indices=[a,b], field=field, mode=ra['mode'], component=ra['component'],
                        chart_ids=[ra['chart_id'],rb['chart_id']], strata=[ra['stratum'],rb['stratum']],
                        fork=fork, targets=targets, controls=view['controls'].tolist(),
                        actual=view['actual'], bands=[list(v) for v in view['bands']], scope_us=list(scope)))
    return pairs, sorted(checked)


def paired_exposure(catalog, updates, *, seed):
    """One auxiliary pair per update; 3 Phira to 1 mania, field/song balanced."""
    if updates < 1:
        raise ValueError('Positive update budget required')
    rng = np.random.default_rng(seed)
    buckets = defaultdict(lambda: defaultdict(list))
    for index, pair in enumerate(catalog):
        buckets[(pair['mode'], pair['field'])][pair['component']].append(index)
    fields = {mode:[f for f in CONTROL_FIELDS if (mode,f) in buckets] for mode in ('phira','mania_4k')}
    if any(not v for v in fields.values()):
        raise ValueError('Both modes need verified real pair support')
    songs = {}
    for key, groups in buckets.items():
        songs[key] = sorted(groups)
        rng.shuffle(songs[key])
        for ids in groups.values():
            rng.shuffle(ids)
    mode_counts, field_counts, song_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    steps = []
    for step in range(1, updates+1):
        mode = 'mania_4k' if step % 4 == 0 else 'phira'
        field = fields[mode][mode_counts[mode] % len(fields[mode])]
        key = (mode,field)
        song = songs[key][field_counts[key] % len(songs[key])]
        ids = buckets[key][song]
        pair = ids[song_counts[(key,song)] % len(ids)]
        steps.append(dict(effective_update=step,pair=pair))
        mode_counts[mode] += 1; field_counts[key] += 1; song_counts[(key,song)] += 1
    return steps

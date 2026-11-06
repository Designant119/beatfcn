"""Opt-in real-positive-pair schedules; the production sampler is unchanged."""
from collections import Counter, defaultdict
from dataclasses import replace
import random

import numpy as np

from ..events.controls import MODE_FIELDS
from ..model.features import CONTROL_FIELDS


def condition_view(example, field):
    """Keep the genuine target; select all conditions or one observed field."""
    if field == 'all':
        return replace(example, controls=example.controls.copy())
    index = CONTROL_FIELDS.index(field)
    if example.controls[index] < 0:
        raise ValueError('single-field positive target has no condition label')
    controls = np.full_like(example.controls, -1)
    controls[index] = example.controls[index]
    return replace(example, controls=controls)


def matched_schedules(pairs, *, steps=1000, block_steps=100, seed=20261025):
    """Equal targets/views per block; only pairing within an update differs.

    The paired arm sees two real targets from the same audio/window under the
    same condition mask. The control gets the same rows, with second rows
    deranged across independent audio inside each mode/field bucket.
    """
    if steps <= 0 or steps % block_steps or block_steps != 100:
        raise ValueError('use complete 100-update comparison blocks')
    buckets = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        for field in ['all', *pair['changed_fields']]:
            if field != 'all' and field not in MODE_FIELDS[pair['mode']]:
                raise ValueError('inapplicable condition field')
            buckets[(pair['mode'],field)][pair['audio_sha256']].append(pair)
    rng = random.Random(seed)
    paired, unpaired, audits = [], [], []
    modes = ['mania_4k','phira']
    for start in range(0,steps,block_steps):
        positions = defaultdict(list)
        block = []
        for offset in range(block_steps):
            step = start+offset
            mode = modes[step % 2]
            fields = ['all', *MODE_FIELDS[mode]]
            field = fields[(step//2) % len(fields)]
            positions[(mode,field)].append(offset)
            block.append(None)
        for key, slots in positions.items():
            available = buckets[key]
            audio = sorted(available)
            rng.shuffle(audio)
            if len(audio)<len(slots):
                raise ValueError(f'{key} needs {len(slots)} independent audio, has {len(audio)}')
            for slot, sha in zip(slots,audio):
                pair = rng.choice(available[sha])
                sides = [0,1]
                rng.shuffle(sides)
                block[slot] = [dict(pair_id=pair['pair_id'],side=side,field=key[1],
                    mode=key[0],audio_sha256=sha) for side in sides]
        control = [[dict(item) for item in batch] for batch in block]
        for slots in positions.values():
            for slot, source in zip(slots,slots[1:]+slots[:1]):
                control[slot][1] = dict(block[source][1])
        identity = lambda item: (item['pair_id'],item['side'],item['field'])
        left = Counter(identity(item) for batch in block for item in batch)
        right = Counter(identity(item) for batch in control for item in batch)
        if left != right: raise AssertionError('arm exposure differs')
        for batch in control:
            a,b = batch
            if a['audio_sha256']==b['audio_sha256'] or (a['mode'],a['field'])!=(b['mode'],b['field']):
                raise AssertionError('unpaired control failed its independent-audio contract')
        audits.append(dict(start_step=start+1,end_step=start+block_steps,
            identical_example_view_multiset=True,cores_per_arm=2*block_steps,
            distinct_example_views=len(left)))
        paired.extend(block);unpaired.extend(control)
    return dict(paired=paired,unpaired=unpaired,blocks=audits)

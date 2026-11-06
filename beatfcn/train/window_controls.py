"""Full-window label coverage experiment; targets and ordinary requests stay fixed."""
import math

import torch

from ..events import controls as C
from ..model import features as F

POLICIES=('density_only','all_applicable')


def encode_window_controls(mode, raw):
    if set(raw)!=set(C.MODE_FIELDS[mode]):raise ValueError('window control coverage differs from mode')
    result=[]
    for field in F.CONTROL_FIELDS:
        value=raw.get(field)
        if value is not None and not math.isfinite(value):raise ValueError('nonfinite window control')
        band=C.band_of(field,value)
        result.append(-1 if band is None else C.BAND_EDGES[field].index(band[0]))
    return result


def window_control_view(batch, policy):
    if policy not in POLICIES:raise ValueError('unknown window control policy')
    result=dict(batch)
    if policy=='all_applicable' and batch.get('_density_cf',False):
        labels=batch.get('full_window_controls')
        original=batch['controls']
        if labels is None:raise ValueError('full window labels missing; regenerate current-version tape')
        if labels.shape!=original.shape or labels.dtype!=torch.long or labels.device!=original.device:
            raise ValueError('full window labels have incompatible shape, dtype or device')
        result['controls']=labels
    return result

"""Inference-only interventions; targets and legal regions stay fixed."""
from contextlib import contextmanager
from dataclasses import replace
import numpy as np

from beatfcn.events.requests import build_reference
from beatfcn.model.batch import SongCache
from beatfcn.train.data import QuantArray


@contextmanager
def without_recent_history(model):
    """Remove both added K-group adapters, preserving ordinary history and masks."""
    import torch
    modules = [getattr(model, name, None) for name in ('lane_history_y', 'lane_history_gru')]
    weights = [(m.weight, m.weight.detach().clone()) for m in modules if m is not None]
    try:
        with torch.no_grad():
            for weight, _ in weights:
                weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for weight, saved in weights:
                weight.copy_(saved)


def swap_reference(sample, donor_events):
    """Same-audio alternate chart, same spans and other inputs; rebuild negative evidence."""
    evidence = []
    references = []
    for i, (old, public) in enumerate(zip(sample.evidence, sample.request['references'], strict=True)):
        new = build_reference(donor_events, old['span'], old['coverage'],
                              np.random.default_rng(sample.request['seed'] + i), old['reference_id'])
        if tuple(new['span']) != tuple(old['span']):
            raise ValueError('alternate reference exceeds the fixed span capacity')
        evidence.append(new)
        references.append({**public, 'events': new['selected']})
    if not evidence:
        raise ValueError('reference swap requires reference evidence')
    return replace(sample, evidence=tuple(evidence), request={**sample.request, 'references': references})


def controlled_reference_swap(sample, donor_events):
    """Two positive-only rhythmic references with genuine content contrast.

    No resampled negative rows can masquerade as style response. Spatial reference
    grants are cleared in both arms; a spatial swap needs independently authorized
    source scenes and fresh projections, not the original chart's cached geometry.
    """
    from copy import deepcopy
    from ..events.reference_policy import controlled_pair, content_fingerprint
    swapped = swap_reference(sample, donor_events)
    pair = controlled_pair(sample.evidence, swapped.evidence)
    result = []
    for original, evidence in zip((sample, swapped), pair):
        request = deepcopy(original.request)
        for public, ref in zip(request['references'], evidence, strict=True):
            public['events'] = ref['selected']
            public['space_attributes'] = False
            public.pop('spatial_source_sha256', None)
        result.append(replace(original, request=request, evidence=tuple(evidence),
                              recipe={**original.recipe, 'reference_diagnostic': 'positive_only_rhythmic_v2',
                                      'positive_content_sha256': content_fingerprint(evidence)}))
    return tuple(result)


def mismatch_audio(target, donor):
    """Resample all donor features to target dimensions; no zero padding or changed masks.

    This is a deliberately inconsistent audio intervention, not a natural time-stretched
    recording. The duration warp is reported by the caller; metrics use target audio.
    """
    def resized(source, count):
        index = np.minimum((np.arange(count) * len(source) / count).astype(int), len(source) - 1)
        if hasattr(source, 'raw'):
            return QuantArray(np.asarray(source.raw[index]).copy(), source.scale)
        return np.asarray(source[index]).copy()
    return SongCache({k: resized(donor.taps[k], len(v)) for k, v in target.taps.items()},
                     resized(donor.summary, len(target.summary)),
                     resized(donor.mel10, len(target.mel10)), target.duration_us)

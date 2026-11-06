"""Extend an exported FullEventModel file with the zero-initialised charter-timing input.

The runner's ``init_weights`` loads ``saved['model']`` strictly (the parameter
name set must equal the new model's). This writes a NEW file whose config has
``beat_features=True`` and whose state dict(s) also hold ``beat_projection.weight``
/ ``.bias`` as zeros, so ``init_weights`` works unchanged and the extended model
initially behaves exactly like the source model (beat features are added
through a zero projection; with timing absent OR present the output is
identical until training moves the new weights).

  .venv/bin/python -m scripts.extend_checkpoint SRC.pt DST.pt

``model`` and (when present) ``ema['model']`` are extended; optimizer / scaler /
RNG state are dropped (the parameter list changed; init_weights never reads
them). Never overwrites; verifies names/shapes against a meta-device model.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.model.full_event_codec import EventVocabulary  # noqa: E402
from beatfcn.model.full_event_model import BEAT_FEATURE_DIM, FullEventConfig, FullEventModel  # noqa: E402

NEW_KEYS = ('beat_projection.weight', 'beat_projection.bias')
DROPPED = ('optimizer', 'scaler', 'cpu_rng', 'cuda_rng', 'mps_rng')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def reference_shapes(config, vocabulary):
    """Parameter/buffer name -> shape of a model with ``config`` (meta device: no memory, no RNG use)."""
    with torch.device('meta'):
        model = FullEventModel(FullEventConfig(**config), EventVocabulary(**(vocabulary or {})))
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def extend_state(state, d):
    if any(k in state for k in NEW_KEYS):
        raise ValueError('State dict already has beat_projection parameters')
    like = state['mel_projection.weight']
    out = dict(state)
    out['beat_projection.weight'] = torch.zeros((d, BEAT_FEATURE_DIM), dtype=like.dtype)
    out['beat_projection.bias'] = torch.zeros((d,), dtype=state.get('mel_projection.bias', like).dtype)
    return out


def extend_checkpoint(source, destination, *, verify=True):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists; never overwritten')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if 'model' not in saved or 'config' not in saved:
        raise ValueError("Expected an exported model file {'model': state_dict, 'config': ...}")
    config = dict(saved['config'])
    if config.get('beat_features'):
        raise ValueError('Source checkpoint already has beat_features=True')
    if not config.get('mel_bins'):
        raise ValueError('beat_features need a mel stream (config.mel_bins > 0)')
    config['beat_features'] = True
    config = asdict(FullEventConfig(**config))  # validates and fills defaults
    out = {k: v for k, v in saved.items() if k not in DROPPED}
    out['config'] = config
    out['model'] = extend_state(saved['model'], config['d'])
    if isinstance(saved.get('ema'), dict) and isinstance(saved['ema'].get('model'), dict):
        out['ema'] = dict(saved['ema'], model=extend_state(saved['ema']['model'], config['d']))
    out['extended_from'] = dict(path=str(source), sha256=file_sha(source), added=list(NEW_KEYS),
                                init='zeros (beat_projection adds exactly 0 until trained)',
                                dropped=[k for k in DROPPED if k in saved])
    if verify:
        expected = reference_shapes(config, saved.get('vocabulary'))
        for name, state in [('model', out['model'])] + ([('ema', out['ema']['model'])] if 'ema' in out else []):
            got = {k: tuple(v.shape) for k, v in state.items()}
            if set(got) != set(expected):
                raise ValueError(f'{name}: extended names differ from a beat_features model: '
                                 f'missing {sorted(set(expected)-set(got))[:5]}, extra {sorted(set(got)-set(expected))[:5]}')
            bad = [k for k in got if got[k] != expected[k]]
            if bad:
                raise ValueError(f'{name}: shape mismatch for {bad[:5]}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name+'.tmp')
    torch.save(out, tmp)
    tmp.replace(destination)
    return dict(source=str(source), destination=str(destination), sha256=file_sha(destination),
                added=list(NEW_KEYS), ema='ema' in out, dropped=out['extended_from']['dropped'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--no-verify', action='store_true', help='skip the meta-device name/shape check')
    args = p.parse_args(argv)
    import json
    print(json.dumps(extend_checkpoint(args.source, args.destination, verify=not args.no_verify), indent=1))


if __name__ == '__main__':
    main()

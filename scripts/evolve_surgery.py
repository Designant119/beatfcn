"""Candidate surgery on an r5-format checkpoint: fewer decoder layers and/or a narrower decoder FFN.

Key remap plus config change only; no optimizer state. Prints parameters and an estimate of MACs per audio second.

  python -m scripts.evolve_surgery --in /data/beatfcn-r5-l322m.pt --out CAND.pt --depth 8 [--pattern even] [--ffn 1792]
  python -m scripts.evolve_surgery --plan reports/evolve-20261104/plans/C2-depth8.json
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beatfcn.model.full_event_codec import EventVocabulary  # noqa: E402
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel  # noqa: E402

LAYER = re.compile(r'^decoder\.(\d+)\.(.+)$')
# Panel means from report E section 1 (cold, r5): one 8 s core.
CORE_TOKENS, HISTORY_N, LOCAL_ROWS, SONG_ROWS, MEMORY_ROWS, CORE_S = 291, 512, 2000, 338, 2430, 8.


def kept_layers(depth, target, pattern='even'):
    """Sorted layer indices. ``even``: even spacing that always keeps the first and the last layer."""
    if pattern != 'even':
        raise ValueError('Unknown kept-layer pattern: '+pattern)
    if target > depth or target < 2:
        raise ValueError('Depth must lie in [2, current depth]')
    return [int(round(i)) for i in np.linspace(0, depth-1, target)]


def shrink_units(up, down, target):
    """Keep the top-norm gated units of one FeedForward; returns (up, down, kept unit indices).

    Unit i is rows i (gate) and ffn+i (value) of ``up`` and column i of ``down``; it contributes
    silu(a_i)*b_i*down_i, so its norm is the product of the three."""
    ffn = down.shape[1]
    score = up[:ffn].norm(dim=1)*up[ffn:].norm(dim=1)*down.norm(dim=0)
    keep = score.float().topk(target).indices.sort().values
    return torch.cat((up[keep], up[ffn+keep])), down[:, keep], keep


def apply_surgery(state, config, *, depth=None, ffn=None, pattern='even'):
    """(new state, new config dict, record); the input is left untouched and unchanged tensors are shared."""
    c = FullEventConfig(**config)
    if c.history_mixer != 'attention':
        raise ValueError('Surgery supports the attention decoder only')
    target_depth = depth or c.decoder_layers
    keep = kept_layers(c.decoder_layers, target_depth, pattern)
    index = {old: new for new, old in enumerate(keep)}
    current_ffn = c.decoder_ffn or c.ffn
    if ffn is not None and not 0 < ffn <= current_ffn:
        raise ValueError('decoder_ffn can only shrink')
    target_ffn = ffn or current_ffn
    out, kept_units = {}, {}
    for key, value in state.items():
        match = LAYER.match(key)
        if match is None:
            out[key] = value
        elif int(match[1]) in index:
            out[f'decoder.{index[int(match[1])]}.{match[2]}'] = value
    if target_ffn != current_ffn:
        for new, old in enumerate(keep):
            up, down, units = shrink_units(out[f'decoder.{new}.ffn.up.weight'], out[f'decoder.{new}.ffn.down.weight'],
                                           target_ffn)
            out[f'decoder.{new}.ffn.up.weight'], out[f'decoder.{new}.ffn.down.weight'] = up, down
            kept_units[old] = int(units.shape[0])
    new_config = asdict(replace(c, decoder_layers=target_depth,
                                decoder_ffn=None if target_ffn == c.ffn and c.decoder_ffn is None else target_ffn))
    return out, new_config, dict(kept_layers=keep, pattern=pattern, from_depth=c.decoder_layers, depth=target_depth,
                                 from_decoder_ffn=current_ffn, decoder_ffn=target_ffn)


def macs_per_audio_second(config):
    """Report E's arithmetic, single sample, from the config only; a device-free proxy for decode cost."""
    c = FullEventConfig(**config)
    d, hd = c.d, c.d//c.heads
    kv = c.kv_heads*hd
    ffn_decoder = 3*d*(c.decoder_ffn or c.ffn)
    ffn_encoder = 3*d*c.ffn
    attend = 2*d
    self_projection = 2*d*d+2*d*kv
    encoder_row = ffn_encoder+self_projection
    encoder = c.encoder_layers*sum(rows*(encoder_row+attend*rows) for rows in (LOCAL_ROWS, SONG_ROWS, 1))
    memory_kv = c.decoder_layers*2*d*kv*MEMORY_ROWS
    layer_step = ffn_decoder+attend*MEMORY_ROWS+self_projection+2*d*d+attend*HISTORY_N
    decoder = CORE_TOKENS*c.decoder_layers*layer_step
    per_core = encoder+memory_kv+decoder
    return dict(macs_per_audio_second=per_core/CORE_S, encoder_share=encoder/per_core, memory_kv_share=memory_kv/per_core,
                decoder_share=decoder/per_core, decoder_row_step_macs=c.decoder_layers*layer_step)


def load_checkpoint_model(path, device, *, dtype=torch.float32):
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = FullEventConfig(**saved['config'])
    vocabulary = EventVocabulary(**saved.get('vocabulary', {}))
    model = FullEventModel(config, vocabulary).to(device)
    model.load_state_dict({k: v.to(dtype) for k, v in saved['model'].items()}, strict=True)
    del saved
    return model.eval()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(source, output, *, depth=None, ffn=None, pattern='even'):
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    state, config, record = apply_surgery(saved['model'], saved['config'], depth=depth, ffn=ffn,
                                          pattern=pattern)
    parameters = sum(v.numel() for v in state.values())
    record.update(source=str(source), source_sha256=file_sha(source), source_parameters=sum(
        v.numel() for v in saved['model'].values()))
    result = dict(schema=saved.get('schema', 'beatfcn.final-model.v1'), config=config,
                  vocabulary=saved.get('vocabulary', {}), weights=saved.get('weights', 'raw'), model=state,
                  parameters=parameters, provenance=saved.get('provenance'), surgery=record)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    torch.save(result, temporary)
    temporary.replace(output)
    summary = dict(output=str(output), parameters=parameters, source_parameters=record['source_parameters'],
                   kept_layers=record['kept_layers'], decoder_ffn=record['decoder_ffn'],
                   **macs_per_audio_second(config),
                   source_macs_per_audio_second=macs_per_audio_second(saved['config'])['macs_per_audio_second'])
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--plan', type=Path, help='evolve plan: surgery.source -> init')
    a.add_argument('--in', dest='source', type=Path)
    a.add_argument('--out', type=Path)
    a.add_argument('--depth', type=int)
    a.add_argument('--ffn', type=int)
    a.add_argument('--pattern', default='even')
    args = a.parse_args()
    if args.plan:
        spec = json.loads(args.plan.read_text())
        surgery = spec['surgery']
        if not surgery:
            a.error('This plan starts from an unchanged checkpoint; no surgery is needed')
        run(surgery['source'], spec['init'], depth=surgery.get('depth'), ffn=surgery.get('ffn'),
            pattern=surgery.get('pattern', 'even'))
    elif args.source and args.out:
        run(args.source, args.out, depth=args.depth, ffn=args.ffn, pattern=args.pattern)
    else:
        a.error('Give --plan, or --in and --out')


if __name__ == '__main__':
    main()

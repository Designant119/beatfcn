"""H100 step-cost attribution for recipe-v2 model flags at the 600M shape.

Times forward+backward (bf16 autocast, eager) of one realistic-shape microbatch
for flag subsets, and records which SDPA kernels run (efficient/flash vs math).
Synthetic tensors with the production shapes; timing only, no learning claim.
"""
import argparse
import json
import statistics
import time
from dataclasses import replace

import torch

from beatfcn.model.full_event_codec import EventVocabulary
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss

FLAGS = ('timed_intensity', 'timed_memory_values', 'onset_head')


def inputs(c, b, device, frames=400, mel=1600, song=400, cells=80):
    g = torch.Generator(device='cpu').manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    x = dict(local=r(b, c.taps, frames, c.bands, c.tap_dim),
             local_times=(torch.arange(frames)[None].float() * .04 - 4).repeat(b, 1),
             local_valid=torch.ones(b, frames, dtype=torch.bool),
             song=r(b, song, c.tap_dim), song_times=(torch.arange(song)[None].float() * .48).repeat(b, 1),
             song_valid=torch.ones(b, song, dtype=torch.bool),
             reference=r(b, 1, c.reference_dim), reference_times=torch.zeros(b, 1),
             reference_valid=torch.zeros(b, 1, dtype=torch.bool),
             controls=r(b, c.controls, c.control_dim), mode=torch.ones(b, dtype=torch.long),
             mel=r(b, mel, c.mel_bins), mel_times=(torch.arange(mel)[None].float() * .01 - 4).repeat(b, 1),
             mel_valid=torch.ones(b, mel, dtype=torch.bool),
             intensity=torch.rand(b, cells, c.intensity_dim, generator=g),
             intensity_times=(torch.arange(cells)[None].float() * .1 + .05).repeat(b, 1),
             intensity_valid=torch.ones(b, cells, dtype=torch.bool))
    return {k: v.to(device) for k, v in x.items()}


def sequence(vocab, b, n, device):
    g = torch.Generator(device='cpu').manual_seed(1)
    seq = dict(tokens=torch.randint(2, vocab.size, (b, n), generator=g),
               fields=torch.randint(0, 10, (b, n), generator=g),
               times=torch.rand(b, n, generator=g) * 8,
               valid=torch.ones(b, n, dtype=torch.bool),
               targets=torch.randint(2, vocab.size, (b, n), generator=g),
               allowed=torch.ones(b, n, vocab.size, dtype=torch.bool))
    return {k: v.to(device) for k, v in seq.items()}


def run(config, flags, *, batch, tokens, iters, device):
    c = replace(config, **{f: f in flags for f in FLAGS})
    vocab = EventVocabulary(relative_blocks=True, tail_query=True)
    model = FullEventModel(c, vocab).to(device).train()
    x, s = inputs(c, batch, device), sequence(vocab, batch, tokens, device)
    times = []
    for i in range(iters + 2):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = event_loss(model(x, s), s)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = event_loss(model(x, s), s)
        loss.backward()
        torch.cuda.synchronize()
    kernels = {}
    for e in prof.key_averages():
        name = e.key.lower()
        if any(k in name for k in ('fmha', 'flash', 'efficient', 'attention', 'softmax', 'bmm')):
            kernels[e.key[:80]] = round(getattr(e, 'device_time_total', getattr(e, 'cuda_time_total', 0)) / 1e3, 2)
    del model
    torch.cuda.empty_cache()
    return dict(flags=sorted(flags), median_s=round(statistics.median(times), 4),
                attention_kernels_ms=dict(sorted(kernels.items(), key=lambda kv: -kv[1])[:8]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale-config', default='bench/rhythm-v2/scale-600m-attention-v1.json')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--iters', type=int, default=6)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    base = FullEventConfig(**json.load(open(a.scale_config))['config'])
    rows = []
    for flags in [(), ('timed_intensity',), ('onset_head',), ('timed_memory_values',),
                  ('timed_intensity', 'onset_head'), FLAGS]:
        rows.append(run(base, set(flags), batch=a.batch, tokens=a.tokens, iters=a.iters, device='cuda'))
        print(json.dumps(rows[-1]), flush=True)
    ref = rows[0]['median_s']
    for row in rows:
        row['relative_to_old'] = round(row['median_s'] / ref, 3)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), rows=rows), indent=1) + '\n')
    print(json.dumps([(r['flags'], r['relative_to_old']) for r in rows]))


if __name__ == '__main__':
    main()

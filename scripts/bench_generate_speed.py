"""Decode-throughput benchmark: fast vs reference ``generate_batch``.

Random-init FullEventModel (no checkpoint, no data); the group-END logit gets
a constant -30 bias and earlier head times are favoured, so rows
run (nearly) to the token cap and throughput is
measured at a fixed step count. Memory length mimics an 8 s core of the
322M mainline (400 MERT frames + 1600 mel rows + song summary + timed
intensity cells + controls).

  .venv/bin/python scripts/bench_generate_speed.py --device cuda --batch-sizes 4,16,64
  .venv/bin/python scripts/bench_generate_speed.py --device cuda --check --cuda-graph off
  .venv/bin/python scripts/bench_generate_speed.py --device cpu --scale small --steps 64

``--check`` first verifies, on the same random model, that fast and reference
give identical greedy and per-row-sampled tokens and reports the max logit
difference of the static decode step; run it on a new GPU image before
trusting it for evaluations (graphs are on by default on CUDA; compare
``--cuda-graph on`` vs ``off``; BEATFCN_DECODE_CUDA_GRAPH=0 disables them).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatfcn.model.full_event_codec import EventRequest  # noqa: E402
from beatfcn.model.full_event_generate import generate_batch  # noqa: E402
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel  # noqa: E402

SCALES = dict(
    # 322M mainline decoder (reports/goal-100usd-20261101 config); encoder slimmed, it runs once per call.
    real=dict(cfg=dict(d=896, heads=14, kv_heads=2, ffn=3584, encoder_layers=1, decoder_layers=16, tap_dim=1024,
                       taps=2, bands=1, band_slots=1, dropout=0., mel_bins=128, intensity_dim=3, qk_norm=True,
                       timed_intensity=True), frames=400, mel=1600, song=100, cells=80),
    small=dict(cfg=dict(d=256, heads=8, kv_heads=2, ffn=768, encoder_layers=1, decoder_layers=6, tap_dim=64,
                        taps=2, bands=1, band_slots=1, dropout=0., mel_bins=32, intensity_dim=3, qk_norm=True,
                        timed_intensity=True), frames=400, mel=1600, song=100, cells=80),
    # Real depth/heads/GQA/memory length, negligible width: device compute ~0, so the
    # time is host overhead (op dispatch + grammar), the regime of the H100 run.
    overhead=dict(cfg=dict(d=112, heads=14, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=16, tap_dim=32,
                           taps=2, bands=1, band_slots=1, dropout=0., mel_bins=16, intensity_dim=3, qk_norm=True,
                           timed_intensity=True), frames=400, mel=1600, song=100, cells=80),
    tiny=dict(cfg=dict(d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2, tap_dim=32,
                       taps=2, bands=1, band_slots=1, dropout=0., mel_bins=16, intensity_dim=3, qk_norm=True,
                       timed_intensity=True), frames=100, mel=400, song=50, cells=80),
)


def build(scale, device, seed=0):
    spec = SCALES[scale]
    torch.manual_seed(seed)
    model = FullEventModel(FullEventConfig(**spec['cfg'])).to(device).eval()
    vocab = model.vocab
    bias = torch.zeros(vocab.size)
    bias[vocab.token('group', 0)] = -30.
    # Prefer the earliest legal head time so the onset cursor crawls and rows
    # keep emitting events until the step cap (a throughput, not quality, run).
    for field in ('head_block', 'head_cell', 'head_ms'):
        size = vocab.sizes[('group', 'head_block', 'head_cell', 'head_ms').index(field)]
        first = vocab.token(field, 0)
        bias[first:first + size] = -2. * torch.arange(size).float()
    bias = bias.to(device)
    model.output.register_forward_hook(lambda module, args, out: out + bias)
    return model, spec


def make_inputs(model, spec, rows, device, seed=1):
    c, g = model.c, torch.Generator().manual_seed(seed)
    t, mel, song, cells = spec['frames'], spec['mel'], spec['song'], spec['cells']
    ones = lambda n: torch.ones(rows, n, dtype=torch.bool)  # noqa: E731
    inputs = dict(
        local=torch.randn(rows, c.taps, t, c.bands, c.tap_dim, generator=g),
        local_times=(torch.arange(t).float() * .02 - 2.).repeat(rows, 1), local_valid=ones(t),
        mel=torch.randn(rows, mel, c.mel_bins, generator=g),
        mel_times=(torch.arange(mel).float() * .005 - 2.).repeat(rows, 1), mel_valid=ones(mel),
        song=torch.randn(rows, song, c.tap_dim, generator=g),
        song_times=(torch.arange(song).float() * 2. - 8.).repeat(rows, 1), song_valid=ones(song),
        reference=torch.randn(rows, 1, c.reference_dim, generator=g),
        reference_times=torch.zeros(rows, 1), reference_valid=torch.zeros(rows, 1, dtype=torch.bool),
        controls=torch.randn(rows, c.controls, c.control_dim, generator=g),
        mode=torch.tensor([i % 2 for i in range(rows)]),
        intensity=torch.rand(rows, cells, c.intensity_dim, generator=g),
        intensity_times=(torch.arange(cells).float() * .1 + .05).repeat(rows, 1),
        intensity_valid=torch.zeros(rows, cells, dtype=torch.bool))
    return {k: v.to(device) for k, v in inputs.items()}


def make_requests(rows):
    out = []
    for i in range(rows):
        mode = 'mania_4k' if i % 2 == 0 else 'phira'
        start = 8_000_000 * (i % 5)
        out.append(EventRequest(mode, start, start + 8_000_000, 180_000_000))
    return out


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def run(model, inputs, requests, steps, *, fast, temperature, cuda_graph, seed=0):
    device = inputs['local'].device
    kwargs = {}
    if temperature is not None:
        gen_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
        kwargs = dict(temperature=temperature,
                      generator=[torch.Generator(device=gen_device).manual_seed(seed + i) for i in range(len(requests))])
    stats = {}
    sync(device)
    tick = time.perf_counter()
    out = generate_batch(model, inputs, requests, steps, fast=fast, cuda_graph=cuda_graph, stats=stats, **kwargs)
    sync(device)
    wall = time.perf_counter() - tick
    tokens = sum(len(r['tokens']) for r in out)
    return out, dict(wall_s=wall, tokens=tokens, tokens_per_s=tokens / wall, per_row_tokens_per_s=tokens / wall / len(requests),
                     **{k: v for k, v in stats.items() if k in ('decode_s', 'grammar_s', 'select_s', 'steps', 'sampling',
                                                              'graph_captures', 'cuda_graph')})


def logit_check(model, inputs, tokens_rows, steps):
    """Teacher-force a generated stream through decode(cache=...) and decode_static."""
    from beatfcn.model.full_event_codec import BOS, EventState
    rows = len(tokens_rows)
    requests = make_requests(rows)
    states = [EventState(r, model.vocab) for r in requests]
    with torch.no_grad():
        memory = model.encode(inputs)
        cache, static = model.start_cache(memory), model.start_static_cache(memory, steps)
        pos = torch.zeros(1, dtype=torch.long, device=memory['values'].device)
        worst = 0.
        for step in range(steps):
            q = [s.query() for s in states]
            dev = memory['values'].device
            prev = torch.tensor([s.tokens[-1] if s.tokens else BOS for s in states], device=dev)[:, None]
            fields = torch.tensor([f for f, _ in q], device=dev)[:, None]
            times = torch.tensor([t for _, t in q], device=dev, dtype=torch.float32)[:, None]
            a = model.decode(prev, fields, times, torch.ones_like(prev, dtype=torch.bool), memory, cache=cache)
            pos.fill_(step)
            b = model.decode_static(prev, fields, times, static, pos, step + 1)
            worst = max(worst, (a.float() - b.float()).abs().max().item())
            for s, toks in zip(states, tokens_rows):
                if step < len(toks) and not s.done:
                    s.apply(toks[step])
            if all(step + 1 >= len(t) for t in tokens_rows):
                break
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--scale', choices=sorted(SCALES), default='real')
    p.add_argument('--batch-sizes', default='4,16,64')
    p.add_argument('--steps', type=int, default=256, help='tokens per row (END is suppressed)')
    p.add_argument('--reference-steps', type=int, help='steps for the reference loop (default --steps)')
    p.add_argument('--temperature', default='1', help="'none' = greedy")
    p.add_argument('--cuda-graph', choices=('auto', 'on', 'off'), default='auto',
                   help='fast-path CUDA-graph replay (auto = module default, on for CUDA)')
    p.add_argument('--skip-reference', action='store_true')
    p.add_argument('--check', action='store_true', help='equivalence check before timing')
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    device = torch.device(args.device)
    temperature = None if args.temperature in ('none', 'greedy') else float(args.temperature)
    args.cuda_graph = dict(auto=None, on=True, off=False)[args.cuda_graph]
    model, spec = build(args.scale, device)
    print(f'torch {torch.__version__} device={device} scale={args.scale} '
          f'params={sum(x.numel() for x in model.parameters())/1e6:.1f}M temperature={temperature} '
          f'cuda_graph={args.cuda_graph}', flush=True)
    report = dict(torch=torch.__version__, device=str(device), scale=args.scale, temperature=temperature,
                  cuda_graph=args.cuda_graph, check=None, runs=[])
    if args.check:
        rows, steps = 4, min(args.steps, 96)
        inputs, requests = make_inputs(model, spec, rows, device), make_requests(rows)
        result = {}
        for label, temp in (('greedy', None), ('sampled', 1.)):
            ref, _ = run(model, inputs, requests, steps, fast=False, temperature=temp, cuda_graph=False)
            fast, _ = run(model, inputs, requests, steps, fast=True, temperature=temp, cuda_graph=args.cuda_graph)
            result[label] = ref == fast
        result['max_logit_abs_diff'] = logit_check(model, inputs, [r['tokens'] for r in ref], steps)
        report['check'] = result
        print('check', json.dumps(result), flush=True)
        if not (result['greedy'] and result['sampled']):
            print('EQUIVALENCE CHECK FAILED', flush=True)
    for rows in [int(x) for x in args.batch_sizes.split(',')]:
        inputs, requests = make_inputs(model, spec, rows, device), make_requests(rows)
        run(model, inputs, requests, 4, fast=True, temperature=temperature, cuda_graph=args.cuda_graph)  # warm caches
        _, fast = run(model, inputs, requests, args.steps, fast=True, temperature=temperature, cuda_graph=args.cuda_graph)
        line = dict(batch=rows, fast=fast)
        if not args.skip_reference:
            run(model, inputs, requests, 2, fast=False, temperature=temperature, cuda_graph=False)
            _, ref = run(model, inputs, requests, args.reference_steps or args.steps, fast=False,
                         temperature=temperature, cuda_graph=False)
            line.update(reference=ref, speedup=fast['tokens_per_s'] / ref['tokens_per_s'])
        report['runs'].append(line)
        msg = (f"B={rows:3d} fast {fast['tokens_per_s']:9.1f} tok/s ({fast['per_row_tokens_per_s']:7.1f}/row) "
               f"[decode {fast['decode_s']:.2f}s grammar {fast['grammar_s']:.2f}s select+sync {fast['select_s']:.2f}s "
               f"{fast.get('sampling')} graph={fast.get('cuda_graph')}]")
        if 'reference' in line:
            msg += f" | ref {line['reference']['tokens_per_s']:9.1f} tok/s | x{line['speedup']:.1f}"
        print(msg, flush=True)
    from beatfcn.model.full_event_generate import _GRAPH_FAILURES
    report['graph_failures'] = list(_GRAPH_FAILURES)
    if _GRAPH_FAILURES:
        print('CUDA-graph fallbacks:', _GRAPH_FAILURES, flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

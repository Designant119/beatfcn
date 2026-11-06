"""G1 scheduler bench: the same few songs, each scheduler at each batch size; rate and token identity vs legacy.

  python -m scripts.bench_sched --checkpoint CKPT --plan PLAN.json --device cuda --json out.json
  python -m scripts.bench_sched --tiny --device cpu --batch-sizes 4,16 --samples 4

Seeds grow with B (at least 8; enough that pairs >= B), because a pair decodes only one core at a time and a batch
size above the number of pairs would measure nothing. Each arm stops at --arm-wall-s, so identity is compared on the
(seed, song, core) triples both arms finished; the rate is real tokens per wall second of the arm, encoder included.
The stats (decode_s, grammar_s, select_s, encoder_s, S, R, occupancy) go to --json for the cost fit.
"""
import argparse
import gc
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from beatfcn.model import full_event_generate as generate  # noqa: E402
from scripts import run_workflow_benchmark as W  # noqa: E402


class SyntheticFeatures:
    """Deterministic per-(audio, core) tensors; song length differs per song, so memory padding is real."""

    def __init__(self, config, songs):
        self.c, self.records = config, {s['audio_sha256']: dict(duration_us=s['duration_us']) for s in songs}
        self.song_rows = {s['audio_sha256']: 4 + index for index, s in enumerate(songs)}

    def window(self, audio, start_us, end_us):
        g = np.random.default_rng([ord(audio[0]), start_us//1000])
        frames = 8 + (end_us-start_us)//1_000_000
        rows = self.song_rows[audio]
        return dict(local=g.standard_normal((self.c.taps, frames, self.c.bands, self.c.tap_dim)).astype(np.float32),
                    local_times=(np.arange(frames)*.75-.5).astype(np.float32), local_valid=np.ones(frames, bool),
                    song=g.standard_normal((rows, self.c.tap_dim)).astype(np.float32),
                    song_times=(np.arange(rows)*4.-start_us/1e6).astype(np.float32), song_valid=np.ones(rows, bool))


def tiny_setup(songs):
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    torch.manual_seed(0)
    config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2, tap_dim=16, bands=4,
                             taps=3, dropout=0.)
    panel = dict(songs=[dict(song_id=f's{i}', audio_sha256=chr(ord('a')+i)*64, mode='mania_4k' if i % 2 == 0 else 'phira',
                             duration_us=(13_000_000 + 3_000_000*i)) for i in range(songs)], split='val')
    return FullEventModel(config).eval(), SyntheticFeatures(config, panel['songs']), panel, 0, 100_000


def checkpoint_setup(a, songs):
    args = SimpleNamespace(plan=a.plan, checkpoint=a.checkpoint, root=a.root, device=a.device, weights=a.weights)
    model, features, _, intensity_dim, cell_us = W.checkpoint_model(args)
    manifest = W.load_tape(a.root, a.tape)
    panel = W.freeze_panel(manifest, W.sha_file(a.root/a.tape))
    by_mode = {mode: [s for s in panel['songs'] if s['mode'] == mode] for mode in ('mania_4k', 'phira')}
    chosen = [s for pair in zip(*by_mode.values()) for s in pair][:songs]
    for song in chosen:
        song['duration_us'] = features.records[song['audio_sha256']]['duration_us']
    return model, features, dict(panel, songs=chosen), intensity_dim, cell_us


def arm_tokens(results):
    """{(seed, song, core start): tokens} of the cores an arm finished."""
    return {(seed, state['chart_id'], core['start_us']): core['tokens'] for seed, (result, _) in results
            for state in result['songs'] for core in state['cores'] if core['complete']}


def compare(reference, tokens):
    shared = sorted(set(reference) & set(tokens))
    different = [key for key in shared if reference[key] != tokens[key]]
    first = None
    if different:
        a, b = reference[different[0]], tokens[different[0]]
        first = dict(core=list(different[0]), position=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                                            min(len(a), len(b))))
    return dict(compared=len(shared), different=len(different), first=first)


def run_arm(a, model, features, panel, intensity_dim, cell_us, scheduler, batch, seeds):
    stats = {}
    args = SimpleNamespace(temperature=a.temperature, top_p=None, seed=seeds[0], device=a.device, batch_size=batch,
                           token_cap=a.token_cap, max_wall_s=a.arm_wall_s, scheduler=scheduler, decode_stats=stats,
                           autocast=a.autocast, compile_decode=a.compile_decode, field_temperature=None)
    generate._GRAPH_RUNNERS.clear()
    gc.collect()
    if a.device.startswith('cuda'):
        torch.cuda.empty_cache()
    tick = time.perf_counter()
    results = W.model_drafts_parallel(args, panel, model, features, intensity_dim=intensity_dim, cell_us=cell_us,
                                      seeds=seeds)
    if a.device.startswith('cuda'):
        torch.cuda.synchronize()
    wall = time.perf_counter()-tick
    tokens = arm_tokens(zip(seeds, results))
    produced = sum(len(t) for t in tokens.values())
    return tokens, dict(scheduler=scheduler, batch=batch, seeds=len(seeds), wall_s=wall, cores=len(tokens),
                        tokens=produced, tok_s=produced/wall, stats=stats)


def line(arm):
    s = arm['stats']
    identity = arm.get('identity') or {}
    occupancy = s.get('occupancy') or 0.
    return (f"B={arm['batch']:3d} {arm['scheduler']:10s} n={arm['seeds']:2d} {arm['tok_s']:9.0f} tok/s "
            f"x{arm.get('speedup', 1.):4.2f} wall {arm['wall_s']:6.1f}s cores {arm['cores']:5d} S {s.get('steps', 0):6d} "
            f"R {s.get('billed_row_steps', 0):8d} occ {occupancy:.2f} "
            f"decode {s.get('decode_s', 0.):6.1f} grammar {s.get('grammar_s', 0.):6.1f} select {s.get('select_s', 0.):6.1f} "
            f"encoder {s.get('encoder_s', 0.):6.1f} E {s.get('encoded_rows', 0):5d} captures {s.get('graph_captures', 0):3d} "
            f"identity {identity.get('compared', '-')}-{identity.get('different', '-')} different")


def bench(a):
    if a.tiny:
        model, features, panel, intensity_dim, cell_us = tiny_setup(a.songs)
    else:
        model, features, panel, intensity_dim, cell_us = checkpoint_setup(a, a.songs)
    print(f"songs {[s['song_id'] for s in panel['songs']]} device {a.device} autocast {a.autocast} "
          f"compile {a.compile_decode}", flush=True)
    report = dict(songs=[s['song_id'] for s in panel['songs']], device=a.device, autocast=a.autocast,
                  compile_decode=a.compile_decode, arm_wall_s=a.arm_wall_s, arms=[])
    schedulers = ['legacy'] + [s for s in a.schedulers.split(',') if s != 'legacy']
    for batch in [int(value) for value in a.batch_sizes.split(',')]:
        seeds = list(range(max(a.samples, math.ceil(batch/len(panel['songs'])))))
        baseline, baseline_rate = None, None
        for scheduler in schedulers:
            tokens, arm = run_arm(a, model, features, panel, intensity_dim, cell_us, scheduler, batch, seeds)
            if baseline is None:
                baseline, baseline_rate = tokens, arm['tok_s']
            else:
                arm.update(identity=compare(baseline, tokens), speedup=arm['tok_s']/baseline_rate)
            report['arms'].append(arm)
            print(line(arm), flush=True)
            if a.json:
                W.write_json_atomic(a.json, report)
    different = sum(arm['identity']['different'] for arm in report['arms'] if 'identity' in arm)
    compared = sum(arm['identity']['compared'] for arm in report['arms'] if 'identity' in arm)
    report.update(cores_compared=compared, cores_different=different, graph_failures=list(generate._GRAPH_FAILURES))
    print(f'token identity vs legacy: {compared-different}/{compared} cores identical; '
          f'CUDA-graph fallbacks: {generate._GRAPH_FAILURES or "none"}', flush=True)
    if a.json:
        W.write_json_atomic(a.json, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--tiny', action='store_true', help='random-init tiny model on synthetic features (CPU check)')
    p.add_argument('--plan', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--tape', default=W.TAPE)
    p.add_argument('--weights', choices=('raw', 'ema'), default='raw')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--songs', type=int, default=5)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--schedulers', default='legacy,grouped,continuous')
    p.add_argument('--batch-sizes', default='20,80,160,320')
    p.add_argument('--arm-wall-s', type=float, default=60.)
    p.add_argument('--temperature', default='1')
    p.add_argument('--token-cap', type=int, default=4096)
    p.add_argument('--autocast', choices=('bf16',))
    p.add_argument('--compile-decode', action='store_true')
    p.add_argument('--json', type=Path)
    a = p.parse_args(argv)
    if a.checkpoint and not a.plan:
        p.error('--checkpoint needs --plan')
    with torch.no_grad():
        return bench(a)


if __name__ == '__main__':
    main()

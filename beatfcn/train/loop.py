"""P1 training loop: sampling (design §6.3), optimisation (§10.2), logging (§10.4).

Sampling order: mode (p ∝ sqrt(#songs)) -> song -> chart -> request (P0 mix) ->
core. Cores that exceed P1 bounds are resampled, never cropped. Every log line
records step time, examples, rows and peak memory so cost can be measured, not
assumed. Validation uses a fixed list of (chart, seed) cores built once.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import time
import numpy as np
import torch
from ..events.requests import sample_request
from ..model import features as F
from ..model.batch import collate
from ..model.p1 import P1Model


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    micro_batch: int = 2
    accumulation: int = 16
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup: int = 500
    weight_decay: float = 0.01
    clip: float = 1.0
    validate_every: int = 500
    log_every: int = 25
    seed: int = 20261022
    val_cores: int = 256


class CoreSampler:
    """charts: list of dicts with keys chart_id, song, mode, duration_us; load(chart_id) -> Chart;
    song_cache(song) -> SongCache."""

    def __init__(self, charts, load, song_cache, seed):
        self.charts, self.load, self.song_cache = charts, load, song_cache
        self.rng = np.random.default_rng(seed)
        by_mode = {}
        for c in charts:
            by_mode.setdefault(c['mode'], {}).setdefault(c['song'], []).append(c)
        self.by_mode = by_mode
        modes = sorted(by_mode)
        w = np.array([math.sqrt(len(by_mode[m])) for m in modes])
        self.modes, self.mode_p = modes, w / w.sum()
        self.skipped = {}

    def core(self, chart_row, seed):
        chart = self.load(chart_row['chart_id'])
        s = sample_request(chart, seed=seed, audio_id=chart_row['song'], audio_duration_us=chart_row['duration_us'])
        if not s.ok:
            raise F.SkipExample('model view quarantine')
        c0 = F.choose_core(s, np.random.default_rng([seed, 1]), chart_row['duration_us'])
        ex = F.build_core(s, chart.events, c0, chart_row['duration_us'])
        bad = F.check_targets(ex)
        if bad:
            raise F.SkipExample('illegal_target:' + ','.join(bad))
        return ex

    def draw(self, cores_per_song: int = 1):
        """Song-bucketed draw (design §10.3): one song cache read serves cores_per_song cores
        from that song's charts/requests. Returns a list of (core, song_cache)."""
        for _ in range(100):
            mode = self.modes[self.rng.choice(len(self.modes), p=self.mode_p)]
            songs = self.by_mode[mode]
            song = list(songs)[self.rng.integers(len(songs))]
            out = []
            for _ in range(4 * cores_per_song):
                row = songs[song][self.rng.integers(len(songs[song]))]
                try:
                    out.append(self.core(row, int(self.rng.integers(2 ** 31))))
                except (F.SkipExample, ValueError, AssertionError) as exc:
                    key = type(exc).__name__ + ':' + str(exc)[:40]
                    self.skipped[key] = self.skipped.get(key, 0) + 1
                if len(out) == cores_per_song:
                    break
            if out:
                cache = self.song_cache(song)
                return [(ex, cache) for ex in out]
        raise RuntimeError(f'could not draw a valid core: {self.skipped}')

    def fixed(self, n, seed):
        rng, out = np.random.default_rng(seed), []
        rows = sorted(self.charts, key=lambda r: r['chart_id'])
        while len(out) < n and rows:
            row = rows[rng.integers(len(rows))]
            s = int(rng.integers(2 ** 31))
            try:
                out.append((self.core(row, s), row['song']))
            except (F.SkipExample, ValueError, AssertionError):
                continue
        return out


def lr_at(step, c: TrainConfig):
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    p = (step - c.warmup) / max(1, c.steps - c.warmup)
    return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


def to_device(bt, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}


def merge(acc, met):
    for k, (l, n, c) in met.items():
        a = acc.setdefault(k, [0.0, 0, 0])
        a[0] += l * n
        a[1] += n
        a[2] += c


def summarize(acc):
    return {k: {'loss': round(l / max(n, 1), 4), 'n': n, 'acc': round(c / max(n, 1), 4)} for k, (l, n, c) in acc.items()}


@torch.no_grad()
def evaluate(model, cores, song_cache, cfg_model, dev, micro_batch):
    model.eval()
    acc = {}
    for i in range(0, len(cores), micro_batch):
        chunk = cores[i:i + micro_batch]
        bt = to_device(collate([c for c, _ in chunk], [song_cache(s) for _, s in chunk], cfg_model.taps, cfg_model.top_tap), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            _, met = model(bt)
        merge(acc, met)
    model.train()
    out = summarize(acc)
    out['total'] = round(sum(v['loss'] for v in out.values()), 4)
    return out


def train(model: P1Model, train_sampler: CoreSampler, val_cores, song_cache, c: TrainConfig, dev, log, save=None):
    torch.manual_seed(c.seed)
    model.to(dev).train()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    other = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': c.weight_decay}, {'params': other, 'weight_decay': 0.0}],
                            lr=c.lr, betas=(0.9, 0.95))
    log({'event': 'start', 'config': asdict(c), 'params': sum(p.numel() for p in model.parameters())})
    best, history, acc, t_log, examples = None, [], {}, time.time(), 0
    for step in range(c.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, c)
        t0 = time.time()
        for _ in range(c.accumulation):
            items = train_sampler.draw(c.micro_batch)
            bt = to_device(collate([a for a, _ in items], [b for _, b in items], model.c.taps, model.c.top_tap), dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                loss, met = model(bt)
            (loss / c.accumulation).backward()
            merge(acc, met)
            examples += len(items)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_s = time.time() - t0
        if (step + 1) % c.log_every == 0:
            rec = {'event': 'train', 'step': step + 1, 'lr': lr_at(step, c), 'grad_norm': float(gn), 'step_seconds': round(step_s, 3),
                   'examples': examples, 'wall': round(time.time() - t_log, 1), 'fields': summarize(acc),
                   'skipped': dict(train_sampler.skipped)}
            if dev.type == 'cuda':
                rec['peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            log(rec)
            acc = {}
        if (step + 1) % c.validate_every == 0 or step + 1 == c.steps:
            v = evaluate(model, val_cores, song_cache, model.c, dev, c.micro_batch)
            history.append(v['total'])
            log({'event': 'val', 'step': step + 1, **v})
            if save and (best is None or v['total'] < best):
                best = v['total']
                save(model, step + 1, v)
    return history

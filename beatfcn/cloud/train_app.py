"""Modal app: P1 training (candidate B) on cached frozen taps.

  .venv/bin/modal run --detach beatfcn/cloud/train_app.py::main --run-id p1-pilot-001 --steps 3000

Data workers build requests/cores on CPU in parallel (torch DataLoader over an IterableDataset);
the GPU only runs the event model. Logs: /data/runs/{run}/log.jsonl; best checkpoint: best.pt.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA = '/data'
app = modal.App('beatfcn-train')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3', 'jsonschema==4.26.0')
         .add_local_python_source('beatfcn'))


def _stream(rows, corpus_args, seed, micro_batch, taps, top_tap):
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.model.batch import collate

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            corpus = Corpus(**corpus_args)
            smp = CoreSampler(corpus.train_rows(rows), corpus.chart, corpus.song, seed * 1000 + wid)
            while True:
                items = []
                while len(items) < micro_batch:  # 4 cores per song bucket -> 2+ songs per microbatch
                    items += smp.draw(min(4, micro_batch - len(items)))
                yield collate([a for a, _ in items], [b for _, b in items], taps, top_tap), dict(smp.skipped)
    return Stream()


@app.function(image=image, gpu='H100', volumes={DATA: data}, cpu=32.0, memory=131072, timeout=6 * 3600)
def train(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, taps_key: str = 'v1_L8-16-24',
          taps: tuple = (8, 16, 24), workers: int = 28, val_cores: int = 256, max_train_songs: int = 0,
          lr: float = 2e-4, warmup: int = 500, validate_every: int = 500):
    import numpy as np
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler, TrainConfig, evaluate, lr_at, merge, summarize, to_device
    run = f'{DATA}/runs/{run_id}'
    os.makedirs(run, exist_ok=True)
    logf = open(f'{run}/log.jsonl', 'a')

    def log(rec):
        rec['ts'] = time.time()
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print(json.dumps(rec)[:600], flush=True)

    have = set(os.listdir(f'{DATA}/taps/{taps_key}'))
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have]
    train_rows = [r for r in rows if r['split'] == 'train']
    if max_train_songs:
        keep = sorted({r['audio_sha256'] for r in train_rows})[:max_train_songs]
        train_rows = [r for r in train_rows if r['audio_sha256'] in set(keep)]
    val_rows = [r for r in rows if r['split'] == 'val']
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': taps}
    mc = P1Config(taps=tuple(taps), top_tap=max(taps))
    tc = TrainConfig(steps=steps, micro_batch=micro_batch, accumulation=accumulation, lr=lr, warmup=warmup,
                     validate_every=validate_every, val_cores=val_cores)
    corpus = Corpus(**corpus_args)
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7)
    t0 = time.time()
    val = vsmp.fixed(val_cores, 20261022)
    log({'event': 'data', 'train_charts': len(train_rows), 'train_songs': len({r['audio_sha256'] for r in train_rows}),
         'val_charts': len(val_rows), 'val_cores': len(val), 'val_build_s': round(time.time() - t0, 1),
         'modes': {m: sum(r['mode'] == m for r in train_rows) for m in ('mania_4k', 'phira')}})
    dev = torch.device('cuda')
    torch.manual_seed(tc.seed)
    model = P1Model(mc).to(dev)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': tc.weight_decay}, {'params': other, 'weight_decay': 0.0}],
                            lr=tc.lr, betas=(0.9, 0.95))
    log({'event': 'start', 'model': asdict(mc), 'train': asdict(tc), 'params': sum(p.numel() for p in model.parameters()),
         'gpu': torch.cuda.get_device_name()})
    stream = torch.utils.data.DataLoader(_stream(train_rows, corpus_args, tc.seed, micro_batch, mc.taps, mc.top_tap),
                                         batch_size=None, num_workers=workers, prefetch_factor=4, persistent_workers=True,
                                         pin_memory=True)
    it = iter(stream)
    best, acc, examples, wait_s, gpu_s = None, {}, 0, 0.0, 0.0
    for step in range(tc.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, tc)
        for _ in range(tc.accumulation):
            tw = time.time()
            bt, skipped = next(it)
            bt = to_device(bt, dev)
            tg = time.time()
            wait_s += tg - tw
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, met = model(bt)
            (loss / tc.accumulation).backward()
            torch.cuda.synchronize()
            gpu_s += time.time() - tg
            merge(acc, met)
            examples += bt['mode'].shape[0]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % tc.log_every == 0:
            log({'event': 'train', 'step': step + 1, 'lr': lr_at(step, tc), 'grad_norm': float(gn), 'examples': examples,
                 'data_wait_s': round(wait_s, 1), 'gpu_s': round(gpu_s, 1), 'elapsed_s': round(time.time() - t0, 1),
                 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'fields': summarize(acc), 'skipped': skipped})
            acc = {}
        if (step + 1) % tc.validate_every == 0 or step + 1 == tc.steps:
            v = evaluate(model, val, corpus.song, mc, dev, micro_batch)
            log({'event': 'val', 'step': step + 1, **v})
            if best is None or v['total'] < best:
                best = v['total']
                torch.save({'model': model.state_dict(), 'config': asdict(mc), 'step': step + 1, 'val': v}, f'{run}/best.pt')
            data.commit()
    data.commit()
    return {'run': run_id, 'best_val_total': best, 'elapsed_s': round(time.time() - t0, 1), 'examples': examples}


@app.local_entrypoint()
def main(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, max_train_songs: int = 0,
         validate_every: int = 500, val_cores: int = 256, warmup: int = 500):
    print(train.remote(run_id, steps=steps, micro_batch=micro_batch, accumulation=accumulation,
                       max_train_songs=max_train_songs, validate_every=validate_every, val_cores=val_cores, warmup=warmup))


@app.function(image=image, volumes={DATA: data}, cpu=8.0, memory=32768, timeout=3600)
def diagnose(n_cores: int = 3000, taps_key: str = 'v1_L8-16-24', select: bool = False):
    """Legality audit on real data: check_targets reasons, then a small model forward on legal cores."""
    import collections
    import traceback
    import numpy as np
    import torch
    from beatfcn.model import features as F
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler
    have = set(os.listdir(f'{DATA}/taps/{taps_key}'))
    rows = [r for r in load_manifest(DATA, select=select) if r['audio_sha256'] in have]
    corpus = Corpus(DATA, taps_key)
    smp = CoreSampler(corpus.train_rows(rows), corpus.chart, corpus.song, 1)
    reasons, examples, legal = collections.Counter(), [], []
    for k in range(n_cores):
        row = smp.charts[smp.rng.integers(len(smp.charts))]
        try:
            chart = corpus.chart(row['chart_id'])
            from beatfcn.events.requests import sample_request
            s = sample_request(chart, seed=k, audio_id=row['song'], audio_duration_us=row['duration_us'])
            if not s.ok:
                reasons['model_view_quarantine'] += 1
                continue
            c0 = F.choose_core(s, np.random.default_rng([k, 1]), row['duration_us'])
            ex = F.build_core(s, chart.events, c0, row['duration_us'])
            bad = F.check_targets(ex)
            for b in bad:
                reasons[b] += 1
                if len(examples) < 30:
                    examples.append({'chart': row['chart_id'], 'reason': b, 'c0': c0, 'kind': s.recipe['kind'],
                                     'groups': [{kk: (v if not isinstance(v, np.ndarray) else v.tolist()) for kk, v in g.items()}
                                                for g in ex.groups][:3]})
            if not bad:
                legal.append((ex, row['song']))
                reasons['legal'] += 1
        except Exception as exc:
            reasons[f'{type(exc).__name__}:{str(exc)[:60]}'] += 1
    torch.manual_seed(0)
    m = P1Model(P1Config(d=64, layers=1, heads=4, ffn=96, ptr=32, gru=48)).eval()
    model_fail = collections.Counter()
    with torch.no_grad():
        for i in range(0, min(len(legal), 600), 4):
            chunk = legal[i:i + 4]
            try:
                m(collate([a for a, _ in chunk], [corpus.song(b) for _, b in chunk], m.c.taps, m.c.top_tap))
                model_fail['ok'] += 1
            except Exception as exc:
                model_fail[str(exc)[:120]] += 1
    out = {'rows': len(rows), 'cores_tried': n_cores, 'reasons': dict(reasons), 'model_forward': dict(model_fail),
           'examples': examples[:12]}
    print(json.dumps(out)[:4000])
    return out


@app.local_entrypoint()
def audit(n_cores: int = 3000):
    r = diagnose.remote(n_cores)
    Path('reports/legality_audit_dry.json').write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ('rows', 'reasons', 'model_forward')}, indent=1))

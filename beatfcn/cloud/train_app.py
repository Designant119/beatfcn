"""Modal app: P1 training (candidate B) on cached frozen taps.

  .venv/bin/modal run --detach beatfcn/cloud/train_app.py::main --run-id p1-pilot-001 --steps 3000

Data workers build requests/cores on CPU in parallel (torch DataLoader over an IterableDataset);
the GPU only runs the event model. Logs: /data/runs/{run}/log.jsonl; best checkpoint: best.pt.
"""
from __future__ import annotations
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA = '/data'
app = modal.App('beatfcn-train')
POOL, RESIDENCE = 6, 8  # per worker and mode: resident songs, draws (of up to 4 cores) each before eviction
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3', 'jsonschema==4.26.0')
         .add_local_python_source('beatfcn'))


def _audit_corpus(corpus_args, rows):
    from beatfcn.train.data import Corpus
    c = Corpus(**corpus_args)
    c.train_rows(rows)
    return c


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
            smp = CoreSampler(corpus.train_rows(rows), corpus.chart, lambda sha: corpus.song(sha, in_memory=True),
                              seed * 1000 + wid, salient=corpus.salient_us, pool=POOL, residence=RESIDENCE)
            while True:
                items = []
                while len(items) < micro_batch:  # 4 cores per song bucket -> 2+ songs per microbatch
                    items += smp.draw(min(4, micro_batch - len(items)))
                yield collate([a for a, _ in items], [b for _, b in items], taps, top_tap), dict(smp.skipped)
    return Stream()


def _pair_stream(rows, corpus_args, seed, n_pairs, taps):
    """Counterfactual support-swap batches from same-recording, same-mode chart pairs (train only)."""
    import torch

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            import numpy as np
            from collections import defaultdict
            from beatfcn.model import cf
            from beatfcn.train.data import Corpus
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng(seed * 7919 + (info.id if info else 0))
            corpus = Corpus(**corpus_args)
            corpus.train_rows(rows)
            groups = defaultdict(list)
            for r in rows:
                groups[(r['audio_sha256'], r['mode'])].append(r)
            keys = [k for k, v in groups.items() if len(v) >= 2]
            heads = {}

            def head_array(cid):
                if cid not in heads:
                    heads[cid] = np.unique(np.array([e.onset_us for e in corpus.chart(cid).events], np.int64))
                return heads[cid]
            while True:
                pairs, songs = [], []
                for _ in range(50 * n_pairs):
                    k = keys[rng.integers(len(keys))]
                    a, b = rng.choice(len(groups[k]), size=2, replace=False)
                    ra, rb = groups[k][a], groups[k][b]
                    try:
                        p = cf.build_pair(head_array(ra['chart_id']), head_array(rb['chart_id']), int(ra['audio_sec'] * 1e6), rng)
                    except ValueError:
                        p = None
                    if p:
                        pairs.append(p)
                        songs.append(corpus.song(k[0]))
                    if len(pairs) == n_pairs:
                        break
                if pairs:
                    yield cf.collate_pairs(pairs, songs, taps, None)
    return Stream()


def _pair_audit(rows, corpus, n_try=600, seed=0):
    """Design §3.4 gate: eligible song groups and independent pair tasks among train charts."""
    import numpy as np
    from collections import defaultdict
    from beatfcn.model import cf
    groups = defaultdict(list)
    for r in rows:
        groups[(r['audio_sha256'], r['mode'])].append(r)
    multi = [k for k, v in groups.items() if len(v) >= 2]
    rng = np.random.default_rng(seed)
    ok_groups, ok = set(), 0
    for _ in range(n_try):
        k = multi[rng.integers(len(multi))]
        a, b = rng.choice(len(groups[k]), size=2, replace=False)
        try:
            ha = np.unique([e.onset_us for e in corpus.chart(groups[k][a]['chart_id']).events])
            hb = np.unique([e.onset_us for e in corpus.chart(groups[k][b]['chart_id']).events])
        except ValueError:
            continue
        if cf.build_pair(ha, hb, int(groups[k][a]['audio_sec'] * 1e6), rng):
            ok += 1
            ok_groups.add(k)
    rate = ok / max(n_try, 1)
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for k, v in groups.items() if len(v) >= 2)
    return {'groups_with_2plus_charts': len(multi), 'sampled': n_try, 'success_rate': round(rate, 3),
            'eligible_groups_seen': len(ok_groups), 'est_eligible_pairs': int(rate * total_pairs),
            'by_mode': {m: sum(1 for k in multi if k[1] == m) for m in ('mania_4k', 'phira')}}


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=12.0, memory=49152, timeout=24 * 3600)
def train(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, taps_key: str = 'v1_L8-16-24',
          taps: tuple = (8, 16, 24), workers: int = 16, val_cores: int = 256, max_train_songs: int = 0,
          lr: float = 2e-4, warmup: int = 500, validate_every: int = 500, tap_dim: int = 1024, songs_file: str = '',
          variant: dict | None = None, cf_every: int = 4, cf_pairs: int = 8, cf_start: int = 2000, val_songs_file: str = '',
          pca: str = '', cf_force: bool = False, stop_at: int = 0, whiten: bool = False):
    """stop_at: pause at this step (checkpointed) while the LR schedule still spans `steps`; a later call
    with the same run_id resumes from last.pt (S1 compares both variants at stop_at, then extends the winner)."""
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

    tdir = f'{DATA}/taps/{taps_key}'  # S20 is written last: songs still being encoded are left out
    have = {s for s in os.listdir(tdir) if os.path.exists(f'{tdir}/{s}/S20.npy')}
    probe = Corpus(root=DATA, taps_key=taps_key, layers=taps, pca=pca, whiten=whiten)
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have]
    n0 = len(rows)
    rows = [r for r in rows if probe.usable(r)]
    dup_rows = sum(1 for r in rows if not r.get('kept'))
    train_rows = [r for r in rows if r['split'] == 'train']
    if songs_file:  # restrict to a fixed song list (pilot / equivalence checks)
        allowed = set(open(songs_file).read().split())
        train_rows = [r for r in train_rows if r['audio_sha256'] in allowed]
    if max_train_songs:
        keep = sorted({r['audio_sha256'] for r in train_rows})[:max_train_songs]
        train_rows = [r for r in train_rows if r['audio_sha256'] in set(keep)]
    val_rows = [r for r in rows if r['split'] == 'val']
    if val_songs_file:  # fixed validation songs so runs with different caches stay comparable
        vallow = set(open(val_songs_file).read().split())
        val_rows = [r for r in val_rows if r['audio_sha256'] in vallow]
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': taps, 'pca': pca, 'whiten': whiten}
    mc = P1Config(taps=tuple(taps), top_tap=max(taps), tap_dim=tap_dim, **(variant or {}))
    tc = TrainConfig(steps=steps, micro_batch=micro_batch, accumulation=accumulation, lr=lr, warmup=warmup,
                     validate_every=validate_every, val_cores=val_cores)
    corpus = Corpus(**corpus_args)
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7, salient=corpus.salient_us)
    t0 = time.time()
    val = vsmp.fixed(val_cores, 20261022)
    log({'event': 'data', 'train_charts': len(train_rows), 'train_songs': len({r['audio_sha256'] for r in train_rows}),
         'val_charts': len(val_rows), 'val_cores': len(val), 'val_build_s': round(time.time() - t0, 1),
         'dup_collapsed_charts': dup_rows, 'dup_ineligible_dropped': n0 - len(rows),
         'modes': {m: sum(r['mode'] == m for r in train_rows) for m in ('mania_4k', 'phira')}})
    dev = torch.device('cuda')
    torch.manual_seed(tc.seed)
    model = P1Model(mc).to(dev)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': tc.weight_decay}, {'params': other, 'weight_decay': 0.0}],
                            lr=tc.lr, betas=(0.9, 0.95), fused=True)
    log({'event': 'start', 'model': asdict(mc), 'train': asdict(tc), 'params': sum(p.numel() for p in model.parameters()),
         'gpu': torch.cuda.get_device_name()})
    start_step, best = 0, None
    if os.path.exists(f'{run}/last.pt'):  # resume after preemption / restart (same run id)
        ck = torch.load(f'{run}/last.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step, best = ck['step'], ck.get('best')
        log({'event': 'resume', 'step': start_step, 'best': best})
    stream = torch.utils.data.DataLoader(_stream(train_rows, corpus_args, tc.seed, micro_batch, mc.taps, mc.top_tap),
                                         batch_size=None, num_workers=workers, prefetch_factor=4, persistent_workers=True,
                                         pin_memory=True)
    it = iter(stream)
    use_cf = False
    if mc.selection:
        audit = _pair_audit(train_rows, _audit_corpus(corpus_args, train_rows))
        use_cf = cf_force or (audit['eligible_groups_seen'] >= 50 and audit['est_eligible_pairs'] >= 300)
        log({'event': 'cf_audit', **audit, 'cf_enabled': use_cf})
        if use_cf:
            cf_it = iter(torch.utils.data.DataLoader(_pair_stream(train_rows, corpus_args, tc.seed, cf_pairs, mc.taps),
                                                     batch_size=None, num_workers=4, prefetch_factor=2, persistent_workers=True))
    from beatfcn.cloud.gpu_util import GpuUtil
    mon = GpuUtil()  # nvidia-smi utilisation per log interval (what the Modal dashboard shows)
    acc, examples, wait_s, gpu_s = {}, 0, 0.0, 0.0
    end = min(stop_at, tc.steps) if stop_at else tc.steps
    for step in range(start_step, end):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, tc)
        for _ in range(tc.accumulation):
            tw = time.time()
            bt, skipped = next(it)
            bt = to_device(bt, dev)
            bt['w_select'] = 0.2 * min(1.0, (step + 1) / 1000)
            tg = time.time()
            wait_s += tg - tw
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, met = model(bt)
            (loss / tc.accumulation).backward()
            gpu_s += time.time() - tg  # host-side issue time (no per-micro-batch sync: the device runs ahead)
            merge(acc, met)
            examples += bt['mode'].shape[0]
        if use_cf and step >= cf_start and step % cf_every == 0:
            from beatfcn.model import cf as CF
            cbt = to_device(next(cf_it), dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l_cf, cf_acc = CF.cf_loss(model, cbt)
            (0.1 * min(1.0, (step - cf_start + 1) / 1000) * l_cf).backward()
            merge(acc, {'cf': (l_cf.detach(), cbt['audio'].shape[0], cf_acc.detach() * cbt['audio'].shape[0])})
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % tc.log_every == 0:
            log({'event': 'train', 'step': step + 1, 'lr': lr_at(step, tc), 'grad_norm': float(gn), 'examples': examples,
                 'data_wait_s': round(wait_s, 1), 'gpu_s': round(gpu_s, 1), 'elapsed_s': round(time.time() - t0, 1),
                 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'gpu_util': mon.take()['util_mean'],
                 'fields': summarize(acc), 'skipped': skipped})
            acc = {}
        if (step + 1) % tc.validate_every == 0 or step + 1 in (tc.steps, end):
            v = evaluate(model, val, corpus.song, mc, dev, micro_batch)
            log({'event': 'val', 'step': step + 1, **v})
            if best is None or v['total'] < best:
                best = v['total']
                torch.save({'model': model.state_dict(), 'config': asdict(mc), 'data': corpus_args, 'step': step + 1, 'val': v}, f'{run}/best.pt')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'config': asdict(mc), 'data': corpus_args, 'step': step + 1,
                        'best': best}, f'{run}/last.pt.tmp')
            os.replace(f'{run}/last.pt.tmp', f'{run}/last.pt')
            data.commit()
    data.commit()
    return {'run': run_id, 'best_val_total': best, 'elapsed_s': round(time.time() - t0, 1), 'examples': examples}


@app.local_entrypoint()
def main(run_id: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4, max_train_songs: int = 0,
         validate_every: int = 500, val_cores: int = 256, warmup: int = 500, taps_key: str = 'v1_L8-16-24',
         taps: str = '8,16,24', tap_dim: int = 1024, songs_file: str = '', variant: str = '{}', cf_start: int = 2000,
         val_songs_file: str = '', pca: str = ''):
    print(train.remote(run_id, steps=steps, micro_batch=micro_batch, accumulation=accumulation, taps_key=taps_key,
                       taps=tuple(int(x) for x in taps.split(',')), max_train_songs=max_train_songs,
                       validate_every=validate_every, val_cores=val_cores, warmup=warmup, tap_dim=tap_dim,
                       songs_file=songs_file, variant=json.loads(variant), cf_start=cf_start, val_songs_file=val_songs_file,
                       pca=pca))


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


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=8.0, memory=32768, timeout=4 * 3600)
def eval_generation(run_id: str, n_cores: int = 200, taps_key: str = 'v1_L8-16-24', greedy: bool = False, seed: int = 20261022,
                    split: str = 'val', pca: str = ''):
    """Generate every fixed val core with the run's best checkpoint and score against hidden targets."""
    import collections
    import numpy as np
    import torch
    from beatfcn.events.requests import sample_request
    from beatfcn.model import features as F
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.sample import generate_core, core_context, match_f1
    from beatfcn.train.data import Corpus, load_manifest
    ck = torch.load(f'{DATA}/runs/{run_id}/best.pt', map_location='cuda', weights_only=False)
    cfg = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(cfg).cuda().eval()
    model.load_state_dict(ck['model'])
    have = set(os.listdir(f'{DATA}/taps/{taps_key}'))
    rows = [r for r in load_manifest(DATA, split=split) if r['audio_sha256'] in have]
    corpus = Corpus(DATA, taps_key, layers=cfg.taps, pca=pca)
    trows = corpus.train_rows(rows)
    rng = np.random.default_rng(seed)
    tol = (10_000, 20_000, 50_000)
    agg = collections.defaultdict(lambda: collections.Counter())
    t0, done, illegal = time.time(), 0, 0
    while done < n_cores:
        row = trows[rng.integers(len(trows))]
        try:
            chart = corpus.chart(row['chart_id'])
            s = sample_request(chart, seed=int(rng.integers(2 ** 31)), audio_id=row['song'], audio_duration_us=row['duration_us'])
            if not s.ok:
                continue
            c0 = F.choose_core(s, np.random.default_rng(done), row['duration_us'])
            li = F.line_index_of(chart.events)
            ex = F.build_core(s, chart.events, c0, row['duration_us'], line_index=li)
            if F.check_targets(ex):
                continue
        except (F.SkipExample, ValueError, AssertionError):
            continue
        song = corpus.song(row['song'])
        holds, runs, heads = core_context(s, ex, li)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen = generate_core(model, ex, song, holds, runs, heads, greedy=greedy, seed=done, device='cuda')
        mode = row['mode']
        gt = [g['t_rel_us'] for g in ex.groups for _ in g['notes']]
        pr = [g['t_rel_us'] for g in gen for _ in g['notes']]
        a = agg[mode]
        for tl in tol:
            tp, npred, ntrue, _ = match_f1(pr, gt, tl)
            a[f'tp{tl // 1000}'] += tp
        a['pred'] += len(pr)
        a['true'] += len(gt)
        a['cores'] += 1
        a['empty_pred'] += not gen
        a['empty_true'] += not ex.groups
        gh = [g['t_rel_us'] for g in ex.groups for n in g['notes'] if 'end_rel_us' in n]
        ph = [g['t_rel_us'] for g in gen for n in g['notes'] if 'end_rel_us' in n]
        tp, _, _, _ = match_f1(ph, gh, 50_000)
        a['hold_tp50'] += tp
        a['hold_pred'] += len(ph)
        a['hold_true'] += len(gh)
        # group-level attribute agreement for exactly matched (<=20 ms) group times
        gtimes = {g['t_rel_us']: g for g in ex.groups}
        for g in gen:
            near = min(gtimes, key=lambda t: abs(t - g['t_rel_us']), default=None)
            if near is not None and abs(near - g['t_rel_us']) <= 20_000:
                tg = gtimes[near]
                a['groups_matched'] += 1
                a['size_equal'] += len(tg['notes']) == len(g['notes'])
                if mode == 'mania_4k':
                    a['lanes_equal'] += sorted(n['lane'] for n in tg['notes']) == sorted(n['lane'] for n in g['notes'])
                a['types_equal'] += sorted(n['type'] for n in tg['notes']) == sorted(n['type'] for n in g['notes'])
        illegal += 0  # generation is mask-constrained; independent validator lands with runtime/constraints
        done += 1
    out = {'run': run_id, 'step': ck['step'], 'cores': done, 'greedy': greedy, 'seconds': round(time.time() - t0, 1)}
    for mode, a in agg.items():
        f = {}
        for tl in tol:
            k = tl // 1000
            p = a[f'tp{k}'] / max(a['pred'], 1)
            r = a[f'tp{k}'] / max(a['true'], 1)
            f[f'f1_{k}ms'] = round(2 * p * r / max(p + r, 1e-9), 4)
            f[f'p_{k}ms'], f[f'r_{k}ms'] = round(p, 4), round(r, 4)
        hp, hr = a['hold_tp50'] / max(a['hold_pred'], 1), a['hold_tp50'] / max(a['hold_true'], 1)
        out[mode] = {**f, 'density_ratio': round(a['pred'] / max(a['true'], 1), 3), 'cores': a['cores'],
                     'empty_pred_rate': round(a['empty_pred'] / max(a['cores'], 1), 3),
                     'empty_true_rate': round(a['empty_true'] / max(a['cores'], 1), 3),
                     'hold_head_f1_50ms': round(2 * hp * hr / max(hp + hr, 1e-9), 4),
                     'group_size_acc': round(a['size_equal'] / max(a['groups_matched'], 1), 4),
                     'group_type_acc': round(a['types_equal'] / max(a['groups_matched'], 1), 4),
                     'group_lane_acc': round(a['lanes_equal'] / max(a['groups_matched'], 1), 4) if mode == 'mania_4k' else None}
    out['split'] = split
    with open(f'{DATA}/runs/{run_id}/eval_generation_{split}{"_greedy" if greedy else ""}.json', 'w') as fo:
        json.dump(out, fo, indent=1)
    data.commit()
    return out


@app.local_entrypoint()
def evaluate(run_id: str, n_cores: int = 200, greedy: bool = False, taps_key: str = 'v1_L8-16-24', split: str = 'val',
             pca: str = ''):
    r = eval_generation.remote(run_id, n_cores=n_cores, greedy=greedy, taps_key=taps_key, split=split, pca=pca)
    Path(f'reports/eval_{run_id}_{split}{"_greedy" if greedy else ""}.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


weights_vol = modal.Volume.from_name('beatfcn-weights', create_if_missing=True)


def _s2_stream(rows, corpus_args, seed, micro_batch, taps, top_tap):
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.model.batch import collate
    from beatfcn.audio import memory as M
    from beatfcn.train import s2 as S2

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            corpus = Corpus(**corpus_args)
            smp = CoreSampler(corpus.train_rows(rows), corpus.chart, lambda sha: corpus.song(sha, in_memory=True),
                              seed * 1000 + (info.id if info else 0), salient=corpus.salient_us, pool=POOL,
                              residence=RESIDENCE)
            pad = M.silence_row()
            while True:
                items = []
                while len(items) < micro_batch:
                    items += smp.draw(min(4, micro_batch - len(items)))
                exs, songs = [a for a, _ in items], [b for _, b in items]
                bt = collate(exs, songs, taps, top_tap)
                cells = [-(-s.mel10.shape[0] // M.FRAMES_PER_CELL) for s in songs]
                plan, il, ir, ih = S2.plan_windows(exs, bt, cells)
                win = S2.mel_windows(plan, [s.mel10 for s in songs], pad).half()
                yield bt, win, il, ir, ih, dict(smp.skipped)
    return Stream()


@app.function(image=image, gpu='H100', volumes={DATA: data, '/weights': weights_vol}, cpu=16.0, memory=65536, timeout=24 * 3600)
def train_s2(run_id: str, init_run: str, steps: int = 3000, micro_batch: int = 8, accumulation: int = 4,
             taps_key: str = 'v3_pca512q_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'pca512_9-16-24',
             workers: int = 14, val_cores: int = 256, lr_event: float = 5e-5, lr_lora: float = 5e-6,
             validate_every: int = 500):
    import numpy as np
    import torch
    from beatfcn.audio.pupu import load_teacher
    from beatfcn.audio import memory as M
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train import s2 as S2
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler, merge, summarize, to_device
    run = f'{DATA}/runs/{run_id}'
    os.makedirs(run, exist_ok=True)
    logf = open(f'{run}/log.jsonl', 'a')

    def log(rec):
        rec['ts'] = time.time()
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print(json.dumps(rec)[:600], flush=True)
    dev = torch.device('cuda')
    ck = torch.load(f'{DATA}/runs/{init_run}/best.pt', map_location='cuda', weights_only=False)
    mc = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(mc).to(dev)
    model.load_state_dict(ck['model'])
    teacher = load_teacher('/weights/teacher_encoder.safetensors', dtype=torch.bfloat16, device='cuda')
    lora = S2.add_lora(teacher)
    z = np.load(f'{DATA}/pca/{pca}.npz')
    mu, comp = torch.from_numpy(z[f'mean_{mc.top_tap}']).cuda(), torch.from_numpy(z[f'comp_{mc.top_tap}']).cuda()
    ti = mc.taps.index(mc.top_tap)
    have = set(os.listdir(f'{DATA}/taps/{taps_key}'))
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': taps, 'pca': pca}
    corpus = Corpus(**corpus_args)
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have and corpus.usable(r)]
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song, 7, salient=corpus.salient_us)
    val = vsmp.fixed(val_cores, 20261022)
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters()], 'lr': lr_event, 'weight_decay': 0.01},
                             {'params': lora, 'lr': lr_lora, 'weight_decay': 0.0}], betas=(0.9, 0.95))
    log({'event': 'start', 'init_run': init_run, 'init_step': ck['step'], 'init_val': ck['val'].get('total'),
         'lora_params': sum(p.numel() for p in lora), 'train_charts': len(train_rows), 'val_cores': len(val)})
    pad = M.silence_row()

    def run_batch(bt, win, il, ir, ih):
        bt = to_device(bt, dev)
        bt = S2.attach_top(bt, teacher, win.to(dev), il, ir, ih, mu, comp, ti)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(bt)

    def validate():
        model.eval()
        teacher.eval()
        acc = {}
        with torch.no_grad():
            for i in range(0, len(val), micro_batch):
                chunk = val[i:i + micro_batch]
                exs, songs = [a for a, _ in chunk], [corpus.song(s) for _, s in chunk]
                bt = collate(exs, songs, mc.taps, mc.top_tap)
                plan, il, ir, ih = S2.plan_windows(exs, bt, [-(-s.mel10.shape[0] // 4) for s in songs])
                _, met = run_batch(bt, S2.mel_windows(plan, [s.mel10 for s in songs], pad).half(), il, ir, ih)
                merge(acc, met)
        model.train()
        out = summarize(acc)
        out['total'] = round(sum(v['loss'] for v in out.values()), 4)
        return out
    v0 = validate()
    log({'event': 'val', 'step': 0, **v0})
    best = v0['total']
    stream = torch.utils.data.DataLoader(_s2_stream(train_rows, corpus_args, 20261024, micro_batch, mc.taps, mc.top_tap),
                                         batch_size=None, num_workers=workers, prefetch_factor=2, persistent_workers=True)
    it, acc, t0 = iter(stream), {}, time.time()
    model.train()
    for step in range(steps):
        warm = min(1.0, (step + 1) / 100)
        for g, base in zip(opt.param_groups, (lr_event, lr_lora)):
            g['lr'] = base * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for _ in range(accumulation):
            bt, win, il, ir, ih, skipped = next(it)
            loss, met = run_batch(bt, win, il, ir, ih)
            (loss / accumulation).backward()
            merge(acc, met)
        gn = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + lora, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0:
            log({'event': 'train', 'step': step + 1, 'grad_norm': float(gn), 'elapsed_s': round(time.time() - t0, 1),
                 'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2), 'fields': summarize(acc)})
            acc = {}
        if (step + 1) % validate_every == 0 or step + 1 == steps:
            v = validate()
            log({'event': 'val', 'step': step + 1, **v})
            if v['total'] < best:
                best = v['total']
                torch.save({'model': model.state_dict(), 'lora': {f'{i}': p.detach().cpu() for i, p in enumerate(lora)},
                            'config': ck['config'], 'step': step + 1, 'val': v, 'init_run': init_run}, f'{run}/best.pt')
            data.commit()
    return {'run': run_id, 'init_val': v0['total'], 'best_val': best, 'improved': best < v0['total']}


@app.local_entrypoint()
def s2(run_id: str, init_run: str, steps: int = 3000):
    print(train_s2.remote(run_id, init_run, steps=steps))


@app.local_entrypoint()
def sweep(configs: str):
    """Several training runs inside ONE app (avoids the app-create rate limit). configs: JSON list of
    {"run_id": ..., **train kwargs}; variant may be a dict. Optional "resources": {"gpu", "cpu", "memory"}
    overrides the container shape (cost-first runs use fewer CPUs / cheaper GPUs)."""
    cfgs = json.loads(Path(configs).read_text()) if configs.endswith('.json') else json.loads(configs)
    calls = []
    for c in cfgs:
        res = c.pop('resources', None)
        fn = train.with_options(**res) if res else train
        calls.append((c['run_id'], fn.spawn(**{k: (tuple(v) if k == 'taps' else v) for k, v in c.items()})))
    for rid, call in calls:
        try:
            print(rid, call.get())
        except Exception as exc:
            print(rid, 'FAILED', repr(exc)[:300])


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=8.0, memory=32768, timeout=4 * 3600)
def eval_controls(run_id: str, n_cores: int = 80, taps_key: str = 'v3_pca512q_L9-16-24', pca: str = 'pca512_9-16-24',
                  seed: int = 20261025):
    """Control sweep (design §13.2 item 5): same core generated at low/mid/high bands of one control;
    report how often the generated statistic moves in the requested direction."""
    import dataclasses
    import numpy as np
    import torch
    from beatfcn.events.requests import sample_request
    from beatfcn.events import controls as C
    from beatfcn.model import features as F
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.sample import generate_core, core_context
    from beatfcn.train.data import Corpus, load_manifest
    ck = torch.load(f'{DATA}/runs/{run_id}/best.pt', map_location='cuda', weights_only=False)
    cfg = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    model = P1Model(cfg).cuda().eval()
    model.load_state_dict(ck['model'])
    have = set(os.listdir(f'{DATA}/taps/{taps_key}'))
    corpus = Corpus(DATA, taps_key, layers=cfg.taps, pca=pca)
    trows = corpus.train_rows([r for r in load_manifest(DATA, split='val') if r['audio_sha256'] in have])
    sweeps = {'density_notes_per_second': [(1.0, 1.5), (3.0, 4.0), (8.0, 10.0)],
              'hold_head_ratio': [(0.0, 0.02), (0.1, 0.2), (0.4, 0.5)]}
    rng = np.random.default_rng(seed)
    res = {f: {'monotonic': 0, 'cores': 0, 'stats': []} for f in sweeps}
    tries = 0
    while min(v['cores'] for v in res.values()) < n_cores and tries < 20 * n_cores:
        tries += 1
        row = trows[rng.integers(len(trows))]
        try:
            chart = corpus.chart(row['chart_id'])
            s = sample_request(chart, seed=int(rng.integers(2 ** 31)), audio_id=row['song'], audio_duration_us=row['duration_us'],
                               mix={'from_scratch': 1.0}, empty_p=0.0, boundary_p=0.0, control_dropout=False)
            c0 = F.choose_core(s, np.random.default_rng(tries), row['duration_us'])
        except (F.SkipExample, ValueError, AssertionError):
            continue
        song = corpus.song(row['song'])
        li = F.line_index_of(chart.events)
        for field, bands in sweeps.items():
            if res[field]['cores'] >= n_cores:
                continue
            vals = []
            for band in bands:
                req = {**s.request, 'controls': {**s.request['controls'], field: band}}
                s2 = dataclasses.replace(s, request=req)
                ex = F.build_core(s2, chart.events, c0, row['duration_us'], line_index=li)
                ex.groups = []
                holds, runs, heads = core_context(s2, ex, li)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gen = generate_core(model, ex, song, holds, runs, heads, seed=tries, device='cuda')
                heads_n = sum(len(g['notes']) for g in gen)
                holds_n = sum('end_rel_us' in n for g in gen for n in g['notes'])
                vals.append(heads_n / 8.0 if field.startswith('density') else holds_n / max(heads_n, 1))
            res[field]['cores'] += 1
            res[field]['monotonic'] += int(vals[0] <= vals[1] <= vals[2] and vals[0] < vals[2])
            res[field]['stats'].append(vals)
    out = {'run': run_id, 'step': ck['step']}
    for f, v in res.items():
        st = np.array(v['stats']) if v['stats'] else np.zeros((0, 3))
        out[f] = {'cores': v['cores'], 'direction_correct': round(v['monotonic'] / max(v['cores'], 1), 3),
                  'mean_by_band': [round(float(x), 3) for x in st.mean(0)] if len(st) else None, 'bands': sweeps[f]}
    with open(f'{DATA}/runs/{run_id}/eval_controls.json', 'w') as fo:
        json.dump(out, fo, indent=1)
    data.commit()
    return out


@app.local_entrypoint()
def controls(run_id: str, n_cores: int = 80):
    r = eval_controls.remote(run_id, n_cores=n_cores)
    Path(f'reports/controls_{run_id}.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


@app.function(image=image, volumes={DATA: data}, cpu=2.0, memory=8192, timeout=1800)
def profile_data(taps_key: str = 'v3_pca512q_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'pca512_9-16-24',
                 n_batches: int = 12, micro_batch: int = 8, pool: int = POOL):
    """Single-process cProfile of the training data path (draw -> collate), to find the CPU hot spots."""
    import cProfile
    import io
    import pstats
    from beatfcn.model.batch import collate
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import CoreSampler
    corpus = Corpus(root=DATA, taps_key=taps_key, layers=taps, pca=pca)
    rows = [r for r in load_manifest(DATA) if r['split'] == 'train']
    load = (lambda sha: corpus.song(sha, in_memory=True)) if pool else corpus.song
    smp = CoreSampler(corpus.train_rows(rows), corpus.chart, load, 1, salient=corpus.salient_us, pool=pool,
                      residence=RESIDENCE)
    for _ in range(2):  # warm caches the same way a long-running worker would
        items = smp.draw(4)
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    for _ in range(n_batches):
        items = []
        while len(items) < micro_batch:
            items += smp.draw(min(4, micro_batch - len(items)))
        collate([a for a, _ in items], [b for _, b in items], taps, max(taps))
    pr.disable()
    dt = time.time() - t0
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats('cumulative').print_stats(45)
    return {'s_per_core': dt / (n_batches * micro_batch), 'skipped': dict(smp.skipped), 'profile': out.getvalue()[-9000:]}


@app.local_entrypoint()
def profile(n_batches: int = 12, pool: int = POOL):
    r = profile_data.remote(n_batches=n_batches, pool=pool)
    print(json.dumps({k: v for k, v in r.items() if k != 'profile'}))
    print(r['profile'])


@app.function(image=image, gpu='L40S', volumes={DATA: data}, cpu=12.0, memory=49152, timeout=3600)
def bench_train_gpu(taps_key: str = 'v4_i8_L9-16-24', taps: tuple = (9, 16, 24), pca: str = 'i8_9-16-24',
                    tap_dim: int = 1024, n_cores: int = 384, steps: int = 24, variant: dict | None = None,
                    configs: tuple = ('pad_m8x4', 'packed_m8x4', 'packed_m32')):
    """GPU-side cost of one optimiser step (32 cores) with data already prepared on the host (pinned).

    Isolates what the device needs per step, so CPU workers can be sized to it, not guessed."""
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus, load_manifest
    from beatfcn.train.loop import to_device
    tdir = f'{DATA}/taps/{taps_key}'
    have = {s for s in os.listdir(tdir) if os.path.exists(f'{tdir}/{s}/S20.npy')}
    rows = [r for r in load_manifest(DATA) if r['audio_sha256'] in have and r['split'] == 'train']
    corpus_args = {'root': DATA, 'taps_key': taps_key, 'layers': tuple(taps), 'pca': pca}
    t0 = time.time()
    loader = torch.utils.data.DataLoader(_stream(rows, corpus_args, 1, 8, tuple(taps), max(taps)), batch_size=None,
                                         num_workers=10, prefetch_factor=2)
    it = iter(loader)
    m8 = [next(it)[0] for _ in range(n_cores // 8)]
    build_s = time.time() - t0
    loader = torch.utils.data.DataLoader(_stream(rows, corpus_args, 2, 32, tuple(taps), max(taps)), batch_size=None,
                                         num_workers=10, prefetch_factor=2)
    it = iter(loader)
    m32 = [next(it)[0] for _ in range(n_cores // 32)]
    pin = lambda bt: {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in bt.items()}
    m8, m32 = [pin(b) for b in m8], [pin(b) for b in m32]
    dev = torch.device('cuda')
    mon, out = GpuUtil(), {'gpu': torch.cuda.get_device_name(), 'songs': len({r['audio_sha256'] for r in rows}),
                           'build_s': round(build_s, 1), 'rows': []}
    for cfg in configs:
        torch.manual_seed(0)
        mc = P1Config(taps=tuple(taps), top_tap=max(taps), tap_dim=tap_dim, **{**(variant or {}), 'packed': 'packed' in cfg})
        model = P1Model(mc).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        fwd = model
        micro, acc = (m8, 4) if '_m8' in cfg else (m32, 1)
        sync = 'sync' in cfg
        k = 0

        def step():
            nonlocal k
            for _ in range(acc):
                bt = to_device(micro[k % len(micro)], dev)
                k += 1
                bt['w_select'] = 0.2
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = fwd(bt)
                (loss / acc).backward()
                if sync:
                    torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        try:
            tw = time.time()
            for _ in range(6 if 'packed' in cfg else 3):  # warm-up (FlexAttention compiles for the first shapes)
                step()
            torch.cuda.synchronize()
            warm = time.time() - tw
            mon.take()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            dt = (time.time() - t) / steps
            row = {'config': cfg, 's_per_step': round(dt, 3), 'util': mon.take()['util_mean'], 'warmup_s': round(warm, 1),
                   'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        except Exception as exc:  # report and continue with the other configs
            row = {'config': cfg, 'error': repr(exc)[:300]}
        out['rows'].append(row)
        print(json.dumps(row), flush=True)
        del model, opt, fwd
        torch.cuda.empty_cache()
    mon.close()
    return out


@app.local_entrypoint()
def bench_gpu(gpus: str = 'L40S,H100', timeout: int = 900):
    calls = {g: bench_train_gpu.with_options(gpu=g, timeout=timeout).spawn() for g in gpus.split(',')}
    for g, c in calls.items():
        print(json.dumps(c.get(), indent=1))

"""Cross-order real-pipeline comparison of asynchronous loading and resident cohorts."""
import hashlib
import json
import math
from pathlib import Path
import time
import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.condition_training_app import ROOT, POLICY, RUNS, check_source
from beatfcn.cloud.rhythm_diagnostic_app import source_digest, save

app = modal.App('beatfcn-condition-cohort-probe')
data = training.data
image = training.image.add_local_file(__file__, '/probe/cohort_probe.py')
STAGE = 'condition-cohort-probe-v1'
RESOURCES = {'gpu': 'L40S', 'cpu': 12, 'memory_gib': 48, 'timeout_s': 420,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}


class ResidentCohorts:
    """Load each consecutive batch cohort before issuing its model computations.

    Unlike asynchronous look-ahead, no background song reads/allocations continue
    while a cohort is being consumed. Every sample and its order are preserved.
    The consumer must not prefetch into the next cohort before finishing this one.
    """

    def __init__(self, batches, pool, *, max_songs, max_batches=128, before_load=None, progress=None):
        if max_songs < 1 or max_batches < 1:
            raise ValueError('cohort bounds must be positive')
        self.batches, self.pool = batches, pool
        self.max_songs, self.max_batches = max_songs, max_batches
        self.before_load, self.progress = before_load, progress

    def __iter__(self):
        pending, keys, number = [], set(), 0

        def consume():
            nonlocal number
            started = time.perf_counter()
            if self.before_load:
                self.before_load()
            drained = time.perf_counter()
            self.pool.request(keys)
            try:
                self.pool.wait_idle()
                if self.progress:
                    self.progress({'cohort': number, 'microbatches': len(pending), 'songs': len(keys),
                                   'drain_s': drained - started, 'load_s': time.perf_counter() - drained})
                number += 1
                yield from pending
            finally:
                self.pool.release(keys)

        for item in self.batches:
            needed = set(item[0]['_audio']['keys'])
            if len(needed) > self.max_songs:
                raise ValueError('one batch exceeds the cohort song bound')
            if pending and (len(pending) >= self.max_batches or len(keys | needed) > self.max_songs):
                yield from consume()
                pending, keys = [], set()
            pending.append(item)
            keys.update(needed)
        if pending:
            yield from consume()


def assess_probe(report):
    if not report.get('complete') or len(report['phases']) != 4:
        raise ValueError('real pipeline probe is incomplete')
    groups = {(r['panel'], r['cohort']): r for r in report['phases']}
    if len(groups) != 4 or set(groups) != {(p, c) for p in ('A', 'B') for c in (False, True)}:
        raise ValueError('cross-order comparison coverage differs')
    for panel in ('A', 'B'):
        a, b = [groups[panel, c] for c in (False, True)]
        if (a['input_provenance_sha256'] != b['input_provenance_sha256'] or
                any(r['microbatches'] != 128 or not r['finite_gradients'] for r in (a, b))):
            raise ValueError('input sequence or finite computation differs')
        if (len(a['losses']) != 256 or len(b['losses']) != 256 or
                not all(math.isfinite(v) for r in (a, b) for v in r['losses']) or
                max(abs(x - y) for x, y in zip(a['losses'], b['losses'])) > 1e-4):
            raise ValueError('same-input forward losses differ beyond the numerical gate')
    baseline = sum(groups[p, False]['wall_s'] for p in ('A', 'B'))
    changed = sum(groups[p, True]['wall_s'] for p in ('A', 'B'))
    if changed >= baseline * .9:
        raise ValueError('cohorts did not improve total pipeline time by at least 10%')
    return {'asynchronous_s': baseline, 'cohort_s': changed, 'speedup': baseline / changed,
            'scope': 'two cross-order panels, including preload time; original long-run guard still required'}


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(12, 12), memory=(49152, 49152),
              timeout=420, retries=0, max_containers=1, scaledown_window=2)
def probe(expected_source, expected_script):
    from collections import deque
    import gc
    import io
    import numpy as np
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import attach_audio, unpack_batch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus
    from beatfcn.train.density import FIELD_INDEX, ranking_loss, replay_rng, rng_state
    from beatfcn.train.loop import to_device
    from beatfcn.train.pinned import PinnedRing
    from beatfcn.train.pool import SongPool
    from beatfcn.train.tape import MetadataTape
    started = time.perf_counter()
    check_source(expected_source)
    if hashlib.sha256(Path('/probe/cohort_probe.py').read_bytes()).hexdigest() != expected_script:
        raise ValueError('probe script changed')
    output = Path(ROOT) / STAGE
    if output.exists():
        raise ValueError('probe already exists')
    output.mkdir()
    torch.set_num_threads(4)
    models = []
    for run, sha in zip(RUNS, ('169546aa06c3f332a4d7bfc4189dd1fe6d085d4fdba574533dfc574f7567a99d',
                               '32389b3ea6927df9d2d141122fec8bfc9515996a94e2d9a29def998223a9b269')):
        payload = Path(f'/data/runs/{run}/last.pt').read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('checkpoint identity differs')
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        model = P1Model(P1Config(**ck['config'])).cuda().train()
        model.load_state_dict(ck['model'])
        models.append(model)
    corpus, rows_sha = Corpus(**ck['data']), ck['rows_sha256']
    del ck, payload
    tape = MetadataTape(ROOT + '/tape', rows_sha256=rows_sha, policy=POLICY, micro_batch=8, skip_batches=600,
                        progress=lambda r: print(json.dumps({'event': 'tape', 'batches': r['batches']}), flush=True)
                        if r['batches'] % 1024 == 0 else None)
    panels = {'A': tape.batches[600:728], 'B': tape.batches[728:856]}
    scales = torch.stack([torch.broadcast_to(torch.as_tensor(np.asarray(corpus.scale[l], np.float32)), (model.c.tap_dim,))
                          for l in model.c.taps]).clone().cuda()
    maxima = {}
    for rows in panels.values():
        for batch, _ in rows:
            for key, value in batch['_packed'].items():
                if key not in maxima or value.numel() > maxima[key].numel():
                    maxima[key] = value
    ring = PinnedRing(depth=40, pin=True)
    ring.preallocate(maxima)
    report = {'complete': False, 'training_updates': 0, 'source_sha256': expected_source,
              'script_sha256': expected_script, 'phases': [], 'startup_s': time.perf_counter() - started,
              'scope': '128 actual microbatches per panel, same order/inputs/weights, two models plus density negatives; no optimizer update.',
              'limitations': 'Short cross-order comparison; cold volume cache can differ. Full training must still pass original 480s utilization guard.'}
    def record():
        report['function_wall_s'] = time.perf_counter() - started
        save(output / 'result.json', report)
        data.commit()
    mon = GpuUtil()
    try:
        for panel, cohort in [('A', False), ('A', True), ('B', True), ('B', False)]:
            if time.perf_counter() - started > 330:
                report['incomplete_reason'] = 'remaining wall-clock reserve too small'
                break
            pool = SongPool(training._song_loader(corpus), model.c.taps, 'cuda', capacity=40, loaders=4, mel=False)
            details = []
            source = ((dict(b), dict(sk)) for b, sk in panels[panel])
            if cohort:
                source = ResidentCohorts(source, pool, max_songs=40, max_batches=128,
                                         before_load=torch.cuda.synchronize, progress=details.append)
            source = iter(source)
            queue, exhausted = deque(), False
            prefetch = 1 if cohort else 16
            torch.manual_seed(20261024)
            torch.cuda.synchronize()
            mon.take()
            begin = time.perf_counter()
            count, input_hash = 0, hashlib.sha256()
            timing = {'get_batch_s': 0., 'attach_s': 0., 'model_issue_s': 0.}
            losses = []
            batch = loss = None
            try:
                while not exhausted or queue:
                    t = time.perf_counter()
                    while not exhausted and len(queue) < prefetch:
                        try:
                            packed, _ = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        # Provenance and every packed value are bound to the immutable tape.
                        input_hash.update(json.dumps(packed['_provenance'], sort_keys=True).encode())
                        packed['_packed'], slot = ring.take(packed['_packed'])
                        batch = unpack_batch(packed)
                        keys = list(batch['_audio']['keys'])
                        pool.request(keys)
                        queue.append((batch, keys, slot))
                    timing['get_batch_s'] += time.perf_counter() - t
                    if not queue:
                        break
                    batch, keys, slot = queue.popleft()
                    t = time.perf_counter()
                    attach_audio(batch, pool.get, 'cuda', scales)
                    batch = to_device(batch, torch.device('cuda'))
                    timing['attach_s'] += time.perf_counter() - t
                    t = time.perf_counter()
                    state = rng_state(torch.device('cuda'))
                    for index, current in enumerate(models):
                        current.zero_grad(set_to_none=True)
                        terms = {} if index and batch.get('_density_cf') else None
                        with replay_rng(state, torch.device('cuda')), torch.autocast('cuda', dtype=torch.bfloat16):
                            loss, _ = current(dict(batch), density_terms=terms)
                        if terms is not None:
                            wrong = dict(batch)
                            wrong['controls'] = batch['controls'].clone()
                            wrong['controls'][:, FIELD_INDEX] = batch['density_cf_wrong_band']
                            negative = {}
                            with replay_rng(state, torch.device('cuda')), torch.autocast('cuda', dtype=torch.bfloat16):
                                unused, _ = current(wrong, density_terms=negative)
                            del unused
                            loss = loss + .12 * ranking_loss(terms, negative)[0]
                        loss.backward()
                        losses.append(loss.detach())
                    timing['model_issue_s'] += time.perf_counter() - t
                    ring.done(slot, torch.cuda.current_stream())
                    pool.release(keys)
                    count += 1
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                util = mon.take()
                loss_values = torch.stack(losses).float().cpu().tolist()
                finite = all(bool(torch.isfinite(p.grad).all()) for current in models for p in current.parameters() if p.grad is not None)
                case = {'panel': panel, 'cohort': cohort, 'wall_s': elapsed, 'microbatches': count,
                        'gpu': util, 'timing': timing, 'preload': details, 'pool': dict(pool.stats),
                        'input_provenance_sha256': input_hash.hexdigest(), 'finite_gradients': finite,
                        'losses': loss_values, 'peak_gib': torch.cuda.max_memory_allocated() / 2**30}
                previous = next((r for r in report['phases'] if r['panel'] == panel), None)
                if previous:
                    case['same_input_order'] = case['input_provenance_sha256'] == previous['input_provenance_sha256']
                    case['max_loss_difference'] = max(abs(a - b) for a, b in zip(loss_values, previous['losses']))
                    if not case['same_input_order'] or len(loss_values) != len(previous['losses']):
                        raise ValueError('cohort changed batch coverage or order')
                report['phases'].append(case)
                print(json.dumps({'event': 'cohort_probe', **{k: v for k, v in case.items() if k != 'losses'}}), flush=True)
                record()
                if not finite or count != 128:
                    raise ValueError('incomplete or nonfinite real-pipeline phase')
            finally:
                source.close()
                pool.pool.shutdown()
                for current in models:
                    current.zero_grad(set_to_none=True)
                del pool, batch, loss, losses
                gc.collect()
    finally:
        mon.close()
    report['complete'] = len(report['phases']) == 4
    record()
    return report


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve, update_stage
    root, sha = training.REPO, source_digest(training.REPO / 'beatfcn')
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ledger = root / 'reports/rhythm_v2_budget.json'
    reserve(ledger, STAGE, RESOURCES, json.loads((root / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((root / 'reports/rhythm_v2_connection_status.json').read_text()),
            app_id=app.app_id, source_sha256=sha)
    try:
        call = probe.spawn(sha, script_sha)
        update_stage(ledger, STAGE, status='running', function_call_id=call.object_id, script_sha256=script_sha)
        result = call.get()
        save(root / f'reports/{STAGE}.json', result)
        update_stage(ledger, STAGE, status='complete' if result['complete'] else 'incomplete', result=result, ended_at=time.time())
    except BaseException as exc:
        update_stage(ledger, STAGE, status='observation_error', error=f'{type(exc).__name__}: {exc}')
        raise

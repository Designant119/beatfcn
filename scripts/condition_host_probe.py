"""Fixed real-input GC/host contention probe. Never writes model checkpoints."""
import hashlib
import json
from pathlib import Path
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.condition_training_app import ROOT, POLICY, RUNS, check_source
from beatfcn.cloud.rhythm_diagnostic_app import source_digest, save

app = modal.App('beatfcn-condition-host-probe')
data = training.data
image = training.image.add_local_file(__file__, '/probe/host_probe.py')
RESOURCES = {'gpu': 'L40S', 'cpu': 12, 'memory_gib': 48, 'timeout_s': 420,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}
STAGE = 'condition-host-gc-v2'


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(12, 12), memory=(49152, 49152),
              timeout=420, retries=0, max_containers=1, scaledown_window=2)
def probe(expected_source: str, expected_script: str):
    import gc
    import io
    import numpy as np
    import torch
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.model.batch import attach_audio, unpack_batch
    from beatfcn.model.p1 import P1Model, P1Config
    from beatfcn.train.data import Corpus
    from beatfcn.train.density import FIELD_INDEX, ranking_loss, replay_rng, rng_state
    from beatfcn.train.loop import to_device
    from beatfcn.train.pool import SongPool
    from beatfcn.train.tape import MetadataTape
    started = time.perf_counter()
    check_source(expected_source)
    if hashlib.sha256(Path('/probe/host_probe.py').read_bytes()).hexdigest() != expected_script:
        raise ValueError('probe script identity differs')
    output = Path(ROOT) / STAGE
    if output.exists():
        raise ValueError('probe already exists; inspect exact call before any retry')
    output.mkdir()
    torch.set_num_threads(4)
    result = {'complete': False, 'training_updates': 0, 'source_sha256': expected_source,
              'script_sha256': expected_script, 'phases': [], 'startup': {},
              'scope': 'Same real microbatches and weights; GC alternation, no optimizer updates.'}
    def record():
        result['function_wall_s'] = time.perf_counter() - started
        save(output / 'result.json', result)
        data.commit()
    def progress(event, **values):
        print(json.dumps({'event': event, 'elapsed_s': time.perf_counter() - started, **values}), flush=True)
    def load_checkpoint(run, expected):
        payload = Path(f'/data/runs/{run}/last.pt').read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('3650 checkpoint changed')
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if ck['step'] != 3650 or 'opt' not in ck:
            raise ValueError('expected resumable 3650 checkpoint')
        model = P1Model(P1Config(**ck['config'])).cuda().train()
        model.load_state_dict(ck['model'])
        return model, ck
    before = time.perf_counter()
    model, ck = load_checkpoint(RUNS[0], '169546aa06c3f332a4d7bfc4189dd1fe6d085d4fdba574533dfc574f7567a99d')
    contrastive, second = load_checkpoint(RUNS[1], '32389b3ea6927df9d2d141122fec8bfc9515996a94e2d9a29def998223a9b269')
    corpus = Corpus(**ck['data'])
    rows_sha = ck['rows_sha256']
    del ck, second
    result['startup']['checkpoint_s'] = time.perf_counter() - before
    progress('models_loaded')
    before = time.perf_counter()
    tape = MetadataTape(ROOT + '/tape', rows_sha256=rows_sha, policy=POLICY, micro_batch=8, skip_batches=600,
                        progress=lambda r: progress('metadata_loading', batches=r['batches']) if r['batches'] % 512 == 0 else None)
    result['startup']['tape_s'] = time.perf_counter() - before
    pool = SongPool(training._song_loader(corpus), model.c.taps, 'cuda', capacity=40, loaders=4, mel=False)
    raw = [dict(b) for b, _ in tape.batches[600:608]]
    keys = sorted({s for b in raw for s in b['_audio']['keys']})
    pool.request(keys)
    scales = torch.stack([torch.broadcast_to(torch.as_tensor(np.asarray(corpus.scale[l], np.float32)), (model.c.tap_dim,))
                          for l in model.c.taps]).clone().cuda()
    batches = []
    for packed in raw:
        batch = unpack_batch(packed)
        attach_audio(batch, pool.get, 'cuda', scales)
        batches.append(to_device(batch, torch.device('cuda')))
    pool.wait_idle()
    torch.cuda.synchronize()
    result['fixture'] = {'microbatches': len(batches), 'cores': len(batches) * 8,
                         'condition_microbatches': sum(bool(b.get('_density_cf')) for b in batches),
                         'songs': len(keys), 'metadata_batches_retained': len(tape.batches)}
    if not result['fixture']['condition_microbatches']:
        raise ValueError('fixture must exercise actual density contrastive computation')
    models = [model, contrastive]
    def compute(batch):
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
        return loss.detach()
    for batch in batches[:2]:
        compute(batch)
    torch.cuda.synchronize()
    progress('fixture_warm', **result['fixture'])
    record()
    pauses, current_gc = [], {}
    def on_gc(phase, info):
        now = time.perf_counter()
        if phase == 'start':
            current_gc[info['generation']] = now
        elif info['generation'] in current_gc:
            pauses.append({'generation': info['generation'], 'seconds': now - current_gc.pop(info['generation']),
                           'collected': info['collected'], 'uncollectable': info['uncollectable']})
    gc.callbacks.append(on_gc)
    was_enabled = gc.isenabled()
    monitor = GpuUtil()
    background = None
    reference_grads = None
    try:
        # Repeat each GC setting; no asynchronous disk reader in the first four cases.
        # The final pair restores real pool loading with exactly the same model inputs.
        for phase, enabled, loading in [('on-a', True, False), ('on-b', True, False),
                                         ('off-a', False, False), ('off-b', False, False),
                                         ('loading-on', True, True), ('loading-off', False, True)]:
            if time.perf_counter() - started > 340:
                result['incomplete_reason'] = 'wall-clock reserve before remaining phases'
                break
            gc.collect()
            pauses.clear()
            if loading:
                background = SongPool(training._song_loader(corpus), model.c.taps, 'cuda', capacity=24, loaders=4, mel=False)
                other = sorted({s for b, _ in tape.batches[640:704] for s in b['_audio']['keys']} - set(keys))[:16]
                background.request(other)
            if enabled:
                gc.enable()
            else:
                gc.disable()
            torch.manual_seed(20261024)
            torch.cuda.synchronize()
            monitor.take()
            before = time.perf_counter()
            for _ in range(2):
                for batch in batches:
                    loss = compute(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            case = {'phase': phase, 'gc_enabled': enabled, 'background_loading': loading,
                    'microbatches': 16, 'paired_model_core_passes': 256, 'wall_s': elapsed,
                    'gpu': monitor.take(), 'gc_events': list(pauses),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'loss': float(loss)}
            gc.enable()
            gradients = {f'{index}:{name}': p.grad.detach().cpu() for index, current in enumerate(models)
                         for name, p in current.named_parameters() if p.grad is not None}
            if reference_grads is None:
                reference_grads = gradients
            case['finite_gradients'] = all(bool(torch.isfinite(g).all()) for g in gradients.values())
            case['gradient_max_abs_difference'] = (max(float((g - reference_grads[name]).abs().max()) for name, g in gradients.items())
                                                   if case['finite_gradients'] else None)
            case['gradient_relative_l2_difference'] = (float((sum((g.double() - reference_grads[name].double()).square().sum()
                                                                 for name, g in gradients.items()) /
                                                              sum(g.double().square().sum() for g in reference_grads.values()).clamp(min=1e-30)).sqrt())
                                                        if case['finite_gradients'] else None)
            case['strict_numerical_gate'] = case['finite_gradients'] and case['gradient_max_abs_difference'] <= 1e-4
            if background is not None:
                background.wait_idle()
                case['background_pool'] = dict(background.stats)
                background.pool.shutdown()
                background = None
            result['phases'].append(case)
            progress('host_probe_phase', **case)
            record()
            if not case['finite_gradients']:
                raise ValueError('nonfinite gradients; failure evidence saved before stopping')
    finally:
        gc.callbacks.remove(on_gc)
        gc.enable() if was_enabled else gc.disable()
        monitor.close()
        pool.pool.shutdown()
        if background is not None:
            background.pool.shutdown()
    result['complete'] = len(result['phases']) == 6
    result['strict_numerical_gate'] = all(p['strict_numerical_gate'] for p in result['phases'])
    record()
    return result


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve, update_stage
    root = training.REPO
    ledger = root / 'reports/rhythm_v2_budget.json'
    sha = source_digest(root / 'beatfcn')
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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

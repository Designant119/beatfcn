"""Bounded, paired density-learning experiment; each dispatch uses the campaign ledger."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import modal

from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import CHECKPOINTS, source_digest, save

REPO = training.REPO
app = modal.App('beatfcn-condition-training')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = training.image.add_local_file(str(REPO / 'bench/capability-v1/manifest.json'), '/bench/manifest.json')
ROOT = '/data/evolution/capability-evolution-20261024/condition-v1'
POLICY = {'request_mix': {'from_scratch': .4, 'middle_holes': .3, 'continuation': .1,
                          'sparse_anchors': .1, 'reference': .1},
          'control_drop_all_p': .2, 'paired_references': False, 'rollout_fraction': 0.,
          'stage_a_arm': 'legacy', 'density_window_every': 4}
VARIANT = {'compact_inner': True, 'lane_history_groups': 0, 'offset_sigma_ms': 0.,
           'selection': False, 'phira_context': 'legacy', 'phira_location': False}
RESOURCES = {
    'prepare': {'cpu': 8, 'memory_gib': 32, 'gpu': None, 'timeout_s': 900},
    'train': {'cpu': 12, 'memory_gib': 48, 'gpu': 'L40S', 'timeout_s': 3000},
}
STAGES = {'prepare': 'shared_cpu_preparation', 'train': 'matched_training'}
RUNS = ['r02-density-supervised', 'r02-density-contrastive']


def check_source(expected):
    import beatfcn
    actual = source_digest(Path(beatfcn.__file__).parent)
    if actual != expected:
        raise ValueError('uploaded source differs from reserved source')


def checkpoint():
    import torch
    path, sha = CHECKPOINTS['3500']
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError('source checkpoint identity differs')
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)


@app.function(image=image, volumes={'/data': data}, cpu=(8, 8), memory=(32768, 32768),
              timeout=900, retries=0, max_containers=1, scaledown_window=2)
def prepare(expected_source_sha256: str):
    from collections import Counter
    import torch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.train.tape import policy_digest
    torch.set_num_threads(1)
    check_source(expected_source_sha256)
    started = time.perf_counter()
    root = Path(ROOT)
    if root.exists():
        raise ValueError('preparation already exists; inspect it, do not overwrite')
    root.mkdir(parents=True)
    tape = root / 'tape'
    tape.mkdir()
    ck = checkpoint()
    ca, cfg, rows_sha = ck['data'], ck['config'], ck['rows_sha256']
    rows, val_rows, info = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'],
                                                '', 0, '', ck['manifest_name'])
    if info['rows_sha256'] != rows_sha:
        raise ValueError('source data selection changed')
    heldout = {r['audio_sha256'] for r in json.loads(Path('/bench/manifest.json').read_text())['rows']}
    if heldout & {r['audio_sha256'] for r in rows}:
        raise ValueError('benchmark songs appear in training selection')
    del ck
    corpus = Corpus(**ca)
    vsmp = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song_info, 7,
                      salient=corpus.salient_us, stage_a_arm='legacy')
    val = training._fixed_val(vsmp, 40, 8)
    if len(val) != 40:
        raise ValueError('fixed validation set is incomplete')
    prep_key = 'rhythm-density-v1'
    prep_path = Path('/data/prep') / (prep_key + '.pt')
    if prep_path.exists():
        raise ValueError('validation preparation already exists')
    torch.save({'val': val, 'audit': None, 'info': info, 'policy': POLICY}, prep_path)
    del corpus, vsmp, val
    seed = 20261024
    loader = torch.utils.data.DataLoader(training._stream(rows, ca, seed, 8, tuple(cfg['taps']), cfg['top_tap'],
                                                         pool=1, training_policy=POLICY),
                                        batch_size=None, num_workers=8, prefetch_factor=4, in_order=False)
    iterator = iter(loader)
    manifest = {'schema': 'beatfcn.metadata_tape.v1', 'complete': False, 'rows_sha256': rows_sha,
                'policy_sha256': policy_digest(POLICY), 'policy': POLICY, 'micro_batch': 8,
                'seed': seed, 'checkpoint': CHECKPOINTS['3500'][0], 'source_sha256': expected_source_sha256,
                'parts': [], 'batches': 0, 'prep_key': prep_key}
    exposure, densities, songs, charts = Counter(), Counter(), set(), set()
    with (tape / 'provenance.jsonl').open('w') as provenance:
        while manifest['batches'] < 4000:
            if time.perf_counter() - started > 820:
                raise TimeoutError('CPU preparation wall-clock bound before completion')
            chunk = [next(iterator) for _ in range(min(32, 4000 - manifest['batches']))]
            for batch, _ in chunk:
                records = batch['_provenance']
                provenance.write(json.dumps(records) + '\n')
                for r in records:
                    exposure[f"{r['mode']}:{r['kind']}:cores"] += 1
                    exposure[f"{r['mode']}:{r['kind']}:notes"] += r['notes']
                    songs.add(r['song'])
                    charts.add(r['chart_id'])
                    if r['kind'] == 'density_window':
                        densities[f"{r['mode']}:band{r['density_true_band']}"] += 1
            buf = io.BytesIO()
            torch.save(chunk, buf)
            raw = buf.getvalue()
            encoded = gzip.compress(raw, compresslevel=1, mtime=0)
            path = tape / f'part-{len(manifest["parts"]):04d}.pt.gz'
            path.write_bytes(encoded)
            manifest['parts'].append({'file': path.name, 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
                                      'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                      'codec': 'gzip', 'batches': len(chunk)})
            manifest['batches'] += len(chunk)
            if manifest['batches'] % 128 == 0:
                save(tape / 'progress.json', {**manifest, 'wall_s': time.perf_counter() - started})
                print(json.dumps({'event': 'condition_metadata', 'batches': manifest['batches'],
                                  'wall_s': time.perf_counter() - started}), flush=True)
    del iterator, loader
    fraction = sum(densities.values()) / 32000
    if not .18 <= fraction <= .32 or len(songs) < 100:
        raise ValueError('unexpected condition exposure or insufficient song diversity')
    manifest.update(complete=True, exposure=dict(exposure), density_bands=dict(densities), density_fraction=fraction,
                    unique_songs=len(songs), unique_charts=len(charts), wall_s=time.perf_counter() - started)
    save(tape / 'manifest.json', manifest)
    data.commit()
    return manifest


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(12, 12), memory=(49152, 49152),
              timeout=3000, retries=0, max_containers=1, scaledown_window=2)
def train(expected_source_sha256: str):
    import torch
    torch.set_num_threads(4)
    check_source(expected_source_sha256)
    ck = checkpoint()
    del ck
    manifest = json.loads((Path(ROOT) / 'tape/manifest.json').read_text())
    if not manifest['complete'] or manifest['source_sha256'] != expected_source_sha256:
        raise ValueError('training must consume the frozen, audited preparation')
    if any((Path('/data/runs') / run).exists() for run in RUNS):
        raise ValueError('paired run already exists; inspect its exact continuation state')
    result = training.train.local(
        run_id=RUNS[0], also=[{'run_id': RUNS[1], 'variant': VARIANT, 'density_cf_weight': .2}],
        steps=8000, stop_at=4500, round_updates=1000, initialize_from=CHECKPOINTS['3500'][0],
        source_sha256=expected_source_sha256, micro_batch=8, accumulation=4, workers=8,
        val_cores=40, val_songs=8, lr=.0002, warmup=500, validate_every=250,
        taps_key='v4_i8_L9-16-24', taps=(9, 16, 24), pca='i8_9-16-24', tap_dim=1024,
        pool_songs=40, pool_per_worker=1, prefetch=16, loaders=4, graph=False,
        min_util=70., check_after_s=480., max_data_wait=.8, check_data_after_s=480., max_wall_s=2850.,
        manifest_name='charts.offset-v2-3d99ff6bd5e7.jsonl', variant=VARIANT,
        training_policy=POLICY, prep=manifest['prep_key'], review_start_step=3500,
        data_seed_step=3500, tape_start_step=3500, metadata_tape=ROOT + '/tape',
        trace_after_s=(300., 480.), trace_updates=2)
    paths = {}
    for run in RUNS:
        path = Path('/data/runs') / run / 'last.pt'
        paths[run] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result.update(checkpoints=paths, source_sha256=expected_source_sha256,
                  complete=result['round_effective_updates'] == 1000 and not result['aborted'])
    save(Path(ROOT) / 'training_result.json', result)
    data.commit()
    return result


@app.local_entrypoint()
def run(phase: str = 'prepare'):
    from beatfcn.train.campaign_budget import reserve, update_stage
    if phase not in STAGES:
        raise ValueError('phase must be prepare or train')
    stage = STAGES[phase]
    ledger = REPO / 'reports/rhythm_v2_budget.json'
    connection = json.loads((REPO / 'reports/rhythm_v2_connection_status.json').read_text())
    rates = json.loads((REPO / 'reports/rhythm_v2_rates.json').read_text())
    sha = source_digest(REPO / 'beatfcn')
    resources = {**RESOURCES[phase], 'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}
    reserve(ledger, stage, resources, rates, connection, app_id=app.app_id, source_sha256=sha)
    try:
        call = (prepare if phase == 'prepare' else train).spawn(sha)
        update_stage(ledger, stage, status='running', function_call_id=call.object_id)
        result = call.get()
        save(REPO / f'reports/condition_v1_{phase}.json', result)
        update_stage(ledger, stage, status='complete' if result['complete'] else 'incomplete',
                     result=result, ended_at=time.time())
        print(json.dumps({'phase': phase, 'complete': result['complete'],
                          'wall_s': result.get('wall_s', result.get('function_wall_s'))}))
    except BaseException as exc:
        update_stage(ledger, stage, status='observation_error', error=f'{type(exc).__name__}: {exc}', observed_at=time.time())
        raise

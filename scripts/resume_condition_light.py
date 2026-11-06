"""Continue the saved paired round with original loading and lightweight CPU/CUDA tracing."""
import hashlib
import io
import json
import math
from pathlib import Path
import time
import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.condition_training_app import ROOT, POLICY, VARIANT, RUNS, check_source
from beatfcn.cloud.rhythm_diagnostic_app import source_digest, save

app = modal.App('beatfcn-condition-light-trace-resume')
data = training.data
image = training.image.add_local_file(__file__, '/resume/light.py')
STAGE = 'matched-training-light-trace-resume-v1'
RESOURCES = {'gpu': 'L40S', 'cpu': 12, 'memory_gib': 48, 'timeout_s': 3000,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}
PARENT = '12c762a006473b3b7b08cf78837b908d3ed59456dbfdb336fdcafc4e0ac4cb58'
CHECKPOINT_HASHES = ('169546aa06c3f332a4d7bfc4189dd1fe6d085d4fdba574533dfc574f7567a99d',
                     '32389b3ea6927df9d2d141122fec8bfc9515996a94e2d9a29def998223a9b269')


def assess_existing_pipeline(report):
    if not report.get('complete'):
        raise ValueError('real pipeline diagnostic is incomplete')
    baseline = [r for r in report['phases'] if not r['cohort']]
    if len(baseline) != 2 or {r['panel'] for r in baseline} != {'A', 'B'}:
        raise ValueError('both unchanged-pipeline diagnostic windows are required')
    for row in baseline:
        if (row['microbatches'] != 128 or not row['finite_gradients'] or
                row['gpu']['util_mean'] is None or row['gpu']['util_mean'] < 70 or
                len(row['losses']) != 256 or not all(math.isfinite(v) for v in row['losses'])):
            raise ValueError('unchanged pipeline has not passed the short finite/utilization diagnostic')
    return {'diagnostic_windows': [{k: r[k] for k in ('panel', 'wall_s', 'gpu', 'microbatches')} for r in baseline],
            'scope': 'Original asynchronous pipeline, no detailed profiling. Does not establish 480-second acceptance.',
            'cohort_experiment': 'rejected: no total-time benefit; not enabled for continuation'}


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(12, 12), memory=(49152, 49152),
              timeout=3000, retries=0, max_containers=1, scaledown_window=2)
def resume(expected_source, script_sha):
    import torch
    torch.set_num_threads(4)
    check_source(expected_source)
    if hashlib.sha256(Path('/resume/light.py').read_bytes()).hexdigest() != script_sha:
        raise ValueError('continuation script changed')
    probe = json.loads((Path(ROOT) / 'condition-cohort-probe-v1/result.json').read_text())
    assessment = assess_existing_pipeline(probe)
    manifest = json.loads((Path(ROOT) / 'tape/manifest.json').read_text())
    if not manifest['complete'] or manifest['source_sha256'] != PARENT or manifest['policy'] != POLICY:
        raise ValueError('original frozen preparation differs')
    for run, sha in zip(RUNS, CHECKPOINT_HASHES):
        payload = Path(f'/data/runs/{run}/last.pt').read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('checkpoint moved; inspect current state before continuing')
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if (ck['step'] != 3650 or 'opt' not in ck or ck['training_policy'] != POLICY or
                ck['metadata_tape'] != ROOT + '/tape' or ck['tape_start_step'] != 3500):
            raise ValueError('continuation identity or optimizer missing')
        if not all(bool(torch.isfinite(t).all()) for t in ck['model'].values() if torch.is_tensor(t) and t.is_floating_point()):
            raise ValueError('nonfinite continuation weights')
        del ck, payload
    result = training.train.local(
        run_id=RUNS[0], also=[{'run_id': RUNS[1], 'variant': VARIANT, 'density_cf_weight': .2}],
        steps=8000, stop_at=4500, round_updates=1000, review_start_step=3500,
        source_sha256=expected_source, micro_batch=8, accumulation=4, workers=8,
        val_cores=40, val_songs=8, lr=.0002, warmup=500, validate_every=250,
        taps_key='v4_i8_L9-16-24', taps=(9, 16, 24), pca='i8_9-16-24', tap_dim=1024,
        pool_songs=40, pool_per_worker=1, prefetch=16, loaders=4, graph=False,
        min_util=70., check_after_s=480., max_data_wait=.8, check_data_after_s=480., max_wall_s=2850.,
        manifest_name='charts.offset-v2-3d99ff6bd5e7.jsonl', variant=VARIANT,
        training_policy=POLICY, prep=manifest['prep_key'], data_seed_step=3500,
        tape_start_step=3500, metadata_tape=ROOT + '/tape',
        trace_after_s=(300., 480.), trace_updates=1, trace_detailed=False)
    paths = {}
    for run in RUNS:
        path = Path('/data/runs') / run / 'last.pt'
        paths[run] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result.update(checkpoints=paths, source_sha256=expected_source, parent_source_sha256=PARENT,
                  resumed_checkpoint_sha256=dict(zip(RUNS, CHECKPOINT_HASHES)), runtime_diagnostic_assessment=assessment,
                  stochastic_continuation='3650 checkpoint has no RNG state; both arms restart with matched masks, weights/optimizer/order preserved',
                  complete=result['round_effective_updates'] == 1000 and not result['aborted'])
    save(Path(ROOT) / 'training_result.json', result)
    data.commit()
    return result


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve, update_stage
    root = training.REPO
    # Fail locally before reserving or dispatching a GPU continuation.
    assess_existing_pipeline(json.loads((root / 'reports/condition-cohort-probe-v1.json').read_text()))
    sha = source_digest(root / 'beatfcn')
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ledger = root / 'reports/rhythm_v2_budget.json'
    reserve(ledger, STAGE, RESOURCES, json.loads((root / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((root / 'reports/rhythm_v2_connection_status.json').read_text()),
            app_id=app.app_id, source_sha256=sha)
    try:
        call = resume.spawn(sha, script_sha)
        update_stage(ledger, STAGE, status='running', function_call_id=call.object_id, script_sha256=script_sha)
        result = call.get()
        save(root / 'reports/condition_v1_resume.json', result)
        update_stage(ledger, STAGE, status='complete' if result['complete'] else 'incomplete', result=result, ended_at=time.time())
    except BaseException as exc:
        update_stage(ledger, STAGE, status='observation_error', error=f'{type(exc).__name__}: {exc}')
        raise

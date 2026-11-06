"""Summarize the saved trace beside the Volume; transfer only the bounded summary."""
import hashlib
import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('beatfcn-condition-trace-summary')
volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.12').add_local_file(
    str(ROOT / 'scripts/analyze_cuda_trace.py'), '/analyzer.py')
STAGE = 'condition-trace-summary-v1'
RESOURCES = {'cpu': 2, 'memory_gib': 4, 'gpu': None, 'timeout_s': 180,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}


@app.function(image=image, volumes={'/data': volume}, cpu=(2, 2), memory=(4096, 4096),
              timeout=180, retries=0, max_containers=1, scaledown_window=2)
def summarize(expected_sha: str):
    import importlib.util
    started = time.perf_counter()
    if hashlib.sha256(Path('/analyzer.py').read_bytes()).hexdigest() != expected_sha:
        raise ValueError('analyzer code identity differs')
    spec = importlib.util.spec_from_file_location('analyzer', '/analyzer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = Path('/data/runs/r02-density-supervised/traces/steps-3550-3551.json')
    result = module.analyze(trace)
    result.update(source_bytes=trace.stat().st_size, analyzer_sha256=expected_sha,
                  function_wall_s=time.perf_counter() - started, gpu_compute=False)
    output = Path('/data/evolution/capability-evolution-20261024/condition-v1/trace_summary.json')
    if output.exists():
        raise ValueError('trace summary already exists; inspect previous stage')
    output.write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    return result


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve, update_stage
    ledger = ROOT / 'reports/rhythm_v2_budget.json'
    sha = hashlib.sha256((ROOT / 'scripts/analyze_cuda_trace.py').read_bytes()).hexdigest()
    reserve(ledger, STAGE, RESOURCES, json.loads((ROOT / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((ROOT / 'reports/rhythm_v2_connection_status.json').read_text()),
            app_id=app.app_id, source_sha256=sha)
    try:
        call = summarize.spawn(sha)
        update_stage(ledger, STAGE, status='running', function_call_id=call.object_id)
        result = call.get()
        (ROOT / 'reports/condition_v1_traces/steps-3550-3551-summary.json').write_text(json.dumps(result, indent=2) + '\n')
        update_stage(ledger, STAGE, status='complete', result={'complete': result['complete'], 'wall_s': result['function_wall_s']}, ended_at=time.time())
    except BaseException as exc:
        update_stage(ledger, STAGE, status='observation_error', error=f'{type(exc).__name__}: {exc}')
        raise

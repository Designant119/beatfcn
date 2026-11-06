"""Run the already reserved R03 evaluation with output-preserving stop diagnostics."""
import hashlib
import json
from pathlib import Path
import sys
import time

import modal

from beatfcn.cloud import window_condition_app as experiment
from beatfcn.cloud.rhythm_diagnostic_app import source_digest, save

ROOT = experiment.REPO
OBSERVER = ROOT / 'scripts/observe_generation.py'
app = modal.App('beatfcn-r03-observed-evaluation')
image = (experiment.image.add_local_file(__file__, '/audit/r03_observed_eval_app.py')
         .add_local_file(str(OBSERVER), '/audit/observe_generation.py'))


def replay_baseline_factual():
    """Verify CUDA outputs against the saved baseline while collecting stop evidence."""
    import torch
    from beatfcn.model import features as F
    from beatfcn.model import sample as sampling
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.evaluation import groups_to_events
    from beatfcn.train.data import Corpus
    from beatfcn.train.control_panel import requested_view
    from beatfcn.train.condition_pairs import blank_context_excerpt
    root = Path(experiment.ROOT)
    old = json.loads((root / 'baseline.json').read_text())
    wanted = {r['case_id']: r for r in old['records'] if r['variant'] == 'unspecified_conditions'}
    panel = json.loads((root / 'panel.json').read_text())
    ck = experiment.checkpoint(); corpus = Corpus(**ck['data'])
    corpus.train_rows([c['row'] for c in panel['cases']])
    model = P1Model(P1Config(**ck['config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model']); del ck
    records = []
    for case in panel['cases']:
        row = case['row']; c0 = case['c0_us']; chart = corpus.chart(row['chart_id'])
        example = blank_context_excerpt(chart, c0, case['duration_us'], row['audio_sha256'], 20261025)
        ex = F.build_core(example, chart.events, c0, case['duration_us'], stage_a_arm='legacy')
        song = corpus.song(row['audio_sha256'], in_memory=True); diagnostics = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            groups = sampling.generate_core(model, requested_view(ex), song, [], [(c0, c0 + F.CORE_US)], [],
                device='cuda', greedy=True, seed=20261025, diagnostics=diagnostics, deadline=time.perf_counter() + 15)
        events = [e.to_json() for e in groups_to_events(groups, c0, row['mode'], placement=False)]
        equal = events == wanted[case['case_id']]['generated_events']
        current = dict(wanted[case['case_id']], diagnostics=diagnostics, observed_replay_output_equal=equal)
        records.append(current)
        save(root / 'baseline_observed.json', dict(records=records, complete=False))
        if not equal or not diagnostics.get('terminated_by_end'):
            raise ValueError('observed CUDA replay differs from the frozen baseline')
        del song
    save(root / 'baseline_observed.json', dict(records=records, complete=True, all_outputs_equal=True))
    return len(records)


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(16384, 16384),
              volumes={'/data': experiment.data}, timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def evaluate(expected_source, expected_plan, expected_wrapper, expected_observer):
    import torch
    torch.set_num_threads(4)
    for path, fingerprint in [('/audit/r03_observed_eval_app.py', expected_wrapper),
                              ('/audit/observe_generation.py', expected_observer)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != fingerprint:
            raise ValueError('observer script identity differs')
    sys.path.insert(0, '/audit')
    from observe_generation import observe_generation
    started = time.perf_counter()
    info = dict(source_sha256=expected_source, plan_sha256=expected_plan, wrapper_sha256=expected_wrapper,
                observer_sha256=expected_observer, changes_generation_rule=False, complete=False)
    path = Path(experiment.ROOT) / 'evaluation_observer.json'
    save(path, info); experiment.data.commit()
    with observe_generation() as records:
        info['baseline_factual_replays'] = replay_baseline_factual()
        info['baseline_factual_outputs_equal'] = True
        result = experiment.evaluate.local(expected_source, expected_plan)
    info.update(complete=result['complete'], observed_generations=len(records),
                wall_s=time.perf_counter() - started, all_observations_finite=all(r['finite'] for r in records))
    save(path, info); experiment.data.commit()
    return dict(result, observer=info)


@app.local_entrypoint()
def main():
    from beatfcn.train.campaign_budget import reserve, update_stage
    ledger = ROOT / 'reports/rhythm_v2_budget.json'; stage = experiment.STAGES['evaluate']
    sha = source_digest(ROOT / 'beatfcn'); plan_sha = hashlib.sha256(experiment.PLAN.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    observer_sha = hashlib.sha256(OBSERVER.read_bytes()).hexdigest()
    resources = dict(experiment.RESOURCES['evaluate'], startup_allowance_s=120, retries=0, max_containers=1)
    reserve(ledger, stage, resources, json.loads((ROOT / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((ROOT / 'reports/rhythm_v2_connection_status.json').read_text()), app_id=app.app_id, source_sha256=sha)
    try:
        call = evaluate.spawn(sha, plan_sha, wrapper_sha, observer_sha)
        update_stage(ledger, stage, status='running', function_call_id=call.object_id,
                     plan_sha256=plan_sha, wrapper_sha256=wrapper_sha, observer_sha256=observer_sha)
        result = call.get(); save(ROOT / 'reports/r03_window_evaluate.json', result)
        update_stage(ledger, stage, status='complete' if result['complete'] else 'incomplete',
                     result_path=experiment.ROOT + '/evaluation.json', ended_at=time.time())
        print(json.dumps(dict(complete=result['complete'], runs=result['runs'], observer=result['observer'])))
    except BaseException as exc:
        update_stage(ledger, stage, status='observation_error', error=f'{type(exc).__name__}: {exc}', observed_at=time.time())
        raise

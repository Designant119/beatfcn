"""Read-only D07: separate onset errors from outer-history and action-head errors."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import modal

from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import save, source_digest

STAGE = 'action-path-diagnostic-v1'
FIXTURE = training.REPO / 'bench/rhythm-v2/action-path-diagnostic-v1.json'
REMOTE = '/data/evolution/capability-evolution-20261024'
RESOURCES = dict(gpu='L40S', cpu=4, memory_gib=16, timeout_s=360, startup_allowance_s=120,
                 retries=0, max_containers=1)
app = modal.App('beatfcn-action-path-diagnostic')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (training.image.add_local_file(__file__, '/probe/action_path_probe_app.py')
         .add_local_file(str(FIXTURE), '/probe/fixture.json'))


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(16384, 16384), volumes={'/data': data},
              timeout=360, retries=0, max_containers=1, scaledown_window=2)
def probe(expected_source, expected_script, expected_fixture):
    import beatfcn
    import torch
    from beatfcn.model import features as F, sample
    from beatfcn.model.action_diagnostics import oracle_onsets, actions_with_true_outer_history
    from beatfcn.model.batch import collate
    from beatfcn.model.evaluation import groups_to_events, legality_issues, score_events
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.condition_pairs import blank_context_excerpt
    from beatfcn.train.control_panel import requested_view
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.probe import capture_decisions, metrics_json, tensor_summary

    started = time.perf_counter(); torch.set_num_threads(4)
    if source_digest(Path(beatfcn.__file__).parent) != expected_source:
        raise ValueError('source identity differs')
    if hashlib.sha256(Path('/probe/action_path_probe_app.py').read_bytes()).hexdigest() != expected_script:
        raise ValueError('probe script identity differs')
    raw = Path('/probe/fixture.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_fixture:
        raise ValueError('diagnostic plan identity differs')
    plan = json.loads(raw)
    output = Path(REMOTE) / STAGE
    if output.exists():
        raise ValueError('diagnostic already exists; inspect its original handle')
    output.mkdir(parents=True)
    panel_raw = (Path(REMOTE) / 'window-conditions-v1/panel.json').read_bytes()
    if hashlib.sha256(panel_raw).hexdigest() != plan['panel_sha256']:
        raise ValueError('panel differs from frozen diagnostic')
    panel = json.loads(panel_raw)
    report = dict(stage=STAGE, complete=False, optimizer_updates=0, checkpoint_saved=False,
        source_sha256=expected_source, script_sha256=expected_script, fixture_sha256=expected_fixture,
        diagnostic_only=True, models=plan['models'], protocol=plan, records=[], data_health=[],
        expected_records=len(panel['cases']) * len(plan['models']) * 4)

    def record():
        report['function_wall_s'] = time.perf_counter() - started
        save(output / 'result.json', report); data.commit()

    record()
    try:
        for label, spec in plan['models'].items():
            payload = Path(spec['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != spec['sha256']:
                raise ValueError('checkpoint identity differs')
            ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            model = P1Model(P1Config(**ck['config'])).cuda().eval().requires_grad_(False)
            model.load_state_dict(ck['model']); corpus = Corpus(**ck['data']); del ck, payload
            corpus.train_rows([c['row'] for c in panel['cases']])
            previous = json.loads((Path(REMOTE) / 'window-conditions-v1' / spec['saved_benchmark']).read_text())
            wanted = {r['case_id']: r for r in previous['records'] if r['variant'] == 'unspecified_conditions'}
            for case in panel['cases']:
                if time.perf_counter() - started > 275:
                    raise TimeoutError('leave time to preserve partial evidence; do not repeat automatically')
                row = case['row']; c0 = case['c0_us']; mode = row['mode']
                path = Path('/data/canon') / row['source'] / f'{row["native_id"]}.json'
                if hashlib.sha256(path.read_bytes()).hexdigest() != case['canon_sha256']:
                    raise ValueError('canonical identity differs')
                chart = corpus.chart(row['chart_id'])
                request = blank_context_excerpt(chart, c0, case['duration_us'], row['audio_sha256'], 20261025)
                ex = requested_view(F.build_core(request, chart.events, c0, case['duration_us'], stage_a_arm='legacy'))
                if F.check_targets(ex):
                    raise ValueError('illegal target')
                public = {**request.request, 'controls': {'allowed_types': request.request['controls']['allowed_types']}}
                truth = [e for g in request.model_view.target_groups for e in g.events]
                if [e.to_json() for e in truth] != wanted[case['case_id']]['truth_events']:
                    raise ValueError('diagnostic target changed')
                song = corpus.song(row['audio_sha256'], in_memory=True)
                identity = dict(model=label, case_id=case['case_id'], chart_id=row['chart_id'], mode=mode,
                                c0_us=c0, truth_events=[e.to_json() for e in truth])
                bt = to_device(collate([ex], [song], model.c.taps, model.c.top_tap), torch.device('cuda'))
                if label == 'baseline':
                    stats = {k: tensor_summary(v) for k, v in bt.items() if torch.is_tensor(v)}
                    if any(s['nan'] or s['posinf'] or s['neginf'] for s in stats.values()):
                        raise ValueError('nonfinite actual input')
                    report['data_health'].append(dict(case_id=case['case_id'], tensors=stats,
                        label_audio_identity_verified=True, controls_missing=bool((bt['controls'] == -1).all())))
                with capture_decisions() as decisions, torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = model(bt)
                report['records'].append(dict(identity, variant='teacher_forced',
                    **metrics_json(loss, metrics), decisions=decisions, diagnostic_only=True))
                del bt
                for variant in ('free', 'oracle_onsets', 'oracle_onsets_and_outer_history'):
                    begin = time.perf_counter(); diagnostics = {}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        if variant == 'oracle_onsets_and_outer_history':
                            groups, diagnostics = actions_with_true_outer_history(model, ex, song, device='cuda')
                            complete = diagnostics['visited_groups'] == len(ex.groups)
                        else:
                            options = dict(device='cuda', greedy=True, seed=20261025, diagnostics=diagnostics,
                                           deadline=min(started + 325, begin + 15))
                            if variant == 'oracle_onsets':
                                with oracle_onsets(model, ex.groups) as observer:
                                    groups = sample.generate_core(model, ex, song, [], [(c0, c0 + F.CORE_US)], [], **options)
                                diagnostics['oracle'] = observer
                            else:
                                groups = sample.generate_core(model, ex, song, [], [(c0, c0 + F.CORE_US)], [], **options)
                            complete = bool(diagnostics.get('terminated_by_end'))
                    events = groups_to_events(groups, c0, mode, placement=False)
                    serial = [e.to_json() for e in events]
                    if variant == 'free' and serial != wanted[case['case_id']]['generated_events']:
                        raise ValueError('free replay changed; do not compare different inference protocols')
                    issues = legality_issues(public, events, placement=False)
                    item = dict(identity, variant=variant, complete=complete, legal=not issues, legality=issues,
                        scores=score_events(events, truth, mode), diagnostics=diagnostics, generated_events=serial,
                        predicted_types=dict(Counter(e.type for e in events)), truth_types=dict(Counter(e.type for e in truth)),
                        diagnostic_only=True, replay_equal=True if variant == 'free' else None,
                        truth_supplied=[] if variant == 'free' else ['onsets'] if variant == 'oracle_onsets' else ['onsets', 'earlier_outer_groups'],
                        wall_s=time.perf_counter() - begin)
                    report['records'].append(item)
                    print(json.dumps(dict(event='action_diagnostic', model=label, case=case['case_id'],
                        variant=variant, complete=complete, legal=not issues, types=item['predicted_types'])), flush=True)
                    if not complete:
                        raise ValueError('diagnostic did not finish; preserve it as incomplete')
                del song
                record()
            del model, corpus
            torch.cuda.empty_cache()
        report.update(complete=len(report['records']) == report['expected_records'],
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        record(); return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'; record(); raise


@app.local_entrypoint()
def main():
    from beatfcn.train.campaign_budget import reserve, update_stage
    root = training.REPO; ledger = root / 'reports/rhythm_v2_budget.json'
    sha = source_digest(root / 'beatfcn'); script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    registration = json.loads((root / 'reports/d07_registration.json').read_text())
    if (sha, script_sha, plan_sha) != tuple(registration[k] for k in ('source_sha256', 'script_sha256', 'plan_sha256')):
        raise ValueError('D07 differs from its pre-dispatch registration')
    reserve(ledger, STAGE, RESOURCES, json.loads((root / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((root / 'reports/rhythm_v2_connection_status.json').read_text()), app_id=app.app_id, source_sha256=sha)
    try:
        call = probe.spawn(sha, script_sha, plan_sha)
        update_stage(ledger, STAGE, status='running', function_call_id=call.object_id,
                     script_sha256=script_sha, plan_sha256=plan_sha)
        result = call.get(); save(root / f'reports/{STAGE}.json', result)
        update_stage(ledger, STAGE, status='complete' if result['complete'] else 'incomplete',
                     result_path=f'{REMOTE}/{STAGE}/result.json', optimizer_updates=0,
                     function_wall_s=result['function_wall_s'], ended_at=time.time())
        print(json.dumps({k: result[k] for k in ('complete', 'optimizer_updates', 'function_wall_s', 'expected_records')}))
    except BaseException as exc:
        update_stage(ledger, STAGE, status='observation_error', error=f'{type(exc).__name__}: {exc}', observed_at=time.time())
        raise

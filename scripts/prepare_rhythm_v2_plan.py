"""Freeze a rhythm-output training plan from read-only billing and preparation evidence.

This writes local artifacts only. It neither imports Modal nor launches compute.
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.train.stage_a import protocol
from beatfcn.spatial.rpe import digest


def read(name):
    return json.loads((ROOT / name).read_text())


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    existing = ROOT / 'reports/rhythm_v2_budget.json'
    if existing.exists() and json.loads(existing.read_text()).get('paid_stages_dispatched'):
        raise ValueError('cannot regenerate a dispatched ledger; reconcile and extend it in place')
    apps = {x['app_id']: x for x in read('reports/rhythm_v2_apps.json')}
    historical = read('reports/evolution_app_lifetimes.json') + read('reports/capability_review_closeout.json')['apps']
    ids = sorted({x['app_id'] for x in historical})
    billing = read('reports/rhythm_v2_billing.json')
    rates = read('reports/rhythm_v2_rates.json')
    # The CLI can include the current, still-accruing hour. Its presence does not
    # prove that a just-finished application's final charge has settled.
    latest_interval_end = max(datetime.fromisoformat(x['interval_start']).replace(tzinfo=timezone.utc) for x in billing) + timedelta(hours=1)
    cutoff = min(latest_interval_end, datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
    archived = {x['app_id']: x for x in historical}
    archived_status_ids = []
    for app_id in ids:
        row = apps.get(app_id, archived[app_id])
        if app_id not in apps:
            archived_status_ids.append(app_id)  # CLI omits older stopped applications
        stopped = datetime.fromisoformat(row['stopped_at'])
        if row['state'] != 'stopped' or int(row['tasks']) or stopped > cutoff:
            raise ValueError('historical compute is still active or outside complete billing intervals')
        if not any(x['object_id'] == app_id for x in billing):
            raise ValueError('historical app has no billing entry; retain old reservation until reconciled')
    actual = sum((Decimal(x['cost']) for x in billing if x['object_id'] in ids), Decimal(0))
    authorization = read('reports/rhythm_campaign_authorization.json')
    cap = Decimal(str(authorization['approved_total_usd']))
    old_allowance, transfer_allowance = Decimal('.40'), Decimal('.01')
    base = actual + old_allowance + transfer_allowance
    resources = {
        'shared_cpu_preparation': {'cpu': 8, 'memory_gib': 32, 'gpu': None, 'timeout_s': 900},
        'matched_training': {'cpu': 12, 'memory_gib': 48, 'gpu': 'L40S', 'timeout_s': 3000},
        'paired_evaluation': {'cpu': 4, 'memory_gib': 16, 'gpu': 'L40S', 'timeout_s': 4500},
    }
    for resource in resources.values():
        hourly = (Decimal(rates['cpu_hour_cost']) * resource['cpu']
                  + Decimal(rates['mem_gib_hour_cost']) * resource['memory_gib'])
        if resource['gpu']:
            hourly += Decimal(rates['gpu_hour_cost_l40s'])
        resource.update(rate_usd_s=float(hourly / 3600),
                        reserved_upper_usd=float(hourly * (resource['timeout_s'] + 120) / 3600),
                        startup_allowance_s=120, retries=0, max_containers=1)
    planned = sum((Decimal(str(x['reserved_upper_usd'])) for x in resources.values()), Decimal(0))
    other = Decimal('.35')  # orchestration, storage, and new preparation transfer reserve
    upper = base + planned + other
    if upper > cap:
        raise ValueError('plan exceeds authorized campaign cap')
    budget = {'schema': 'rhythm_v2_budget_v1', 'approved_total_usd': float(cap),
              'reconciled_app_ids': ids, 'billing_through_exclusive_utc': cutoff.isoformat(),
              'archived_stopped_status_app_ids': archived_status_ids,
              'reported_historical_compute_usd': float(actual), 'prior_other_reserve_usd': float(old_allowance),
              'completed_readonly_transfer_reserve_usd': float(transfer_allowance),
              'historical_accounted_upper_usd': float(base), 'available_before_plan_usd': float(cap - base),
              'planned_stages': resources, 'planned_other_reserve_usd': float(other),
              'planned_new_upper_usd': float(planned + other), 'campaign_planned_upper_usd': float(upper),
              'unallocated_margin_usd': float(cap - upper), 'paid_stages_dispatched': [],
              'authorization': authorization,
              'policy': 'No independent old ledger. Debit all new stages here before dispatch; retain failed reservations.'}
    plan = protocol()
    plan.update(schema='beatfcn.rhythm_evolution_v2', status='plan_frozen_execution_preparation_pending',
                training_authorized=True, launch_enabled=False,
                intent={'output': 'rhythm_only', 'spatial_input': 'allowed_as_matched_experiment',
                        'phira_output_fields': ['onset', 'type', 'hold_end', 'multiplicity'],
                        'mania_output': 'existing timing/type/duration/four-key lane contract unchanged',
                        'spatial_output': 'deferred', 'human_study': 'deferred'},
                campaign='capability-evolution-20261024', round='r02-rhythm-context',
                budget='reports/rhythm_v2_budget.json', total_budget_usd=float(cap), campaign_cap_usd=float(cap),
                arms_execution='shared frozen schedule; paired updates; one GPU container; no independent resampling',
                stop_after_updates_per_arm=1000, next_round_requires_review=True,
                validation_every_updates=250,
                execution_preconditions=['training-source and split audit; development data excluded from training',
                                         'shared schedule and validation regenerated under new input policy',
                                         'source checkpoint bytes and source bundle match frozen hashes',
                                         'paired training runner, control tests and monitor accept projected batches',
                                         'current billing and rates rechecked before reserving dispatch'],
                performance={'execution': 'eager', 'graph': False, 'metadata': 'CPU prepared and verified before GPU lease',
                             'util_guard_percent': 70, 'util_check_after_training_s': 480,
                             'max_data_wait_fraction': .8, 'data_wait_check_after_s': 480,
                             'on_guard': 'checkpoint every arm, stop, diagnose; no automatic restart',
                             'judge': ['finite effective updates', 'end_to_end_time', 'total_cost', 'benchmark quality']},
                resources=resources)
    plan['common']['paired_references'] = False
    plan['common']['learning_rate']['interpretation'] = 'cosine schedule resumes at 3500; no new 500-step warmup'
    plan['common']['train_split'] = 'original full eligible train split; never substitute development fixtures'
    plan['common']['unsupported_geometry'] = 'keep rhythm truth and explicit unsupported status; identical cohort in all arms'
    plan['common']['retained_scene_training_fraction'] = 0.0
    plan['monitor_requirements'] = ['per-arm effective updates and latest checkpoint identity',
                                    'per-head validation metrics every 250 updates; not a substitute for generation',
                                    'benchmark scores, completion denominators and source checkpoint every 1000 updates',
                                    'GPU utilization, allocated/reserved/peak VRAM, CPU preparation and data wait',
                                    'cost reservations and actual wall time; explicit stop or failure reason']
    plan['evaluation'] = {
        'main_manifest': 'bench/phira-stage-a-v1/benchmark-v2.json',
        'main_requests_per_model': 120, 'models': ['source_3500', 'legacy_4500', 'removed_4500', 'operation_4500'],
        'historical_4500': 'reference only; different intervention and old request policy',
        'decode': 'greedy; same seed and timeout in every arm; include failures in denominator',
        'baseline_3500_adapter': 'regenerate under repaired legacy policy; not the old v1 scores',
        'retained_scene_manifest': 'bench/phira-stage-a-v1/retained-scene-benchmark.json',
        'retained_scene_interpretation': 'separate transfer test; not part of the primary comparison',
        'primary': {'contrast': 'operation_4500 minus removed_4500',
                    'population': 'Phira ordinary holes and reference requests',
                    'metric': 'song-averaged note_f1_20ms; empty targets separate',
                    'limitation': 'reference fidelity measure, not sufficient for quality promotion'},
        'promotion_gates': ['zero locked-region changes and illegal types',
                            'completion no worse than paired comparator',
                            'paired song-level 95% interval lower bound above zero for primary gain',
                            'note F1 lower bound above -0.02 for predeclared noninferiority checks; provisional margin',
                            'density/control response and hold behavior reviewed; no gain by flooding notes',
                            'geometry masking/mismatch supports genuine input use',
                            'ordinary no-scene scratch and mania reported separately',
                            'fully/partly/unsupported input and every missing result reported'],
        'secondary': ['onset precision/recall and matched timing error with coverage',
                      'per-type note F1; drag separate from non-drag',
                      'hold endpoint and duration accuracy', 'density and control band response',
                      'positive-only reference swap with genuinely different content',
                      'audio anchors and shifted chance baseline; no universal drag onset requirement',
                      'continuous drag candidate diagnostics, not gesture truth'],
        'sealed_test': 'untouched', 'budget_exhaustion': 'partial evaluation blocks promotion; no cherry-picked success'}
    plan['evaluation']['assistance_evidence'] = {
        'reference_edit_proxy': 'unique timing points, typed heads, full notes including hold tails; insert/delete/review sensitivity',
        'audio_baseline': 'fixed audio-only peak baseline on the same requests; not implemented yet',
        'limits': 'reference matching is not human time or aesthetic quality; do not promote on this proxy alone',
        'controls': 'report unconditioned, visible-context-derived and explicit user-band requests separately; no hidden-target-derived controls',
        'artifact': 'export playable/editable rhythm drafts with provenance and per-request evidence before claiming assistance'}
    plan['pretraining_diagnostics'] = ['fixed non-oracle density sweep on both checkpoints before choosing the next training intervention',
                                      'frozen-checkpoint history path probe; K8 Y/GRU separated',
                                      'teacher-forced lane choice baseline and group-set accuracy; preserve targets and legality',
                                      'audit feature/target visibility and full-vs-incremental decode equivalence']
    plan['next_training_decision'] = ('Condition/history diagnostics precede paid spatial-arm training. Preserve the three prepared spatial arms; '
                                    'if conditions fail, first run a matched conditional-learning intervention and baseline, reviewed after 1000 updates.')
    dependencies = ['beatfcn/train/stage_a.py', 'bench/phira-stage-a-v1/benchmark-v2.json',
                    'bench/phira-stage-a-v1/retained-scene-benchmark.json', 'reports/phira_stage_a_test_summary.json']
    plan['dependency_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in dependencies}
    plan['plan_sha256'] = digest(plan)
    save('reports/rhythm_v2_budget.json', budget)
    save('bench/rhythm-v2/plan.json', plan)
    for arm in plan['arms']:
        save(f'bench/rhythm-v2/{arm}.json', {'plan_sha256': plan['plan_sha256'], 'arm': arm,
             'run_id': f'r02-rhythm-{arm}', 'initialize_from': plan['checkpoint'],
             'checkpoint_sha256': plan['checkpoint_sha256'], 'stop_at': 4500, 'round_updates': 1000,
             'variant': {'phira_context': arm, 'phira_location': False, 'selection': False,
                         'lane_history_groups': 0, 'offset_sigma_ms': 0.0, 'compact_inner': True}})
    print(json.dumps({'plan': 'bench/rhythm-v2/plan.json', 'historical_accounted_usd': float(base),
                      'planned_new_upper_usd': float(planned + other), 'campaign_upper_usd': float(upper),
                      'paid_stages_dispatched': 0}))


if __name__ == '__main__':
    main()

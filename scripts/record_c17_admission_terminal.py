"""Record raw arithmetic and a separately authored engineering postmortem; no NN."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.archive_council import append_event, events

ROOT = Path(__file__).resolve().parents[1]
ID = 'c17-masked-native-events-v1'
ART = ROOT / 'artifacts' / ID
DEST = ROOT / 'reports/council-shared-total-c17'
BASE = ROOT / 'reports/local-integration-v1'


def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rel(p):
    return str(p.relative_to(ROOT))


def main():
    plan_path = ROOT / 'bench/rhythm-v2' / (ID + '.json')
    plan = read(plan_path)
    extra = read(ART / 'matched-controls-plan.json')
    admission = read(ART / 'admission.json')
    report = read(BASE / (ID + '.json'))
    profile_path = BASE / 'c17-masked-native-admission-profile.json'
    profile = read(profile_path)
    narrative = BASE / 'c17-masked-native-events-review.md'
    receipt_path = BASE / 'c17-masked-native-events-mechanism-review.json'
    facts_path = BASE / 'c17-masked-native-events-terminal-facts.json'
    if receipt_path.exists() or facts_path.exists():
        raise FileExistsError('Preserve terminal analysis and append future revisions separately')
    checked = []
    for value, prefix in [(plan, ART / 'source'), (extra, ART / 'source/matched-control-addendum')]:
        for name, expected in value['source_sha256'].items():
            path = prefix / name
            if sha(path) != expected:
                raise ValueError('Frozen execution source drift: ' + name)
            checked.append(dict(path=rel(path), sha256=expected, bytes=path.stat().st_size))
    checkpoint = ROOT / admission['failure_checkpoint']
    if sha(checkpoint) != admission['failure_checkpoint_sha256']:
        raise ValueError('Failure checkpoint drift')
    if admission['optimizer_updates'] != 2 or set(report['stages']) != {'admission'}:
        raise ValueError('Unexpected actual stage/update state')
    if not profile['integer_replay_equal'] or not profile['weight_digest_unchanged'] or profile['new_optimizer_updates']:
        raise ValueError('Posthoc replay is not a frozen zero-update control')
    cost = admission['actual_cost_probe']
    probe = cost['prediction']
    types = dict(Counter(e['type'] for e in probe['events']))
    # The full trace is deliberately left only in the immutable raw receipt.
    trace = probe.get('trace', [])
    active = sum(v.get('active', False) for v in trace)
    facts = dict(candidate=ID, status='engineering_runtime_estimate_gate_failed',
        parameters=plan['parameters'], source_objects_verified=checked,
        core_source_objects=len(plan['source_sha256']), extra_source_objects=len(extra['source_sha256']),
        plan_sha256=sha(plan_path), extra_plan_sha256=sha(ART / 'matched-controls-plan.json'),
        actual_global_optimizer_updates=2, CPU_optimizer_updates=1, MPS_checkpoint_optimizer_updates=1,
        health_updates=dict(A=0, C=0), quality_updates=dict(A=0, C=0),
        new_dev_free_outcomes=0, new_dev_proper_outcomes=0,
        actual_attempt_wall_s=report['wall_s'], preparation_wall_s=report['preparation_wall_s'],
        admission_wall_s=admission['wall_s'], cpu_mps=admission['cpu_mps'],
        supervised_branch_gradient_norms=admission['branches'],
        failure=admission['failure'], peak_sampled_physical_bytes=admission['telemetry']['peak_sampled_physical_bytes'],
        free_train_probe=dict(index=cost['index'], complete=probe['complete'], final_events=len(probe['events']),
            types=types, slot_updates=len(trace), active_over_two_sweeps=active, free_s=cost['free_s'],
            proper_three_rates_s=cost['proper_three_rates_s'],
            original_main_estimate_s=cost['full_panel_linear_estimate_s'], original_evaluation_cap_s=1200,
            full_plus_extra_nominal_estimate_s=cost['full_panel_linear_estimate_s'] + extra['eligible']*(cost['free_s'] + cost['proper_three_rates_s']/3),
            scope='One fixed training request after MPS Adam1;not independent-song quality or measured full-panel wall'),
        posthoc_profile=dict(path=rel(profile_path), sha256=sha(profile_path), optimizer_updates=0,
            integer_replay_equal=profile['integer_replay_equal'], active_probability_max_difference=profile['active_probability_max_difference'],
            weight_digest_unchanged=profile['weight_digest_unchanged'], synchronized_profile_wall_s=profile['synchronized_profile_wall_s'],
            process_wall_s=read(ART / 'posthoc-profile-runtime.json')['wall_s'],
            interpretation='Intrusive synchronized single replay;forward includes attention;not additive original critical path'),
        failure_checkpoint=dict(path=rel(checkpoint), sha256=sha(checkpoint), bytes=checkpoint.stat().st_size,
            scope='Hashed file only by this arithmetic script;optimizer1 established by admission producer'),
        artifact_bytes=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file()),
        confirmation_scored=False, cloud_calls=0, cloud_spend_usd=0, scaling_authorized=False)
    if len(trace) != 512 or len(probe['events']) != 192:
        raise ValueError('Unexpected actual training-request sampler trace')
    write(facts_path, facts)
    evidence = [plan_path, ART / 'matched-controls-plan.json', ART / 'admission.json',
        BASE / (ID + '.json'), profile_path, ART / 'posthoc-profile-runtime.json', facts_path, narrative]
    receipt = dict(candidate=ID, analysis_status='complete', identity={rel(p): sha(p) for p in evidence},
        layers=dict(health='Engineering CPU/MPS 8requests+one Adam each passed numerical/gradients;cost estimate1423.099>1200 stopped admission;A/C health0',
            quality='A/C quality0;dev free/proper0;confirm sealed;no learning or F1 conclusion',
            structure_conditions='Training-request sampler16blocks512slots complete;192events/390active draws;frozen replay identical;62extra inputs preflight only'),
        mechanism_hypotheses=[
            dict(id='repeated_static_and_endpoint_work', observation='64cross calls/417tail scores/1197categorical draws;static context K/V repeated;interval mass recomputes legal bucket counts',
                 mechanism='Redundant projections and exact-prior bookkeeping consume execution time without new model information',
                 prediction='One exact-cache/vectorization intervention can reduce cost with same probabilities/support/gradients/RNG',
                 evidence=[rel(profile_path),'beatfcn/model/masked_native_events.py','beatfcn/train/masked_native_events.py'],
                 judgment='Repeated work observed;speed benefit and exact optimized equivalence unmeasured'),
            dict(id='resource_guard_overhead', observation='Guard clear1.133891s/16calls in intrusive3.786880s replay',
                 mechanism='Resource checks/GC/synchronization/cache clearing contribute independent overhead',
                 prediction='Disaggregate costs;only equivalent monitoring optimization can bypass repeated overhead;cleanup-frequency changes require memory/Adam control',
                 evidence=[rel(profile_path),'beatfcn/train/managed_mps.py','beatfcn/train/mps_request_lifecycle.py'],
                 judgment='Aggregate measured;not uniquely attributed to GC/filesystem/MPSGraphCache;do not delete guards'),
            dict(id='transient_workload_estimate', observation='One training request afterMPS1 outputs192events and77holds;fixed16 network calls remain',
                 mechanism='Transient density and hold branching can overestimate eventual A/C panel cost but cannot guarantee reduction',
                 prediction='Prospectively freeze small training workload panel and conservative cost equation before new measurement',
                 evidence=[rel(ART/'admission.json'),rel(facts_path)],
                 judgment='Plausible calibration limitation;original gate stays failed;not permission for faster-case search'),
            dict(id='numeric_support_or_generation_crash', observation='Device/loss/gradient<=1e-4;actualAdam finite;integer frozen replay and full sampler complete',
                 mechanism='If numerical/support crash caused stop these checks or terminal error would differ',
                 prediction='Exact runtime repair preserves legal completed outputs;healthy training still separately required',
                 evidence=[rel(ART/'admission.json'),rel(profile_path)],
                 judgment='Unsupported as present failure cause;quality and capacity unjudged')],
        controls=['Actual eight-request matched CPU/MPS forward/backward/Adam;hidden-target input invariance',
            'Native/COPY/duplicates/NULL and exact duration-prior regression;small neural scope separated',
            'Frozen same-weight/same-RNG training-request replay;new optimizer/dev outcomes0',
            'Matched range/donor input preflight only;no acoustic quality result'],
        missing_controls=[dict(question='Which exact execution change makes evaluation feasible?',
                comparison='Original versus one static-KV or duration-mass cache change;full probabilities/gradients if touched/native support/RNG and runtime',
                decision_if='Use only equality+cost-supported fix;failed parity requires numerical repair,not relaxed tolerance'),
            dict(question='Guard versus model contribution?',comparison='Disaggregate guard children while retaining watchdog and caps;changed cleanup schedule also gets real Adam/free memory curve',
                decision_if='Optimize monitoring only if equivalent;otherwise retain original checks'),
            dict(question='Single-case cost representativeness?',comparison='Frozen bounded training-only workload panel including empty/dense/hold/COPY with full main+extra equation',
                decision_if='New separately frozen resource contract may repartition stage caps within9000s;never rewrite original gate'),
            dict(question='Can the learning law fit/generalize?',comparison='Actual discarded100A/C then common-initial fresh400A/C and original free/proper/negative panels',
                decision_if='Only future actual outcomes judge learning;runtime fix alone supports neither quality nor scaling')],
        lessons=['Engineering cost and learning failure have different causal scope',
            'Intrusive nested timers are not an additive critical path or universal benchmark',
            'One-step training-request density is neither independent quality nor proof of terminal collapse'],
        successor_contract=dict(required_changes=['C18 independent raw-evidence review before repaired execution',
                'Preserve law/native support/controls/RNG;one probability-preserving execution intervention and full correctness+cost preflight',
                'If resource contract changes,freeze new identity with unchanged global/resource limits;health and quality fresh original initial state'],
            forbidden_repeats=['No blind rerun or old admission weights as quality warmstart',
                'No shorter chain/deleted hold/dropped controls/threshold or seed fishing as purported exact speed repair',
                'No relaxed parity/resource limits or gate-only rejection of masked learning family']),
        closure_reason='The frozen C17 execution path is closed after its registered single-training-request cost estimate1423.099s exceeded1200s. Actual numerical/gradient/Adam and legal full sampler checks passed;frozen replay is identical and measured several competing overheads. No health or independent-song quality updates/outcomes ran. This is a bounded engineering admission failure,not evidence against capacity,type learning,masked generation or its family. C18 must choose a verified execution repair before fresh health/quality.',
        source_record=rel(narrative))
    write(receipt_path, receipt)
    registry_path = ROOT / 'reports/local-evolution-v1/mechanism-reviews/index.json'
    registry = read(registry_path)
    if any(r['id'] == ID for r in registry['candidates']):
        raise ValueError('Duplicate terminal registry')
    registry['candidates'].append(dict(id=ID,analysis_status='complete',review=rel(receipt_path),
        review_sha256=sha(receipt_path),markdown=rel(narrative),narrative_sha256=sha(narrative)))
    write(registry_path,registry)
    state_path = BASE / 'state.json'
    state = read(state_path)
    state['previous_experiment'] = state['current_experiment']
    state['current_experiment'] = dict(id=ID,status='engineering_admission_failed_mechanism_closed',
        parameters=plan['parameters'], admission_updates=2,global_optimizer_updates=2,
        health_updates=dict(A=0,C=0),quality_updates=dict(A=0,C=0),free_completed=0,conditional_completed=0,
        supervised_experiment_wall_s=report['wall_s'],posthoc_profile_wall_s=facts['posthoc_profile']['process_wall_s'],
        artifact_bytes=facts['artifact_bytes'], report=rel(BASE/(ID+'.json')),review=rel(narrative),
        research_supported=None,learning_judgment='not_tested',cloud_spend_usd=0)
    state.update(status='c17_engineering_terminal_review_complete_c18_pending',runtime_probe_running=False,
        diagnostic_process_running=False,plan=rel(plan_path),terminal_receipt=rel(receipt_path),
        admission_receipt=rel(ART/'admission.json'),quality_evaluation='not_started',
        training_scope='C17actualCPU1+MPS1;health0;quality0;dev0;posthoc zero-update train replay;cloud0',
        registered_parameters=plan['parameters'],new_neural_updates=2,health_updates=0,quality_updates=0,
        next_entry='C18 independent runtime/law-preserving repair review;no live training/no automatic rerun',
        last_update_at=datetime.now(timezone.utc).isoformat())
    state['c17'].update(new_optimizer_updates=2,new_neural_scope='Actuald256 admission CPU/MPS8each+Adam1each;one free train probe and frozen timing replay;not health/dev',
        sol_action_status='tested_failed_engineering_cost_only_learning_unjudged',experiment_complete=True,
        mechanism_closed=True,mechanism_review=rel(narrative))
    write(state_path,state)
    budget_path = ROOT / 'reports/rhythm_v2_budget.json'
    budget=read(budget_path)
    budget['local_cpu_diagnostics'][ID]=dict(status='engineering_admission_failed_mechanism_review_complete',
        plan=rel(plan_path),report=rel(BASE/(ID+'.json')),parameters=plan['parameters'],
        preparation_wall_s=report['preparation_wall_s'],wall_s=report['wall_s'],wall_cap_s=9000,
        optimizer_updates=2,max_optimizer_updates=1002,health_updates=0,quality_updates=dict(A=0,C=0),
        free_dev_completed=0,conditional_dev_completed=0,peak_physical_bytes=facts['peak_sampled_physical_bytes'],
        artifact_bytes=facts['artifact_bytes'],mechanism_review=rel(narrative),research_supported=None,
        posthoc_profile=dict(optimizer_updates=0,new_dev_outcomes=0,wall_s=facts['posthoc_profile']['process_wall_s'],report=rel(profile_path)),
        unit_regressions='5 core+2 matched control tests;separate small neural regression scope;not quality',
        extra_input_preflight=rel(ART/'matched-control-input-preflight-runtime.json'),
        new_cloud_spend_usd=0,new_cloud_compute_calls=0)
    write(budget_path,budget)
    old=[r for r in events(ROOT) if r['id']=='C17-S-R01'][-1]
    event=dict(old,event_id='C17-S-R01-engineering-admission-terminal-20261030',status='tested_failed',
        rationale='Complete candidate implemented and actual CPU/MPS numerical/Adam/sampler checks passed;registered1423.099s>1200s cost estimate stops before health/quality. Deep review complete;learning law and family unjudged;C18 exact execution repair next.',
        evidence=[rel(narrative),rel(receipt_path),rel(facts_path),rel(ART/'admission.json'),rel(profile_path)],
        revisit='C18 chooses probability-preserving repair or separately frozen bounded resource repartition;validate same-state parity/cost,then discarded health and original fresh quality;no scaling',
        recorded_at=datetime.now(timezone.utc).isoformat(),recording_basis='Actual terminal raw receipts plus one zero-update same-training-request profile;not independent quality')
    append_event(ROOT,event)
    manifest_path=ROOT/'reports/council-archive/manifest.json'
    manifest=read(manifest_path)
    for a in manifest['attempts']:
        if a['id']=='C17-G':a['note']='Full original/input/audit archived;both current designs explicitly declined;not experimentally failed'
        if a['id']=='C17-S':a['note']='Full candidate implemented;actual admission numerics passed but registered cost estimate failed;mechanism review complete;health/quality0;family unjudged'
    write(manifest_path,manifest)
    print(dict(status='recorded',source_objects=len(checked),updates=2,health=0,quality=0,
               train_sampler_events=len(probe['events']),profile_new_updates=0,cloud=0))


if __name__=='__main__':
    main()

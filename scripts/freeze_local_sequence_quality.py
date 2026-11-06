"""Freeze matched D87 learning curves and a heldout, inference-safe stage panel."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ('control', 'candidate')
    plans = {name:ROOT/f'bench/rhythm-v2/d87-sequence-{name}-v1.json' for name in names}
    protocol_path = 'bench/rhythm-v2/d87-sequence-stage-protocol-v1.json'
    if any(path.exists() for path in [*plans.values(), ROOT/protocol_path]):
        raise FileExistsError('Do not overwrite a frozen comparison')
    base_path = 'bench/rhythm-v2/d68-local-baseline-v3.json'
    base = json.loads((ROOT/base_path).read_text())
    tape = json.loads((ROOT/base['tape']).read_text())
    records = []
    for chart in tape['charts']:
        for row in chart['requests']:
            records.append(dict(row, chart_id=chart['chart_id'], mode=chart['mode'],
                audio_sha256=chart['audio_sha256'], component=chart['component'],
                stratum=chart['stratum'], split=chart['split']))
    historical = 'bench/rhythm-v2/d83-condition-protocol-v1.json'
    active = json.loads((ROOT/historical).read_text())['songs']
    cases = []
    components = set()

    def case(index, role):
        row = records[index]
        return dict(record_index=index, role=role, mode=row['mode'], chart_id=row['chart_id'],
            component=row['component'], audio_sha256=row['audio_sha256'], stratum=row['stratum'],
            scope_us=[row['request']['start_us'], row['request']['end_us']])

    for song in active:
        matched = [i for i,r in enumerate(records) if r['chart_id']==song['chart_id']
                   and [r['request']['start_us'],r['request']['end_us']]==song['scope_us']]
        if len(matched) != 1 or records[matched[0]]['split'] != 'val' or records[matched[0]]['event_count'] <= 0:
            raise ValueError('Historical activity scope is not a unique real heldout request')
        cases.append(case(matched[0], 'active'))
        components.add(song['component'])
    empty_candidates = []
    for index, row in sorted(enumerate(records), key=lambda x:(x[1]['component'], x[1]['request']['start_us'], x[1]['chart_id'])):
        if (row['split']=='val' and not row['event_count'] and not row['incoming_holds']
                and row['request']['end_us']-row['request']['start_us']==8_000_000):
            empty_candidates.append(index)
            if row['mode']=='phira' and row['component'] not in components and len(cases)<8:
                cases.append(case(index, 'reference_rest'))
                components.add(row['component'])
    if len(cases) != 8 or len(components) != 8:
        raise ValueError('Need four activity and four independent reference-rest requests')
    protocol = dict(id='D87-stage-panel-v1', cases=cases, cpu_threads=2, token_cap=4096, max_wall_s=120,
        source_sha256={p:sha(ROOT/p) for p in (base['tape'],historical)},
        available_full_8s_rest_requests={m:sum(records[i]['mode']==m for i in empty_candidates) for m in ('phira','mania_4k')},
        selection='Activity scopes unchanged from audio-only D68 protocol; reference-rest scopes sorted by component/time/chart, train excluded and distinct components. Labels are not inference conditions.',
        early_stop={'at_step':500,'all_four_activity_direct_end_at_both_250_and_500':True,
                    'all_four_activity_token_capped_at_500':True},
        no_quality_promotion_from_small_panel=True)
    write_json(ROOT/protocol_path, protocol)
    source_paths = set(base['code_sha256']) | {
        'beatfcn/train/sequence_likelihood.py', 'scripts/run_local_sequence_quality.py',
        'scripts/freeze_local_sequence_quality.py', 'scripts/audit_local_sequence_stage.py',
        'scripts/probe_full_event_model.py', 'scripts/freeze_local_song_review.py',
        'scripts/review_local_song_rollouts.py', 'scripts/build_phira_preview.py',
        'scripts/run_local_song_generation.py', 'tests/test_local_sequence_quality.py'}
    code = {path:sha(ROOT/path) for path in sorted(source_paths)}
    old_changed = [path for path,expected in base['code_sha256'].items() if sha(ROOT/path) != expected]
    # These affect the intended causal contract; differing review or unrelated
    # archived files cannot silently change these training inputs and mechanics.
    core = ['beatfcn/model/full_event_model.py','beatfcn/model/full_event_codec.py',
        'beatfcn/train/local_request_panel.py','beatfcn/train/local_features.py',
        'beatfcn/train/local_shapes.py','beatfcn/train/event_conditions.py']
    if set(core) & set(old_changed):
        raise ValueError('D68 core implementation changed; inspect before claiming one-variable comparison')
    initial = dict(checkpoint='artifacts/d68-local-baseline/checkpoints/step-0002.pt',
        report='reports/d68_local_baseline_v1.json', driver_probe='reports/d68_driver_probe_v1.json',
        plan='bench/rhythm-v2/d68-local-baseline-v1.json')
    inputs = dict(base['inputs_sha256'])
    additions = [*initial.values(), 'reports/d86_sequence_likelihood_probe_v1.json',
        'reports/d86_sequence_likelihood_review_v1.json','reports/d86_sequence_weight_contract_v2.json',
        'design/D87_SEQUENCE_NLL_QUALITY_v1.md',protocol_path,base_path]
    inputs.update({path:sha(ROOT/path) for path in additions})
    for path, expected in inputs.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Previously frozen input no longer matches: '+path)
    quality_gates = dict(full_songs=20,minimum_nonempty_songs=18,
        note_f1_macro_gain_over_matched_control=.02,note_f1_macro_floor=.10337650151132332,
        minimum_songs_not_worse=12,tap_under10ms_fraction_reduction=.05,
        tap_under10ms_fraction_absolute_alternative=.05,require_tap_intervals=True,
        minimum_real_20ms_matches_per_nontap_type=1,ez_hd_macro_max_regression=.01,
        reference_rest_mean_extra_notes_max=1,manual_twenty_song_review_required=True,
        final_candidate_requires_all_conditions_reference_and_convergence=True)
    for name in names:
        plan = deepcopy(base)
        plan.update(id=f'D87-sequence-{name}-1000-v1',loss_mode='request_mean' if name=='control' else 'fixed_mode_sequence',
            normalizers=dict(mania_mean_tokens=268.106,phira_mean_tokens=249.457),
            health='reports/d86_sequence_likelihood_probe_v1.json',initialize_from=initial,
            physical_microbatch_requests=1,checkpoint_every=100,
            output=f'reports/d87_sequence_{name}_v1.json',
            checkpoint_dir=f'artifacts/d87-sequence-{name}-v1/checkpoints',
            generation_output=f'reports/d87_sequence_{name}_generation_v1.json',
            generation_dir=f'artifacts/d87-sequence-{name}-v1/generation-1000',
            source_snapshot=f'inputs/code_baselines/d87-sequence-{name}-v1',
            stage_protocol=protocol_path,stage_report_pattern=f'reports/d87_{name}_stage_{{step}}_v1.json',
            code_sha256=code,inputs_sha256=inputs,
            quality_gates=quality_gates,quality_promotion_allowed=False,
            hypothesis='D86 same hypothesis: only per-request CE denominator changes; matched new control avoids D68 physical-batch history confound.',
            baseline='D68 historical F1 .0833765 and short tap intervals .3648; new matched-control X measured before interpreting candidate, required deltas frozen.',
            correctness_revision='Both arms fixed audio shapes/microbatch1 throughout, same restored common2, unified validation CE, CPU full-song generation.',
            failure_policy='Inspect250/500 frozen panel; stop on registered collapse/cap or correctness error. No automatic continuation beyond1000 or scaling.',
            historical_code_changes=old_changed)
        snapshot = ROOT/plan['source_snapshot']
        snapshot.mkdir(parents=True,exist_ok=False)
        for source in code:
            dest=snapshot/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
        write_json(plans[name],plan)
    receipt = dict(id='D87-preparation-v1',complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
        plans={name:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for name,path in plans.items()},
        stage_protocol=protocol_path,stage_protocol_sha256=sha(ROOT/protocol_path),
        historical_changed_code=old_changed,cases=cases,
        only_arm_differences=['loss_mode','id and output/snapshot paths'],quality_gates=quality_gates)
    write_json(ROOT/'reports/d87_preparation_v1.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()

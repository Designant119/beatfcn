"""Close the failed D97 objective experiment, retaining all raw evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from scripts.closeout_local_capacity import ROOT, read, save, sha

# Written only after all ten actual sheets were viewed in this chat.
DETAILS = {
    '28324': '24–32秒364tap对67个人谱对象，持续drag与稀疏tap被细碎点击替代。',
    '12626': '184–192秒模型缺失、人谱10对象含长hold；缺失来自32秒后token失败，不能解释为音乐留白。',
    '38724': '136–144秒266tap对144个混合对象，flick/hold/drag全部消失。',
    '60952': '88–96秒404tap对18drag，强弱周期与类型选择未得到有用表达。',
    '19959': '32–40秒308tap对20个tap/drag，弱音频区仍密集点击。',
    '29617': '0–8秒462tap对23个混合对象，模型在开头近静音区域也连续出音。',
    '28312': '80–88秒310tap对40对象，规律hold/flick结构缺失。',
    '49151': '0–8秒451tap对3个稀疏tap，低负担起段严重误放。',
    '70898': '136–144秒373tap对28对象，人谱hold/drag层次全失。',
    '2357': '24–32秒267tap对58对象，人谱长hold/drag被短点击替代。',
    '9047': '48–56秒291tap对114对象，没有表达51秒后flick/drag/hold类型转换。',
    '35274': '16–24秒248tap对28对象，长hold持续结构丢失。',
    '64151': '16–24秒553tap对18个稀疏对象；全曲4212对135，低负担失效。',
    '18170': '16–24秒437tap对47对象，drag向tap/hold的转换缺失。',
    '40131': '24–32秒456tap对68对象，细碎点击掩盖可辨的稀疏节奏与少量flick。',
    '76445': '112–120秒265tap对17对象，长hold与稀疏flick/drag均缺失。',
    '56430': '176–184秒无模型输出、人谱20对象；这是评测deadline导致的未覆盖，不是模型主动留白。',
    '38848': '32–40秒523tap对38对象，长hold全丢；整曲后段因deadline未完成。',
    '62420': '16–24秒264tap对64对象，没有hold/flick/drag；整曲尾部未完成。',
    '65921': '24–32秒401tap对95对象，类型坍缩；后续长段因deadline未生成。',
}


def main():
    disposition_path = 'reports/d97_candidate_disposition_v1.json'
    review_path = 'reports/d97_assistant_review_1000_v1.json'
    if (ROOT/disposition_path).exists() or (ROOT/review_path).exists():
        raise FileExistsError('Closeout already exists; do not double-count')
    if 'scripts.run_local_semantic_objective' in subprocess.run(
            ['ps','-p','16417','-o','command='],capture_output=True,text=True).stdout:
        raise RuntimeError('Exact training handle still live')
    plan_path = 'bench/rhythm-v2/d97-semantic-quality-v1.json'
    plan = read(plan_path)
    training, generation = read(plan['output']), read(plan['generation_output'])
    auto_path = 'reports/d97_review_1000_v1.json'
    comp_path = 'reports/d97_quality_comparison_1000_v1.json'
    cond_path = 'reports/d97_sequence_condition_response_v1.json'
    audio_path = 'reports/d97_sequence_audio_response_v1.json'
    automatic, comparison, condition, audio = map(read, (auto_path,comp_path,cond_path,audio_path))
    if not (training['complete'] and training['updates'] == 1000 and not training.get('error')
            and generation['execution_complete'] and len(automatic['records']) == 20
            and not comparison['numerical_gates_passed'] and condition['execution_complete']
            and not condition.get('error') and not condition['expansion_allowed']
            and audio['complete'] and not audio.get('error')):
        raise ValueError('Missing terminal evidence or unexpected verdict')
    for evidence in (generation,automatic,condition,audio):
        if evidence['checkpoint_sha256'] != training['checkpoint_sha256']:
            raise ValueError('Checkpoint identity differs')
    for p,expected in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
        if sha(p) != expected:
            raise ValueError('Frozen source changed: '+p)
    if sha(training['checkpoint']) != training['checkpoint_sha256'] or sha(plan_path) != training['plan_sha256']:
        raise ValueError('Plan or checkpoint changed')
    inspection_path = 'artifacts/d97-review-v1/inspection/manifest.json'
    inspection = read(inspection_path)
    if len(inspection['sheets']) != 10 or inspection['report_sha256'] != sha(auto_path):
        raise ValueError('Need exact twenty-song full/detail evidence')
    for sheet in inspection['sheets']:
        if sha(sheet['file']) != sheet['sha256']:
            raise ValueError('Sheet changed')
        for source in sheet['sources']:
            if sha(source['path']) != source['sha256']:
                raise ValueError('Sheet source changed')
    if {'phira:'+i for i in DETAILS} != {r['chart_id'] for r in automatic['records']}:
        raise ValueError('Incomplete actual review coverage')
    cohort = read('reports/d66_local_cohort_status.json')
    before = dict(objective=4,architecture=4,data_representation=5,training_dynamics=2)
    if cohort['current_local_experiment_failures'] != before:
        raise ValueError('Counter changed; reconcile instead of overwriting')
    after = dict(before,objective=5)
    s = comparison['summary']['candidate']
    aggregate = {k:v for k,v in s.items() if k not in ('type_scores','multiplicity')}
    aggregate.update(model_types=s['predicted_types'], generated_notes=s['predicted_notes'], macro_full_note_f1=s['macro_f1'])
    paths = [plan_path,plan['output'],plan['generation_output'],auto_path,comp_path,cond_path,audio_path,
             inspection_path,'reports/d97_stage_1000_v1.json','scripts/closeout_local_semantic_objective.py']
    sources = {p:sha(p) for p in paths}
    rows = [dict(chart_id=r['chart_id'],title=r['title'],stratum=r['stratum'],notes=r['notes'],
        full_note_f1=r['full_note_f1'],complete=r['complete'],failure=r['failure'],preview=r['preview'],
        whole_song_observation=f"原始输出{r['notes']['model']}个全tap，对照{r['notes']['human']}个人谱对象；全曲密集点击且hold/flick/drag消失。"
            +(f"未完成原因：{r['failure']}。" if r['failure'] else ''),
        detail_observation=DETAILS[r['chart_id'].split(':')[1]],detail_scope_us=r['audio_selected_detail_scope_us'])
        for r in automatic['records']]
    save(review_path,dict(id='D97-assistant-review-1000-v1',complete=True,songs=20,
        checkpoint_sha256=training['checkpoint_sha256'],automatic_evidence_sha256=sha(auto_path),
        source_sha256=sources,artifact_sha256={s['file']:s['sha256'] for s in inspection['sheets']},
        visually_reviewed_songs=20,visually_reviewed_sheets=10,timeline_visual_review_complete=True,
        audio_audition_complete=False,native_playtest_complete=False,rows=rows,aggregate=aggregate,summary=aggregate,
        quality_passed=False,converged_candidates=0,optimizer_updates=0,new_cloud_compute_calls=0,
        limitations=['Five outputs incomplete; their missing regions are failures, not musical rests.',
            'Actual audio-flux and raw timing/type plots reviewed; no audio audition or native playtest.',
            'Density failure stopped other conditions. Reference remains unknown/untrained.',
            'All four audio swaps change tokens, showing sensitivity only, not correct musical understanding.',
            'Auxiliary Phira placement is not model spatial ability.']))
    runner = 'scripts/run_local_semantic_objective.py'
    archive = 'inputs/code_baselines/d97-closed-v1/'+runner
    if (ROOT/archive).exists():
        raise FileExistsError(archive)
    (ROOT/archive).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/archive).write_bytes((ROOT/runner).read_bytes())
    if sha(archive) != plan['code_sha256'][runner]:
        raise ValueError('Owned runner archive mismatch')
    (ROOT/runner).write_text('"""D97 A rejected after the frozen 1000-update quality comparison."""\n'
        'raise SystemExit("D97 A closed; no automatic resume. See reports/d97_candidate_disposition_v1.json. '
        'Original source, weights and raw outputs are archived.")\n')
    next_route = ('Switch away from loss-weight tuning. Inspect Data/Representation: conditional exposure and '
        'teacher-prefix versus generated-prefix decision failure, and genuine visible-reference support after clock/duplicate audit. '
        'Freeze one new hypothesis before any quality training; do not revive old k8/relative/grid losses or auto-run B from rejected A. '
        'B SCST is untested, not disproven; retain scoring/sampling mechanics but require a separately qualified base and real signal gates. '
        'Keep the original12h deadline and0new paid compute.')
    save(disposition_path,dict(id='D97-semantic-objective-rejected-v1',status='rejected',dimension='objective',complete=True,
        quality_passed=False,converged_candidates=0,resume_allowed=False,scaling_allowed=False,automatic_execution_allowed=False,
        updates=1000,new_updates=998,checkpoint=training['checkpoint'],checkpoint_sha256=training['checkpoint_sha256'],
        numerical_gates=comparison['gates'],failure_counters_before=before,failure_counters_after=after,
        counting='One supervised semantic-weight hypothesis. B was not trained and adds no failure count.',
        review=review_path,next_allowed_route=next_route,source_sha256={**sources,review_path:sha(review_path)},
        rollback=dict(owned_file=runner,archived_source=archive,archive_sha256=sha(archive),disabled_entry_sha256=sha(runner),
            production_model_unchanged=sha('beatfcn/model/full_event_model.py')==plan['code_sha256']['beatfcn/model/full_event_model.py'],
            retained='All shared code/data, original runtime snapshot, checkpoints, raw outputs and isolated scoring/replay diagnostics.'),
        created_at_unix=time.time(),new_cloud_compute_calls=0))
    cohort.update(phase='d97_closed_quality_failed',active_stage=None,active_auxiliary_stages=[],active_aux_stages=[],
        current_local_experiment_failures=after,candidate_disposition=disposition_path,whole_song_review=review_path,
        current_review='reports/d97_paired1000_review_v1.md',next_action=next_route,quality_passed=False,converged_candidates=0,
        updated_at_unix=time.time())
    save('reports/d66_local_cohort_status.json',cohort)
    state = read('reports/d97_current_state_v1.json')
    state.update(status='A_closed_quality_failed_B_not_dispatched',current_quality_pid=None,
        current_quality_session=None,disposition=disposition_path,review=review_path,next_action=next_route,
        failure_counters=after,updated_at_unix=time.time())
    state['A'].update(status='rejected',quality_passed=False,resume_allowed=False)
    state['B'].update(status='not_dispatched_no_qualified_A_base',actual_signal_gate_passed=None)
    save('reports/d97_current_state_v1.json',state)
    budget = read('reports/rhythm_v2_budget.json'); old = deepcopy(budget)
    key = 'd97-condition-outcome-v1'
    budget['local_cpu_diagnostics'][key].update(status='A_closed_quality_failed_B_not_dispatched',
        dispatch_pid=None,dispatch_session_id=None,child_pid=None,quality_pid=None,quality_session_id=None,
        quality_verified_updates=1000,quality_new_updates=998,optimizer_updates=1000,
        train_s=training['train_s'],preparation_s=training['preparation_s'],evaluation_s=training['evaluation_s'],
        final_attempt_wall_s=training['attempt_wall_s'],generation_wall_s=generation['wall_s'],
        condition_wall_s=condition['wall_s'],audio_wall_s=audio['wall_s'],
        stage1000_wall_s=read('reports/d97_stage_1000_v1.json')['wall_s'],review=review_path,disposition=disposition_path,
        reference_transfer_bytes=read('reports/d97_reference_download_v1.json')['transferred_bytes'],
        timing_note='Phase durations overlap attempt wall time; do not sum both accounting views.',
        new_cloud_spend_usd=0,new_cloud_compute_calls=0)
    check=deepcopy(budget);check['local_cpu_diagnostics'][key]=old['local_cpu_diagnostics'][key]
    assert check==old
    save('reports/rhythm_v2_budget.json',budget)
    target=read('reports/training_monitor_target.json')
    target['current_work_label']='D97 A结案失败：F1 .07807、128439全tap、15/20完整、密度0/4有向0/8在带；B未训练，0收敛。参考原14歌经已知clock混淆检查仅8歌保留差异；仍0准入独立风格验证歌。新增云0。'
    target['local_song_training'].update(status='completed_quality_failed',pid=None,session_id=None,
        artifact_index='artifacts/d97-review-v1/index.html',assistant_review=review_path)
    for field,path in [('review',auto_path),('timeline_review',review_path),('condition_audit',cond_path),('audio_audit',audio_path)]:
        target['local_song_training'][field]=dict(report=Path(path).name,sha256=sha(path))
    target['quality_disposition']=dict(report=Path(disposition_path).name,sha256=sha(disposition_path))
    target['local_cohort']['review']='reports/d97_paired1000_review_v1.md'
    save('reports/training_monitor_target.json',target)
    print(json.dumps(dict(disposition=disposition_path,failures=after,budget_other_fields_unchanged=True)))


if __name__ == '__main__':
    main()

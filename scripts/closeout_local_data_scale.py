"""Record the actual D94 twenty-song review and retire only its owned runner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def save(name, value):
    p = ROOT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(p)


# Observed in all ten rendered sheets, not inferred only from scalar scores.
OBSERVATIONS = [
    ('28324', '长段密集tap带和整块留白，原谱的drag串、hold和flick全部丢失。', '24–32秒96tap替代67个混合对象，tap细碎分布未保持人谱选择。'),
    ('12626', '连续tap带在约96–104、112–128秒等处断开，其他类型全无。', '184–192秒模型空，人谱10个对象含长hold，音频仍有明显变化。'),
    ('38724', '287秒整曲空，人谱4284个混合对象；不是生成未完成。', '136–144秒仍空，人谱144个对象，含大量tap和flick。'),
    ('60952', '2296tap覆盖几乎全曲，对照371个flick/drag，低难过密且类型全错。', '88–96秒79tap替代18drag；不能因艺术drag不逐点对应起音而删人谱。'),
    ('19959', 'tap块之间出现大段留白，人谱的连续drag和长hold没有延续。', '32–40秒模型空，人谱20个tap/drag，音频峰仍存在。'),
    ('29617', '只在约56–72、136–160和192–200秒有tap块，漏掉大部分人谱结构。', '0–8秒模型空，人谱23个对象含长hold和其他类型。'),
    ('28312', '1548tap几乎覆盖全曲，对照676个混合对象；hold和flick完全丢失。', '80–88秒155tap替代40个含长短hold和flick的对象，缺少采音层次。'),
    ('49151', '1108tap对350个人谱对象；大片tap带没有保留人谱tap/hold变化。', '0–8秒模型空，人谱3个稀疏tap。'),
    ('70898', '只有若干tap块，没表达原谱中段大量长hold。', '136–144秒模型空，人谱28个可见对象含延续hold和drag。'),
    ('2357', '约8–88秒近连续tap，未保留hold、flick和drag。', '24–32秒82tap对58个混合对象，后半drag选择和hold持续信息缺失。'),
    ('9047', '两段tap带中间有约8秒空白，混合类型结构丢失。', '48–56秒100tap对114个混合对象；数目接近但长hold与flick/drag均消失。'),
    ('35274', '1799与1777的总数接近，模型仍为几乎连续tap，人谱有明显类型和负担变化。', '16–24秒74tap对28个以长hold为主的对象，点击不能替代持续信息。'),
    ('64151', '253tap对135个人谱对象，稀疏采音变成更密的点击带。', '16–24秒44tap对18个混合对象，flick和drag丢失。'),
    ('18170', '长tap带与块状留白，早期drag和后续类型转换均未表达。', '16–24秒88tap对47个对象，人谱drag向tap/hold的变化没有出现。'),
    ('40131', '1019tap覆盖多数时段，长hold、flick/drag切换和尾段均有缺失。', '24–32秒97tap对68个对象，细碎tap未保留flick及tap节奏。'),
    ('76445', '仅约16–32秒产生155tap，其余曲段空白，对照724个人谱对象。', '112–120秒模型空，人谱17个含长hold、flick、drag的对象。'),
    ('56430', '1257tap对687个人谱对象，长段tap带及块状留白替代混合类型。', '176–184秒模型空，人谱20个tap/hold/drag对象。'),
    ('38848', '1995tap近乎覆盖全曲，对740个人谱对象；未表达hold主导和留白结构。', '32–40秒83tap对38个对象，原谱长hold被细碎点击替代。'),
    ('62420', '139秒整曲空，人谱1140个对象；完整结束不等于可用。', '16–24秒模型空，人谱64个混合对象，音频仍有活动。'),
    ('65921', '3655tap对2137个人谱对象，连续过密tap和中间块状留白；本轮无同刻巨量堆叠。', '24–32秒194tap对95个对象，缺少hold/flick/drag和原谱间隔变化。'),
]


def main():
    disposition_path = 'reports/d94_candidate_disposition_v1.json'
    if (ROOT/disposition_path).exists():
        raise RuntimeError('D94 already closed; do not increment failure counters again')
    live = subprocess.run(['ps', '-p', '47619', '-o', 'command='], capture_output=True, text=True).stdout
    if 'scripts.run_local_data_scale' in live:
        raise RuntimeError('The owned training/generation process is still live')
    plan_path = 'bench/rhythm-v2/d94-data-scale-quality-v1.json'
    plan = read(plan_path)
    train = read(plan['output'])
    generation = read(plan['generation_output'])
    automatic_path = 'reports/d94_review_1000_v1.json'
    automatic = read(automatic_path)
    comparison_path = 'reports/d94_quality_comparison_1000_v1.json'
    comparison = read(comparison_path)
    condition_path = 'reports/d94_sequence_condition_response_v1.json'
    audio_path = 'reports/d94_sequence_audio_response_v1.json'
    condition, audio = read(condition_path), read(audio_path)
    if (not train['complete'] or train.get('error') or train['updates'] != 1000
            or not generation['execution_complete'] or generation['completed_songs'] != 20
            or len(automatic['records']) != 20 or comparison['numerical_gates_passed']
            or not condition['execution_complete'] or condition['expansion_allowed']
            or not audio['complete']):
        raise ValueError('Missing completed evidence or a different quality disposition')
    for evidence in (generation, automatic, condition, audio):
        if evidence['checkpoint_sha256'] != train['checkpoint_sha256']:
            raise ValueError('Evidence refers to different weights')
    if sha(train['checkpoint']) != train['checkpoint_sha256'] or sha(plan_path) != train['plan_sha256']:
        raise ValueError('Frozen model or plan changed')
    runner = 'scripts/run_local_data_scale.py'
    model = 'beatfcn/model/full_event_model.py'
    if any(sha(p) != plan['code_sha256'][p] for p in (runner, model)):
        raise ValueError('Owned runner or production model changed; do not overwrite shared edits')
    inspection_path = 'artifacts/d94-review-v1/inspection/manifest.json'
    inspection = read(inspection_path)
    if len(inspection['sheets']) != 10:
        raise ValueError('Missing full-song/detail sheets')
    for p, expected in inspection['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Rendered source changed: '+p)
    for sheet in inspection['sheets']:
        if sha(sheet['file']) != sheet['sha256']:
            raise ValueError('Rendered sheet changed')
    cohort = read('reports/d66_local_cohort_status.json')
    before = dict(objective=4, architecture=3, data_representation=4, training_dynamics=2)
    if cohort['current_local_experiment_failures'] != before:
        raise ValueError('Failure counters changed; reconcile rather than overwrite')
    after = dict(before, data_representation=5)
    observations = {'phira:'+i: (whole, detail) for i, whole, detail in OBSERVATIONS}
    if set(observations) != {r['chart_id'] for r in automatic['records']}:
        raise ValueError('Twenty actual observations required')
    rows = [dict(chart_id=r['chart_id'], title=r['title'], stratum=r['stratum'], notes=r['notes'],
                 full_note_f1=r['full_note_f1'], whole_song_observation=observations[r['chart_id']][0],
                 detail_observation=observations[r['chart_id']][1],
                 detail_scope_us=r['audio_selected_detail_scope_us'], preview=r['preview'])
            for r in automatic['records']]
    s = comparison['summary']['candidate']
    aggregate = dict(complete_legal_songs=s['complete_legal_songs'], nonempty_songs=s['nonempty_songs'],
        generated_notes=s['predicted_notes'], human_notes=s['human_notes'], model_types=s['predicted_types'],
        macro_full_note_f1=s['macro_f1'], tap_positive_intervals=s['tap_positive_intervals'],
        tap_under10ms_fraction=s['tap_short_fraction'], zero_interval_tap_pairs=s['tap_zero_adjacent_pairs'],
        max_same_onset=s['max_notes_same_timestamp'])
    sources = [plan_path, plan['output'], plan['generation_output'], automatic_path, comparison_path,
               condition_path, audio_path, 'reports/d94_music_comparison_v1.json', inspection_path,
               'reports/d94_stage_1000_v1.json', 'reports/d94_comparison_identity_fix_v1.json',
               'scripts/closeout_local_data_scale.py']
    review_path = 'reports/d94_assistant_review_1000_v1.json'
    save(review_path, dict(id='D94-assistant-review-1000-v1', complete=True,
        checkpoint_sha256=train['checkpoint_sha256'], automatic_evidence_sha256=sha(automatic_path),
        source_sha256={p:sha(p) for p in sources},
        artifact_sha256={p['file']:p['sha256'] for p in inspection['sheets']},
        visually_reviewed_songs=20, visually_reviewed_sheets=10, songs=20,
        timeline_visual_review_complete=True, audio_audition_complete=False, native_playtest_complete=False,
        spatial_output='Static auxiliary placement only; no learned Phira spatial output.',
        aggregate=aggregate, summary=aggregate, rows=rows, quality_passed=False, converged_candidates=0,
        optimizer_updates=0, new_cloud_compute_calls=0, condition_report=condition_path, audio_report=audio_path,
        limitations=['No actual audio audition or native playtest; timelines and audio-flux plots were inspected.',
                    'All fields beyond failed density are unproven; human-reference continuation is untrained.',
                    'Fixed1000 updates/4000 requests expose more songs fewer times; token count is6.19% higher.',
                    'One seed and this horizon cannot establish that more data is intrinsically harmful.',
                    'Reference matching and estimated audio onsets are proxies, not unique correct charts.']))
    archive = 'inputs/code_baselines/d94-closed-v1/scripts/run_local_data_scale.py'
    dest = ROOT/archive
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_bytes((ROOT/runner).read_bytes())
    if sha(archive) != plan['code_sha256'][runner]:
        raise ValueError('Owned execution archive differs')
    (ROOT/runner).write_text('"""D94 closed after failed1000-step full-song quality evaluation."""\n'
        'raise SystemExit("D94 rejected; no automatic resume. See reports/d94_candidate_disposition_v1.json. '
        'Original source and all data/outputs are archived.")\n')
    next_route = ('Switch away from the data-pool intervention. Prepare the untested7.89M capacity comparison '
        'on the same333-song panel, same audio/target/conditions/loss/exposure and20-song evaluation. '
        'First verify initialization comparability and existing health scope; only run missing necessary '
        'real-data gates. Do not silently extend D94, change losses/audio simultaneously, or call this '
        'a converged candidate. All-condition response and human-reference continuation remain required.')
    save(disposition_path, dict(id='D94-data-scale-rejected-v1', status='rejected', dimension='data_representation',
        complete=True, quality_passed=False, converged_candidates=0, resume_allowed=False,
        scaling_allowed=False, automatic_execution_allowed=False, updates=1000, new_updates=998,
        checkpoint=train['checkpoint'], checkpoint_sha256=train['checkpoint_sha256'],
        plan_sha256=sha(plan_path), numerical_gates=comparison['gates'], whole_twenty_song_review=True,
        failure_counters_before=before, failure_counters_after=after,
        counting='One data-pool quality hypothesis; health/preparation do not add or clear failures.',
        decisive_failures=['MacroF1 .061348213 versus matched control .076097527;2whole songs empty.',
                          'All22657 emitted events tap; no hold/flick/drag or generated chords.',
                          'Density4/4 identical,0/4 directed,1/8 in band; expansion stopped.',
                          'Low-difficulty overproduction, unsupported blank blocks and repetitive tap streams.'],
        interpretation='More independent songs alone failed at the fixed exposure budget. Data value at longer '
            'exposure or different capacity remains unresolved; accurate corpus/cache assets are retained.',
        next_allowed_route=next_route, review=review_path,
        rollback=dict(owned_file=runner, archived_source=archive, archive_sha256=sha(archive),
                      disabled_entry_sha256=sha(runner), production_model_unchanged=sha(model)==plan['code_sha256'][model],
                      retained='All verified data, features, checkpoints, raw outputs, diagnostics and shared work.'),
        source_sha256={p:sha(p) for p in sources+[review_path]}, created_at_unix=time.time(), new_cloud_compute_calls=0))
    budget_path = 'reports/rhythm_v2_budget.json'
    budget = read(budget_path)
    old_budget = deepcopy(budget)
    entry = budget['local_cpu_diagnostics']['d94_data_scale_quality']
    entry.update(status='completed_quality_failed', quality_updates=1000, new_quality_updates=998,
                 pid=None, session_id=None, disposition=disposition_path, review=review_path,
                 complete_songs=20, nonempty_songs=18, new_cloud_spend_usd=0,
                 train_s=train['train_s'], preparation_s=train['preparation_s'], generation_s=generation['wall_s'],
                 last_phase_end_to_end_s=902.60, comparison_identity_repair_tests=4)
    check = deepcopy(budget)
    check['local_cpu_diagnostics']['d94_data_scale_quality'] = old_budget['local_cpu_diagnostics']['d94_data_scale_quality']
    assert check == old_budget
    save(budget_path, budget)
    cohort.update(phase='d94_closed_quality_failed', active_stage=None, active_auxiliary_stages=[],
                  active_aux_stages=[], current_candidate_effective_updates=1000, quality_passed=False,
                  converged_candidates=0, current_local_experiment_failures=after, candidate_disposition=disposition_path,
                  whole_song_review=review_path, current_review='reports/d94_paired1000_review_v1.md',
                  next_action=next_route, updated_at_unix=time.time())
    save('reports/d66_local_cohort_status.json', cohort)
    state = read('reports/d94_current_state_v1.json')
    state.update(phase='closed_quality_failed', pid=None, session_id=None, observed_at_unix=time.time(),
                 latest_persisted_update=1000, review=review_path, disposition=disposition_path, next_action=next_route)
    save('reports/d94_current_state_v1.json', state)
    target = read('reports/training_monitor_target.json')
    target['current_work_label'] = 'D94 1000步20曲已复盘失败：F1 .06135、全tap、2空谱、密度0/4；不续训，0收敛。'
    target['local_song_training'].update(pid=None, session_id=None, status='completed_quality_failed',
        artifact_index='artifacts/d94-review-v1/index.html', assistant_review=review_path)
    for key, p in [('review',automatic_path),('timeline_review',review_path),
                   ('condition_audit',condition_path),('audio_audit',audio_path)]:
        target['local_song_training'][key] = dict(report=Path(p).name, sha256=sha(p))
    target['quality_disposition'] = dict(report=Path(disposition_path).name, sha256=sha(disposition_path))
    target['local_cohort']['review'] = 'reports/d94_paired1000_review_v1.md'
    save('reports/training_monitor_target.json', target)
    print(json.dumps(dict(disposition=disposition_path, review=review_path, aggregate=aggregate,
                          failures=after, budget_other_fields_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()

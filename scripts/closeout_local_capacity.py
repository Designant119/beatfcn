"""Archive the reviewed D95 failure without changing shared model/data code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def save(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


# Recorded after actually viewing all five whole-song and five detail sheets.
OBSERVATIONS = [
    ('28324', '只在约64–72秒出现265tap，其余大段空白；原谱连续drag、长hold与flick全丢。', '24–32秒模型空，人谱67个混合对象，音频仍有清晰周期峰。'),
    ('12626', '只在约24–40秒产生715tap，整曲其余部分漏掉，类型全部坍缩。', '184–192秒模型空，人谱10个对象且有两段长hold；音频仍活动。'),
    ('38724', '287秒只出现两小段tap带，共700个，对照4284个人谱对象。', '136–144秒模型空，人谱144个对象，包括大量tap和flick。'),
    ('60952', '186秒整曲空，人谱371个flick/drag；合法结束不等于可用。', '88–96秒模型空，人谱18drag，音频节律峰明确；不能把艺术drag当脏标签。'),
    ('19959', '147秒整曲空，人谱411个对象且有连续drag与hold。', '32–40秒模型空，人谱20个tap/drag，音乐仍有变化。'),
    ('29617', '218秒整曲空，人谱995个混合对象。', '0–8秒模型空，人谱23个对象；长hold与后续flick/drag均缺失。'),
    ('28312', '1851tap集中成数块，长段空白，未保留676个人谱对象的类型和结构。', '80–88秒模型空，人谱40个含hold/flick的对象，音频并未静止。'),
    ('49151', '189秒整曲空，人谱350个tap/hold。', '0–8秒模型空，人谱3个稀疏tap；不能由局部少音推断整曲为空合理。'),
    ('70898', '151秒整曲空，人谱395个对象，中段大量hold未表达。', '136–144秒模型空，人谱28个可见对象，含延续hold与drag。'),
    ('2357', '2181tap对570个人谱对象，后半段近连续点击带，其他类型全消失。', '24–32秒245tap对58个混合对象；细碎点击不能替代长hold、flick和drag。'),
    ('9047', '2301tap分成数个整块，块间空白，人谱的类型切换和持续信息丢失。', '48–56秒模型空，人谱114个对象，含成组flick/drag和hold。'),
    ('35274', '4253tap对1777个人谱对象，多个长点击带，中间整块留白。', '16–24秒模型空，人谱28个对象以长hold为主；音频持续有周期变化。'),
    ('64151', '71秒整曲空，人谱135个稀疏对象，低负担内容完全未生成。', '16–24秒模型空，人谱18个混合对象，音频仍有显著峰。'),
    ('18170', '181秒整曲空，人谱1109个对象；早期drag与后续混合段全漏。', '16–24秒模型空，人谱47个对象，包含drag向tap/hold的转换。'),
    ('40131', '1386tap集中为数段，对照829个混合对象；长hold和类型变化缺失。', '24–32秒模型空，人谱68个tap/flick等对象且音频节律仍在。'),
    ('76445', '1991tap对724个人谱对象，块状过密点击与长空段并存。', '112–120秒模型空，人谱17个含长hold、flick和drag的对象。'),
    ('56430', '193秒整曲空，人谱687个混合对象。', '176–184秒模型空，人谱20个tap/hold/drag对象，波形仍有活动。'),
    ('38848', '仅尾段两块552tap，前约160秒大段漏失；740个人谱对象的hold结构不见。', '32–40秒模型空，人谱38个对象且长hold明显。'),
    ('62420', '140秒整曲空，人谱1140个混合对象；不以单个严格时钟F1归因音乐能力。', '16–24秒模型空，人谱64个对象，音频周期峰明显。'),
    ('65921', '2827tap集中为五块，大片漏段；原谱2137个对象的混合类型和采音层次全失。', '24–32秒模型空，人谱95个对象；这不是长hold造成的无新头。'),
]


def main():
    disposition_path = 'reports/d95_candidate_disposition_v1.json'
    review_path = 'reports/d95_assistant_review_1000_v1.json'
    if any((ROOT / p).exists() for p in (disposition_path, review_path)):
        raise RuntimeError('D95 already reviewed/closed; reconcile instead of counting twice')
    live = subprocess.run(['ps', '-p', '66922', '-o', 'command='], capture_output=True, text=True).stdout
    if 'scripts.run_local_capacity' in live:
        raise RuntimeError('D95 training/generation still live')
    plan_path = 'bench/rhythm-v2/d95-capacity-quality-v1.json'
    plan = read(plan_path)
    training, generation = read(plan['output']), read(plan['generation_output'])
    automatic_path = 'reports/d95_review_1000_v1.json'
    comparison_path = 'reports/d95_quality_comparison_1000_v1.json'
    condition_path = 'reports/d95_sequence_condition_response_v1.json'
    audio_path = 'reports/d95_sequence_audio_response_v1.json'
    automatic, comparison = read(automatic_path), read(comparison_path)
    condition, audio = read(condition_path), read(audio_path)
    if (not training['complete'] or training.get('error') or training['updates'] != 1000
            or not generation['execution_complete'] or generation['completed_songs'] != 20
            or len(automatic['records']) != 20 or comparison['numerical_gates_passed']
            or not condition['execution_complete'] or condition.get('error')
            or condition['expansion_allowed'] or not audio['complete'] or audio.get('error')):
        raise ValueError('Missing terminal evidence or unexpected verdict')
    for evidence in (generation, automatic, condition, audio):
        if evidence['checkpoint_sha256'] != training['checkpoint_sha256']:
            raise ValueError('Evidence has different weights')
    if sha(training['checkpoint']) != training['checkpoint_sha256'] or sha(plan_path) != training['plan_sha256']:
        raise ValueError('Checkpoint/plan changed')
    runner = 'scripts/run_local_capacity.py'
    model = 'beatfcn/model/full_event_model.py'
    if any(sha(p) != plan['code_sha256'][p] for p in (runner, model)):
        raise ValueError('Owned runner or shared model changed; do not overwrite')
    inspection_path = 'artifacts/d95-review-v1/inspection/manifest.json'
    inspection = read(inspection_path)
    if len(inspection['sheets']) != 10:
        raise ValueError('Need all twenty full-song and detail figures')
    for p, expected in inspection['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Rendered source changed: ' + p)
    for sheet in inspection['sheets']:
        if sha(sheet['file']) != sheet['sha256']:
            raise ValueError('Rendered sheet changed')
    observations = {'phira:' + i: (whole, detail) for i, whole, detail in OBSERVATIONS}
    if set(observations) != {r['chart_id'] for r in automatic['records']}:
        raise ValueError('Require twenty actual observations')
    cohort = read('reports/d66_local_cohort_status.json')
    before = dict(objective=4, architecture=3, data_representation=5, training_dynamics=2)
    if cohort['current_local_experiment_failures'] != before:
        raise ValueError('Failure counters changed')
    after = dict(before, architecture=4)
    archive = 'inputs/code_baselines/d95-closed-v1/scripts/run_local_capacity.py'
    if (ROOT / archive).exists():
        raise FileExistsError(archive)
    s = comparison['summary']['candidate']
    aggregate = {k: v for k, v in s.items() if k not in ('type_scores', 'multiplicity')}
    aggregate.update(model_types=s['predicted_types'], generated_notes=s['predicted_notes'],
                     macro_full_note_f1=s['macro_f1'])
    sources = [plan_path, plan['output'], plan['generation_output'], automatic_path, comparison_path,
        condition_path, audio_path, inspection_path, 'reports/d95_stage_1000_v1.json',
        'reports/d95_music_comparison_v1.json', 'reports/d95_matched_learning_curve_v1.json',
        'reports/d95_training_support_v1.json', 'scripts/closeout_local_capacity.py']
    rows = [dict(chart_id=r['chart_id'], title=r['title'], stratum=r['stratum'], notes=r['notes'],
        full_note_f1=r['full_note_f1'], whole_song_observation=observations[r['chart_id']][0],
        detail_observation=observations[r['chart_id']][1], detail_scope_us=r['audio_selected_detail_scope_us'],
        preview=r['preview']) for r in automatic['records']]
    save(review_path, dict(id='D95-assistant-review-1000-v1', complete=True,
        checkpoint_sha256=training['checkpoint_sha256'], source_sha256={p: sha(p) for p in sources},
        automatic_evidence_sha256=sha(automatic_path), songs=20,
        artifact_sha256={s['file']: s['sha256'] for s in inspection['sheets']},
        visually_reviewed_songs=20, visually_reviewed_sheets=10,
        timeline_visual_review_complete=True, audio_audition_complete=False, native_playtest_complete=False,
        spatial_output='Auxiliary static placement, not learned Phira spatial output.',
        aggregate=aggregate, summary=aggregate, rows=rows, quality_passed=False,
        converged_candidates=0, optimizer_updates=0, new_cloud_compute_calls=0,
        limitations=['Raw timing/type and audio-flux plots inspected; no actual listening/native playtest.',
            'Failed density stopped wider condition testing; reference continuation remains untrained.',
            'One seed/fixed1000 horizon and common optimizer do not establish optimal capacity scaling.',
            'All four audio-swap pairs are empty: no evidence of response here, not proof of global audio independence.',
            'Audio anchoring is an onset proxy, not the unique correct chart. Missing/empty metrics remain explicit.']))
    destination = ROOT / archive
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT / runner).read_bytes())
    if sha(archive) != plan['code_sha256'][runner]:
        raise ValueError('Archive differs from frozen owned runner')
    (ROOT / runner).write_text('"""D95 closed after failed1000-step full-song quality evaluation."""\n'
        'raise SystemExit("D95 rejected; no automatic resume. See reports/d95_candidate_disposition_v1.json. '
        'Original source and all data/outputs are archived.")\n')
    next_route = ('Switch away from capacity tuning to the unresolved audio representation question. '
        'First prepare a bounded local MERT2 runtime/operator/resource gate in an isolated environment; '
        'do not change the existing experiment environment or launch whole-library encoding yet. '
        'Use reports/d95_mert2_local_feasibility_research.md: Pupu two-slot versus MERT single-vector '
        'requires an explicit common interface and a fresh matched Pupu control, not copied fake bands. '
        'Preserve full tails, clocks,20-song review and budget; no paid compute. '
        'All-condition response and authorized human-reference continuation remain final requirements.')
    decisive = [f'MacroF1 {s["macro_f1"]:.9f} versus matched control .061348213;9whole songs empty.',
        'All19022 emitted events tap; no hold/flick/drag or generated chords.',
        '59.96percent of positive tap gaps below10ms, versus36.10percent in control.',
        'Density4/4 identical empty outputs;0/4 directed and0/8 in band; expansion stopped.',
        'Actual twenty-song review shows severe blank blocks and dense tap bursts, no usable draft selected.']
    save(disposition_path, dict(id='D95-capacity-rejected-v1', status='rejected', dimension='architecture',
        complete=True, quality_passed=False, converged_candidates=0, resume_allowed=False,
        scaling_allowed=False, automatic_execution_allowed=False, updates=1000, new_updates=998,
        checkpoint=training['checkpoint'], checkpoint_sha256=training['checkpoint_sha256'], plan_sha256=sha(plan_path),
        numerical_gates=comparison['gates'], whole_twenty_song_review=True,
        failure_counters_before=before, failure_counters_after=after,
        counting='One width/FFN capacity hypothesis; initialization/health/preparation do not add or clear quality failures.',
        decisive_failures=decisive, next_allowed_route=next_route, review=review_path,
        interpretation='Capacity alone regressed at fixed exposure and optimizer. Not a claim about optimally trained larger models.',
        rollback=dict(owned_file=runner, archived_source=archive, archive_sha256=sha(archive),
            disabled_entry_sha256=sha(runner), production_model_unchanged=sha(model)==plan['code_sha256'][model],
            retained='All verified data, shared code, features, checkpoints, raw outputs and diagnostics.'),
        source_sha256={p: sha(p) for p in sources+[review_path]}, created_at_unix=time.time(), new_cloud_compute_calls=0))
    budget_path = 'reports/rhythm_v2_budget.json'
    budget = read(budget_path)
    old_budget = deepcopy(budget)
    entry = budget['local_cpu_diagnostics']['d95_capacity_quality']
    entry.update(phase='closed_quality_failed', pid=None, session_id=None, quality_updates=1000,
        new_quality_updates=998, complete_songs=20, nonempty_songs=s['nonempty_songs'],
        train_s=training['train_s'], preparation_s=training['preparation_s'], evaluation_s=training['evaluation_s'],
        train250_to500_wall_s=281.8411595840007, final_training_generation_attempt_wall_s=training['attempt_wall_s'],
        generation_s=generation['wall_s'], stage500_wall_s=1.1623714169836603,
        stage1000_wall_s=read('reports/d95_stage_1000_v1.json')['wall_s'],
        condition_wall_s=condition['wall_s'], audio_wall_s=audio['wall_s'],
        timing_note='Phase wall times overlap train/preparation/eval/generation categories; do not sum both accounting views.',
        actual_whole_song_visual_review=True, new_cloud_spend_usd=0, disposition=disposition_path, review=review_path,
        research='reports/d95_mert2_local_feasibility_research.md')
    check = deepcopy(budget)
    check['local_cpu_diagnostics']['d95_capacity_quality'] = old_budget['local_cpu_diagnostics']['d95_capacity_quality']
    assert check == old_budget
    save(budget_path, budget)
    cohort.update(phase='d95_closed_quality_failed', active_stage=None, active_auxiliary_stages=[], active_aux_stages=[],
        current_candidate_effective_updates=1000, quality_passed=False, converged_candidates=0,
        current_local_experiment_failures=after, candidate_disposition=disposition_path,
        whole_song_review=review_path, current_review='reports/d95_paired1000_review_v1.md',
        next_action=next_route, updated_at_unix=time.time())
    save('reports/d66_local_cohort_status.json', cohort)
    state = read('reports/d95_current_state_v1.json')
    state.update(phase='closed_quality_failed', pid=None, session_id=None, observed_at_unix=time.time(),
        latest_persisted_update=1000, quality_passed=False, review=review_path, disposition=disposition_path,
        training_generation_session24221_exit_code=0, next_action=next_route)
    save('reports/d95_current_state_v1.json', state)
    target = read('reports/training_monitor_target.json')
    target['current_work_label'] = 'D95 20整曲复盘失败：F1 .04084、19022全tap、9空谱、密度0/4；0收敛，不续训。'
    target['local_song_training'].update(pid=None, session_id=None, status='completed_quality_failed',
        artifact_index='artifacts/d95-review-v1/index.html', assistant_review=review_path)
    for key, p in [('review', automatic_path), ('timeline_review', review_path),
                   ('condition_audit', condition_path), ('audio_audit', audio_path)]:
        target['local_song_training'][key] = dict(report=Path(p).name, sha256=sha(p))
    target['quality_disposition'] = dict(report=Path(disposition_path).name, sha256=sha(disposition_path))
    target['local_cohort']['review'] = 'reports/d95_paired1000_review_v1.md'
    save('reports/training_monitor_target.json', target)
    print(json.dumps(dict(disposition=disposition_path, review=review_path, aggregate=aggregate,
        failures=after, budget_other_fields_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()

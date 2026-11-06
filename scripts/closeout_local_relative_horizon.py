"""Record the completed D90 review; never trains or changes production weights."""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def save(path, value):
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(dest)


# Actual full-song and active-detail visual observations, in frozen review order.
OBSERVATIONS = [
    ('28324', '约8–80和96–128秒为规则tap带，后段缺失；人谱大量drag、hold和flick未保留。', '24–32秒48个近等间隔tap，替代人谱67个含drag串的对象。'),
    ('12626', 'tap密度近恒定，约96–104秒留空；人谱类型与密度变化未学到。', '184–192秒模型空，人谱10个对象含长hold；此处音频仍有活动。'),
    ('38724', '几段块状留白之间是平直tap流，未表现人谱的混合类型和密度层次。', '136–144秒54个规则tap对人谱144个混合对象，长flick串缺失。'),
    ('60952', '897个tap对人谱371个flick/drag，类型完全不对应且数量过多。', '88–96秒模型空，人谱18个drag；不能把这些drag不逐个对应起音视为脏标签。'),
    ('19959', 'tap块之间大段留空，人谱drag仍连续分布于这些区域。', '32–40秒模型空，人谱20个tap/drag，音频峰仍存在。'),
    ('29617', '开头约64秒空白，后续只有零散tap块；人谱从开头就有四种类型。', '0–8秒模型空，人谱23个对象含长hold和drag。'),
    ('28312', '2287个近连续tap对676个人谱对象，过密且缺少类型变化。', '80–88秒163个近等间隔tap对40个人谱对象；人谱有短hold和flick节奏。'),
    ('49151', '前段零散8秒块，约104秒后tap连续；未追随人谱tap/hold的结构。', '0–8秒模型空，人谱3个稀疏tap，不能以全空代替低难采音。'),
    ('70898', '模型约40秒后才开始，只有两段tap带；人谱约20秒起已有hold。', '136–144秒模型空，人谱28个含hold和drag的对象。'),
    ('2357', '约8–88秒近恒定tap流，其他类型全部消失。', '24–32秒41个规则tap对58个人谱对象，缺少hold、flick和后半drag串。'),
    ('9047', '两段规则tap带，中间约72–80秒空白，人谱混合类型未延续。', '48–56秒48个规则tap对114个人谱对象，未表现中段hold和flick/drag变化。'),
    ('35274', 'tap密度平直且约152–160秒留空；人谱类型和负担变化未表达。', '16–24秒41tap对28个人谱对象；人谱以长hold为主，模型均匀点击替代。'),
    ('64151', '早段稀疏后出现均匀tap尾部，273个输出约为135个人谱对象两倍。', '16–24秒38个规则tap对18个人谱对象，忽略稀疏tap间的flick/drag选择。'),
    ('18170', '约16–136秒连续tap，后续另有短带；早期drag和混合类型完全缺失。', '16–24秒44tap对47个人谱对象，数量接近但未表现drag到tap/hold的转换。'),
    ('40131', '约8–96秒规则tap带，后段缺失，人谱类型切换未表达。', '24–32秒44个等间隔tap对68个人谱对象，漏掉flick并简化tap节奏。'),
    ('76445', '总数718对724接近，仍有长中段空白和孤立tap块；总数接近不等于有用。', '112–120秒模型空，人谱17个含长hold、flick、drag的对象。'),
    ('56430', '约24–40、64–80、104–112和144–168秒有tap块，其余大量空白；人谱四类型仍遍布全曲。', '176–184秒模型空，人谱20个对象含hold、tap和drag。'),
    ('38848', '约8–192秒几乎连续规则tap，尾部更密；人谱有明确留白和hold主导段。', '32–40秒35个规则tap对38个人谱对象，总数接近但hold/flick及起音选择不同。'),
    ('62420', '只有约16–32、72–80和104–128秒tap块，长段音频活动被漏掉。', '16–24秒54个规则tap对64个人谱对象，未表达人谱tap间隔变化及hold。'),
    ('65921', '未完成：仅早期约16–25秒生成2258tap后触发core_token_limit，后面不是合理留白。', '24–32秒片段1946tap挤在约24–25.1秒内；原始计数另证同一时刻1271tap，不能由图像重叠掩盖。'),
]


def multiplicity(events):
    counts = Counter(int(e['onset_us']) for e in events)
    taps = sorted(int(e['onset_us']) for e in events if e['type'] == 'tap')
    gaps = [b - a for a, b in zip(taps, taps[1:])]
    positive = sum(g > 0 for g in gaps)
    short = sum(0 < g < 10000 for g in gaps)
    zeros = sum(g == 0 for g in gaps)
    peak = max(counts.values(), default=0)
    return dict(notes=len(events), exact_onset_extras=sum(n-1 for n in counts.values()),
                max_same_onset=peak, max_onset_us=[t for t, n in counts.items() if n == peak],
                tap_intervals=len(gaps), positive_tap_intervals=positive,
                tap_zero_intervals=zeros, positive_tap_under10ms=short,
                original_positive_only_fraction=short/positive if positive else None,
                zero_or_under10ms_fraction=(zeros+short)/len(gaps) if gaps else None)


def main():
    disposition_path = 'reports/d90_candidate_disposition_v1.json'
    if (ROOT / disposition_path).exists():
        raise SystemExit('D90 already closed; do not increment failures or overwrite evidence twice.')
    plan_path = 'bench/rhythm-v2/d90-relative-horizon-v1.json'
    plan = read(plan_path)
    train_path = 'reports/d90_relative_horizon_v1.json'
    train = read(train_path)
    auto_path = 'reports/d90_relative_review_1000_v1.json'
    auto = read(auto_path)
    comparison_path = 'reports/d90_quality_comparison_1000_v1.json'
    comparison = read(comparison_path)
    condition_path = 'reports/d90_relative_condition_response_v1.json'
    condition = read(condition_path)
    audio_path = 'reports/d90_relative_audio_response_v1.json'
    audio = read(audio_path)
    music_path = 'reports/d90_music_comparison_v1.json'
    manifest_path = 'artifacts/d90-relative-review-v1/inspection/manifest.json'
    manifest = read(manifest_path)
    cohort = read('reports/d66_local_cohort_status.json')
    before = dict(objective=4, architecture=3, data_representation=4, training_dynamics=1)
    assert cohort['current_local_experiment_failures'] == before
    assert train['complete'] and train['updates'] == 1000 and train['new_updates'] == 500
    assert sha(plan_path) == train['plan_sha256']
    checkpoint_sha = sha(train['checkpoint'])
    assert checkpoint_sha == train['checkpoint_sha256'] == auto['checkpoint_sha256']
    assert condition['checkpoint_sha256'] == audio['checkpoint_sha256'] == checkpoint_sha
    assert len(auto['records']) == 20 and auto['complete_songs'] == 19
    assert manifest['report_sha256'] == sha(auto_path) and len(manifest['sheets']) == 10
    for sheet in manifest['sheets']:
        assert sha(sheet['file']) == sheet['sha256']
        for source in sheet['sources']:
            assert sha(source['path']) == source['sha256']
    runner_path = 'scripts/run_local_relative_horizon.py'
    runner_sha = sha(runner_path)
    assert runner_sha == plan['code_sha256'][runner_path]
    assert runner_sha == sha(f"{plan['source_snapshot']}/{runner_path}")
    model_path = 'beatfcn/model/full_event_model.py'
    model_sha = sha(model_path)
    assert model_sha == '1dbb1fee55ca6fd815bb6a82f749353ca1a242b75014ccc713ecbb69f15696d0'

    # Frozen outputs only: supplementary diagnostic, no changed gate or inference.
    multiplicity_rows = []
    evidence = {}
    for arm, path in [('control', 'reports/d87_control_review_1000_v1.json'), ('candidate', auto_path)]:
        for row in read(path)['records']:
            source = row['source_generation']
            assert sha(source) == row['source_generation_sha256']
            evidence[source] = sha(source)
            multiplicity_rows.append(dict(arm=arm, chart_id=row['chart_id'], complete=row['complete'],
                                          **multiplicity(read(source)['events'])))
    for row in auto['records']:
        source = str(Path(row['preview']).parent / 'human-timeline.csv')
        evidence[source] = sha(source)
        with (ROOT/source).open() as stream:
            events = list(csv.DictReader(stream))
        multiplicity_rows.append(dict(arm='human', chart_id=row['chart_id'], complete=True,
                                      **multiplicity(events)))
    audit_path = 'reports/d90_multiplicity_diagnostic_v1.json'
    totals = {}
    for arm in ('control', 'candidate', 'human'):
        rows = [r for r in multiplicity_rows if r['arm'] == arm]
        totals[arm] = {k: sum(r[k] for r in rows) for k in (
            'notes', 'exact_onset_extras', 'tap_intervals', 'positive_tap_intervals',
            'tap_zero_intervals', 'positive_tap_under10ms')}
        totals[arm]['max_same_onset'] = max(r['max_same_onset'] for r in rows)
    save(audit_path, dict(id='D90-zero-interval-diagnostic-v1', complete=True,
        optimizer_updates=0, new_cloud_compute_calls=0, rows=multiplicity_rows, totals=totals,
        source_sha256=evidence, quality_passed=False,
        interpretation='Descriptive multiplicity audit of unchanged raw outputs and human charts. '
        'The registered positive-interval metric excludes zero intervals. Normal simultaneous chords '
        'remain legitimate; do not delete them or silently cap outputs. 1271 same-time taps are a '
        'visible collapse hidden by that metric. No retrospective success threshold.'))

    rows = []
    model_types = Counter()
    for row, (chart, whole, detail) in zip(auto['records'], OBSERVATIONS):
        assert row['chart_id'] == f'phira:{chart}'
        model_types.update(row['types']['model'])
        rows.append({**{k: row[k] for k in ('chart_id', 'title', 'stratum', 'notes', 'types', 'full_note_f1', 'complete', 'failure')},
            'full_song_observation': whole, 'active_detail_observation': detail,
            'usable_draft': False, 'preview': row['preview'],
            'review_scope': 'Actual visual review of raw timeline and audio flux; no audition or native playtest'})
    s = comparison['summary']['candidate']
    aggregate = dict(complete_legal_songs=s['complete_legal_songs'], nonempty_songs=s['nonempty_songs'],
        generated_notes=s['predicted_notes'], human_notes=s['human_notes'], model_types=dict(model_types),
        macro_full_note_f1=s['macro_f1'], tap_positive_intervals=s['tap_positive_intervals'],
        tap_under10ms_fraction=s['tap_short_fraction'],
        zero_interval_tap_pairs=totals['candidate']['tap_zero_intervals'],
        max_same_onset=totals['candidate']['max_same_onset'])
    sources = [train_path, plan_path, auto_path, comparison_path, condition_path, audio_path,
               music_path, audit_path, manifest_path, 'reports/d90_relative_stage_1000_v1.json']
    review_path = 'reports/d90_relative_assistant_review_1000_v1.json'
    save(review_path, dict(id='D90-relative-assistant-review-1000-v1', complete=True,
        checkpoint_sha256=checkpoint_sha, source_sha256={p:sha(p) for p in sources},
        artifact_sha256={r['file']:r['sha256'] for r in manifest['sheets']},
        visually_reviewed_songs=20, visually_reviewed_sheets=10, songs=20,
        timeline_visual_review_complete=True, audio_audition_complete=False, native_playtest_complete=False,
        spatial_output='Static auxiliary placement only; no learned spatial output.',
        summary=aggregate, aggregate=aggregate, rows=rows, quality_passed=False, converged_candidates=0,
        optimizer_updates=0, new_cloud_compute_calls=0,
        condition_report=condition_path, audio_report=audio_path,
        automatic_evidence_sha256=sha(auto_path),
        limitations=['Only visual timing/type/flux review; no audio audition or native playtest.',
        'All 20 raw attempts retained, including the incomplete token-capped song.',
        'Reference F1 and estimated onset peaks do not define a unique correct chart.',
        'Density sentinel failed; other applicable conditions and reference continuation remain unestablished.',
        'Positive tap interval score excludes zero gaps; companion multiplicity audit exposes collapse.']))

    archive = 'inputs/code_baselines/d90-closed-v1/scripts/run_local_relative_horizon.py'
    (ROOT/archive).parent.mkdir(parents=True, exist_ok=True)
    if (ROOT/archive).exists():
        assert sha(archive) == runner_sha
    else:
        (ROOT/archive).write_bytes((ROOT/runner_path).read_bytes())
    stub = '''"""Closed D90: matched 1000-update horizon failed whole-song quality."""


def run(*args, **kwargs):
    raise RuntimeError(
        'D90 is closed: 19/20 complete, all-tap collapse and density nonresponse. '
        'No training/resume is permitted through this entry. See '
        'reports/d90_candidate_disposition_v1.json and the archived original '
        'inputs/code_baselines/d90-closed-v1/scripts/run_local_relative_horizon.py.'
    )


if __name__ == '__main__':
    run()
'''
    (ROOT/runner_path).write_text(stub)
    after = {**before, 'training_dynamics': 2}
    save(disposition_path, dict(id='D90-quality-disposition-1000-v1',
        state='Commit or Rollback / failed route removed from active execution',
        dimension='training_dynamics', hypothesis_id='D90-matched-horizon-500-to1000',
        complete=True, quality_passed=False, baseline_promoted=False, converged_candidates=0,
        updates=1000, new_updates=500, inherited_updates=500, checkpoint=train['checkpoint'],
        checkpoint_sha256=checkpoint_sha, plan_sha256=sha(plan_path),
        numerical_gates=comparison['gates'], whole_twenty_song_review=True,
        generation_completed_songs=19, audio_audition_complete=False, native_playtest_complete=False,
        interpretation='500-step all-END was insufficient to predict later activity, but the registered '
        '1000-step result still fails completeness, types, conditions and musical usefulness. '
        'No evidence supporting another horizon extension or architecture promotion.',
        decisive_failures=['19/20 complete: phira:65921 core_token_limit; 1271 taps at one timestamp.',
            'All16616 emitted events are tap; hold/flick/drag absent.',
            f"Macro full-note F1 {s['macro_f1']:.9f} versus control .076097527; gain/floor fail.",
            'Density4/4 identical,0/4 directed,0/8 inband; expansion stopped.',
            'Many regular tap bands and unsupported silent blocks; no usable draft selected.'],
        failure_counters_before=before, failure_counters_after=after,
        counting='One D90 horizon hypothesis increments dynamics1to2 once. D88 failed representation '
        'and older R20/R25 history remain; these are current-phase counters, not all-time totals.',
        next_allowed_route='Switch away from TrainingDynamics. First isolate why free generation '
        'repeats a timing/type pattern despite audio input; use existing raw cases and bounded '
        'zero-update architecture/objective diagnostics before any new one-variable intervention. '
        'No2000step extension, LR tuning, token-cap increase or stacking rejected defaults.',
        rollback=dict(owned_file=runner_path, original_source_sha256=runner_sha,
            archived_source=archive, archive_sha256=sha(archive), disabled_entry_sha256=sha(runner_path),
            production_model_unchanged=sha(model_path)==model_sha,
            scope='Only the owned D90 execution entry disabled; checkpoints, diagnostics and other work preserved.'),
        source_sha256={p:sha(p) for p in sources+[review_path]}, created_at_unix=time.time(),
        new_cloud_compute_calls=0, additional_cloud_cost_usd=0))

    budget_path = 'reports/rhythm_v2_budget.json'
    budget = read(budget_path)
    budget_before = deepcopy(budget)
    entry = budget['local_cpu_diagnostics']['d90_matched_horizon']
    entry.update(status='completed_quality_failed', observed_at_unix=time.time(),
        reported_updates=1000, reported_additional_updates=500, inherited_updates=500,
        new_cloud_compute_calls=0, additional_usd=0, disposition=disposition_path,
        review=review_path, train_s=train['train_s'], preparation_s=train['preparation_s'],
        evaluation_s=train['evaluation_s'], complete_songs=19, attempted_songs=20,
        scope='D90 additional500updates and recordedlocalgeneration/audits only; D88parent and allfuture reserves preserved.')
    budget_check = deepcopy(budget)
    budget_check['local_cpu_diagnostics']['d90_matched_horizon'] = budget_before['local_cpu_diagnostics']['d90_matched_horizon']
    assert budget_check == budget_before
    save(budget_path, budget)

    cohort.update(phase='d90_closed_quality_failed', active_stage=None,
        active_auxiliary_stages=[], active_aux_stages=[], quality_passed=False, converged_candidates=0,
        current_candidate_effective_updates=1000, candidate_disposition=disposition_path,
        current_local_experiment_failures=after, whole_song_review=review_path,
        current_review='reports/d90_paired1000_review_v1.md', updated_at_unix=time.time(),
        pending=['No live training. D90 failed1000quality,19/20complete; no automatic resume.',
            'Next switch dimension: investigate repetitive free-generation mechanism before a new intervention.',
            'All-condition response, reference continuation and two converged candidates remain unmet.'],
        next_action=read(disposition_path)['next_allowed_route'],
        current_running_verification='reports/d90_closeout_verification_v1.json')
    save('reports/d66_local_cohort_status.json', cohort)
    target = read('reports/training_monitor_target.json')
    target['plan_file'] = plan_path
    target['current_work_label'] = 'D90 1000步已复盘失败：19/20完成，全部tap、密度0响应，同刻1271tap暴露指标盲区。无训练，0收敛。'
    target['local_cohort']['review'] = 'reports/d90_paired1000_review_v1.md'
    for key in ('local_song_training', 'comparison_candidate_training'):
        item = target[key]
        item.update(pid=None, session_id=None, status='completed_quality_failed', assistant_review=review_path)
        for name, path in [('review', auto_path), ('timeline_review', review_path),
                           ('condition_audit', condition_path), ('audio_audit', audio_path)]:
            item[name] = dict(report=Path(path).name, sha256=sha(path))
    target['quality_disposition'] = dict(report=Path(disposition_path).name, sha256=sha(disposition_path))
    save('reports/training_monitor_target.json', target)
    print(json.dumps(dict(complete=True, disposition=disposition_path, review=review_path,
        aggregate=aggregate, failure_counters=after, budget_other_fields_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()

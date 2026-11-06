import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('train_monitor', Path(__file__).resolve().parents[1] / 'scripts/train_monitor.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = module.Monitor()
        self.monitor.ingest('run', {'event': 'warm_start', 'step': 3500, 'ts': 10})

    def test_external_baseline_requires_matching_review_and_never_promotes_training(self):
        from hashlib import sha256
        state = self.monitor.snapshot('run', now=20)
        original_metrics = dict(state['metrics'])
        target = {'external_baseline': dict(review='review.json', benchmark='benchmark.json', scope_label='one song')}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            path = reports / 'benchmark.json'
            path.write_text(module.json.dumps({'charts': {'external': {'notes': 394}}}))
            review = dict(complete=True, screen_passed=True, hold_control_observation={'failure': True},
                          new_cloud_spend_usd=0, source_sha256={'reports/benchmark.json': sha256(path.read_bytes()).hexdigest()})
            (reports / 'review.json').write_text(module.json.dumps(review))
            module.external_baseline_status(state, target, reports)
            self.assertEqual(state['external_baseline']['status'], 'complete')
            self.assertFalse(state['external_baseline']['model_promotion'])
            self.assertEqual(state['external_baseline']['optimizer_updates'], 0)
            self.assertEqual(state['metrics'], original_metrics)
            path.write_text('{}')
            module.external_baseline_status(state, target, reports)
            self.assertEqual(state['external_baseline']['status'], 'evidence_incomplete')
            self.assertNotIn('charts', state['external_baseline'])

    def test_audio_coordinates_do_not_promote_model_or_hide_stale_evidence(self):
        from hashlib import sha256
        state = self.monitor.snapshot('run', now=20)
        original_metrics = dict(state['metrics'])
        target = {'audio_coordinate_diagnostic': dict(review='review.json', audit='audit.json')}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            path = reports / 'audit.json'
            path.write_text('{"complete":true}')
            review = dict(complete=True, charts=12, coordinate_screen_passed=False, new_cloud_spend_usd=0,
                monitor_summary={'mania_4k': dict(heads_20ms=.95, hold_ends_20ms=.97,
                    drag_burst_20ms=None, heads_phase_excess_20ms=.30)},
                source_sha256={'reports/audit.json': sha256(path.read_bytes()).hexdigest()})
            (reports / 'review.json').write_text(module.json.dumps(review))
            module.audio_coordinate_status(state, target, reports)
            self.assertEqual(state['audio_coordinate_diagnostic']['status'], 'complete')
            self.assertFalse(state['audio_coordinate_diagnostic']['model_promotion'])
            self.assertEqual(state['audio_coordinate_diagnostic']['optimizer_updates'], 0)
            self.assertEqual(state['metrics'], original_metrics)
            rendered = module.render_text(state)
            self.assertIn('不是生成质量提升', rendered)
            self.assertIn('95.0%', rendered)
            path.write_text('{}')
            module.audio_coordinate_status(state, target, reports)
            self.assertEqual(state['audio_coordinate_diagnostic']['status'], 'evidence_incomplete')
            self.assertNotIn('summary', state['audio_coordinate_diagnostic'])

    def test_data_expansion_does_not_claim_training_or_model_improvement(self):
        state = self.monitor.snapshot('run', now=20)
        metrics = dict(state['metrics'])
        target = {'local_data': dict(transfer='transfer.json', processing='processing.json', audit='audit.json',
                   pid=123, script='scripts/expand.py', plan='bench/expansion.json', phase='download', expected_archives=128)}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'transfer.json').write_text(module.json.dumps({'complete':False,
                'records':[{'status':'downloaded'}], 'transferred_bytes':1024}))
            module.local_data_status(state, target, reports,
                lambda _: 'python scripts/expand.py --plan bench/expansion.json --phase download')
            self.assertEqual(state['local_data']['status'], 'downloading')
            self.assertFalse(state['local_data']['model_promotion'])
            self.assertEqual(state['metrics'], metrics)
            text = module.render_text(state)
            self.assertIn('1/128 个谱包', text)
            self.assertIn('未启动新训练', text)
            module.local_data_status(state, target, reports, lambda _: '')
            self.assertEqual(state['local_data']['status'], 'process_not_confirmed')

    def test_completed_rejected_generation_is_not_rendered_as_missing(self):
        state = self.monitor.snapshot('run', now=20)
        row = dict(macro_full_note_f1=.18, macro_timing_point_f1=.62,
                   first_onset_matches=2, passed_fields=0, empty_windows=0)
        state.update(round_label='R07B', r07={'review': {'results': {'candidate': row},
            'comparison': {'short_window_gate_passed': False}}})
        text = module.render_text(state)
        self.assertIn('R07B 不通过，不晋升', text)
        self.assertIn('条件 0/13', text)
        self.assertIn('整曲制谱实用性尚未验收', text)
        self.assertNotIn('尚无本轮生成评测结果', text)
        self.assertNotIn('条件响应、音频/参考交换和人评校准：尚未完成', text)

    def test_local_fit_requires_live_process_and_preserves_formal_metrics(self):
        state = self.monitor.snapshot('run', now=20)
        original_metrics = dict(state['metrics'])
        target = {'local_probe': {'result':'fit.json', 'resources':'resources.json',
                  'steps':100, 'script':'scripts/fit.py', 'plan':'bench/fit.json'}}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'fit.json').write_text(module.json.dumps({'optimizer_updates':5,
                'history':[{'step':5,'loss':.5}], 'model_promotion':False}))
            (reports/'resources.json').write_text(module.json.dumps({'pid':123,'complete':False}))
            module.local_probe_status(state,target,reports,lambda _: 'python scripts/fit.py --plan bench/fit.json')
            self.assertEqual(state['local_probe']['status'],'running')
            self.assertEqual(state['metrics'],original_metrics)
            self.assertIn('5/100 次更新',module.render_text(state))
            self.assertIn('不代表当前 CPU 探针',module.render_text(state))
            module.local_probe_status(state,target,reports,lambda _: '')
            self.assertEqual(state['local_probe']['status'],'process_not_confirmed')

    def test_local_fit_passing_does_not_promote_model(self):
        state = self.monitor.snapshot('run', now=20)
        state['status']='experiment_rejected'
        target = {'local_probe': {'result':'fit.json', 'resources':'resources.json',
                  'steps':100, 'script':'scripts/fit.py', 'plan':'bench/fit.json'}}
        with tempfile.TemporaryDirectory() as directory:
            reports=Path(directory)
            (reports/'fit.json').write_text(module.json.dumps({'complete':True,'probe_passed':True,'optimizer_updates':100}))
            (reports/'resources.json').write_text(module.json.dumps({'complete':True,'exit_code':0}))
            module.local_probe_status(state,target,reports,lambda _: '')
        self.assertEqual(state['local_probe']['status'],'passed')
        self.assertFalse(state['local_probe']['model_promotion'])
        self.assertEqual(state['status'],'experiment_rejected')

    def test_local_generation_summary_keeps_missing_fourth_round_and_rare_types(self):
        state = self.monitor.snapshot('run', now=20)
        target = {'local_probe': dict(result='fit.json', resources='resources.json',
                  steps=100, script='scripts/fit.py', plan='bench/fit.json')}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            summary = {
                'mania_4k': dict(first_f1=.21, final_f1=.22, all_four_rounds=True,
                    per_type_scope='fourth_round', per_type={'hold': dict(pred=0, true=21)}),
                'phira': dict(first_f1=.13, last_available_f1=.99, final_f1=.99,
                    all_four_rounds=False, per_type_scope='last_available_diagnostic_only')}
            (reports/'fit.json').write_text(module.json.dumps(dict(complete=True,
                probe_passed=False, optimizer_updates=100, summary=summary)))
            (reports/'resources.json').write_text(module.json.dumps(dict(complete=True, exit_code=1)))
            module.local_probe_status(state, target, reports, lambda _: '')
        text = module.render_text(state)
        self.assertIn('最近正式训练记录（历史）', text)
        self.assertIn('训练片段，非整曲 benchmark', text)
        self.assertIn('| mania_4k | 0.21 | 0.22 | hold 0/21 |', text)
        self.assertIn('| phira | 0.13 | 未完成 | 未完成 |', text)
        self.assertNotIn('0.99', text)
        self.assertFalse(state['local_probe']['model_promotion'])

    def test_memorised_generation_success_does_not_hide_condition_failure(self):
        state = self.monitor.snapshot('run', now=20)
        target = {'local_probe': {'result':'fit.json', 'resources':'resources.json',
                  'steps':100, 'script':'scripts/fit.py', 'plan':'bench/fit.json'}}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'fit.json').write_text(module.json.dumps({
                'complete':True,'probe_passed':False,'optimizer_updates':100,
                'history':[{'step':100,'loss':.0011747876}],
                'binding_gate':{'passed':False,'passed_fields':[],'total_fields':13},
                'generation_gate':{'passed':True,'cores':[{'chart_id':'test:1','full_note_f1':1.}]}}))
            (reports/'resources.json').write_text(module.json.dumps({'complete':True,'exit_code':1}))
            module.local_probe_status(state,target,reports,lambda _: '')
        text = module.render_text(state)
        self.assertIn('本机固定批探针未通过',text)
        self.assertIn('0.001175',text)
        self.assertIn('0/13 达标',text)
        self.assertIn('test:1=1.0000',text)
        self.assertIn('不是生成控制通过数',text)
        self.assertFalse(state['local_probe']['model_promotion'])

    def test_active_diagnostic_retains_model_metrics_and_uses_own_stage_status(self):
        state=self.monitor.snapshot('run',now=20)
        state.update(status='experiment_rejected',r04={'review':{'candidate_f1':.06}})
        for stage_status,expected in [('running','diagnosing'),('complete','review_required'),
                                      ('observation_error','evaluation_observation_error')]:
            module.active_diagnostic_status(state,{'paid_stages_dispatched':[
                {'name':'d08','status':stage_status}]},{'active_diagnostic':'d08'})
            self.assertEqual(state['status'],expected)
            self.assertEqual(state['r04']['review']['candidate_f1'],.06)
            self.assertEqual(state['current_work']['round'],'d08')

    def test_single_field_pair_probe_reports_its_scope_and_generated_swap(self):
        state=self.monitor.snapshot('run',now=20)
        target={'local_probe':dict(result='fit.json',resources='resources.json',steps=100,
            script='scripts/fit.py',plan='bench/fit.json',scope_label='四个固定片段、两项单字段条件')}
        with tempfile.TemporaryDirectory() as directory:
            reports=Path(directory)
            (reports/'fit.json').write_text(module.json.dumps(dict(complete=True,probe_passed=False,
                optimizer_updates=100,binding_gate=dict(kind='generation_swap',passed_fields=['phira:drag_ratio'],total_fields=2))))
            (reports/'resources.json').write_text(module.json.dumps(dict(complete=True,exit_code=1)))
            module.local_probe_status(state,target,reports,lambda _: '')
        text=module.render_text(state)
        self.assertIn('四个固定片段、两项单字段条件',text)
        self.assertIn('记忆配对条件移除/交换：**1/2 达标**',text)
        self.assertNotIn('固定批条件评分间隔',text)
        self.assertNotIn('两个固定训练片段',text)

    def test_r05_preflight_keeps_historical_model_separate(self):
        state=self.monitor.snapshot('run',now=20)
        with tempfile.TemporaryDirectory() as directory:
            module.r05_campaign_status(state,{'paid_stages_dispatched':[
                {'name':'r05-direct-condition-preflight','status':'running'}]},
                {'phase':'preflight'},Path(directory))
        text=module.render_text(state)
        self.assertEqual(state['status'],'probing')
        self.assertFalse(state['training_complete'])
        self.assertIn('R05 尚无训练后生成 benchmark',text)
        self.assertIn('不能当作 R05 指标',text)
        self.assertIn('R05 尚未完成正式训练',text)

    def test_r06_preflight_does_not_claim_r05_metrics_as_new_quality(self):
        state=self.monitor.snapshot('run',now=20)
        with tempfile.TemporaryDirectory() as directory:
            module.r06_campaign_status(state,{'paid_stages_dispatched':[
                {'name':'r06-ordered-time-preflight','status':'running'}]},
                {'phase':'preflight'},Path(directory))
        text=module.render_text(state)
        self.assertEqual(state['status'],'probing')
        self.assertEqual(state['round_label'],'R06')
        self.assertIn('R06 尚无训练后生成 benchmark',text)
        self.assertIn('不能当作 R06 指标',text)
        self.assertIn('R06 尚未完成正式训练',text)

    def test_r06_review_resolves_only_the_pending_review_alert(self):
        state=self.monitor.snapshot('run',now=20)
        state['alerts']=['已到 1000 次有效更新复盘点：先审查 benchmark 再继续', '预算需要核对']
        with tempfile.TemporaryDirectory() as directory:
            reports=Path(directory)
            (reports/'r06_review.json').write_text(module.json.dumps({
                'comparison':{'short_window_gate_passed':False}}))
            (reports/'r06_rollback.json').write_text('{"source_restored_exactly": true}')
            module.r06_campaign_status(state,{'paid_stages_dispatched':[
                {'name':'r06-ordered-time-evaluate','status':'complete'}]},
                {'phase':'evaluate'},reports)
        self.assertEqual(state['status'],'experiment_rejected')
        self.assertEqual(state['alerts'],['预算需要核对'])
        self.assertTrue(state['r06']['rollback']['source_restored_exactly'])

    def test_r07_preflight_does_not_claim_r06_metrics_as_new_quality(self):
        state=self.monitor.snapshot('run',now=20)
        with tempfile.TemporaryDirectory() as directory:
            module.r07_campaign_status(state,{'paid_stages_dispatched':[
                {'name':'r07-blank-context-preflight','status':'running'}]},
                {'phase':'preflight'},Path(directory))
        text=module.render_text(state)
        self.assertEqual(state['status'],'probing')
        self.assertEqual(state['round_label'],'R07')
        self.assertIn('R07 尚无训练后生成 benchmark',text)
        self.assertIn('不能当作 R07 指标',text)
        self.assertIn('R07 尚未完成正式训练',text)

    def test_r07_review_resolves_only_the_pending_review_alert(self):
        state=self.monitor.snapshot('run',now=20)
        state['alerts']=['已到 1000 次有效更新复盘点：先审查 benchmark 再继续', '预算需要核对']
        with tempfile.TemporaryDirectory() as directory:
            reports=Path(directory)
            (reports/'r07_review.json').write_text(module.json.dumps({
                'comparison':{'short_window_gate_passed':False}}))
            (reports/'r07_rollback.json').write_text('{"source_restored_exactly": true}')
            module.r07_campaign_status(state,{'paid_stages_dispatched':[
                {'name':'r07-blank-context-evaluate','status':'complete'}]},
                {'phase':'evaluate'},reports)
        self.assertEqual(state['status'],'experiment_rejected')
        self.assertEqual(state['alerts'],['预算需要核对'])
        self.assertTrue(state['r07']['rollback']['source_restored_exactly'])

    def test_r07b_does_not_inherit_interrupted_r07_state(self):
        state = self.monitor.snapshot('run', now=20)
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'r07_infrastructure_review.json').write_text(module.json.dumps({
                'status': 'infrastructure_interrupted', 'effective_updates': 450}))
            module.r07_campaign_status(state, {'paid_stages_dispatched': [
                {'name': 'r07b-blank-context-preflight', 'status': 'running'}]},
                {'phase': 'preflight', 'report_prefix': 'r07b'}, reports)
        self.assertEqual(state['status'], 'probing')
        self.assertEqual(state['round_label'], 'R07B')
        self.assertIn('R07B 尚无训练后生成 benchmark', module.render_text(state))
        self.assertNotIn('450/1000', module.render_text(state))

    def test_r07_infrastructure_stop_is_not_a_quality_verdict(self):
        state = self.monitor.snapshot('run', now=20)
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'r07_infrastructure_review.json').write_text(module.json.dumps({
                'status': 'infrastructure_interrupted', 'effective_updates': 450}))
            module.r07_campaign_status(state, {'paid_stages_dispatched': [
                {'name': 'r07-blank-context-train', 'status': 'failed', 'error': 'GPU guard'}]},
                {'phase': 'train'}, reports)
        self.assertEqual(state['status'], 'infrastructure_interrupted')
        self.assertFalse(state['training_complete'])
        self.assertIn('450/1000', module.render_text(state))
        self.assertIn('不能裁决模型质量', module.render_text(state))

    def test_r03_does_not_inherit_old_cancelled_evaluation_or_claim_candidate_metrics(self):
        state = self.monitor.snapshot('run', now=20)
        target = dict(phase='preflight', console_file='probe.log')
        budget = {'paid_stages_dispatched': [{'name': 'paired_evaluation', 'status': 'stopped_by_user'},
                                            {'name': 'r03-window-preflight', 'status': 'running'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'r03_window_review.json').write_text(module.json.dumps({'results': {'baseline': {
                'passed_fields': 0, 'macro_full_note_f1': .2, 'macro_timing_point_f1': .4,
                'macro_reference_edit_saved_fraction': -.4}}}))
            module.r03_campaign_status(state, budget, target, path)
        self.assertEqual(state['status'], 'probing')
        text = module.render_text(state)
        self.assertIn('baseline | 0/13 | 20.00%', text)
        self.assertIn('不能把基线当新模型指标', text)
        self.assertNotIn('评测已按要求停止', text)

    def test_new_campaign_displays_paired_runs_budget_and_historical_control_scope(self):
        state = self.monitor.snapshot('run', now=20)
        state['paired_runs'] = [self.monitor.snapshot('unstarted', now=20)]
        state['rhythm_budget'] = {'approved_total_usd': 15, 'historical_accounted_upper_usd': 3.75,
                                  'campaign_committed_upper_usd': 10.158}
        state['condition_diagnostic'] = {'controls': {'4500/phira': {
            'achieved_by_band': {'low': {'mean': 3.03}, 'mid': {'mean': 3.03}, 'high': {'mean': 3.04}},
            'positive_monotonic_songs': 1, 'paired_songs': 8}}}
        text = module.render_text(state)
        self.assertIn('$15.00', text)
        self.assertIn('4500/phira | 3.03 | 3.03 | 3.04 | 1/8', text)
        self.assertIn('上述结果不能当作新模型指标', text)

    def test_r03_finished_evaluation_marks_gpu_metrics_historical(self):
        state = self.monitor.snapshot('run', now=1000)
        target = dict(phase='evaluate', console_file='eval.log')
        budget = {'paid_stages_dispatched': [{'name': 'r03-window-evaluate', 'status': 'complete'}]}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / 'r03_window_train.json').write_text('{"complete": true}')
            module.r03_campaign_status(state, budget, target, reports)
        text = module.render_text(state)
        self.assertIn('不代表当前评测资源占用', text)
        self.assertNotIn('生成 benchmark 中', text)
        self.assertEqual(state['current_work']['status'], 'awaiting_review')

    def test_r04_preflight_failure_is_not_running_or_completed_training(self):
        state = self.monitor.snapshot('run', now=1000)
        budget = {'paid_stages_dispatched': [{'name': 'r04-initial-preflight', 'status': 'failed',
                                              'error': 'legacy gradient changed: band'}]}
        with tempfile.TemporaryDirectory() as directory:
            module.r04_campaign_status(state, budget, {'phase': 'preflight'}, Path(directory))
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['current_work']['status'], 'needs_diagnosis')
        self.assertFalse(state['training_complete'])
        text = module.render_text(state)
        self.assertIn('legacy gradient changed: band', text)
        self.assertIn('R04 尚未完成正式训练', text)
        self.assertNotIn('本轮训练已完成', text)

    def test_r04_new_diagnostic_stage_does_not_inherit_failed_attempt(self):
        state = self.monitor.snapshot('run', now=1000)
        budget = {'paid_stages_dispatched': [
            {'name': 'r04-initial-preflight', 'status': 'failed'},
            {'name': 'r04-initial-preflight-deterministic', 'status': 'running'}]}
        with tempfile.TemporaryDirectory() as directory:
            module.r04_campaign_status(state, budget,
                {'phase': 'preflight', 'stage_name': 'r04-initial-preflight-deterministic'}, Path(directory))
        self.assertEqual(state['status'], 'probing')
        self.assertFalse(state['training_complete'])

    def test_r04_final_event_metrics_are_visible_even_when_candidate_is_rejected(self):
        state = self.monitor.snapshot('run', now=1000)
        state['r04'] = {'review': {'results': {'candidate': {
            'macro_full_note_f1': .2, 'macro_timing_point_f1': .4, 'passed_fields': 0,
            'first_onset_matches': 2, 'empty_windows': 1, 'aggregate_count_ratio': 1.2}},
            'comparison': {'short_window_gate_passed': False}}}
        text = module.render_text(state)
        self.assertIn('完整音符 F1 20.00%', text)
        self.assertIn('首起音 ±20ms：2/8', text)
        self.assertIn('不通过，不晋升该模型', text)
        self.assertNotIn('R04 尚无训练后生成 benchmark', text)

    def test_completed_r04_rejection_does_not_keep_waiting_for_review(self):
        state = self.monitor.snapshot('run', now=1000)
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports/'r04_review.json').write_text('{"comparison": {"short_window_gate_passed": false}}')
            module.r04_campaign_status(state,
                {'paid_stages_dispatched': [{'name': 'r04-initial-evaluate', 'status': 'complete'}]},
                {'phase': 'evaluate'}, reports)
        self.assertEqual(state['status'], 'experiment_rejected')
        self.assertEqual(state['current_work']['status'], 'rejected')

    def test_condition_evaluation_keeps_failure_denominator_and_marks_training_metrics_historical(self):
        state = self.monitor.snapshot('run', now=1000)
        state['paired_runs'] = [self.monitor.snapshot('run', now=1000)]
        budget = {'paid_stages_dispatched': [
            {'name': 'matched-training-light-trace-resume-v1', 'result': {'complete': True}},
            {'name': 'paired_evaluation', 'status': 'incomplete', 'result': {
                'complete': False, 'coverage': {'candidate': {'requested': 264, 'complete_legal': 260}}}}]}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / 'condition_v1_benchmark_summary.json').write_text(module.json.dumps({
                'complete': False, 'expected': 792, 'received': 792, 'cells': {}, 'controls': {}}))
            module.condition_campaign_status(state, budget, reports, now=1000)
        self.assertEqual(state['status'], 'evaluation_incomplete')
        self.assertEqual(state['paired_runs'][0]['status'], 'review_required')
        text = module.render_text(state)
        self.assertIn('合法完成 260/264', text)
        self.assertIn('不能作完整验收', text)
        self.assertIn('不代表当前评测资源占用', text)
        self.assertNotIn('尚无本轮生成评测结果', text)

    def test_stale_evaluation_and_observation_errors_do_not_claim_remote_job_stopped(self):
        state = self.monitor.snapshot('run', now=1000)
        stage = {'name': 'paired_evaluation', 'status': 'running', 'reserved_at': 10}
        budget = {'paid_stages_dispatched': [stage]}
        with tempfile.TemporaryDirectory() as directory:
            module.condition_campaign_status(state, budget, Path(directory), now=1000)
            self.assertEqual(state['status'], 'evaluation_log_stale')
            stage['status'] = 'observation_error'
            module.condition_campaign_status(state, budget, Path(directory), now=1000)
        self.assertEqual(state['status'], 'evaluation_observation_error')
        self.assertIn('需核实原任务', module.render_text(state))

    def test_user_stopped_partial_evaluation_does_not_label_unrecorded_as_failure(self):
        state = self.monitor.snapshot('run', now=1000)
        budget = {'paid_stages_dispatched': [{'name': 'paired_evaluation', 'status': 'stopped_by_user'}]}
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / 'condition_v1_benchmark_summary.json').write_text(module.json.dumps({
                'complete': False, 'expected': 792, 'received': 234, 'failed_recorded': 0,
                'not_recorded': 558, 'cells': {}, 'controls': {}}))
            module.condition_campaign_status(state, budget, reports, now=1000)
        self.assertEqual(state['status'], 'evaluation_stopped_by_user')
        text = module.render_text(state)
        self.assertIn('已记录失败 0 条，未形成记录 558 条', text)
        self.assertNotIn('GPU 空闲', text)

    def test_failed_update_is_not_counted_and_failure_does_not_become_stale(self):
        self.monitor.ingest('run', {'event': 'nonfinite_gradient', 'attempted_step': 3527,
                                    'completed_updates': 26, 'ts': 20})
        state = self.monitor.snapshot('run', now=1000)
        self.assertEqual((state['step'], state['effective_updates'], state['status']), (3526, 26, 'failed'))

    def test_legacy_truncated_failure_remains_actionable(self):
        parsed = module.console_record('run {"event": "nonfinite_gradient", "attempted_step": 3527, "completed_updates": 26, "bad_parameters": {"song', 20)
        self.monitor.ingest(*parsed)
        state = self.monitor.snapshot('run', now=21)
        self.assertEqual((state['status'], state['step']), ('failed', 3526))

    def test_missing_metrics_are_unknown_and_old_logs_are_stale(self):
        self.monitor.ingest('run', {'event': 'train', 'step': 3525, 'ts': 20})
        state = self.monitor.snapshot('run', now=1000)
        self.assertIsNone(state['metrics']['loss'])
        self.assertIsNone(state['metrics']['gpu_util_percent'])
        self.assertEqual(state['status'], 'log_stale')

    def test_nonfinite_loss_alert_and_strict_json_snapshot(self):
        self.monitor.ingest('run', {'event': 'train', 'step': 3525, 'ts': 20,
                                    'fields': {'offset': {'loss': float('nan')}}})
        state = self.monitor.snapshot('run', now=21)
        self.assertEqual(state['status'], 'failed')
        self.assertIsNone(state['metrics']['loss'])
        module.json.dumps(state, allow_nan=False)

    def test_review_boundary_duplicate_and_ledger_failure(self):
        record = {'event': 'train', 'step': 4500, 'ts': 20}
        self.monitor.ingest('run', record)
        self.monitor.ingest('run', record)
        state = self.monitor.snapshot('run', now=21)
        self.assertEqual(state['effective_updates'], 1000)
        self.assertEqual(len(state['alerts']), 1)
        ledger = {'rounds': {'r': {'plan': {'training': {'run_id': 'run'}},
                                  'status': 'needs_diagnosis', 'started_at': 10, 'error': 'failed'}}}
        self.assertEqual(self.monitor.snapshot('run', now=21, ledger=ledger)['status'], 'failed')

    def test_resume_keeps_original_review_boundary(self):
        self.monitor.ingest('resume', {'event': 'warm_start', 'step': 3900, 'round_start_step': 3500, 'ts': 10})
        self.monitor.ingest('resume', {'event': 'train', 'step': 4500, 'ts': 20})
        self.assertEqual(self.monitor.snapshot('resume', now=21)['effective_updates'], 1000)

    def test_confirmed_resume_clears_current_guard_alert_but_retains_failure_history(self):
        self.monitor.ingest('run', {'event': 'aborted', 'step': 3650, 'reason': 'low util', 'ts': 20})
        self.monitor.ingest('run', {'event': 'resume', 'step': 3650, 'round_start_step': 3500, 'ts': 30})
        self.monitor.ingest('run', {'event': 'aborted', 'step': 3650, 'reason': 'old cloud log', 'ts': 21})
        state = self.monitor.snapshot('run', now=31)
        self.assertEqual(state['status'], 'starting')
        self.assertEqual(state['effective_updates'], 150)
        self.assertEqual(state['alerts'], [])
        self.assertEqual(state['previous_failure']['reason'], 'low util')

    def test_guard_is_reported_directly_from_console_before_cloud_commit(self):
        self.monitor.ingest('run', {'event': 'aborted', 'step': 3900, 'reason': 'GPU util below guard', 'ts': 20})
        s = self.monitor.snapshot('run', now=21)
        self.assertEqual((s['status'], s['effective_updates']), ('stopped_by_guard', 400))
        self.assertIn('GPU util below guard', s['alerts'][0])

    def test_latest_validation_does_not_regress_when_old_cloud_log_arrives(self):
        for step, ts in [(4000, 40), (3750, 30), (3900, 35)]:
            self.monitor.ingest('run', {'event': 'val', 'step': step, 'ts': ts, 'total': 10})
        state = self.monitor.snapshot('run', now=41)
        self.assertEqual(state['model_progress']['latest_validation']['step'], 4000)

    def test_paired_arms_share_only_initial_validation_not_their_learning_curves(self):
        for run in ('run', 'other'):
            for event in ({'event': 'data', 'prep': 'fixed'},
                          {'event': 'data_identity', 'rows_sha256': 'rows'},
                          {'event': 'start', 'model': {'d': 512}},
                          {'event': 'warm_start', 'step': 3500}):
                self.monitor.ingest(run, dict(event, ts=11))
        self.monitor.ingest('other', {'event': 'round_baseline', 'step': 3500, 'ts': 12, 'total': 10})
        self.monitor.ingest('run', {'event': 'val', 'step': 4000, 'ts': 20, 'total': 9})
        self.monitor.ingest('other', {'event': 'val', 'step': 4000, 'ts': 21, 'total': 8})
        progress = self.monitor.snapshot('run', now=22)['model_progress']
        self.assertEqual(progress['baseline']['total'], 10)
        self.assertEqual([(r['step'], r['total']) for r in progress['history']], [(3500, 10), (4000, 9)])

    def test_resume_compares_same_fixed_validation_and_reports_regression(self):
        for run in ('run', 'next'):
            for event in ({'event': 'data', 'prep': 'fixed'},
                          {'event': 'data_identity', 'rows_sha256': 'rows'},
                          {'event': 'start', 'model': {'offset_sigma_ms': 1.5}}):
                self.monitor.ingest(run, dict(event, ts=1))
        self.monitor.ingest('run', {'event': 'round_baseline', 'step': 3500, 'ts': 11, 'total': 11,
                                    'lane_choice': {'acc': .67, 'n': 100}})
        self.monitor.ingest('next', {'event': 'warm_start', 'step': 3900, 'round_start_step': 3500, 'ts': 20})
        self.monitor.ingest('next', {'event': 'val', 'step': 4000, 'ts': 30, 'total': 10,
                                     'lane_choice': {'acc': .66, 'n': 100}, 'offset': {'acc': 0, 'n': 0}})
        state = self.monitor.snapshot('next', now=31)
        self.assertEqual(state['model_progress']['baseline']['step'], 3500)
        text = module.render_text(state)
        self.assertIn('66.00%', text)
        self.assertIn('-1.00 个百分点', text)
        self.assertIn('毫秒偏移 | 未测 | 未测', text)
        self.assertIn('尚无本轮生成评测结果', text)
        self.monitor.runs['next']['validation_prep'] = 'different'
        self.assertIsNone(self.monitor.snapshot('next', now=31)['model_progress']['baseline'])

    def test_partial_benchmark_keeps_failed_requests_in_denominator(self):
        benchmark = {'checkpoints': ['last.pt'], 'expected_per_checkpoint': 24,
                     'records': {'last.pt': [{'complete': True, 'legality': []}, {'complete': False}]}}
        text = module.render_text(self.monitor.snapshot('run', now=20, benchmark=benchmark))
        self.assertIn('完成 1/24，合法完成 1/24', text)
        self.assertIn('部分完成，不能作完整验收', text)

    def test_generation_failure_does_not_erase_completed_training_round(self):
        ledger = {'approved_usd': 10, 'stages': {}, 'rounds': {'round': {
            'plan': {'training': {'run_id': 'run'}}, 'status': 'needs_diagnosis',
            'training_result': {'round_effective_updates': 1000, 'aborted': None},
            'error': 'RuntimeError: negative dimension'}}}
        state = self.monitor.snapshot('run', now=20, ledger=ledger)
        self.assertEqual(state['status'], 'evaluation_failed')
        state['user_paused'] = True
        text = module.render_text(state)
        self.assertIn('训练轮次完成，生成评测失败', text)
        self.assertIn('已按用户要求暂停', text)
        self.assertIn('negative dimension', text)

    def test_review_keeps_missing_and_failed_requests_visible_while_training_paused(self):
        state = self.monitor.snapshot('run', now=20)
        state.update(user_paused=True, evaluation_review={
            'coverage': {'3500': {'attempted': 12, 'expected': 120, 'completed': 10, 'missing': 108}},
            'groups': {'3500/baseline/phira/reference': {'completed': 2, 'note_f1_20ms': {'mean_complete_only': .25}}}})
        text = module.render_text(state)
        self.assertIn('12/120 | 10 | 108', text)
        self.assertIn('评测进行中', text)
        self.assertIn('25.00%', text)
        self.assertIn('当前另行授权的工作仅为既有检查点评测', text)
        self.assertNotIn('暂停后续训练和付费评测', text)


if __name__ == '__main__':
    unittest.main()

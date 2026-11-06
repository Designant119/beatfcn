from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train_monitor as monitor


class LocalSongMonitorTest(unittest.TestCase):
    def test_registered_early_stop_does_not_claim_whole_song_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); reports=root/'reports'; reports.mkdir()
            (root/'plan.json').write_text('{}'); (root/'code.py').write_text('code')
            result=dict(plan_sha256=sha256((root/'plan.json').read_bytes()).hexdigest(),
                source_sha256={'code.py':sha256((root/'code.py').read_bytes()).hexdigest()},
                updates=500,curve=[],parameters=2370051,valid_tokens=100,requests=100,
                train_s=2.,preparation_s=1.,evaluation_s=1.,phase='explicit_driver_probe_yield',
                checkpoint_sha256='current', complete=False)
            (reports/'result.json').write_text(json.dumps(result))
            config=dict(result='result.json',plan='plan.json',pid=None,generation='missing.json')
            target=dict(local_song_training=config)
            disposition=dict(complete=True,quality_passed=False,early_stop_triggered=True,
                checkpoint_sha256='old',plan_sha256=result['plan_sha256'],updates=500,
                stop_reason='250/500活动片段全空',stage_summary=[{'step':250},{'step':500}])
            path=reports/'disposition.json'
            for checkpoint, expected in [('old','yielded'),('current','early_stopped_quality_failed')]:
                disposition['checkpoint_sha256']=checkpoint;path.write_text(json.dumps(disposition))
                config['early_stop_disposition']=dict(report=path.name,sha256=sha256(path.read_bytes()).hexdigest())
                snapshot=monitor.Monitor().snapshot('new',now=20)
                monitor.local_song_training_status(snapshot,target,reports,lambda pid:'')
                item=snapshot['local_song_training']
                self.assertEqual(item['status'],expected)
                self.assertNotIn('review_summary',item)
                self.assertNotIn('timeline_review',item)
                self.assertFalse(snapshot['training_complete'])
                self.assertFalse(item['model_promotion'])
            self.assertIn('未进行本候选的1000步20首整曲评测',monitor.render_text(snapshot))
            path.write_text('{}')  # A stale hash cannot authorize terminal status.
            monitor.local_song_training_status(snapshot,target,reports,lambda pid:'')
            self.assertEqual(snapshot['local_song_training']['status'],'yielded')

    def test_new_baseline_never_imports_old_quality_or_unreviewed_generation_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); reports = root/'reports'; reports.mkdir()
            (root/'plan.json').write_text('{}'); (root/'code.py').write_text('code')
            result = dict(plan_sha256=sha256((root/'plan.json').read_bytes()).hexdigest(),
                source_sha256={'code.py': sha256((root/'code.py').read_bytes()).hexdigest()},
                updates=25, curve=[], parameters=2370051, valid_tokens=100, requests=100,
                train_s=2., preparation_s=1., evaluation_s=1., phase='training', latest_train_ce=1.)
            result['checkpoint_sha256'] = 'current'
            (reports/'result.json').write_text(json.dumps(result))
            target = dict(local_song_training=dict(result='result.json', plan='plan.json', pid=10, generation='generation.json'))
            snapshot = monitor.Monitor().snapshot('new', now=20)
            snapshot['training_complete'] = True  # A previous cloud campaign has finished.
            target['local_song_training']['run_id'] = 'current-local'
            monitor.local_song_training_status(snapshot, target, reports,
                lambda pid: 'python -m scripts.run_local_song_baseline --plan plan.json')
            self.assertEqual(snapshot['local_song_training']['status'], 'running')
            self.assertEqual(snapshot['effective_updates'], 25)
            self.assertEqual(snapshot['run_id'], 'current-local')
            self.assertFalse(snapshot['training_complete'])
            self.assertEqual(snapshot['local_song_training']['condition_response'], 'unmeasured')
            self.assertIn('teacher-forced', monitor.render_text(snapshot))
            target['local_song_training'].update(script='scripts.run_local_interaction_candidate',label='D83')
            monitor.local_song_training_status(snapshot, target, reports,
                lambda pid: 'python -m scripts.run_local_interaction_candidate --plan plan.json')
            self.assertEqual(snapshot['local_song_training']['status'], 'running')
            self.assertIn('D83', monitor.render_text(snapshot))
            review = dict(checkpoint_sha256='old', records=[dict(complete=True, full_note_f1=.5,
                notes=dict(model=10, human=20)) for _ in range(20)])
            review_path = reports/'review.json'; review_path.write_text(json.dumps(review))
            target['local_song_training']['review'] = dict(report='review.json', sha256=sha256(review_path.read_bytes()).hexdigest())
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertNotIn('review_summary', snapshot['local_song_training'])
            review['checkpoint_sha256'] = 'current'; review_path.write_text(json.dumps(review))
            target['local_song_training']['review']['sha256'] = sha256(review_path.read_bytes()).hexdigest()
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['review_summary']['raw_full_note_f1_mean'], .5)
            self.assertFalse(snapshot['local_song_training']['model_promotion'])
            timeline = dict(checkpoint_sha256='current',songs=20,timeline_visual_review_complete=True,
                automatic_evidence_sha256=target['local_song_training']['review']['sha256'],
                aggregate=dict(model_types={}),audio_audition_complete=False,native_playtest_complete=False,quality_passed=False)
            timeline_path=reports/'timeline.json';timeline_path.write_text(json.dumps(timeline))
            target['local_song_training'].update(artifact_index='artifacts/current/index.html',
                timeline_review=dict(report='timeline.json',sha256=sha256(timeline_path.read_bytes()).hexdigest()))
            monitor.local_song_training_status(snapshot,target,reports,lambda pid:'')
            self.assertEqual(snapshot['local_song_training']['status'],'reviewed_failed')
            self.assertIn('artifacts/current/index.html',monitor.render_text(snapshot))
            prior = dict(target['local_song_training'], pid=None, label='prior')
            target['prior_song_quality'] = prior
            target['local_song_training'] = {k:v for k,v in target['local_song_training'].items() if k!='review'}
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertNotIn('review_summary', snapshot['local_song_training'])
            self.assertEqual(snapshot['local_song_training']['previous_quality']['review_summary']['raw_full_note_f1_mean'], .5)
            self.assertIn('不属于当前候选', monitor.render_text(snapshot))
            target['local_song_training']['generation_runner'] = dict(pid=20, plan='plan.json', script='scripts.run_local_song_generation')
            (reports/'generation.json').write_text(json.dumps(dict(checkpoint_sha256='current',
                plan_sha256=result['plan_sha256'], backend='cpu', execution_complete=False, completed_songs=3, song_files={})))
            monitor.local_song_training_status(snapshot, target, reports,
                lambda pid: 'python -m scripts.run_local_song_generation --plan plan.json' if pid == 20 else '')
            self.assertEqual(snapshot['local_song_training']['status'], 'running')
            self.assertEqual(snapshot['local_song_training']['whole_songs']['backend'], 'cpu')
            generation = json.loads((reports/'generation.json').read_text())
            generation.pop('backend')
            (reports/'generation.json').write_text(json.dumps(generation))
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['whole_songs']['backend'], 'unrecorded')
            target['local_song_training']['generation_backend'] = 'cpu'
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['whole_songs']['backend'], 'cpu')
            audit = dict(checkpoint_sha256='current', complete=True,
                by_field={f:dict(sentinel_passed=False,pairs=4,identical_pairs=0,in_band_outputs=4)
                          for f in ('density_notes_per_second','hold_head_ratio')},
                combined_gate=dict(passed=True))
            audit_path = reports/'audit.json'; audit_path.write_text(json.dumps(audit))
            target['local_song_training']['condition_audit'] = dict(report='audit.json', sha256=sha256(audit_path.read_bytes()).hexdigest())
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['condition_response'], 'combined_sentinel_only')
            self.assertFalse(snapshot['local_song_training']['model_promotion'])
            audit.update(complete=False, execution_complete=True, expansion_allowed=False,
                by_field={'density_notes_per_second': dict(sentinel_passed=False, pairs=4,
                    identical_pairs=4, sufficiently_directed_pairs=0, in_band_outputs=0)},
                unmeasured_fields=['hold_head_ratio'], stop_reason='Required density response failed')
            audit_path.write_text(json.dumps(audit))
            target['local_song_training']['condition_audit']['sha256'] = sha256(audit_path.read_bytes()).hexdigest()
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['condition_response'], 'required_field_failed_expansion_stopped')
            self.assertEqual(snapshot['local_song_training']['condition_audit']['unmeasured_fields'], ['hold_head_ratio'])
            audio=dict(checkpoint_sha256='current',complete=True,rows=[{} for _ in range(4)],identical_pairs=4)
            audio_path=reports/'audio.json';audio_path.write_text(json.dumps(audio))
            target['local_song_training']['audio_audit']=dict(report='audio.json',sha256=sha256(audio_path.read_bytes()).hexdigest())
            monitor.local_song_training_status(snapshot,target,reports,lambda pid:'')
            self.assertEqual(snapshot['local_song_training']['audio_audit']['identical_pairs'],4)
            self.assertIn('音频错配：4/4',monitor.render_text(snapshot))
            (root/'code.py').write_text('changed')
            monitor.local_song_training_status(snapshot, target, reports, lambda pid: '')
            self.assertEqual(snapshot['local_song_training']['status'], 'evidence_incomplete')

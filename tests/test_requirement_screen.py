from dataclasses import asdict
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig,FullEventModel
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.periodic_requirements import requirement_due,evaluate_training_checkpoint
from beatfcn.train.requirement_diagnostics import diagnose,summarize_requirements
from beatfcn.train.requirement_screen import audio_intervention,prepare_case,run_requirement_screen
from scripts.freeze_requirement_screen import freeze


def event(time=100000,kind='tap',lane=-1,end=None):
    return dict(onset_us=time,end_us=time if end is None else end,type=kind,lane=lane)


def fixture():
    from tests.test_full_event_model import inputs
    config=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=48,encoder_layers=1,
        decoder_layers=2,tap_dim=16,taps=2,bands=1,dropout=.1,intensity_dim=3)
    base={k:v[0].numpy()for k,v in inputs(config).items()if k not in('controls','mode','reference','reference_valid','reference_times')}
    base.update(mel=np.arange(31*128,dtype=np.float32).reshape(31,128),
        mel_times=np.arange(31,dtype=np.float32)*.01,mel_valid=np.ones(31,bool))
    class Features:
        records={}
        def window(self,audio,*_):
            result={k:v.copy()for k,v in base.items()}
            for stream in ('local','song','mel'):result[stream]+=int(audio[0],16)*.01
            return result
    class Panel:
        records=[];charts=[];features=Features();targets_by_index=[]
        def targets(self,index):return self.targets_by_index[index]
    panel=Panel();samples=[]
    for index in range(6):
        mode='phira'if index<3 else'mania_4k';audio=f'{index+1:x}'*64
        request=asdict(EventRequest(mode,0,1000000,10000000))
        request['copy_times']=[];request['known_carry']=[]
        truth=[]if index%3==2 else [RhythmEvent(100000,100000,'tap',-1 if mode=='phira' else 0),
             RhythmEvent(300000,600000,'hold',-1 if mode=='phira' else 1)]
        panel.features.records[audio]=dict(duration_us=10000000)
        target=f'{index+8:x}'*64;row=dict(index=index,chart_id=f'chart{index}',audio_sha256=audio,
            component=audio,mode=mode,split='val',request=request,chart_index=index)
        panel.records.append(row);panel.charts.append(dict(target_sha256=target));panel.targets_by_index.append(truth)
        samples.append(dict(index=index,chart_id=row['chart_id'],audio_sha256=audio,component=audio,mode=mode,split='val',
            request=request,target_sha256=target,event_count=len(truth),type_counts={},case_role='natural_silence'if not truth else'positive'))
    protocol=freeze(dict(samples=samples,tape_sha256='a'*64,selection='CPU fixture'))
    return FullEventModel(config),panel,protocol


class RequirementMetricTests(unittest.TestCase):
    def test_empty_positive_cannot_score_as_human_similarity_success(self):
        request=asdict(EventRequest('phira',0,1000000,1000000))
        result=diagnose(dict(complete=True,events=[]),[event()],request)
        self.assertEqual(result['accuracy']['scores']['note_50ms']['f1'],0.)
        self.assertIsNone(result['human_statistical_similarity']['type_js_nats'])
        self.assertTrue(result['failures']['empty_positive'])
        self.assertFalse(result['failures']['complete_nonempty_positive'])

    def test_correct_type_count_at_wrong_time_has_zero_accuracy(self):
        request=asdict(EventRequest('phira',0,1000000,1000000))
        result=diagnose(dict(complete=True,events=[event(800000)]),[event()],request)
        self.assertEqual(result['accuracy']['per_type']['tap']['tp'],0)
        self.assertEqual(result['human_statistical_similarity']['type_js_nats'],0.)
        self.assertFalse(result['failures']['illegal_event_indices'])

    def test_unspecified_density_is_distinct_from_explicit_zero(self):
        absent=grid_density_curve([],0,1000000)
        zero=grid_density_curve([dict(start_us=0,end_us=1000000,density=0.)],0,1000000)
        self.assertFalse(absent['intensity_valid'].any())
        self.assertTrue(zero['intensity_valid'].all())
        np.testing.assert_equal(absent['intensity'],zero['intensity'])

    def test_partial_correct_prefix_is_not_complete_accuracy(self):
        request=asdict(EventRequest('phira',0,1000000,1000000));pred=dict(complete=False,events=[event()],deadline_reached=True)
        diagnostics=diagnose(pred,[event()],request)
        record=dict(index=0,condition='free',case_role='positive',teacher_joint_ce=1.,prediction=pred,diagnostics=diagnostics)
        result=summarize_requirements([record],dict(samples=[dict(case_role='positive')],anchors=[]))
        self.assertIsNone(result['full_note_f1_50ms'])
        self.assertEqual(result['prefix_note_f1_diagnostic'],1.)
        self.assertFalse(result['all_empty_hack'])
        self.assertIsNone(result['product_requirements_satisfied'])

    def test_repeating_template_and_only_start_are_reported_without_editing_events(self):
        request=asdict(EventRequest('phira',0,8000000,8000000));events=[event(cell*1000000+offset)for cell in range(3)for offset in (1000,2000)]
        result=diagnose(dict(complete=True,events=events),[event(7000000)],request)
        self.assertEqual(result['failures']['predicted_pattern']['largest_repeated_second_pattern_fraction'],1.)
        self.assertLess(result['failures']['predicted_pattern']['head_span_fraction'],.3)
        self.assertEqual(len(events),6)


class RequirementRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_interventions_preserve_request_controls_and_roll_only_local_acoustics(self):
        _,panel,protocol=fixture()
        base=prepare_case(panel,protocol,dict(index=0,condition='free'))['inputs']
        zero=audio_intervention(base,'audio_zero');rolled=audio_intervention(base,'local_roll',roll_s=.08)
        for stream in ('local','mel','song'):self.assertTrue((zero[stream]==0).all())
        for key in ('controls','intensity','intensity_valid','local_times','mel_times','song'):
            np.testing.assert_equal(base[key],rolled[key])
        self.assertFalse(np.array_equal(base['local'],rolled['local']))
        self.assertFalse(np.array_equal(base['mel'],rolled['mel']))
        low=prepare_case(panel,protocol,dict(index=0,condition='density_low'))
        high=prepare_case(panel,protocol,dict(index=0,condition='density_high'))
        self.assertTrue(high['inputs']['intensity_valid'].all())
        self.assertAlmostEqual(sum(s['density']for s in high['requested_curve']),4*sum(s['density']for s in low['requested_curve']))

    def test_real_model_serial_screen_preserves_weights_gradients_rng_and_training_flags(self):
        model,panel,protocol=fixture();model.train();model.decoder[0].eval()
        first=next(model.parameters());first.grad=torch.ones_like(first)
        before={k:v.clone()for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();flags=[m.training for m in model.modules()]
        with tempfile.TemporaryDirectory()as directory,contextlib.redirect_stdout(io.StringIO()):
            path=Path(directory)/'raw.json'
            result=run_requirement_screen(model,panel,protocol,checkpoint=dict(sha256='f'*64,optimizer_updates=10,exposure_step=160),
                plan_sha256='e'*64,output=path,device='cpu',max_wall_s=30,max_tokens=3,batch_size=8)
            saved=json.loads(path.read_text())
        self.assertEqual(len(result['records']),20)
        self.assertTrue(result['all_cases_measured'])
        self.assertTrue(all('tokens'in r['prediction']and'deadline_reached'in r['prediction']for r in saved['records']))
        self.assertEqual([m.training for m in model.modules()],flags)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        torch.testing.assert_close(first.grad,torch.ones_like(first))
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_cadence_counts_optimizer_updates_and_refuses_partial_boundary(self):
        settings=dict(every_optimizer_updates=250)
        self.assertFalse(requirement_due(250//16,settings));self.assertFalse(requirement_due(249,settings));self.assertTrue(requirement_due(250,settings))
        with self.assertRaises(ValueError):
            evaluate_training_checkpoint(None,None,dict(requirement_eval={},accum_steps=16),
                dict(updates=251,optimizer_updates=15),root='.',device='cpu',remaining_wall_s=150)

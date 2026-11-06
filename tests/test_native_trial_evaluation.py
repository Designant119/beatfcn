import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.exact_cell_support import ExactCellSupport
from beatfcn.train.field_integrity import validate,coverage,replay_fields
from beatfcn.train.native_cell_chain import decode_cells
from beatfcn.train.native_trial_evaluation import score_outcome,acoustic_pairs,outcome_tasks,ORDER
from beatfcn.train.native_trial_evaluation import evaluate
from scripts import run_native_cell_recovery as engine
from scripts.run_local_portfolio_scale import write,sha


ROOT=Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_partial_decode_is_zero_but_keeps_raw_match_and_count(self):
        target=[RhythmEvent(1000,1000,'drag')]
        score,raw=score_outcome(engine,target,target,False)
        self.assertEqual(raw['full_note_f1'],1.)
        self.assertEqual(score['full_note_f1'],0.)
        self.assertEqual(score['full_note_f1_10ms'],0.)
        self.assertEqual(score['onset_10ms']['f1'],0.)
        self.assertEqual(score['per_type']['drag']['tp'],0)
        self.assertEqual(score['per_type']['drag']['pred'],1)
        self.assertEqual(score['per_type']['drag']['true'],1)

    def test_failed_control_cannot_qualify_positive_acoustic_gain(self):
        target=[RhythmEvent(1000,1000,'tap')];s,_=score_outcome(engine,target,target,True)
        base=dict(index=1,component='c',prediction=dict(complete=True),scores=s,
            control=dict(core_coverage={k:dict(fraction=1.) for k in ('local','mel','song')}))
        bad=copy.deepcopy(base);bad['prediction']['complete']=False;bad['scores']=score_outcome(engine,[],target,False)[0]
        result=acoustic_pairs([base],[bad]);self.assertFalse(result['complete'])
        self.assertEqual(result['positive_covered_components'],0)
        self.assertEqual(result['full_denominator_components'],1)

    def test_fixed_task_order_and_all_170_denominators(self):
        plan=json.loads((ROOT/'bench/rhythm-v2/c12-native-cell-recovery-v1.json').read_text())
        tasks=outcome_tasks(engine,plan)
        self.assertEqual(tuple(tasks),ORDER);self.assertEqual(sum(map(len,tasks.values())),170)
        self.assertEqual(len(tasks['draft']),32);self.assertEqual(len(tasks['empty']),2)

    def test_endpoint_probability_changes_cannot_hide_behind_equal_integers(self):
        path=ROOT/'artifacts/c13-frozen-health-completion-v1/fields-3817.json'
        f=json.loads(path.read_text());self.assertEqual(validate(f)['maximum_probability_diff'],0.)
        row=next(r for r in f['arms']['base'] if r.get('endpoint'))
        trace=next(t for t in row['endpoint']['levels'] if len(t['probability'])>1)
        p=trace['probability'];at=int(np.argmax(p));other=(at+1)%len(p);amount=min(p[at]/2,.01)
        p[at]-=amount;p[other]+=amount
        with self.assertRaisesRegex(ValueError,'BASE probability'):validate(f)

    def test_support_length_mismatch_is_not_zip_truncated(self):
        f=json.loads((ROOT/'artifacts/c13-frozen-health-completion-v1/fields-3817.json').read_text())
        row=next(r for r in f['arms']['base'] if r.get('endpoint'))
        row['endpoint']['levels'].pop()
        with self.assertRaisesRegex(ValueError,'3levels'):validate(f)

    def test_coverage_retains_outside_queries(self):
        result=coverage([0.,.01],[True,True],[-.006,0.,.01,.016],.01)
        self.assertEqual(result.tolist(),[False,True,True,False])

    def test_resource_stop_preserves_all_expected_outcomes(self):
        plan=json.loads((ROOT/'bench/rhythm-v2/c12-native-cell-recovery-v1.json').read_text())
        tasks=outcome_tasks(engine,plan);groups={};target=[RhythmEvent(1000,1000,'tap')]
        s,raw=score_outcome(engine,[],target,False)
        request=EventRequest('phira',0,8000000,16000000)
        null=engine.output_null([],target,request,plan['shifts_us'])
        for name,items in tasks.items():
            groups[name]=[dict(index=i,component=str(j//2),view='draft',status='not_attempted',target=[dict(onset_us=1000,end_us=1000,type='tap')],
                prediction=dict(complete=False,events=[]),scores=copy.deepcopy(s),raw_partial_scores=copy.deepcopy(raw),
                output_null=copy.deepcopy(null),output_null_onset10=[0.]*len(null),control={}) for j,(i,*_) in enumerate(items)]
        class StopGuard:
            start=0.;wall_s=100.;phase='fixture_stop'
            def check(self,*_):raise MemoryError('Synthetic physical stop')
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);art=base/'artifacts';art.mkdir();write(art/'data-identity.json',dict(prior=[1,1,1,1]))
            path=base/'metadata.json';write(path,groups);plan.update(expected_free_metadata='metadata.json',expected_free_metadata_sha256=sha(path))
            stub=SimpleNamespace(ART=art,ROOT=base,read=lambda name:json.loads((base/name).read_text()),sha=sha,write=write,specs=engine.specs)
            report={};evaluate(stub,None,plan,None,None,StopGuard(),report)
            self.assertEqual(report['all_free_outcomes'],170);self.assertFalse(report['research_qualified'])
            self.assertEqual(report['evaluation']['draft']['requests'],32)
            self.assertEqual(report['evaluation']['draft']['complete'],0)
            self.assertEqual(len(report['p2']),8)
            self.assertEqual(sum(len(json.loads((art/f'{name}.json').read_text())) for name in ORDER),170)

    def test_audited_reader_on_actual_hold_model_path(self):
        torch.set_num_threads(4);torch.manual_seed(312)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
            taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128,dropout=0.)
        model=NativeCellChain(c,density_reference=6.).eval()
        request=EventRequest('phira',0,5000,12000,copy_times=(1501,1502,3501),allowed_types=('hold',))
        heads,allowed=ExactCellSupport().addresses(request)
        output=dict(x=torch.randn(1,1,32),song=torch.randn(1,1,32),reference=torch.randn(1,1,32),
            song_values=torch.randn(1,2,32),song_times=torch.tensor([[0.,.5]]),song_valid=torch.ones(1,2,dtype=torch.bool),count=torch.full((1,1,66),-20.))
        output['count'][0,0,1]=20.
        inputs=dict(mode=torch.zeros(1,dtype=torch.long),mel=torch.randn(1,4,128),mel_times=torch.tensor([[-.01,0.,.01,.02]]),
            mel_valid=torch.ones(1,4,dtype=torch.bool),joint_frame_times=torch.zeros(1,1),
            local_times=torch.tensor([[0.]]),local_valid=torch.tensor([[True]]),song_times=torch.tensor([[0.,.5]]),song_valid=torch.tensor([[True,True]]))
        import os
        device=os.environ.get('REPLAY_TEST_DEVICE','cpu');model.to(device)
        output={k:v.to(device) for k,v in output.items()};inputs={k:v.to(device) for k,v in inputs.items()}
        layout=dict(heads=heads,allowed=allowed);pred=decode_cells(model,inputs,request,layout,trace=True,output=output)
        targets=[RhythmEvent(**e) for e in pred['events']]
        result=replay_fields(model,output,inputs,request,layout,pred,targets,dict(zero=(output,inputs),shift=(output,inputs)))
        self.assertEqual(result['full_probability_certificate']['status'],'passed')
        self.assertEqual(len(result['source_coverage']),12)
        self.assertEqual(len(result['arms']['base'][0]['coverage_paths']['endpoint_free']['levels']),3)


if __name__=='__main__':unittest.main()

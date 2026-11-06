from copy import deepcopy
from types import SimpleNamespace
import os
import json
import unittest

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.exact_cell_support import ExactCellSupport
from beatfcn.train.native_cell_chain import decode_cells
from beatfcn.train.native_diagnostic_priors import fit_priors
from beatfcn.train.c14_type_probe import type_arm,tt_diagnostic,self_diagnostic,_item,summarize_conditionals


class TypeProbeTests(unittest.TestCase):
    def test_actual_small_model_paths_and_no_source_mutation(self):
        torch.set_num_threads(4);torch.manual_seed(312)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128,dropout=0.)
        device=os.environ.get('REPLAY_TEST_DEVICE','cpu');model=NativeCellChain(c,density_reference=6.).to(device).eval()
        request=EventRequest('phira',0,5000,12000,copy_times=(1501,1502,3501),allowed_types=('hold',))
        heads,allowed=ExactCellSupport().addresses(request)
        out=dict(x=torch.randn(1,1,32),song=torch.randn(1,1,32),reference=torch.randn(1,1,32),song_values=torch.randn(1,3,32),song_times=torch.tensor([[0.,.5,1.]]),song_valid=torch.ones(1,3,dtype=torch.bool),count=torch.full((1,1,66),-20.))
        out['count'][0,0,1]=20.
        inputs=dict(mode=torch.zeros(1,dtype=torch.long),mel=torch.randn(1,4,128),mel_times=torch.tensor([[-.01,0.,.01,.02]]),mel_valid=torch.ones(1,4,dtype=torch.bool),joint_frame_times=torch.zeros(1,1))
        out={k:v.to(device) for k,v in out.items()};inputs={k:v.to(device) for k,v in inputs.items()};original_song=out['song'].clone();original_count=out['count'].clone()
        pred=decode_cells(model,inputs,request,dict(heads=heads,allowed=allowed),trace=True,output=out)
        truth=[RhythmEvent(**e) for e in pred['events']];counts,members=ExactCellSupport().encode(truth,request)
        target=dict(heads=heads,allowed=allowed,counts=counts,members=[[m for m in members if m.cell==i] for i in range(len(counts))])
        prior=fit_priors([[dict(onset_us=0,end_us=1000,type='hold')]],[],[1]*4)
        cells=torch.tensor([0],device=device);state=torch.zeros(1,192,device=device);feature=torch.zeros(1,32,device=device)
        raw=model.kind_logits(out,state,feature,cells).detach().clone()
        for arm in ('B','Z','T'):
            with type_arm(model,out,inputs,arm):
                altered=model.kind_logits(out,state,feature,cells).detach()
                if arm=='B':torch.testing.assert_close(altered,raw)
                else:self.assertFalse(torch.allclose(altered,raw))
                tt=tt_diagnostic(model,out,inputs,request,target,prior,None)
                own=self_diagnostic(model,out,inputs,request,target,prior,None)
            self.assertEqual(tt['summary']['events'],1);self.assertEqual(own['summary']['events'],1)
            json.dumps(tt);json.dumps(own)
            self.assertEqual(tt['summary']['support_failures'],0)
            self.assertTrue(torch.equal(out['song'],original_song));self.assertTrue(torch.equal(out['count'],original_count))
        self.assertEqual(model.kind_logits.__func__,NativeCellChain.kind_logits)

    def test_unsupported_singleton_is_infinite_not_forced_success(self):
        prior=fit_priors([],[],[1]*4);request=EventRequest('phira',0,8000,12000)
        truth=SimpleNamespace(kind=1,head=1000,end_us=4000);previous=RhythmEvent(3000,3000,'drag')
        item=_item(prior,request,0,1,truth,previous,[False,False,False,True],[0,0,0,1],False,None,None,False)
        result=summarize_conditionals([item])
        self.assertEqual(result['forced'],0);self.assertEqual(result['events'],1)
        self.assertEqual(result['support_failures'],1);self.assertIsNone(result['type_nll_sum'])
        self.assertEqual(result['aggregate_semantics'],'infinite')

    def test_nonforced_numpy_mask_is_json_serializable(self):
        prior=fit_priors([],[],[1]*4);request=EventRequest('phira',0,8000,12000)
        truth=SimpleNamespace(kind=0,head=1000,end_us=1000)
        item=_item(prior,request,0,0,truth,None,np.array([True]*4),[.4,.2,.2,.2],True,1.,0.,True)
        self.assertIs(type(item['forced']),bool)
        self.assertFalse(json.loads(json.dumps(item))['forced'])


if __name__=='__main__':unittest.main()

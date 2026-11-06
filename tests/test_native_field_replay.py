import unittest
from dataclasses import replace
import os

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.exact_cell_support import ExactCellSupport
from beatfcn.train.multiset_endpoint_diagnostic import endpoint_diagnostic
from beatfcn.train.native_cell_chain import decode_cells
from beatfcn.train.native_field_replay import controlled_heads,controlled_end_keys,replay_fields


class ReplayTests(unittest.TestCase):
    def test_exact_branch_identity_and_fixed_machine_replay(self):
        torch.manual_seed(312)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
            taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128,dropout=0.)
        model=NativeCellChain(c,density_reference=6.).eval()
        request=EventRequest('phira',0,5000,12000,copy_times=(1501,1502,3501),allowed_types=('hold',))
        heads,allowed=ExactCellSupport().addresses(request)
        output=dict(x=torch.randn(1,1,32),song=torch.randn(1,1,32),reference=torch.randn(1,1,32),
            song_values=torch.randn(1,2,32),song_times=torch.tensor([[0.,.5]]),song_valid=torch.ones(1,2,dtype=torch.bool),
            count=torch.full((1,1,66),-20.))
        output['count'][0,0,1]=20.
        inputs=dict(mode=torch.zeros(1,dtype=torch.long),mel=torch.randn(1,4,128),mel_times=torch.tensor([[-.01,0.,.01,.02]]),
            mel_valid=torch.ones(1,4,dtype=torch.bool),joint_frame_times=torch.zeros(1,1))
        device=os.environ.get('REPLAY_TEST_DEVICE','cpu');model.to(device)
        output={k:v.to(device) for k,v in output.items()};inputs={k:v.to(device) for k,v in inputs.items()}
        cells=torch.tensor([0],device=device);a=model.address_features(output,inputs,heads,cells,0)
        b,_=controlled_heads(model,output,inputs,heads,cells,0)
        torch.testing.assert_close(a,b)
        ends=[1502,2000,3501,12000]
        torch.testing.assert_close(model.endpoint_keys(output,inputs,ends,0),controlled_end_keys(model,output,inputs,ends,0)[0])
        torch.testing.assert_close(b,controlled_heads(model,output,inputs,heads+300000,cells,300000)[0])
        torch.testing.assert_close(controlled_end_keys(model,output,inputs,ends,0)[0],controlled_end_keys(model,output,inputs,np.asarray(ends)+300000,300000)[0])
        layout=dict(heads=heads,allowed=allowed)
        pred=decode_cells(model,inputs,request,layout,trace=True,output=output)
        self.assertTrue(pred['complete']);self.assertEqual(len(pred['events']),1)
        targets=[RhythmEvent(**e) for e in pred['events']]
        result=replay_fields(model,output,inputs,request,layout,pred,targets,{},health=True)
        self.assertEqual(result['certificate']['coverage_members'],1)
        self.assertLessEqual(result['field_identity_max_diff'],1e-4)
        self.assertEqual(result['arms']['base'][0]['endpoint']['end_us'],targets[0].end_us)
        self.assertEqual(result['arms']['base'][0]['endpoint_true_path']['end_us'],targets[0].end_us)
        complete=replay_fields(model,output,inputs,request,layout,pred,targets,dict(zero=(output,inputs),shift=(output,inputs)),health=False)
        self.assertEqual(len(complete['arms']),12)

    def test_same_head_different_tails_are_not_duration_errors(self):
        p=[RhythmEvent(1000,9000,'hold'),RhythmEvent(1000,2000,'hold')]
        t=list(reversed(p));r=endpoint_diagnostic(p,t)
        self.assertEqual(r['matches'],2);self.assertEqual(r['mae_ms'],0.)
        self.assertEqual(len(r['pairs']),2)


if __name__=='__main__':unittest.main()

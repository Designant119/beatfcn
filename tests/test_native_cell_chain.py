from itertools import product
from dataclasses import replace
import unittest

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.native_cell_chain import NativeCellChain
from beatfcn.train.exact_cell_support import ExactEndpointTree
from beatfcn.train.native_cell_chain import endpoint_decision,decode_cells


class NativeCellChainTests(unittest.TestCase):
    def test_canonical_count_member_distribution_normalizes_and_keeps_duplicates(self):
        # A tiny completely enumerated multiset law. Count followed by canonical
        # members is normalized, while conditional greedy is not named jointMAP.
        count=[.2,.3,.5];p=[.1,.2,.3,.4];total=count[0];duplicates=0
        for n in (1,2):
            for path in product(range(4),repeat=n):
                if tuple(sorted(path))!=path:continue
                probability=count[n];minimum=0
                for j in path:
                    probability*=p[j]/sum(p[minimum:]);minimum=j
                total+=probability
                duplicates+=n==2 and path[0]==path[1]
        self.assertAlmostEqual(total,1.);self.assertEqual(duplicates,4)

    def test_endpoint_neural_tree_exact_normalization_and_origin_shift(self):
        torch.manual_seed(91)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
            taps=2,bands=1,band_slots=1,tap_dim=1024,ssm_state=8,ssm_headdim=16,mel_bins=128)
        m=NativeCellChain(c,density_reference=6.)
        request=EventRequest('phira',0,5000,5000,copy_times=(1501,1502,3501))
        out=dict(x=torch.zeros(1,1,32),song_values=torch.zeros(1,2,32),song_times=torch.tensor([[0.,.5]]),song_valid=torch.ones(1,2,dtype=torch.bool))
        inputs=dict(mel=torch.zeros(1,2,128),mel_times=torch.tensor([[0.,.01]]),mel_valid=torch.ones(1,2,dtype=torch.bool))
        state=torch.zeros(192);feature=torch.zeros(32);tree=ExactEndpointTree(request,1501)
        leaves=[u for b in tree.blocks() for c in tree.cells(b) for u in tree.leaves(b,c)]
        probability=0.;losses=[]
        for end in leaves:
            value,nll,_=endpoint_decision(m,out,inputs,state,feature,request,1501,None,target=end)
            self.assertEqual(value,end);probability+=float((-nll).exp().detach());losses.append(nll)
        self.assertAlmostEqual(probability,1.,places=5)
        sum(losses).backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.endpoint_key.parameters() if p.grad is not None))
        shifted=replace(request,start_us=300000,end_us=305000,duration_us=305000,copy_times=tuple(u+300000 for u in request.copy_times))
        _,n1,_=endpoint_decision(m,out,inputs,state,feature,request,1501,None,target=3501)
        _,n2,_=endpoint_decision(m,out,inputs,state,feature,shifted,301501,None,target=303501)
        torch.testing.assert_close(n1,n2)

    def test_decode_interface_rejects_hidden_label_dictionary(self):
        with self.assertRaises(ValueError):decode_cells(None,{},None,dict(heads=[],allowed=[],members=[]))


if __name__=='__main__':unittest.main()

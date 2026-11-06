from dataclasses import replace
import itertools
import unittest

import numpy as np
import torch

from beatfcn.model.arrangement_iohmm import forward_log_probability,prior_path
from beatfcn.model.whole_song_arrangement import WholeSongArrangement,arrangement_loss,generate_arrangement
from beatfcn.train.canonical_probability import encode_canonical
from beatfcn.train.whole_song_mel import whole_summary,multiscale_flux
from beatfcn.model.event_canvas import RhythmEvent
from tests import test_local_event_package as old


class ArrangementTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(91)

    def test_sparse_hmm_forward_matches_path_enumeration_and_unlabelled_is_not_empty(self):
        initial=torch.randn(2,dtype=torch.float64).log_softmax(-1)
        transition=torch.randn(4,2,2,dtype=torch.float64).log_softmax(-1)
        emission=torch.zeros(4,2,dtype=torch.float64)
        emission[1]=torch.tensor([-.2,-1.8]);emission[3]=torch.tensor([-2.,-.4])
        terms=[]
        for path in itertools.product(range(2),repeat=4):
            terms.append(initial[path[0]]+sum(emission[b,path[b]] for b in range(4))+
                sum(transition[b,path[b-1],path[b]] for b in range(1,4)))
        torch.testing.assert_close(forward_log_probability(initial,transition,emission),torch.logsumexp(torch.stack(terms),0))
        zero=forward_log_probability(initial,transition,torch.zeros_like(emission))
        self.assertLess(abs(float(zero)),1e-12)
        chosen=prior_path(initial,transition)
        prior_terms=[(p,float(initial[p[0]]+sum(transition[b,p[b-1],p[b]] for b in range(1,4)))) for p in itertools.product(range(2),repeat=4)]
        self.assertEqual(tuple(chosen),max(prior_terms,key=lambda p:p[1])[0])

    def test_whole_song_summary_preserves_order_and_last_partial_block(self):
        mel=np.repeat(np.arange(205,dtype=np.float32)[:,None],128,axis=1)
        summary,times=whole_summary(mel,205*240)
        self.assertEqual(summary.shape,(2,2060))
        np.testing.assert_allclose(times,[1.,2.025],atol=1e-7)
        np.testing.assert_allclose(summary[0,:128],24.5)
        np.testing.assert_allclose(summary[0,512:640],74.5)
        np.testing.assert_allclose(multiscale_flux(mel)[0,128:],0)

    def test_real_emission_graph_backpropagates_through_state_and_acoustic_routes(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        model=WholeSongArrangement(old.EventPackageTests.small_model().c,density_reference=4).eval()
        inputs={k:v for k,v in inputs.items() if not k.startswith(('local','song'))}
        inputs['mel_flux']=torch.randn(1,8,32)
        # Small test config has8 mel bins, so whole summaries have140 fields.
        inputs['whole_song']=torch.randn(1,2,140)
        inputs['whole_song_times']=torch.tensor([[1.,3.]])
        inputs['whole_song_valid']=torch.ones(1,2,dtype=torch.bool)
        events=[RhythmEvent(4000,2012000,'hold'),RhythmEvent(5000,5000,'flick')]
        target={k:torch.from_numpy(v)[None] for k,v in encode_canonical(events,request).items()}
        loss,parts=arrangement_loss(model,inputs,target,request)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(int(parts['valid_groups']),8)
        loss.backward();parameters=dict(model.named_parameters())
        for name in ('fine_projection.weight','song_input.weight','state_film.weight','count_gate.weight','member_gru.weight_ih','initial_read.weight'):
            self.assertIsNotNone(parameters[name].grad,name)
            self.assertTrue(torch.isfinite(parameters[name].grad).all(),name)
        # Endpoint is a genuine prior-path sampler; no posterior target enters.
        with torch.no_grad():model.count_gate.weight.zero_();model.count_gate.bias.fill_(-100.)
        prediction,info=generate_arrangement(model,inputs,request,seed=41)
        self.assertEqual(prediction,[]);self.assertEqual(len(info['prior_state_path']),2)
        self.assertFalse(any('local_projection' in n or 'song_projection' in n for n,_ in model.named_parameters()))


if __name__=='__main__':unittest.main()

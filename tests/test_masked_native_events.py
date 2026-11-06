from collections import Counter
import math
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.masked_native_events import MaskedNativeEvents
from beatfcn.train.masked_native_events import (
    SLOTS,public_layout,target_state,state_events,empty_state,corruption,stable_rng,
    duration_interval_mass,endpoint_factor,tuple_loss,decode_masked,
)
from beatfcn.train.native_diagnostic_priors import fit_priors,duration_probability


def toy_inputs(device='cpu'):
    rng=torch.Generator().manual_seed(34)
    values=dict(local=torch.randn(1,2,8,1,1024,generator=rng)*.1,
        local_times=torch.arange(8)[None]*.04,local_valid=torch.ones(1,8,dtype=torch.bool),
        mel=torch.randn(1,32,128,generator=rng)*.1,mel_times=torch.arange(32)[None]*.01,
        mel_valid=torch.ones(1,32,dtype=torch.bool),song=torch.randn(1,8,1024,generator=rng)*.1,
        song_times=torch.arange(8)[None]*.5,song_valid=torch.ones(1,8,dtype=torch.bool),
        reference=torch.zeros(1,3,16),reference_times=torch.zeros(1,3),reference_valid=torch.tensor([[True,False,False]]),
        controls=torch.zeros(1,8,6),mode=torch.ones(1,dtype=torch.long))
    return {k:v.to(device) for k,v in values.items()}


def small_model():
    torch.manual_seed(878)
    config=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=64,encoder_layers=1,decoder_layers=0,
                          taps=2,bands=1,band_slots=1,tap_dim=1024,mel_bins=128)
    return MaskedNativeEvents(config)


class MaskedEventsTests(unittest.TestCase):
    def test_exact_multiset_null_holes_capacity_and_protection(self):
        r=EventRequest('phira',0,4000,12000,copy_times=(1501,1502,7001))
        events=[RhythmEvent(1501,7001,'hold')]*2+[RhythmEvent(1502,1502,k) for k in ('tap','flick','drag')]
        layout=public_layout(dict(request=r,known_regions=()))
        state=target_state(events,r,layout)
        self.assertEqual(Counter(state_events(state)),Counter(events))
        state['status'][10]=2
        for k in ('head','kind','end'):state[k][10]=state[k][0]
        state['status'][0]=1
        self.assertEqual(Counter(state_events(state)),Counter(events))
        target_state([RhythmEvent(1000,1000,'tap')]*256,r,layout)
        with self.assertRaises(ValueError):target_state(events*60,r,layout)
        protected=public_layout(dict(request=r,known_regions=((0,4000),)))
        self.assertEqual(len(protected['heads']),0)
        with self.assertRaises(ValueError):target_state(events,r,protected)

    def test_mask_is_label_independent_and_wipes_unknown_fields(self):
        a=empty_state();a['status'][:]=1
        b={k:v.copy() for k,v in a.items()};b['status'][:100]=2;b['head'][:100]=1501;b['end'][:100]=7001
        x,ma,ra=corruption(a,stable_rng(42),.25)
        y,mb,rb=corruption(b,stable_rng(42),.25)
        self.assertTrue(np.array_equal(ma,mb));self.assertEqual(ra,rb)
        for key in ('head','kind','end'):self.assertTrue((y[key][ma]==0).all())
        self.assertTrue((y['status'][ma]==0).all())

    def test_exact_duration_prior_zero_residual_and_gradient(self):
        torch.set_num_threads(4);model=small_model()
        r=EventRequest('phira',0,4000,12000,copy_times=(1501,1502,7001))
        event=RhythmEvent(1501,7001,'hold');layout=public_layout(dict(request=r))
        prior=fit_priors([], [dict(onset_us=1501,end_us=7001,type='hold')], [1,1,1,1])
        x=toy_inputs();memory=model.encode(x,layout,0);out=model(memory,empty_state())
        address=int(np.flatnonzero(layout['heads']==1501)[0])
        _,nll,_=endpoint_factor(model,out,0,address,r,prior,target=7001)
        expected=-math.log(duration_probability(prior,1501,7001,12000,r.copy_times))
        self.assertLess(abs(float(nll.detach())-expected),1e-5)
        total=duration_interval_mass(prior,1501,12000,r.copy_times,1502,12000)
        self.assertAlmostEqual(total,1.,places=12)
        nll.backward();self.assertGreater(float(model.tail_residual.weight.grad.abs().sum()),0)

    def test_cpu_mps_whole_tuple_gradient_and_forced_null(self):
        if not torch.backends.mps.is_available():self.skipTest('MPS unavailable')
        torch.set_num_threads(4)
        request=EventRequest('phira',0,8000,12000,copy_times=(1501,7001))
        layout=public_layout(dict(request=request));events=[RhythmEvent(1000,1000,'tap'),RhythmEvent(1501,7001,'hold'),RhythmEvent(2000,2000,'flick'),RhythmEvent(2000,2000,'drag')]
        clean=target_state(events,request,layout);prior=fit_priors([], [dict(onset_us=1501,end_us=7001,type='hold')],[1,1,1,1])
        base=small_model().state_dict();results=[]
        for device in ('cpu','mps'):
            model=small_model().to(device);model.load_state_dict(base)
            memory=model.encode(toy_inputs(device),layout,0);out=model(memory,empty_state())
            loss,_=tuple_loss(model,out,request,dict(layout=layout,state=clean),np.ones(SLOTS,bool),1.,prior)
            loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            results.append((float(loss.detach().cpu()),{n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}))
        self.assertLess(abs(results[0][0]-results[1][0]),1e-4)
        self.assertEqual(set(results[0][1]),set(results[1][1]))
        self.assertLess(max(float((results[0][1][n]-results[1][1][n]).abs().max()) for n in results[0][1]),1e-4)
        model=small_model();layout=public_layout(dict(request=request,known_regions=((0,8000),)))
        x=toy_inputs();memory=model.encode(x,layout,0);out=model(memory,empty_state())
        clean=target_state([],request,layout)
        loss,_=tuple_loss(model,out,request,dict(layout=layout,state=clean),np.ones(SLOTS,bool),1.,prior)
        self.assertEqual(float(loss.detach()),0.)
        p=decode_masked(model,x,request,layout,prior,seed=1,public_id='same')
        self.assertTrue(p['complete']);self.assertEqual(p['events'],[]);self.assertEqual(len(p['trace']),512)

    def test_nonempty_sampling_multiset_and_fixed_uniform_replay(self):
        torch.set_num_threads(4);model=small_model()
        with torch.no_grad():model.active_read.weight.zero_();model.active_read.bias.fill_(20.)
        request=EventRequest('phira',0,8000,12000,copy_times=(1501,1502,7001))
        layout=public_layout(dict(request=request));prior=fit_priors([],[],[1,1,1,1]);x=toy_inputs()
        a=decode_masked(model,x,request,layout,prior,seed=543,public_id='fixed')
        b=decode_masked(model,x,request,layout,prior,seed=543,public_id='fixed')
        self.assertTrue(a['complete']);self.assertEqual(len(a['events']),256)
        self.assertEqual(a,b);self.assertEqual(len(a['trace']),512)
        self.assertEqual(len({v['type'] for v in a['events']}),4)
        target_state([RhythmEvent(**v) for v in a['events']],request,layout)


if __name__=='__main__':unittest.main()

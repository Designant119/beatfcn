from dataclasses import replace
import itertools
import unittest

import numpy as np
import torch

from beatfcn.model.grouped_chart_process import GroupedChartProcess,nonempty_count_law,decide_nonempty_counts,grouped_process_loss,generate_grouped_process
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.local_typed_canvas import request_layout
from tests import test_local_event_package as old


class GroupedProcessTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5)

    def test_nonempty_law_is_exact_conditional_probability_not_forced_type(self):
        p=torch.tensor([[.7,.2,.1],[.5,.3,.2]],dtype=torch.float64)
        total=0.
        for c in itertools.product(range(3),repeat=2):
            if sum(c):
                nll=nonempty_count_law(p.log(),torch.tensor(c))
                expected=p[0,c[0]]*p[1,c[1]]/(1-p[0,0]*p[1,0])
                torch.testing.assert_close((-nll).exp(),expected);total+=float((-nll).exp())
        self.assertAlmostEqual(total,1.)
        rng=np.random.default_rng(80)
        for _ in range(50):self.assertGreater(int(decide_nonempty_counts(p.log(),rng=rng).sum()),0)
        c=decide_nonempty_counts(p.log());self.assertGreater(int(c.sum()),0)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS regression')
    def test_absolute_microsecond_clock_does_not_corrupt_candidate_phase_on_mps(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        request=replace(request,start_us=136_000_000,end_us=136_080_000,duration_us=140_000_000,copy_times=(136_001_234,))
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(request).items()})
        cpu=GroupedChartProcess(old.EventPackageTests.small_model().c,density_reference=4,history_dropout=0.).eval()
        gpu=GroupedChartProcess(old.EventPackageTests.small_model().c,density_reference=4,history_dropout=0.).to('mps').eval()
        gpu.load_state_dict(cpu.state_dict())
        with torch.no_grad():
            a=cpu.context(inputs,request);b=gpu.context({k:v.to('mps') for k,v in inputs.items()},request)
        torch.testing.assert_close(a['keys'],b['keys'].cpu(),atol=1e-4,rtol=0)

    def test_strict_group_time_keeps_same_head_mixed_duplicates_and_legal_stop(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        model=GroupedChartProcess(old.EventPackageTests.small_model().c,density_reference=4,history_dropout=0.).eval()
        events=[RhythmEvent(4000,2012000,'hold')]*2+[RhythmEvent(4000,4000,'tap'),RhythmEvent(5000,5000,'flick')]
        loss,parts=grouped_process_loss(model,inputs,events,request)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        params=dict(model.named_parameters())
        for name in ('chart_gru.weight_ih','chart_query.weight','count_gate.weight','tail_block.weight','frame_read.weight'):
            self.assertIsNotNone(params[name].grad,name);self.assertTrue(torch.isfinite(params[name].grad).all(),name)
        context=model.context(inputs,request)
        first=model.head_law(context,model.chart_start,request)
        after=model.head_law(context,model.chart_start,request,previous_head=4000)
        self.assertTrue(torch.isfinite(first[-1]));self.assertTrue(torch.isfinite(after[-1]))
        self.assertTrue(torch.isneginf(after[:-1][context['head']<=4000]).all())
        candidate=int((context['head']==4000).nonzero()[0])
        counts=torch.tensor([1,2,0,0]);decoded,_,_,_=model.emit_group(context,model.chart_start,candidate,counts,inputs,request,events=events[:3])
        self.assertEqual(sorted(decoded),sorted(events[:3]))
        with torch.no_grad():model.stop_read.weight.zero_();model.stop_read.bias.fill_(10000.)
        pred,info=generate_grouped_process(model,inputs,request,seed=51)
        self.assertEqual(pred,[]);self.assertEqual(info['stop_step'],0)


if __name__=='__main__':unittest.main()

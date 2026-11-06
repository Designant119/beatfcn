from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model.event_code_flow import EventCodeFlow,encode_event_code,decode_event_code,flow_loss,generate_flow,CODE_DIM
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.canonical_probability import encode_canonical
from beatfcn.train.local_typed_canvas import request_layout
from tests import test_local_event_package as old
from tests.test_local_mamba_canvas import small_config


class EventCodeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(4)

    def test_finite_code_roundtrip_preserves_all_native_and_authorized_copy_multisets(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        request=replace(request,copy_times=(1234,2345,3000))
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(request).items()})
        # The native alias at3000 is retained once as an address, never by
        # deleting repeated events in the output.
        events=[RhythmEvent(1234,2345,'hold')]*2+[RhythmEvent(1000,1000,'tap'),RhythmEvent(1234,1234,'flick'),RhythmEvent(3000,3_999_000,'hold')]
        target={k:torch.from_numpy(v)[None] for k,v in encode_canonical(events,request).items()}
        code=encode_event_code(target,request)
        self.assertEqual(code.shape,(1,200,4,CODE_DIM))
        self.assertEqual(decode_event_code(code,inputs,request),sorted(events))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS exact-clock regression')
    def test_absolute_tail_digit_conversion_is_done_exactly_before_mps_transfer(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        request=replace(request,start_us=136_000_000,end_us=136_080_000,duration_us=140_000_000)
        events=[RhythmEvent(136_001_000,137_001_000,'hold')]
        target={k:torch.from_numpy(v)[None] for k,v in encode_canonical(events,request).items()}
        cpu=encode_event_code(target,request)
        gpu=encode_event_code({k:v.to('mps') for k,v in target.items()},request).cpu()
        torch.testing.assert_close(cpu,gpu,atol=0,rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS actual operator support')
    def test_partial_request_context_and_velocity_use_supported_mps_operators(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        cpu=EventCodeFlow(small_config(),density_reference=4).eval()
        gpu=EventCodeFlow(small_config(),density_reference=4).to('mps').eval();gpu.load_state_dict(cpu.state_dict())
        state=torch.randn(1,200,4,CODE_DIM);t=torch.tensor([.5])
        with torch.no_grad():
            a=cpu(inputs,state,t);b=gpu({k:v.to('mps') for k,v in inputs.items()},state.to('mps'),t.to('mps'))
            audio=gpu.encode_audio({k:v.to('mps') for k,v in inputs.items()})
        torch.testing.assert_close(a,b.cpu(),atol=1e-4,rtol=0)
        self.assertEqual(float(audio[:,2:].abs().sum()),0)

    def test_true_gaussian_velocity_objective_reaches_frozen_shape_audio_and_flow_heads(self):
        request,inputs,_=old.EventPackageTests.synthetic_example()
        events=[RhythmEvent(4000,2012000,'hold'),RhythmEvent(5000,5000,'flick')]
        target={k:torch.from_numpy(v)[None] for k,v in encode_canonical(events,request).items()}
        model=EventCodeFlow(small_config(),density_reference=4).eval()
        code=encode_event_code(target,request);noise=torch.randn_like(code);t=torch.tensor([.37])
        expected=model(inputs,(1-t[:,None,None,None])*noise+t[:,None,None,None]*code,t)
        valid=inputs['canvas_valid'][...,None,None]
        direct=((expected-(code-noise)).square()*valid).sum()/(valid.sum()*4*CODE_DIM)
        loss=flow_loss(model,inputs,target,request,time_value=t,noise=noise)
        torch.testing.assert_close(loss,direct);loss.backward()
        params=dict(model.named_parameters())
        for name in ('code_input.weight','flow_output.2.weight','frame_read.weight','time_projection.weight'):
            self.assertIsNotNone(params[name].grad,name);self.assertTrue(torch.isfinite(params[name].grad).all(),name)

    def test_multistep_solver_encodes_audio_once_and_empty_is_a_real_decision(self):
        request,inputs,_=old.EventPackageTests.synthetic_example();model=EventCodeFlow(small_config(),density_reference=4).eval()
        # Prescribed velocity transports every draw to the all-empty code in
        # one Euler step, while still exercising declared integration calls.
        def velocity(inp,state,t,*,audio=None):return (-torch.ones_like(state)-state)/(1-t[:,None,None,None])
        with patch.object(model,'forward',side_effect=velocity),patch.object(model,'encode_audio',wraps=model.encode_audio) as encode:
            events,info=generate_flow(model,inputs,request,seed=19,steps=4)
        self.assertEqual(events,[]);self.assertEqual(encode.call_count,1);self.assertEqual(info['ode_steps'],4)


if __name__=='__main__':unittest.main()

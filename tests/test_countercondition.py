import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig,FullEventModel,event_loss
from beatfcn.train.event_conditions import encode_controls,request_log_probability
from beatfcn.train.countercondition import countercondition_objective,validated_countercondition
from tests.test_full_event_model import inputs


class CounterconditionTests(unittest.TestCase):
    def test_rejects_noncontradiction_scope_change_and_extra_field(self):
        scope=(0,8_000_000);field='hold_head_ratio';band=[.1,.2]
        true=encode_controls('phira',{field:band},scope,origin_us=0)
        opposing=validated_countercondition('phira',field,.15,band,[.3,.4],scope,true)
        self.assertTrue(np.array_equal(true[:,2:],opposing[:,2:]))
        with self.assertRaises(ValueError):validated_countercondition('phira',field,.15,band,band,scope,true)
        with self.assertRaises(ValueError):validated_countercondition('phira',field,.15,band,[.3,.4],(0,4_000_000),true)
        extra=encode_controls('phira',{field:band,'drag_ratio':[0.,.02]},scope,origin_us=0)
        with self.assertRaises(ValueError):validated_countercondition('phira',field,.15,band,[.3,.4],scope,extra)

    def test_zero_weight_and_shared_encoder_gradient_equal_naive_loss(self):
        torch.manual_seed(1)
        c=FullEventConfig(d=32,heads=4,kv_heads=2,ffn=48,encoder_layers=1,decoder_layers=2,tap_dim=16,bands=4,dropout=0.)
        model=FullEventModel(c).eval();naive=copy.deepcopy(model)
        inp=inputs(c);q=EventRequest('mania_4k',0,8_000_000,10_000_000)
        seq={k:torch.from_numpy(v)[None] for k,v in teacher_sequence(encode_events([RhythmEvent(1000,1000,'tap',0)],q),q).items()}
        opp=inp['controls']+.1
        actual,_=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=0.)
        torch.testing.assert_close(actual,event_loss(model(inp,seq),seq),atol=0,rtol=0)
        actual,_=countercondition_objective(model,inp,seq,opposing_controls=opp,weight=.5,margin=.1)
        actual.backward()
        pos=naive(inp,seq);neg=naive({**inp,'controls':opp},seq)
        expected=event_loss(pos,seq)+.5*F.softplus(.1-(request_log_probability(pos,seq)-request_log_probability(neg,seq))).mean()
        expected.backward();torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        for p,n in zip(model.parameters(),naive.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,n.grad,atol=2e-6,rtol=2e-5)

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.model import features as F, sample
from beatfcn.model.batch import collate
from beatfcn.model.information_diagnostics import decisions, prefix_states, response
from beatfcn.model.p1 import P1Model
from beatfcn.train.condition_pairs import blank_context_excerpt
from beatfcn.train.control_panel import requested_view
from events_helpers import _chart, DURATION
from test_p1_model import CFG, song


class InformationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        chart = _chart('mania_4k', [Event(str(i), 8_123_000+i*250_000, 8_123_000+i*250_000,
            'tap', LaneLoc(i % 4), 'test') for i in range(8)])
        request = blank_context_excerpt(chart, 8*F.SEC, DURATION, 'audio', 2)
        self.ex = requested_view(F.build_core(request, chart.events, 8*F.SEC, DURATION, stage_a_arm='legacy'))
        torch.manual_seed(13)
        self.model = P1Model(CFG).eval().requires_grad_(False)
        self.song = song(0)

    def test_full_and_actual_incremental_prefixes_match_without_future_targets(self):
        bt = collate([self.ex], [self.song], CFG.taps, CFG.top_tap)
        h, pointer, _ = self.model.encode(bt)
        before = copy.deepcopy(self.model.state_dict()); rng = torch.get_rng_state().clone()
        for row, prefix, hp, pp in prefix_states(self.model, self.ex, self.song, [0,4]):
            self.assertEqual(int(prefix['grp_valid'].sum()), row)
            torch.testing.assert_close(hp, h[:,row:row+1], atol=2e-5, rtol=2e-5)
            a = decisions(self.model, bt, h[:,row:row+1], pointer, self.ex.groups[row], row)
            b = decisions(self.model, prefix, hp, pp, self.ex.groups[row], row)
            self.assertTrue(a['target_informed'])
            for value in response(a,b).values():
                self.assertLess(value['maximum_absolute_difference'], 2e-5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key,value in self.model.state_dict().items():
            torch.testing.assert_close(before[key],value,atol=0,rtol=0)

    def test_type_distribution_is_actual_inner_head_at_fixed_onset(self):
        row,bt,h,pa = next(prefix_states(self.model,self.ex,self.song,[0]))
        group=self.ex.groups[0]; result=decisions(self.model,bt,h,pa,group,0)
        observed=[]; original=self.model.type_h.forward
        def observe(x):
            value=original(x); observed.append(value.clone()); return value
        with patch.object(self.model.type_h,'forward',side_effect=observe):
            sample._inner(self.model,bt,h[:,0],pa,group['t_rel_us'],group['right_new'],group['busy_lanes'],
                self.song,self.ex,self.ex.c0_us+group['t_rel_us'],[],[(8*F.SEC,16*F.SEC)],
                torch.Generator().manual_seed(2),.8,.95,True,'cpu',history_row=0)
        allowed=bt['allowed_types'] & self.model.mode_types[bt['mode']]
        expected=observed[0].masked_fill(~allowed,-1e9).softmax(-1)[0].numpy()
        np.testing.assert_allclose(result['type_probability'],expected,atol=1e-7)

    def test_probability_responses_and_invalid_inputs(self):
        a={k+'_probability':[.8,.2] for k in ('time','type','continuation','lane')}
        self.assertTrue(all(v['total_variation']==0 for v in response(a,a).values()))
        b=copy.deepcopy(a);b['type_probability']=[.1,.9]
        self.assertAlmostEqual(response(a,b)['type']['total_variation'],.7)
        self.assertTrue(response(a,b)['type']['argmax_changed'])
        for bad in ([float('nan'),0],[-.1,1.1],[.1,.1],None,[1.]):
            b['type_probability']=bad
            with self.assertRaises(ValueError):response(a,b)


if __name__=='__main__':unittest.main()

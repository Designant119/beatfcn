import copy
import math
import unittest

import torch

from beatfcn.events import controls as C
from beatfcn.model import features as F
from beatfcn.model.batch import collate,pack_batch,unpack_batch
from beatfcn.train.density import FIELD_INDEX
from beatfcn.train.loop import CoreSampler
from beatfcn.train.window_controls import encode_window_controls,window_control_view
from events_helpers import synthetic_mania,synthetic_phira,DURATION
from test_p1_model import CFG,song


class WindowControlsTests(unittest.TestCase):
    def test_isolated_label_view_roundtrip_does_not_change_other_inputs(self):
        for chart in (synthetic_mania(),synthetic_phira()):
            row=dict(chart_id='c',song='a',mode=chart.mode,duration_us=DURATION)
            sampler=CoreSampler([row],lambda _:chart,lambda _:None,seed=3,stage_a_arm='legacy')
            sampler.density_window=True
            baseline=sampler.core(row,4,'from_scratch',8*F.SEC)
            expected=encode_window_controls(chart.mode,C.raw_controls(chart.mode,chart.events,[(8*F.SEC,16*F.SEC)]))
            self.assertEqual(expected[FIELD_INDEX],int(baseline.controls[FIELD_INDEX]))
            bt=collate([baseline],[song(0)],CFG.taps,CFG.top_tap)
            bt['_density_cf']=True
            bt['full_window_controls']=torch.tensor([expected])
            restored=unpack_batch(pack_batch(copy.deepcopy(bt)))
            changed=window_control_view(restored,'all_applicable')
            torch.testing.assert_close(changed['controls'],bt['full_window_controls'])
            for key in restored:
                if key!='controls':self.assertIs(changed[key],restored[key])
            torch.testing.assert_close(restored['controls'],bt['controls'])
            ordinary={**restored,'_density_cf':False}
            self.assertIs(window_control_view(ordinary,'all_applicable')['controls'],ordinary['controls'])
            self.assertIs(window_control_view(restored,'density_only')['controls'],restored['controls'])

    def test_missing_is_not_zero_and_inapplicable_fields_stay_missing(self):
        raw=C.raw_controls('mania_4k',[],[(0,8*F.SEC)])
        values=encode_window_controls('mania_4k',raw)
        self.assertEqual(values[FIELD_INDEX],0)
        self.assertEqual(values[F.CONTROL_FIELDS.index('hold_head_ratio')],-1)
        self.assertEqual(values[F.CONTROL_FIELDS.index('flick_ratio')],-1)
        raw['density_notes_per_second']=math.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):encode_window_controls('mania_4k',raw)

    def test_old_tape_or_bad_labels_fail_before_training(self):
        bt={'controls':torch.full((2,8),-1),'_density_cf':True}
        with self.assertRaisesRegex(ValueError,'labels missing'):window_control_view(bt,'all_applicable')
        bt['full_window_controls']=torch.zeros(2,7,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'incompatible'):window_control_view(bt,'all_applicable')
        with self.assertRaisesRegex(ValueError,'unknown'):window_control_view(bt,'invented')


if __name__=='__main__':unittest.main()

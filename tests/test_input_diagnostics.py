from dataclasses import replace
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from beatfcn.events.requests import build_request
from beatfcn.events.canonical import LaneLoc
from beatfcn.model.batch import SongCache
from beatfcn.model.input_diagnostics import without_recent_history, swap_reference, mismatch_audio
from beatfcn.train.data import QuantArray
from events_helpers import synthetic_mania, DURATION

class DiagnosticTests(unittest.TestCase):
    def test_history_adapters_restore_even_on_failure(self):
        model=SimpleNamespace(lane_history_y=torch.nn.Linear(3,4,bias=False),lane_history_gru=torch.nn.Linear(3,4,bias=False))
        saved=model.lane_history_y.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with without_recent_history(model):
                self.assertEqual(model.lane_history_y.weight.count_nonzero(),0)
                self.assertEqual(model.lane_history_gru.weight.count_nonzero(),0)
                raise RuntimeError('test')
        torch.testing.assert_close(model.lane_history_y.weight,saved)

    def test_swap_changes_encoded_evidence_not_truth_or_controls(self):
        chart=synthetic_mania()
        s=build_request(chart,kind='reference',seed=7,audio_id='a',audio_duration_us=DURATION,control_dropout=False)
        donor=[replace(e,location=LaneLoc((e.location.lane+1)%4)) for e in chart.events]
        b=swap_reference(s,donor)
        self.assertNotEqual(s.evidence,b.evidence)
        self.assertEqual(s.hidden,b.hidden)
        self.assertEqual(s.model_view,b.model_view)
        for k in ('regions','observed_events','controls','audio_id'):
            self.assertEqual(s.request[k],b.request[k])
        self.assertEqual(b.request['references'][0]['events'],b.evidence[0]['selected'])

    def test_audio_changes_all_channels_without_shape_or_duration_change(self):
        a=SongCache({9:QuantArray(np.zeros((12,8,3),np.int8),np.ones(3))},np.zeros((3,8,4)),np.zeros((48,128)),480000)
        b=SongCache({9:QuantArray(np.ones((10,8,3),np.int8),np.ones(3))},np.ones((2,8,4)),np.ones((40,128)),400000)
        c=mismatch_audio(a,b)
        self.assertEqual(c.duration_us,a.duration_us)
        for x,y in ((c.taps[9].raw,a.taps[9].raw),(c.summary,a.summary),(c.mel10,a.mel10)):
            self.assertEqual(x.shape,y.shape)
            self.assertTrue((x==1).all())
            self.assertTrue((y==0).all())

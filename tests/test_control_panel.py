from dataclasses import replace
import unittest

import numpy as np

from beatfcn.model import features as F
from beatfcn.train.control_panel import REQUEST_BANDS, requested_view, response_gate, field_decision, feasible_witness
from beatfcn.events import controls as C
from beatfcn.events.canonical import MODE_TYPES
from beatfcn.model.evaluation import legality_issues
from events_helpers import synthetic_mania, DURATION
from beatfcn.events.requests import build_request


class ControlPanelTests(unittest.TestCase):
    def test_user_request_does_not_reuse_hidden_chart_conditions(self):
        chart = synthetic_mania()
        sample = build_request(chart, kind='from_scratch', seed=3, audio_id='test-audio', audio_duration_us=DURATION,
                               control_dropout=False, empty_overlay=None)
        example = F.build_core(sample, chart.events, 0, DURATION, stage_a_arm='legacy')
        a = requested_view(example, 'hold_head_ratio', [.2, .3])
        b = requested_view(replace(example, controls=np.zeros_like(example.controls)),
                           'hold_head_ratio', [.2, .3])
        np.testing.assert_equal(a.controls, b.controls)
        self.assertEqual(int((a.controls >= 0).sum()), 1)
        self.assertTrue((requested_view(example).controls == -1).all())
        for key, value in vars(example).items():
            if key != 'controls':
                self.assertIs(getattr(a, key), value)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            requested_view(example, 'drag_ratio', [.2, .3])

    def test_flat_or_invalid_generation_cannot_pass(self):
        field = 'density_notes_per_second'
        rows = [dict(requested=band, achieved={field: value}, complete=True, legal=True)
                for band, value in zip(REQUEST_BANDS[field], [1.25, 3.5, 7.])]
        self.assertTrue(response_gate(rows, field)['passed'])
        self.assertFalse(response_gate(rows[:2], field)['passed'])
        rows[1]['achieved'][field] = 1.25
        self.assertFalse(response_gate(rows, field)['passed'])
        rows[1]['achieved'][field] = 3.5
        rows[1]['complete'] = False
        self.assertFalse(response_gate(rows, field)['passed'])

    def test_futility_keeps_untested_cases_distinct_from_failures(self):
        decision = field_decision([{'passed': False}, {'passed': False}])
        self.assertEqual(decision, dict(tested=2, successes=0, remaining=2, failed=True, passed=False))
        self.assertFalse(field_decision([{'passed': True}, {'passed': False}])['failed'])
        self.assertTrue(field_decision([{'passed': True}] * 3 + [{'passed': False}])['passed'])

    def test_every_requested_band_has_a_legal_blank_canvas_witness(self):
        for mode, fields in C.MODE_FIELDS.items():
            for field in fields:
                for level, band in enumerate(REQUEST_BANDS[field]):
                    with self.subTest(mode=mode, field=field, band=band):
                        events = feasible_witness(mode, field, level)
                        request = dict(mode=mode, audio_duration_us=F.CORE_US, observed_events=[],
                            controls={'allowed_types': list(MODE_TYPES[mode]), field: band},
                            regions=[dict(region_id='core', start_us=0, end_us=F.CORE_US, state='editable')])
                        self.assertEqual(legality_issues(request, events, placement=False), [])
                        self.assertEqual(C.band_of(field, C.raw_controls(mode, events, [(0, F.CORE_US)])[field]), band)


if __name__ == '__main__':
    unittest.main()

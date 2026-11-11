import copy
import unittest

from beatfcn.events.controls import MODE_FIELDS
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.train.condition_audit import inspect_sample, summarize
from beatfcn.train.density import complete_window
from events_helpers import synthetic_mania, synthetic_phira, DURATION


class ConditionAuditTests(unittest.TestCase):
    def test_scope_difference_is_not_reported_as_wrong_label(self):
        chart = synthetic_mania()
        s = build_request(chart, kind='from_scratch', seed=4, audio_id='a', audio_duration_us=DURATION,
                          control_dropout=False)
        before = copy.deepcopy(s.request)
        r = inspect_sample(s, 8*F.SEC, DURATION)
        self.assertFalse(r['exact_scope'])
        self.assertTrue(all(not f['label_inconsistent'] for f in r['fields'].values()))
        self.assertEqual(s.request, before)
        s.request['controls']['hold_head_ratio'] = [9., 10.]
        self.assertTrue(inspect_sample(s, 8*F.SEC, DURATION)['fields']['hold_head_ratio']['label_inconsistent'])

    def test_density_window_only_supplies_density_and_keeps_other_fields_missing(self):
        for chart in (synthetic_mania(), synthetic_phira()):
            s = build_request(chart, kind='from_scratch', seed=4, audio_id='a', audio_duration_us=DURATION)
            s = complete_window(s, chart, 8*F.SEC, DURATION, 4)
            r = inspect_sample(s, 8*F.SEC, DURATION)
            r['variant'] = 'density_window'
            self.assertTrue(r['exact_scope'])
            supplied = [k for k, v in r['fields'].items() if v['requested_band'] is not None]
            self.assertEqual(supplied, ['density_notes_per_second'])
            self.assertEqual(r['fields']['density_notes_per_second']['local_band_differs'], False)
            cell, = summarize([r]).values()
            self.assertEqual(cell['fields']['hold_head_ratio']['missing'], 1)
            self.assertEqual(set(r['fields']), set(MODE_FIELDS[chart.mode]))  # 字段按模式裁剪


if __name__ == '__main__':
    unittest.main()

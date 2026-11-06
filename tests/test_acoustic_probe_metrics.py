import unittest
from scripts.acoustic_probe_metrics import event_metrics, physical_events, in_interval


class AcousticMetricTests(unittest.TestCase):
    def event(self, head=1_000_000, tail=None, kind='tap', lane=0):
        return dict(onset_us=head, end_us=head if tail is None else tail, type=kind, lane=lane)

    def test_wrong_lane_is_one_false_positive_and_one_false_negative(self):
        r = event_metrics([self.event(lane=1)], [self.event()], 'mania_4k')
        self.assertEqual(r['matched'], 0)
        self.assertEqual(r['full_event_false_positive'], 1)
        self.assertEqual(r['full_event_false_negative'], 1)

    def test_multiplicity_and_complete_tail_are_scored(self):
        e = self.event(kind='drag', lane=-1)
        r = event_metrics([e], [e, e], 'phira')
        self.assertEqual(r['matched'], 1)
        self.assertEqual(r['exact_onset_group_recall'], 0)
        a = self.event(tail=3_000_000, kind='hold')
        b = self.event(tail=3_100_000, kind='hold')
        self.assertEqual(event_metrics([a], [b], 'mania_4k')['matched'], 0)

    def test_physical_interval_keeps_full_crossing_tail(self):
        e = self.event(head=15_000_000, tail=20_000_000, kind='hold')
        x = physical_events([e], dict(source_clip_start_us=2_000_000))
        selected = in_interval(x, [10_000_000, 18_000_000])
        self.assertEqual(selected[0]['end_us'], 22_000_000)
        self.assertEqual(e['end_us'], 20_000_000)


if __name__ == '__main__':
    unittest.main()

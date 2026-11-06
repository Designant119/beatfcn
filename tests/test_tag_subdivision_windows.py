import unittest

import numpy as np

from scripts.tag_subdivision_windows import subdivision, tag, window_key

BEAT = 500_000  # 120 BPM
SEG = [[0.0, float(BEAT), 4]]


class FakePanel:
    def __init__(self, charts):
        self.charts = charts
        self.records = []
        for ci, (mode, onsets) in enumerate(charts):
            for start in (0, 8_000_000):
                self.records.append(dict(split='train', chart_index=ci, chart_id=f'c{ci}', mode=mode,
                                         request=dict(start_us=start, end_us=start + 8_000_000)))
        self.records.append(dict(split='val', chart_index=0, chart_id='c0', mode=charts[0][0],
                                 request=dict(start_us=0, end_us=8_000_000)))

    def _arrays(self, ci):
        return None, dict(onset_us=np.asarray(self.charts[ci][1], np.int64))


class SubdivisionTagTest(unittest.TestCase):
    def test_classes(self):
        starts, beats = np.array([0.]), np.array([float(BEAT)])
        self.assertEqual(subdivision(1_000_000, starts, beats), '1/1')
        self.assertEqual(subdivision(1_250_000, starts, beats), '1/2')
        self.assertEqual(subdivision(1_125_000, starts, beats), '1/4')
        self.assertEqual(subdivision(1_000_000 + BEAT // 3, starts, beats), 'triplet')
        self.assertEqual(subdivision(1_062_500, starts, beats), '1/8')
        self.assertEqual(subdivision(1_031_000, starts, beats), 'other')

    def test_window_keys_and_train_only(self):
        trip = [int(b * BEAT + k * BEAT / 3) for b in range(4) for k in range(3)]
        fine = [int(b * BEAT + k * BEAT / 4) for b in range(4) for k in range(4)]
        plain = [b * BEAT for b in range(8)]
        self.assertEqual(window_key('phira', np.array(trip), SEG)[0], 'phira:triplet')
        self.assertEqual(window_key('phira', np.array(fine), SEG)[0], 'phira:fine')
        self.assertEqual(window_key('phira', np.array(plain), SEG)[0], 'phira:plain')
        self.assertEqual(window_key('phira', np.array(trip), None)[0], 'phira:plain')
        panel = FakePanel([('phira', trip), ('mania_4k', [8_000_000 + x for x in fine])])
        timing = {'c0': dict(valid=True, segments=SEG), 'c1': dict(valid=True, segments=SEG)}
        tags, totals = tag(panel, timing)
        self.assertEqual(len(tags), 4)  # val record excluded
        self.assertEqual(tags[0], 'phira:triplet')
        self.assertEqual(tags[1], 'phira:plain')      # second window of chart 0 is empty
        self.assertEqual(tags[3], 'mania_4k:fine')
        self.assertEqual(sum(totals.values()), 4)


if __name__ == '__main__':
    unittest.main()

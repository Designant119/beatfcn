import unittest

from beatfcn.events.phira_layout import LEFT, MIN_SEP, RIGHT, layout_report, two_thumb_layout


def ev(t_ms, typ='tap', end_ms=None):
    t = int(t_ms * 1000)
    return dict(onset_us=t, end_us=int((end_ms if end_ms is not None else t_ms) * 1000), type=typ, lane=-1)


class PhiraLayoutTest(unittest.TestCase):
    def test_chords_split_hands_and_everything_on_screen(self):
        events = [ev(0), ev(0), ev(500), ev(625), ev(750), ev(875), ev(1000), ev(1000), ev(2000, 'hold', 2600),
                  ev(2100), ev(2200), ev(2300, 'drag'), ev(2350, 'drag'), ev(2400, 'drag'), ev(3500, 'flick')]
        lay = two_thumb_layout(events)
        self.assertEqual(len(lay), len(events))
        self.assertEqual(sorted((r['t_us'], r['end_us'], r['type']) for r in lay),
                         sorted((e['onset_us'], e['end_us'], e['type']) for e in events))
        for r in lay:
            self.assertTrue(LEFT[0] <= r['x'] <= RIGHT[1])
        chord = [r['x'] for r in lay if r['t_us'] == 0]
        self.assertTrue(min(chord) < 0 < max(chord))
        self.assertGreaterEqual(abs(chord[0] - chord[1]), MIN_SEP - 1)
        hold = next(r for r in lay if r['type'] == 'hold')
        during = [r for r in lay if hold['t_us'] < r['t_us'] < hold['end_us']]
        self.assertTrue(all((r['x'] < 0) != (hold['x'] < 0) for r in during))  # the other hand plays them

    def test_stream_alternates_and_report(self):
        events = [ev(i * 120) for i in range(16)]
        lay = two_thumb_layout(events)
        sides = [r['x'] < 0 for r in sorted(lay, key=lambda r: r['t_us'])]
        self.assertTrue(all(a != b for a, b in zip(sides, sides[1:])))
        rep = layout_report(lay)
        self.assertEqual(rep['near_too_close'], 0)
        self.assertAlmostEqual(rep['left_share'], 0.5)

    def test_deterministic(self):
        events = [ev(i * 300) for i in range(30)]
        self.assertEqual(two_thumb_layout(events, seed=3), two_thumb_layout(events, seed=3))


if __name__ == '__main__':
    unittest.main()

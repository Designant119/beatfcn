import unittest

from beatfcn.events.difficulty_presets import DifficultyPresets, mania_star, monotone


def stream(nps, seconds=60, lanes=4):
    step = int(1e6 / nps)
    return [(t, t, 0, i % lanes) for i, t in enumerate(range(0, seconds * 1_000_000, step))]


class DifficultyPresetsTest(unittest.TestCase):
    def test_star_grows_with_density(self):
        stars = [mania_star(stream(n)) for n in (2, 4, 8, 16)]
        self.assertEqual(stars, sorted(stars))
        self.assertGreater(stars[-1], 2 * stars[0])

    def test_monotone(self):
        x, y = monotone([1, 2, 3, 4], [1.0, 3.0, 2.0, 4.0], [1, 1, 1, 1])
        self.assertEqual(x, [1, 2, 3, 4])
        self.assertEqual(y, [1.0, 2.5, 2.5, 4.0])

    def test_bundled_table_monotone_in_range(self):
        d = DifficultyPresets.bundled()
        phira = [d.heads_per_s('phira', lv) for lv in range(8, 17)]
        mania = [d.heads_per_s('mania_4k', s / 2) for s in range(2, 15)]
        self.assertEqual(phira, sorted(phira))
        self.assertEqual(mania, sorted(mania))
        self.assertTrue(1.0 < phira[0] < phira[-1] < 8.0)


if __name__ == '__main__':
    unittest.main()

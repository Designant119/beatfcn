from collections import Counter
import unittest

from beatfcn.train.local_sampling import LocalRequestSampler


class LocalRequestSamplerTests(unittest.TestCase):
    def records(self):
        return [dict(mode=mode, component=f'{mode}:{song}', chart_id=f'{mode}:{song}:{chart}',
                     split=split, stratum='test')
                for mode in ('phira', 'mania_4k') for split in ('train', 'val')
                for song, variants, windows in [('short', 1, 1), ('long', 5, 20)]
                for chart in range(variants) for _ in range(windows)]

    def test_song_exposure_is_independent_of_length_and_variant_count(self):
        rows = self.records()
        sampler = LocalRequestSampler(rows, seed=1)
        seen = [i for _ in range(100) for i in sampler.next_batch()]
        self.assertTrue(all(rows[i]['split'] == 'train' for i in seen))
        self.assertEqual(Counter(rows[i]['mode'] for i in seen), {'phira': 300, 'mania_4k': 100})
        self.assertEqual(sampler.exposure['song'],
            {'phira:short': 150, 'phira:long': 150, 'mania_4k:short': 50, 'mania_4k:long': 50})
        self.assertEqual([sampler.exposure['chart'][f'phira:long:{i}'] for i in range(5)], [30]*5)

    def test_resume_replays_the_same_next_requests(self):
        rows = self.records()
        a, b = LocalRequestSampler(rows, seed=9), LocalRequestSampler(rows, seed=9)
        for _ in range(7):
            a.next_batch()
        b.load_state_dict(a.state_dict())
        self.assertEqual([a.next_batch() for _ in range(9)], [b.next_batch() for _ in range(9)])

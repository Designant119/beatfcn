import unittest

from beatfcn.train.local_cohort import connected_songs,estimate_feature_bytes,stratified_song_order


class LocalCohortTests(unittest.TestCase):
    def test_alias_chain_exposes_leakage_across_groups_and_audio(self):
        rows=[dict(chart_id='a',audio_sha256='x',song_group='g1',split='train'),
              dict(chart_id='b',audio_sha256='x',song_group='g2',split='train'),
              dict(chart_id='c',audio_sha256='y',song_group='g2',split='val')]
        identity,splits=connected_songs(rows)
        self.assertEqual(len(set(identity.values())),1)
        self.assertEqual(splits[identity['a']],{'train','val'})

    def test_nested_prefixes_preserve_rare_strata_without_duplicate_songs(self):
        rows=[dict(chart_id=str(i),component=str(i//2),stratum='rare' if i==1 else 'common') for i in range(20)]
        order=stratified_song_order(rows,seed=3,used={'9'})
        self.assertEqual(order,stratified_song_order(list(reversed(rows)),seed=3,used={'9'}))
        self.assertEqual(len(order),9)
        self.assertEqual(len({r['component'] for r in order}),9)
        self.assertTrue(any(r['stratum']=='rare' for r in order[:2]))
        self.assertTrue({r['component'] for r in order[:3]} <= {r['component'] for r in order[:8]})

    def test_estimate_keeps_full_band_and_layer_inputs(self):
        self.assertEqual(estimate_feature_bytes(1.),3*(128+25*8*1024)+128+100*128*2)


if __name__=='__main__':unittest.main()

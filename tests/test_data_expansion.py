import copy
import unittest

from beatfcn.ingest.expansion import (declared_burden, incremental_splits, phira_acquisition_eligible,
                                    phira_evidence, select_phira, osu_metadata_fields)


def chart(cid, name, level='EZ Lv.3', difficulty=3, votes=3, rating=.85):
    return dict(id=cid, name=name, level=level, difficulty=difficulty, rating=rating,
                ratingCount=votes, reviewed=True, tags=['plain'], file=f'https://example.org/{cid}')


class Expansion(unittest.TestCase):
    def test_native_osu_metadata_whitespace_does_not_reject_valid_mania(self):
        fields = osu_metadata_fields('osu file format v14\r\n[General]\r\nAudioFilename: audio.mp3\r\n'
                                     'Mode: 3\r\n[Metadata]\r\nBeatmapID:5149987\r\n'
                                     'BeatmapSetID: 2382059\r\n[Difficulty]\r\nCircleSize:4\r\n')
        self.assertEqual(fields, dict(AudioFilename='audio.mp3', Mode='3', BeatmapID='5149987',
                                      BeatmapSetID='2382059', CircleSize='4'))

    def test_low_difficulty_and_low_popularity_are_not_invalidity(self):
        row = chart(1, 'Song')
        self.assertTrue(phira_acquisition_eligible(row))
        self.assertFalse(phira_evidence(row)['legacy_rating_gate'])
        self.assertFalse(phira_evidence(row)['musical_quality_verified'])
        self.assertEqual(phira_evidence(row)['confidence'], 'limited')

    def test_unreviewed_or_performance_remains_separate(self):
        self.assertFalse(phira_acquisition_eligible(dict(chart(1, 'Song'), reviewed=False)))
        self.assertFalse(phira_acquisition_eligible(dict(chart(1, 'Song'), tags=['visual'])))

    def test_mislabeled_level_does_not_fake_low_difficulty(self):
        self.assertEqual(declared_burden(chart(1, 'Song', 'IN Lv.15.3', 3)), ('10+', True))

    def test_does_not_prefer_harder_or_more_popular_chart(self):
        rows = [chart(1, 'Song', 'AT Lv.16', 16, 10000, .99), chart(2, 'Song', votes=50)]
        self.assertEqual(select_phira(rows, [], 1)[0]['id'], 2)
        self.assertEqual(select_phira(rows, ['1'], 1)[0]['id'], 2)
        self.assertEqual(select_phira(rows, [], 2), select_phira(list(reversed(rows)), [], 2))

    def test_training_acquisition_respects_current_rating_and_vote_gate(self):
        rows = [chart(1, 'Below votes', votes=49, rating=.99),
                chart(2, 'Boundary', votes=50, rating=.75),
                chart(3, 'Below score', votes=1000, rating=.749)]
        self.assertEqual([r['id'] for r in select_phira(rows, [], 10)], [2])

    def test_preserves_frozen_validation_membership(self):
        old = [dict(chart_id='osu:1', source='osu', title='Song', audio_sha256='a',
                    split='val', song_group='frozen-group')]
        original = copy.deepcopy(old)
        new = [dict(chart_id='phira:2', source='phira', title='Song (TV size)', audio_sha256='b')]
        result = incremental_splits(old, new)[0]
        self.assertEqual((result['split'], result['song_group']), ('val', 'frozen-group'))
        self.assertEqual(old, original)

    def test_cross_split_bridge_is_not_admitted(self):
        old = [dict(chart_id='osu:1', title='One', audio_sha256='a', split='train', song_group='g1'),
               dict(chart_id='osu:2', title='Two', audio_sha256='b', split='val', song_group='g2')]
        new = [dict(chart_id='phira:3', title='One', title_alt='Two', audio_sha256='c')]
        result = incremental_splits(old, new)[0]
        self.assertTrue(result['split_conflict'])
        self.assertIsNone(result['split'])

    def test_name_only_new_source_inherits_old_heldout_song_with_different_audio(self):
        old = [dict(chart_id='osu:1', title='Same Song', audio_sha256='mp3',
                    split='test', song_group='frozen-test')]
        original = copy.deepcopy(old)
        new = [dict(chart_id='phigros_resource:2', name='Same Song (TV size)', audio_sha256='vorbis')]
        result = incremental_splits(old, new)[0]
        self.assertEqual((result['split'], result['song_group']), ('test', 'frozen-test'))
        self.assertEqual(result['split_evidence']['existing_chart_ids'], ['osu:1'])
        self.assertEqual(old, original)

    def test_name_and_alt_title_bridge_does_not_leak_old_split(self):
        old = [dict(chart_id='osu:1', title='One', audio_sha256='a', split='train', song_group='g1'),
               dict(chart_id='phira:2', title='Two', audio_sha256='b', split='val', song_group='g2')]
        new = [dict(chart_id='phigros_resource:3', name='One', title_alt='Two', audio_sha256='c')]
        result = incremental_splits(old, new)[0]
        self.assertTrue(result['split_conflict'])
        self.assertIsNone(result['split'])


if __name__ == '__main__':
    unittest.main()

import copy
import unittest

from scripts.audit_phigros_overlap import linked, signature


def doc():
    return dict(mode='phira', source_format='phira_official_v3', format_version='3', events=[
        dict(event_id='a', onset_us=1_000_000, end_us=2_000_000, type='hold',
             location=dict(kind='line', line_id='0', side='above', native_x=0.), native_ref='chart'),
        dict(event_id='b', onset_us=3_000_000, end_us=3_000_000, type='drag',
             location=dict(kind='line', line_id='1', side='above', native_x=1.), native_ref='chart')])


class OverlapIdentityTest(unittest.TestCase):
    def test_global_shift_is_a_review_candidate_not_an_exact_target(self):
        original = doc()
        shifted = copy.deepcopy(original)
        for event in shifted['events']:
            event['onset_us'] += 80_000
            event['end_us'] += 80_000
        a, b = signature(original), signature(shifted)
        self.assertEqual(a['relative'], b['relative'])
        self.assertNotEqual(a['absolute'], b['absolute'])
        self.assertEqual(b['first_onset_us']-a['first_onset_us'], 80_000)

    def test_stacked_drag_multiplicity_is_part_of_both_identities(self):
        original = doc()
        stacked = copy.deepcopy(original)
        stacked['events'].append(dict(stacked['events'][-1], event_id='c'))
        a, b = signature(original), signature(stacked)
        self.assertEqual((a['status'], b['status']), ('ok', 'ok'))
        self.assertNotEqual(a['absolute'], b['absolute'])
        self.assertNotEqual(a['relative'], b['relative'])

    def test_unrelated_titles_are_not_linked_by_missing_metadata(self):
        self.assertFalse(linked({'audio_sha256':'a'}, {'audio_sha256':'b'}))
        self.assertTrue(linked({'audio_sha256':'a','name':'A (TV size)'},
                               {'audio_sha256':'b','title':'A'}))

    def test_mania_cannot_use_a_signature_that_discards_lanes(self):
        with self.assertRaisesRegex(ValueError, 'must not discard mania lanes'):
            signature(dict(doc(), mode='mania_4k'))


if __name__ == '__main__':
    unittest.main()

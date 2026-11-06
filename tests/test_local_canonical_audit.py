from copy import deepcopy
from dataclasses import replace
import unittest

from beatfcn.events.canonical import Event, LaneLoc, LineLoc
from beatfcn.train.local_canonical_audit import audit_document, rhythm_fingerprint


class LocalCanonicalAuditTests(unittest.TestCase):
    def fixture(self):
        event = Event('a', 1_000_000, 1_000_000, 'drag', LineLoc('0', 'above', 0.))
        row = dict(chart_id='phira:1', source='phira', native_id='1', mode='phira',
                   audio_sha256='b'*64, audio_sec=10., split='train',
                   offset_decision=dict(decision_version=2, shift_ms=0, c_ms=0))
        doc = dict(row, schema='beatfcn.canon.v1', source_format='phira_rpe',
                   source_sha256='a'*64, offset_sec='0',
                   timing=dict(offset_sec='0', segments=[dict(start_beat='0', bpm='60', start_sec='0')]),
                   native_beats={'a': ['1', '1']}, events=[event.to_json()], fakes=[])
        return doc, row

    def test_consistency_does_not_grant_native_or_clock_admission(self):
        doc, row = self.fixture()
        result, events = audit_document(doc, row)
        self.assertTrue(result['canonical_integrity_passed'])
        self.assertEqual(len(events), 1)
        self.assertEqual(result['playable_types'], {'drag': 1})
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['audio_clock_verified'])

    def test_playable_fake_and_shared_event_id_fail_closed(self):
        doc, row = self.fixture()
        doc['events'][0]['fake'] = True
        with self.assertRaisesRegex(ValueError, 'non-playable'):
            audit_document(doc, row)
        doc['events'][0]['fake'] = False
        fake = deepcopy(doc['events'][0]); fake['fake'] = True; doc['fakes'] = [fake]
        with self.assertRaisesRegex(ValueError, 'Duplicate event identity'):
            audit_document(doc, row)

    def test_frozen_shift_bounds_are_checked_without_clipping(self):
        doc, row = self.fixture()
        row['offset_decision']['shift_ms'] = -1001
        result, events = audit_document(doc, row)
        self.assertFalse(result['canonical_integrity_passed'])
        self.assertEqual(result['out_of_bounds_count'], 1)
        self.assertEqual(events, ())
        self.assertEqual(doc['events'][0]['onset_us'], 1_000_000)

    def test_internal_beat_disagreement_is_not_audio_alignment_evidence(self):
        doc, row = self.fixture()
        doc['native_beats']['a'] = ['2', '2']
        result, events = audit_document(doc, row)
        self.assertEqual(result['internal_clock']['max_error_us'], 1_000_000)
        self.assertFalse(result['canonical_integrity_passed'])
        self.assertEqual(events, ())

    def test_segment_precision_matches_integer_microsecond_storage(self):
        doc, row = self.fixture()
        segment = dict(start_beat='2', bpm='60', start_sec='2000000000000001/1000000000000000')
        doc['timing']['segments'].append(segment)
        result, _ = audit_document(doc, row)
        self.assertTrue(result['canonical_integrity_passed'])
        self.assertLess(result['internal_clock']['max_segment_residual_us'], 1.)
        segment['start_sec'] = '2001/1000'
        result, events = audit_document(doc, row)
        self.assertFalse(result['canonical_integrity_passed'])
        self.assertEqual(events, ())

    def test_rhythm_identity_ignores_geometry_but_keeps_multiplicity_and_lanes(self):
        a = Event('a', 1000, 1000, 'drag', LineLoc('0', 'above', 0.))
        b = replace(a, event_id='b', location=LineLoc('8', 'below', 500.))
        self.assertEqual(rhythm_fingerprint('phira', [a]), rhythm_fingerprint('phira', [b]))
        self.assertNotEqual(rhythm_fingerprint('phira', [a]), rhythm_fingerprint('phira', [a, b]))
        a = replace(a, type='tap', location=LaneLoc(0))
        b = replace(a, location=LaneLoc(1))
        self.assertNotEqual(rhythm_fingerprint('mania_4k', [a]), rhythm_fingerprint('mania_4k', [b]))


if __name__ == '__main__':
    unittest.main()

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from beatfcn.model.local_song_rollout import generate_complete_songs


class WholeSongTests(unittest.TestCase):
    def inputs(self, features, audio, request, controls, *, sources=()):
        self.references.append(sources)
        return dict(local=np.zeros((3, 1, 8, 1024), np.float32), mode=np.asarray(1),
                    controls=controls, reference=np.zeros((1, 16), np.float32)), []

    def test_own_hold_crosses_core_and_partial_final_core_is_generated(self):
        self.references = []
        requests = []
        def predict(model, inputs, reqs, cap, **kw):
            requests.extend(reqs)
            return [dict(complete=True, tokens=[], events=[dict(onset_us=7_000_000, end_us=10_000_000,
                type='hold', lane=-1)] if r.start_us == 0 else []) for r in reqs]
        song = dict(chart_id='test', audio_sha256='a'*64, mode='phira', duration_us=17_123_000)
        with patch('beatfcn.model.local_song_rollout.request_inputs', self.inputs):
            result = generate_complete_songs(SimpleNamespace(training=False), None, [song], device='cpu', generator=predict)
        self.assertTrue(result['complete'])
        self.assertEqual([r.start_us for r in requests], [0, 8_000_000, 16_000_000])
        self.assertEqual(requests[-1].end_us, 17_123_000)
        self.assertEqual(requests[1].known_carry[0].end_us, 10_000_000)
        self.assertFalse(requests[2].known_carry)
        self.assertTrue(all(not r for r in self.references))

    def test_incomplete_core_is_never_skipped_or_called_a_complete_song(self):
        self.references = []
        def predict(model, inputs, reqs, cap, **kw):
            return [dict(complete=False, tokens=[], events=[], token_limit_reached=True) for r in reqs]
        song = dict(chart_id='test', audio_sha256='a'*64, mode='phira', duration_us=17_000_000)
        with patch('beatfcn.model.local_song_rollout.request_inputs', self.inputs):
            result = generate_complete_songs(SimpleNamespace(training=False), None, [song], device='cpu', generator=predict)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['songs'][0]['cores']), 1)
        self.assertEqual(result['songs'][0]['failure'], 'core_token_limit')
        with self.assertRaisesRegex(ValueError, 'hidden'):
            generate_complete_songs(SimpleNamespace(training=False), None, [dict(song, events=[])], device='cpu', generator=predict)

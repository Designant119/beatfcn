from copy import deepcopy
import unittest

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from scripts.audit_local_candidate_conditions import request_legality, summarize, validate_checkpoint


class CandidateConditionAuditTest(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(fields=[dict(name='density', minimum_delta=2.), dict(name='hold', minimum_delta=.15)],
            songs=[dict(audio_sha256=str(i)) for i in range(4)],
            combined_gate=dict(minimum_directed_pairs=6, minimum_in_band_outputs=8))
        self.rows = [dict(field=f['name'], audio_sha256=s['audio_sha256'], complete=True,
            identical_token_stream=False, direction_delta=f['minimum_delta'], in_band=[True, False],
            independent_legality=[[], []]) for f in self.protocol['fields'] for s in self.protocol['songs']]

    def test_combined_and_historical_field_gate_stay_distinct(self):
        result = summarize(self.rows, self.protocol)
        self.assertTrue(result['complete'])
        self.assertTrue(result['combined_gate']['passed'])
        self.assertEqual(result['combined_gate']['in_band_outputs'], 8)
        self.assertFalse(any(r['sentinel_passed'] for r in result['by_field'].values()))

    def test_identical_missing_undefined_or_illegal_output_never_passes(self):
        for mutation in ('identical', 'undefined', 'illegal', 'incomplete', 'duplicate'):
            rows = deepcopy(self.rows)
            if mutation == 'identical':
                for r in rows:
                    r['identical_token_stream'] = True
            elif mutation == 'undefined':
                for r in rows:
                    r['direction_delta'] = None
                    r['in_band'] = [False, False]
            elif mutation == 'illegal':
                rows[0]['independent_legality'][0] = ['outside request']
            elif mutation == 'incomplete':
                rows[0]['complete'] = False
            else:
                rows[-1] = rows[-2]
            with self.subTest(mutation=mutation):
                self.assertFalse(summarize(rows, self.protocol)['combined_gate']['passed'])
        self.assertFalse(summarize(self.rows[:-1], self.protocol)['complete'])

    def test_no_audit_of_running_or_different_candidate(self):
        training = dict(config={'d':128}, model_module='model', key_module='key', normalizer_module='transform', normalizer='stats')
        saved = dict(updates=1000, config=training['config'], plan_sha256='plan',
            candidate_components={k:v for k,v in training.items() if k != 'config'})
        report = dict(complete=True, updates=1000, plan_sha256='plan', checkpoint_sha256='ckpt')
        validate_checkpoint(saved, training, 'plan', report, 'ckpt')
        for change in [dict(complete=False),dict(error='failure'),dict(updates=999),dict(checkpoint_sha256='old')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_checkpoint(saved, training, 'plan', {**report, **change}, 'ckpt')
        with self.assertRaises(ValueError):
            validate_checkpoint({**saved, 'candidate_components':{}}, training, 'plan', report, 'ckpt')

    def test_song_legal_but_out_of_core_heads_fail_and_outgoing_holds_remain_legal(self):
        request = EventRequest('phira', 8_000_000, 16_000_000, 24_000_000)
        self.assertEqual(request_legality([RhythmEvent(10_000_000,20_000_000,'hold')], request), [])
        for head in (7_999_999,16_000_000):
            issues = request_legality([RhythmEvent(head,head,'tap')], request)
            self.assertEqual(issues[0]['code'], 'outside_requested_head_scope')

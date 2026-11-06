import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.events.requests import build_request
from beatfcn.model import features as F, sample
from beatfcn.model.p1 import P1Model
from scripts.observe_generation import observe_generation, pointer_summary
from events_helpers import synthetic_mania, DURATION
from test_p1_model import CFG, song


class ObserverTests(unittest.TestCase):
    def test_individual_rank_is_distinct_from_total_probability(self):
        logits = torch.tensor([[[0., -.1, -.1, -.1, 100.]]])
        mask = torch.tensor([[[True, True, True, True, False]]])
        record = pointer_summary(logits, mask, 0)
        self.assertTrue(record['greedy_end'])
        self.assertLess(record['end_probability'], .5)
        self.assertTrue(record['end_beats_individual_but_not_total_note_mass'])
        self.assertEqual(record['legal_note_choices'], 3)
        self.assertAlmostEqual(record['end_probability'] + record['any_note_probability'], 1., places=6)

    def test_actual_generation_and_rng_are_unchanged_and_patch_restored(self):
        chart = synthetic_mania(7)
        request = build_request(chart, kind='from_scratch', seed=7, audio_id='audio',
                                audio_duration_us=DURATION, empty_overlay=None)
        ex = F.build_core(request, chart.events, 8 * F.SEC, DURATION)
        holds, runs, heads = sample.core_context(request, ex, {})
        torch.manual_seed(8); model = P1Model(CFG).eval(); audio = song(0)
        original = sample.generate_core
        common = dict(greedy=True, seed=42)
        first = {}; second = {}
        with patch.object(F, 'MAX_GROUPS', 8):
            expected = original(model, ex, audio, holds, runs, heads, diagnostics=first, **common)
            rng = torch.get_rng_state().clone()
            with observe_generation() as records:
                actual = sample.generate_core(model, ex, audio, holds, runs, heads, diagnostics=second, **common)
            np.testing.assert_equal(expected, actual)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        observed = second.pop('time_pointer_observation')
        self.assertEqual(first, second)
        self.assertGreater(observed['calls'], 0)
        self.assertFalse(observed['output_rule_changed'])
        self.assertEqual(len(records), 1)
        self.assertIs(sample.generate_core, original)
        self.assertNotIn('time_logits', model.__dict__)

    def test_restore_on_generation_error(self):
        model = P1Model(CFG).eval()
        original = sample.generate_core
        with patch.object(sample, 'generate_core', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with observe_generation():
                    sample.generate_core(model)
        self.assertIs(sample.generate_core, original)
        self.assertNotIn('time_logits', model.__dict__)


if __name__ == '__main__':
    unittest.main()

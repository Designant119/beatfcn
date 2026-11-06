import copy
from dataclasses import replace
import unittest

import numpy as np
import torch

from beatfcn.events import controls as C
from beatfcn.events.canonical import Event
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.batch import collate
from beatfcn.model.p1 import P1Model
from beatfcn.train.density import complete_window, FIELD, FIELD_INDEX, ranking_loss, rng_state, replay_rng
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_p1_model import CFG, song


def window_batch():
    examples, samples = [], []
    for chart in (synthetic_mania(), synthetic_phira()):
        sample = build_request(chart, kind='from_scratch', seed=4, audio_id='a', audio_duration_us=DURATION)
        sample = complete_window(sample, chart, 8 * F.SEC, DURATION, seed=4)
        examples.append(F.build_core(sample, chart.events, 8 * F.SEC, DURATION, stage_a_arm='legacy'))
        samples.append(sample)
    bt = collate(examples, [song(0), song(1)], CFG.taps, CFG.top_tap)
    return bt, samples


class DensityTrainingTests(unittest.TestCase):
    def test_sampler_marks_complete_windows_and_retains_input_policy_provenance(self):
        from beatfcn.train.loop import CoreSampler
        chart = synthetic_mania()
        row = {'chart_id': 'c', 'song': 'a', 'mode': chart.mode, 'duration_us': DURATION}
        sampler = CoreSampler([row], lambda _: chart, lambda _: None, seed=3, stage_a_arm='legacy')
        sampler.density_window = True
        ex = sampler.core(row, 4, 'from_scratch', 8 * F.SEC)
        self.assertEqual(ex.provenance['kind'], 'density_window')
        self.assertEqual(ex.provenance['density_scope_us'], [8 * F.SEC, 16 * F.SEC])
        self.assertEqual(ex.provenance['context_policy'], 'authorized_rhythm_prefix_v1')
        self.assertNotEqual(ex.controls[FIELD_INDEX], ex.provenance['density_wrong_band'])

    def test_full_window_density_and_boundary_holds_are_not_thinned(self):
        _, samples = window_batch()
        for chart, sample in zip((synthetic_mania(), synthetic_phira()), samples):
            obs = [Event.from_json(e) for e in sample.request['observed_events']]
            self.assertEqual({e.event_id for e in obs + list(sample.hidden)}, {e.event_id for e in chart.events})
            editable = [(r['start_us'], r['end_us']) for r in sample.request['regions'] if r['state'] == 'editable']
            self.assertEqual(editable, [(8 * F.SEC, 16 * F.SEC)])
            raw = C.raw_controls(chart.mode, obs + list(sample.hidden), editable)[FIELD]
            self.assertEqual(C.band_of(FIELD, raw), sample.request['controls'][FIELD])
            self.assertTrue(all(8 * F.SEC <= e.onset_us < 16 * F.SEC and e.occupancy()[1] <= 16 * F.SEC
                                for e in sample.hidden))
            edges = C.BAND_EDGES[FIELD]
            wrong = sample.recipe['density_wrong_band']
            self.assertFalse(edges[wrong] <= raw < edges[wrong + 1])
            self.assertEqual(sample.evidence, ())

    def test_condition_scores_preserve_loss_and_compact_dense_equivalence(self):
        bt, _ = window_batch()
        torch.manual_seed(3)
        model = P1Model(CFG).eval()
        compact = P1Model(replace(CFG, compact_inner=True)).eval()
        compact.load_state_dict(model.state_dict())
        terms = {}
        loss, _ = model(bt, density_terms=terms)
        original, _ = model(bt)
        torch.testing.assert_close(loss, original)
        other = {}
        compact(bt, density_terms=other)
        for key in ('time', 'cont'):
            self.assertEqual(terms[key].shape, (2,))
            self.assertTrue(torch.isfinite(terms[key]).all())
            torch.testing.assert_close(terms[key], other[key], rtol=1e-5, atol=1e-6)

    def test_control_only_swap_has_finite_gradients_into_condition_embeddings(self):
        bt, samples = window_batch()
        original = copy.deepcopy(bt)
        torch.manual_seed(4)
        model = P1Model(replace(CFG, dropout=.1)).train()
        before = rng_state(torch.device('cpu'))
        positive = {}
        loss, _ = model(bt, density_terms=positive)
        after = torch.get_rng_state().clone()
        wrong = dict(bt)
        wrong['controls'] = bt['controls'].clone()
        wrong['controls'][:, FIELD_INDEX] = torch.tensor([s.recipe['density_wrong_band'] for s in samples])
        negative = {}
        with replay_rng(before, torch.device('cpu')):
            model(wrong, density_terms=negative)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        contrast, _, count = ranking_loss(positive, negative)
        self.assertEqual(count, 2)
        contrast_gradient, = torch.autograd.grad(contrast, model.ctrl_e.weight, retain_graph=True)
        self.assertTrue(torch.isfinite(contrast_gradient).all())
        self.assertGreater(float(contrast_gradient.abs().sum()), 0)
        (loss + .2 * contrast).backward()
        self.assertTrue(torch.isfinite(contrast))
        self.assertTrue(torch.isfinite(model.ctrl_e.weight.grad).all())
        self.assertGreater(float(model.ctrl_e.weight.grad.abs().sum()), 0)
        for key, value in original.items():
            if torch.is_tensor(value):
                # encode promotes frozen fp16 inputs to the model dtype in place.
                torch.testing.assert_close(bt[key], value, check_dtype=False)

    def test_ranking_gradient_direction_and_rng_restoration_on_error(self):
        p = torch.tensor([.5, .8], requires_grad=True)
        n = torch.tensor([.4, .6], requires_grad=True)
        zero = torch.zeros(2)
        loss, _, _ = ranking_loss({'time': p, 'cont': zero}, {'time': n, 'cont': zero})
        loss.backward()
        self.assertTrue((p.grad > 0).all())
        self.assertTrue((n.grad < 0).all())
        device = torch.device('cpu')
        before = rng_state(device)
        torch.rand(3)
        after = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with replay_rng(before, device):
                torch.rand(7)
                raise RuntimeError('test')
        self.assertTrue(torch.equal(torch.get_rng_state(), after))


if __name__ == '__main__':
    unittest.main()

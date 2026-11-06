import math
import unittest

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_event_package import (
    EventPackageCanvas, generate_package, hurdle_count_probabilities, legal_tail_nll, package_loss,
)
from beatfcn.train.local_typed_canvas import encode_canvas, request_layout


class RejectTargetReads(dict):
    def __getitem__(self, key):
        if key in {'target', 'targets', 'count', 'slot_valid', 'offset',
                   'tail_block', 'tail_cell', 'tail_ms'}:
            raise AssertionError(f'inference attempted to read target field {key}')
        return super().__getitem__(key)


class EventPackageTests(unittest.TestCase):
    @staticmethod
    def synthetic_example():
        request = EventRequest('phira', 0, 80_000, 4_000_000)
        inputs = dict(
            local=torch.randn(1, 3, 2, 4, 16),
            local_times=torch.tensor([[.02, .06]]),
            local_valid=torch.ones(1, 2, dtype=torch.bool),
            song=torch.randn(1, 4, 16),
            song_times=torch.arange(4)[None] * .5 + .25,
            song_valid=torch.ones(1, 4, dtype=torch.bool),
            mel=torch.randn(1, 8, 8),
            mel_times=torch.arange(8)[None] * .01 + .005,
            mel_valid=torch.ones(1, 8, dtype=torch.bool),
            reference=torch.zeros(1, 1, 16),
            reference_times=torch.zeros(1, 1),
            reference_valid=torch.ones(1, 1, dtype=torch.bool),
            controls=torch.zeros(1, 8, 6),
            mode=torch.tensor([1]),
        )
        inputs.update({
            key: torch.from_numpy(np.asarray(value))[None]
            for key, value in request_layout(request).items()
        })
        events = [
            RhythmEvent(4_000, 2_012_000, 'hold'),
            RhythmEvent(5_000, 2_011_000, 'hold'),
            RhythmEvent(5_000, 5_000, 'flick'),
        ]
        target = {
            key: torch.from_numpy(value)[None]
            for key, value in encode_canvas(events, request).items()
        }
        return request, inputs, target

    @staticmethod
    def small_model():
        config = FullEventConfig(
            d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1,
            decoder_layers=0, tap_dim=16, bands=4, dropout=0., mel_bins=8,
        )
        return EventPackageCanvas(config, density_reference=4).eval()

    def test_hurdle_count_is_one_normalized_distribution(self):
        gate = torch.tensor([0.0, math.log(.25 / .75)])
        positive = torch.zeros(2, 16)

        probability = hurdle_count_probabilities(gate, positive)

        expected = torch.stack((
            torch.cat((torch.tensor([.5]), torch.full((16,), .5 / 16))),
            torch.cat((torch.tensor([.75]), torch.full((16,), .25 / 16))),
        ))
        torch.testing.assert_close(probability, expected)
        torch.testing.assert_close(probability.sum(-1), torch.ones(2))

    def test_legal_tail_nll_matches_explicit_endpoint_enumeration(self):
        block = torch.linspace(-1, 1, 601)
        cell = torch.linspace(.3, -.2, 50)
        ms = torch.linspace(-.4, .5, 40)
        lo, hi, target = 1998, 4003, 2001

        values = torch.arange(lo, hi + 1)
        scores = block[values // 2000] + cell[(values % 2000) // 40] + ms[values % 40]
        expected = torch.logsumexp(scores, 0) - scores[target - lo]

        actual = legal_tail_nll(block, cell, ms, lo, hi, target)
        torch.testing.assert_close(actual, expected)

    def test_real_synthetic_loss_is_finite_and_reaches_all_readouts(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        request, inputs, target = self.synthetic_example()
        model = self.small_model()

        loss, parts = package_loss(model, inputs, target, [request])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {'count', 'onset', 'tail'})
        loss.backward()

        parameters = dict(model.named_parameters())
        for name in (
            'count_gate.weight', 'count_positive.weight', 'member_gru.weight_ih',
            'onset_read.weight', 'tail_block.weight', 'tail_cell.weight',
            'tail_ms.weight', 'frame_read.weight',
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.norm()), 0, name)

    def test_generation_preserves_duplicates_and_empty_without_target_counts(self):
        torch.manual_seed(11)
        _, inputs, _ = self.synthetic_example()
        request = EventRequest('phira', 0, 80_000, 4_000_000, allowed_types=('tap',))
        model = self.small_model()
        with torch.no_grad():
            model.count_gate.weight.zero_()
            model.count_gate.bias.fill_(20.)
            model.count_positive.weight.zero_()
            model.count_positive.bias.zero_()
            model.count_positive.bias.view(4, 16)[:, 1] = 10.
            model.onset_read.weight.zero_()
            model.onset_read.bias.zero_()
            model.onset_read.bias[4] = 10.

        inference_inputs = RejectTargetReads(inputs)
        events = generate_package(model, inference_inputs, request)
        self.assertEqual(events, [RhythmEvent(4_000, 4_000, 'tap')] * 2
            + [RhythmEvent(44_000, 44_000, 'tap')] * 2)
        self.assertTrue(all(isinstance(event, RhythmEvent) for event in events))

        with torch.no_grad():
            model.count_gate.bias.fill_(-20.)
        self.assertEqual(generate_package(model, inference_inputs, request), [])


if __name__ == '__main__':
    unittest.main()

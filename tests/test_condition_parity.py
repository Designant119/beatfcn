"""Train/eval/generation parity of the unconditional request's condition inputs."""
import unittest

import numpy as np

from beatfcn.train.counterfactual_exposure import build_exposure, unconditional_inputs
from beatfcn.train.exposure_loader import CounterfactualExposureDataset
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request

CONDITION_KEYS = ('controls', 'intensity', 'intensity_times', 'intensity_valid')


class TrainPanel(LocalRequestPanel):
    """The benchmark fixture's real-shaped records/features as LocalRequestPanel train rows."""

    def __init__(self):
        from tests.test_benchmark_v2 import fixture
        _, base = fixture()
        self.features, self._targets = base.features, base.targets_by_index
        self.records = [dict(r, split='train') for r in base.records]
        self.charts = base.charts

    def targets(self, index):
        return list(self._targets[index])


class UnconditionalParityTests(unittest.TestCase):
    def setUp(self):
        self.panel = TrainPanel()
        exposure = build_exposure(list(range(len(self.panel.records))), steps=4, seed=1)
        exposure['pool_identity'] = 'fake'
        self.dataset = CounterfactualExposureDataset(self.panel, exposure, microbatch=4,
            vocabulary=dict(relative_blocks=True, tail_query=True), cell_us=100_000)

    def train_dropped(self, index):
        row, audit = self.dataset.sample(index, None, edit_seed=0, curve_specified=False, specified_fields=[])
        self.assertEqual(audit['shown_fields'], [])
        return row['inputs']

    def assert_same(self, a, b, keys=CONDITION_KEYS):
        for key in keys:
            self.assertEqual(np.asarray(a[key]).shape, np.asarray(b[key]).shape, key)
            self.assertEqual(np.asarray(a[key]).dtype, np.asarray(b[key]).dtype, key)
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)

    def test_fully_dropped_training_sample_equals_every_unconditional_eval_request(self):
        from beatfcn.train.audio_dependence import base_inputs
        from beatfcn.train.learning_evaluation import case_inputs
        from beatfcn.train.density_curve import grid_density_curve
        for mode in ('phira', 'mania_4k'):
            index = next(i for i, r in enumerate(self.panel.records) if r['mode'] == mode)
            request = event_request(self.panel.records[index]['request'])
            train = self.train_dropped(index)
            # Training renders 100 ms cells, all unspecified.
            self.assertEqual(len(train['intensity']), 80)
            self.assertFalse(np.asarray(train['intensity_valid']).any())
            # Teacher screen / benchmark v2 inputs: identical, audio included.
            teacher, _ = base_inputs(self.panel, index, cell_us=100_000)
            self.assert_same(train, teacher, tuple(train))
            free, _, _, curve = case_inputs(self.panel, dict(index=index, condition='free', cell_us=100_000))
            self.assertIsNone(curve)
            self.assert_same(train, free, tuple(train))
            # Shared renderer and the runner's whole-song generation provider.
            self.assert_same(train, unconditional_inputs(mode, request))
            provider = grid_density_curve([], request.start_us, request.end_us, cell_us=100_000, scale=16.)
            self.assert_same(train, provider, CONDITION_KEYS[1:])
            # Controls keep the applicability/scope columns (not all zeros).
            self.assertGreater(np.asarray(train['controls'])[:, 3].sum(), 0)

    def test_requested_density_curve_uses_the_training_coarse_cells(self):
        from beatfcn.train.learning_evaluation import case_inputs
        index = next(i for i, r in enumerate(self.panel.records) if r['types'])
        inputs, _, _, curve = case_inputs(self.panel, dict(index=index, condition='density_low', cell_us=100_000,
                                                           density_factor=.5, curve_cell_us=2_000_000))
        self.assertEqual([s['end_us']-s['start_us'] for s in curve], [2_000_000]*4)
        row, audit = self.dataset.sample(index, None, edit_seed=0, curve_specified=True, specified_fields=[])
        self.assertEqual([s['start_us'] for s in curve], [s['start_us'] for s in audit['curve_segments']])
        np.testing.assert_allclose([s['density'] for s in curve],
                                   [.5*s['density'] for s in audit['curve_segments']])
        self.assertEqual(np.asarray(inputs['intensity']).shape, np.asarray(row['inputs']['intensity']).shape)


if __name__ == '__main__':
    unittest.main()

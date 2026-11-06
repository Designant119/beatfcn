import unittest

import numpy as np
import torch

from beatfcn.train.exposure_loader import ExposureDataset, build_exposure_loader


class FakePanel:
    def __init__(self):
        self.records = [
            dict(split='train', controls=np.ones((8, 6), np.float32),
                 audio_sha256='a', chart_id='c', chart_index=0)
            for _ in range(4)
        ]

    def batch(self, indices, *, controls=None, device='cpu'):
        batch = len(indices)
        inputs = dict(
            local=torch.ones(batch, 2, 20, 1, 4),
            local_times=torch.ones(batch, 20),
            local_valid=torch.ones(batch, 20, dtype=torch.bool),
            mel=torch.ones(batch, 30, 2),
            mel_times=torch.ones(batch, 30),
            mel_valid=torch.ones(batch, 30, dtype=torch.bool),
            song=torch.ones(batch, 6, 4),
            song_times=torch.ones(batch, 6),
            song_valid=torch.ones(batch, 6, dtype=torch.bool),
            reference=torch.ones(batch, 1, 16),
            reference_times=torch.ones(batch, 1),
            reference_valid=torch.ones(batch, 1, dtype=torch.bool),
            controls=torch.as_tensor(np.stack(controls), dtype=torch.float32),
            mode=torch.zeros(batch, dtype=torch.int64),
        )
        sequence = dict(
            tokens=torch.ones(batch, 5, dtype=torch.int64),
            fields=torch.ones(batch, 5, dtype=torch.int64),
            times=torch.ones(batch, 5),
            targets=torch.ones(batch, 5, dtype=torch.int64),
            allowed=torch.ones(batch, 5, 7, dtype=torch.bool),
            valid=torch.ones(batch, 5, dtype=torch.bool),
        )
        return inputs, sequence


class ExposureDatasetTest(unittest.TestCase):
    def exposure(self):
        return dict(steps=[dict(effective_update=i+1, indices=[0, 1, 2, 3],
            specified_condition_fields=[[0], [1], [2], [3]]) for i in range(3)])

    def test_resume_starts_at_next_unconsumed_item(self):
        loader = build_exposure_loader(FakePanel(), self.exposure(), microbatch=4,
            start_update=2, num_workers=0, pin_memory=False)
        self.assertEqual([item['step'] for item in loader], [3])

    def test_loader_does_not_advance_model_initialization_rng(self):
        state = torch.get_rng_state().clone()
        list(build_exposure_loader(FakePanel(), self.exposure(), microbatch=4,
            num_workers=0, pin_memory=False))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_bucket_overflow_and_malformed_exposure_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bucket'):
            ExposureDataset(FakePanel(), self.exposure(), microbatch=4,
                sequence_buckets=[4])[0]
        exposure = self.exposure()
        exposure['steps'][0]['specified_condition_fields'] = [[0]]
        with self.assertRaises(ValueError):
            ExposureDataset(FakePanel(), exposure, microbatch=4)[0]

    def test_controls_hiding_and_fixed_shapes(self):
        exposure = dict(steps=[dict(effective_update=1, indices=[0, 1, 2, 3],
            specified_condition_fields=[[0], [1], [2], [3]])])
        dataset = ExposureDataset(FakePanel(), exposure, microbatch=4,
            input_shapes=dict(local=24, mel=32, song=8, reference=2),
            sequence_buckets=[8, 16])
        item = dataset[0]
        self.assertEqual(item['step'], 1)
        self.assertEqual(len(item['batches']), 1)
        ids, inputs, sequence = item['batches'][0]
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(tuple(inputs['local'].shape), (4, 2, 24, 1, 4))
        self.assertEqual(tuple(inputs['mel'].shape), (4, 32, 2))
        self.assertEqual(tuple(sequence['targets'].shape), (4, 8))
        # Request 0 specifies only field 0; its other fields are masked.
        controls = inputs['controls']
        self.assertTrue(torch.all(controls[0, 1:, :3] == 0))
        self.assertTrue(torch.all(controls[0, 0, :3] == 1))
        self.assertTrue(torch.all(controls[1, 1, :3] == 1))
        self.assertTrue(torch.all(controls[1, 0, :3] == 0))


    def test_multiprocess_loader_roundtrip(self):
        exposure = dict(steps=[dict(effective_update=1, indices=[0, 1, 2, 3],
            specified_condition_fields=[[0], [1], [2], [3]])])
        loader = build_exposure_loader(FakePanel(), exposure, microbatch=4,
            input_shapes=dict(local=24, mel=32, song=8, reference=2),
            sequence_buckets=[8, 16], num_workers=2, pin_memory=False,
            prefetch_factor=2, persistent_workers=False)
        items = list(loader)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['step'], 1)
        self.assertEqual(tuple(items[0]['batches'][0][1]['local'].shape), (4, 2, 24, 1, 4))


if __name__ == '__main__':
    unittest.main()


class CounterfactualResumeTests(unittest.TestCase):
    def test_resume_reads_next_frozen_exposure_group(self):
        from beatfcn.train.exposure_loader import CounterfactualExposureDataset
        from beatfcn.model.event_canvas import RhythmEvent
        class Panel(FakePanel):
            def __init__(self):
                self.records = [dict(split='train', request=dict(mode='mania_4k',
                    start_us=0, end_us=1000000, duration_us=1000000)) for _ in range(4)]
            def row(self, index):
                inputs, _ = self.batch([index], controls=[np.zeros((8, 6), np.float32)], device='cpu')
                return dict(inputs={k: v[0].numpy() for k, v in inputs.items()})
            def targets(self, index):
                return [RhythmEvent(0, 0, 'tap', index)]
        from beatfcn.train.counterfactual_exposure import build_exposure
        # counterfactual-exposure-v2 (the rates-only v1 exposure is retired).
        exposure = build_exposure([0, 1, 2, 3], steps=3, seed=0)
        dataset = CounterfactualExposureDataset(Panel(), exposure, microbatch=4,
            start_update=2)
        self.assertEqual(dataset[0]['step'], 3)
        with self.assertRaises(IndexError):
            dataset[1]


class GroupedExposureTests(unittest.TestCase):
    def test_grouped_loader_keeps_order_and_resume_position(self):
        exposure = dict(steps=[dict(effective_update=i+1, indices=[0,1,2,3],
            specified_condition_fields=[[0],[1],[2],[3]]) for i in range(4)])
        loader = build_exposure_loader(FakePanel(), exposure, microbatch=8,
            start_update=2, num_workers=0, pin_memory=False)
        items = list(loader)
        self.assertEqual([item['step'] for item in items], [4])
        self.assertEqual(items[0]['indices'], [0,1,2,3,0,1,2,3])
        self.assertEqual(items[0]['batches'][0][1]['local'].shape[0], 8)
        self.assertEqual(items[0]['batches'][0][2]['valid'].sum().item(), 40)

import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from beatfcn.events.canonical import Event, LaneLoc
from beatfcn.train.exposure_loader import build_exposure_loader
from beatfcn.train.intensity_panel import IntensityPanel
from beatfcn.train.local_panel import collate
from beatfcn.train.local_shapes import pad_input_shapes, pad_sequence_shapes
from beatfcn.train.mert_panel import MertRequestPanel
from beatfcn.train.reference_intensity import ReferenceIntensityPanel
from scripts.prepare_local_request_tape import requests_for_chart
from scripts.bench_exposure_loader import benchmark_loader


def make_panel(root):
    features = root/'features'/('a'*64)
    features.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for name in ('L12', 'L23'):
        np.save(features/f'{name}.npy', rng.normal(size=(450, 1024)).astype(np.float16))
    np.save(features/'times_s.npy', np.arange(450)*.04+.02)
    np.save(features/'mel10_complete.npy', rng.normal(size=(1800, 128)).astype(np.float16))
    (features/'receipt.json').write_text(json.dumps(dict(complete=True,
        metadata=dict(real_samples=18*24000))))
    events = [Event('tap', 8_100_000, 8_100_000, 'tap', LaneLoc(0)),
              Event('hold', 7_000_000, 9_000_000, 'hold', LaneLoc(1))]
    rows, tokens, targets = requests_for_chart(events, 'mania_4k', 18_000_000, 'train')
    np.save(root/'tokens.npy', tokens)
    np.savez(root/'targets.npz', **targets)
    chart = dict(requests=rows, chart_id='c', audio_sha256='a'*64, component='song',
        split='train', mode='mania_4k', stratum='test', duration_us=18_000_000,
        token_path='tokens.npy', target_path='targets.npz')
    (root/'tape.json').write_text(json.dumps(dict(complete=True, failures=[], charts=[chart])))
    return MertRequestPanel(root, 'tape.json', root/'features')


class LoaderPipelineTests(unittest.TestCase):
    def test_benchmark_counts_requests_microbatches_and_valid_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = make_panel(Path(folder))
            exposure = dict(steps=[dict(effective_update=i+1, indices=[0, 1, 2, 1],
                specified_condition_fields=[[]]*4) for i in range(3)])
            loader = build_exposure_loader(panel, exposure, microbatch=2,
                num_workers=0, pin_memory=False)
            result = benchmark_loader(loader, batches=2, warmup=1)
            self.assertAlmostEqual(result['requests_per_s']/result['exposure_items_per_s'], 4.)
            self.assertAlmostEqual(result['physical_microbatches_per_s']/result['exposure_items_per_s'], 2.)
            expected = sum(len(panel.sequence(i)['valid']) for i in [0, 1, 2, 1])
            self.assertAlmostEqual(result['valid_tokens_per_s']/result['exposure_items_per_s'], expected)

    def test_final_shape_collate_matches_old_pad_and_preserves_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = make_panel(Path(folder))
            rows = [panel.row(0), panel.row(1)]
            shapes = dict(local=400, mel=1600, song=48, reference=2)
            old_inputs, old_seq = collate(rows)
            expected = (pad_input_shapes(old_inputs, shapes), pad_sequence_shapes(old_seq, 256))
            actual = collate(rows, input_shapes=shapes, sequence_buckets=[256])
            for old_section, new_section in zip(expected, actual):
                for key in old_section:
                    self.assertTrue(torch.equal(old_section[key], new_section[key]), key)

    def test_real_mert_panel_caches_rebuild_with_spawn_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = ReferenceIntensityPanel(make_panel(Path(folder)), source='audio', version=1)
            panel.row(0)  # Populated LRU functions and memmaps must not be pickled.
            restored = pickle.loads(pickle.dumps(panel))
            np.testing.assert_array_equal(restored.row(1)['inputs']['intensity_times'],
                panel.row(1)['inputs']['intensity_times'])
            exposure = dict(steps=[dict(effective_update=i+1, indices=[0, 1, 2, 1],
                specified_condition_fields=[[]]*4) for i in range(2)])
            loader = build_exposure_loader(restored, exposure, microbatch=2,
                start_update=1, num_workers=1, pin_memory=False,
                persistent_workers=False, multiprocessing_context='spawn')
            items = list(loader)
            self.assertEqual([item['step'] for item in items], [2])
            self.assertEqual(items[0]['batches'][0][2]['valid'].sum().item(),
                sum(len(panel.sequence(i)['valid']) for i in [0, 1]))

    def test_mert_summary_matches_original_recipe_including_short_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = make_panel(Path(folder))
            l23 = np.load(Path(folder)/'features'/('a'*64)/'L23.npy')
            expected = np.stack([np.asarray(l23[i:i+12], np.float32).mean(axis=0)
                                 for i in range(0, len(l23), 12)])
            np.testing.assert_array_equal(panel.features._build_summary('a'*64), expected)

    def test_both_curve_paths_use_relative_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = make_panel(Path(folder))
            for wrapper in (IntensityPanel(panel, version=1), ReferenceIntensityPanel(panel, source='audio', version=1)):
                curve = wrapper.curve_for(1)
                self.assertEqual(float(curve['intensity_times'][0]), .25)
                self.assertEqual(float(curve['intensity_times'][-1]), 7.75)


if __name__ == '__main__':
    unittest.main()

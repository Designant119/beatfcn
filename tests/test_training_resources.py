import unittest
from unittest.mock import patch

from beatfcn.train.training_resources import TrainingResources


class TrainingResourceTests(unittest.TestCase):
    def test_owner_threshold_is_85_after_480_seconds(self):
        with patch('beatfcn.train.training_resources.GpuUtil') as sampler:
            sampler.return_value.take.return_value = dict(util_mean=80., util_p10=80., samples=2)
            guard = TrainingResources()
            guard.observe(elapsed_s=479., data_wait_s=0.)
            with self.assertRaisesRegex(RuntimeError, '<85%'):
                guard.observe(elapsed_s=480., data_wait_s=0.)

    def test_gate_uses_last_eight_interval_means(self):
        with patch('beatfcn.train.training_resources.GpuUtil') as sampler:
            guard = TrainingResources()
            sampler.return_value.take.return_value = dict(util_mean=100., util_p10=100., samples=2)
            for _ in range(8):
                guard.observe(elapsed_s=479., data_wait_s=0.)
            sampler.return_value.take.return_value = dict(util_mean=0., util_p10=0., samples=2)
            result = guard.observe(elapsed_s=480., data_wait_s=0.)
            self.assertEqual(result['util_rolling_mean'], 87.5)
            with self.assertRaisesRegex(RuntimeError, '<85%'):
                guard.observe(elapsed_s=481., data_wait_s=0.)

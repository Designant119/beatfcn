import unittest

try:
    import modal
except ModuleNotFoundError:
    raise unittest.SkipTest('modal is not installed in this environment')

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import modal
import numpy as np

from beatfcn.cloud.full_database_app import StableGpuMonitor, verify_feature_folder
from beatfcn.train.full_database import commit_feature, feature_payloads
from scripts.run_full_database_mert_pilot import (
    normalized_rates, poll_call, recompute_gpu_acceptance)


class Encoder:
    revision = 'test-revision'
    spec = None

    def encode(self, wave):
        samples = len(wave)
        frames = (samples + 959) // 960
        values = np.zeros((2, frames, 1024), dtype=np.float16)
        return dict(layers=values, times_s=np.arange(frames) * .04 + .015,
            intervals_s=np.stack((np.arange(frames) * .04,
                np.minimum((np.arange(frames) + 1) * .04, samples / 24000)), axis=-1),
            metadata=dict(revision=self.revision, real_samples=samples))


class Sampler:
    instances = []

    def __init__(self, period_ms):
        self.period_ms = period_ms
        self.calls = 0
        self.closed = False
        self.__class__.instances.append(self)

    def take(self):
        self.calls += 1
        if self.calls == 1:
            return dict(util_mean=None, samples=0)
        return dict(util_mean=90., samples=1)

    def close(self):
        self.closed = True


class MertPilotRuntimeTests(unittest.TestCase):
    def test_builtin_polling_timeout_is_retried_and_remote_timeout_propagates(self):
        class Call:
            def __init__(self):
                self.calls = 0

            def get(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError()
                return {'complete': True}

        call = Call()
        self.assertTrue(poll_call(call, modal, threading.Event(), time.time() + 5)['complete'])
        self.assertEqual(call.calls, 2)

        class Failed:
            def get(self, timeout):
                raise modal.exception.FunctionTimeoutError('deadline')

        with self.assertRaises(modal.exception.FunctionTimeoutError):
            poll_call(Failed(), modal, threading.Event(), time.time() + 5)

    def test_raw_gpu_acceptance_is_recomputed_and_missing_samples_fail(self):
        value = dict(applicable=True, raw_interval_means=[90.] * 17,
            raw_sample_counts=[60] * 17, raw_interval_durations_s=[30.] * 16 + [20.],
            error_interval_count=0, stable_observed_s=500, rolling_stop_reason=None)
        result = recompute_gpu_acceptance(value)
        self.assertTrue(result['passed'])
        self.assertEqual(result['stable_period_mean_utilization_percent'], 90.)
        value['raw_sample_counts'][3] = 0
        self.assertFalse(recompute_gpu_acceptance(value)['passed'])
        value['raw_sample_counts'][3] = 60
        value['raw_interval_means'][-1] = 20.
        self.assertFalse(recompute_gpu_acceptance(value)['passed'])
        value['raw_interval_means'][-1] = 90.
        value['raw_interval_durations_s'].pop()
        result = recompute_gpu_acceptance(value)
        self.assertFalse(result['passed'])
        self.assertFalse(result['interval_coverage_valid'])

    def test_monitor_starts_sampler_at_boundary_and_close_takes_final_interval(self):
        Sampler.instances = []
        with mock.patch('beatfcn.cloud.gpu_util.GpuUtil', Sampler):
            monitor = StableGpuMonitor()
            self.assertEqual(Sampler.instances, [])
            monitor.start()
            self.assertEqual(Sampler.instances[0].calls, 1)  # setup drain, not an interval
            monitor.close()
        summary = monitor.summary()
        self.assertEqual(summary['raw_interval_means'], [90.])
        self.assertEqual(summary['raw_sample_counts'], [1])
        self.assertEqual(summary['interval_count'], 1)
        self.assertTrue(Sampler.instances[0].closed)
        self.assertFalse(summary['passed'])  # shorter than 480s

    def test_monitor_missing_final_observation_fails_closed(self):
        class Missing(Sampler):
            def take(self):
                self.calls += 1
                return dict(util_mean=None, samples=0)

        with mock.patch('beatfcn.cloud.gpu_util.GpuUtil', Missing):
            monitor = StableGpuMonitor()
            monitor.start()
            monitor.close()
        summary = monitor.summary()
        self.assertEqual(summary['error_interval_count'], 1)
        self.assertEqual(summary['rolling_stop_reason'], 'gpu_observation_unavailable')
        self.assertFalse(summary['passed'])

    def test_remote_feature_reopen_checks_all_files_and_clock(self):
        wave = np.linspace(-1, 1, 960, dtype=np.float32)
        encoder = Encoder()
        features, mel, expected = feature_payloads(wave, encoder)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original.audio';original.write_bytes(b'original')
            folder = root / 'out' / 'features' / ('a' * 64)
            commit_feature('a' * 64, original, root / 'out', root / 'tmp',
                features, mel, expected)
            job = dict(audio_sha256='a' * 64, expected_mel_frames=4)
            result = verify_feature_folder(folder, job, encoder.revision)
            self.assertTrue(result['remote_reopened'])
            self.assertEqual(result['shapes']['L12.npy'], [1, 1024])
            self.assertEqual(result['shapes']['mel10_complete.npy'], [4, 128])
            receipt = json.loads((folder / 'receipt.json').read_text())
            path = folder / 'L12.npy'
            np.save(path, np.zeros((2, 1024), dtype=np.float16), allow_pickle=False)
            receipt['files']['L12.npy'] = dict(bytes=path.stat().st_size,
                sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest())
            (folder / 'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'shape/clock'):
                verify_feature_folder(folder, job, encoder.revision)

    def test_remote_feature_reopen_preserves_non_aligned_complete_tail(self):
        wave = np.linspace(-1, 1, 1000, dtype=np.float32)
        encoder = Encoder()
        features, mel, expected = feature_payloads(wave, encoder)
        self.assertEqual(expected, 2)
        self.assertEqual(mel['mel'].shape, (8, 128))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original.audio';original.write_bytes(b'original')
            folder = root / 'out' / 'features' / ('b' * 64)
            commit_feature('b' * 64, original, root / 'out', root / 'tmp',
                features, mel, expected)
            # The old mel10 receipt truncated to one 40ms cell; MERT output retains both.
            job = dict(audio_sha256='b' * 64, expected_mel_frames=4)
            result = verify_feature_folder(folder, job, encoder.revision)
            self.assertEqual(result['complete_tail_frames'], 2)
            self.assertEqual(result['legacy_truncated_mel_frames'], 4)
            self.assertEqual(result['shapes']['mel10_complete.npy'], [8, 128])

    def test_unadmitted_plan_is_not_persisted_and_resource_drift_fails(self):
        from scripts.run_full_database_mert_pilot import build_contract, contract_digest
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / 'pilot.json'
            with mock.patch('scripts.run_full_database_mert_pilot.PLAN', plan), \
                    mock.patch('scripts.run_full_database_mert_pilot.gate', return_value=['blocked']):
                with self.assertRaisesRegex(ValueError, 'blocked'):
                    build_contract()
                self.assertFalse(plan.exists())
            with mock.patch('scripts.run_full_database_mert_pilot.PLAN', plan), \
                    mock.patch('scripts.run_full_database_mert_pilot.gate', return_value=[]):
                contract = build_contract()
                self.assertTrue(plan.exists())
                contract['resources']['gpu'] = 'H100'
                contract['contract_sha256'] = contract_digest(contract)
                plan.write_text(json.dumps(contract))
                with self.assertRaisesRegex(ValueError, 'resource envelope'):
                    build_contract()

    def test_pilot_cpu_rate_is_not_halved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rates.json'
            path.write_text(json.dumps(dict(gpu_hour_cost_l40s='1.95',
                cpu_hour_cost='0.04730', mem_gib_hour_cost='0.00800')))
            with mock.patch('scripts.run_full_database_mert_pilot.RATE_PATH', path):
                rates = normalized_rates()
            self.assertEqual(Decimal(rates['cpu_usd_s']), Decimal('0.04730') / Decimal(3600))
            self.assertEqual(Decimal(rates['gpu_usd_s']), Decimal('1.95') / Decimal(3600))


if __name__ == '__main__':
    unittest.main()

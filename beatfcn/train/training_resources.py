"""Use the existing interval sampler and last-eight resource gate for training."""
from types import SimpleNamespace

from beatfcn.cloud.gpu_util import GpuUtil
from beatfcn.train.hn2_uncached_amp_runtime import ResourceGuard


class TrainingResources:
    def __init__(self, minimum_util=85.):
        self.sampler = GpuUtil()
        self.guard = ResourceGuard(SimpleNamespace(check_data_after_s=480.,
            max_data_wait=.8, check_after_s=480., min_util=float(minimum_util)))
        self.minimum_util = float(minimum_util)
        self.stable_sum = 0.
        self.stable_samples = 0
        self.previous_elapsed = 0.

    def observe(self, *, elapsed_s, data_wait_s):
        sample = self.sampler.take()
        if self.previous_elapsed >= 480. and sample['samples']:
            self.stable_sum += sample['util_mean'] * sample['samples']
            self.stable_samples += sample['samples']
        self.previous_elapsed = elapsed_s
        try:
            rolling = self.guard.observe(sample['util_mean'], elapsed_s=elapsed_s,
                data_wait_s=data_wait_s)
        except RuntimeError as exc:
            raise RuntimeError(str(exc).replace('<70%', f'<{self.minimum_util:g}%')) from exc
        return dict(**sample, **rolling, minimum_util=self.minimum_util,
            stable_mean=(self.stable_sum/self.stable_samples if self.stable_samples else None))

    def close(self):
        self.sampler.close()

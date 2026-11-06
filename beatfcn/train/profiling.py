"""Short, explicit CPU/CUDA trace windows inside the real training pipeline."""
import contextlib
from pathlib import Path


class TraceWindows:
    def __init__(self, directory, after_seconds=(), updates=2, cuda=True, detailed=True):
        self.directory = Path(directory)
        self.pending = sorted(float(t) for t in after_seconds)
        self.updates, self.cuda = updates, cuda
        self.detailed = detailed
        self.prof = self.cpu = None
        self.count = 0

    def begin(self, step, elapsed):
        if self.prof is not None or not self.pending or elapsed < self.pending[0]:
            return
        import cProfile
        import torch
        self.pending.pop(0)
        self.start_step, self.count = step, 0
        self.directory.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=activities, record_shapes=self.detailed, profile_memory=self.detailed)
        self.prof.start()
        if self.detailed:
            self.cpu = cProfile.Profile()
            self.cpu.enable()

    def scope(self, name):
        if self.prof is None:
            return contextlib.nullcontext()
        import torch
        return torch.profiler.record_function(name)

    def end(self, step, force=False):
        if self.prof is None:
            return None
        self.count += 1
        if not force and self.count < self.updates:
            return None
        import io
        import pstats
        if self.cpu is not None:
            self.cpu.disable()
        self.prof.stop()
        stem = self.directory / f'steps-{self.start_step + 1}-{step + 1}'
        self.prof.export_chrome_trace(str(stem) + '.json')
        if self.cpu is not None:
            self.cpu.dump_stats(str(stem) + '.pstats')
            text = io.StringIO()
            pstats.Stats(self.cpu, stream=text).sort_stats('cumulative').print_stats(45)
            Path(str(stem) + '.txt').write_text(text.getvalue())
        result = {'event': 'trace_saved', 'start_step': self.start_step + 1, 'end_step': step + 1,
                  'trace': str(stem) + '.json', 'cpu_profile': str(stem) + '.txt' if self.detailed else None,
                  'detailed': self.detailed, 'note': 'diagnostic window includes profiler overhead'}
        self.prof = self.cpu = None
        return result

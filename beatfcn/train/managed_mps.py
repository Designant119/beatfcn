"""Per-boundary GPU checks plus own-process physical telemetry every 0.5s."""
import os
from pathlib import Path
import signal
import threading
import time

from .mps_request_lifecycle import memory_sample,clear_request_cache
from .process_memory import physical_footprint
from scripts.check_experiment_storage import admission,GIB


class ManagedMPS:
    def __init__(self,root,artifact,wall_s):
        self.root=Path(root);self.artifact=Path(artifact);self.start=time.monotonic();self.wall_s=wall_s
        self.phase='start';self.rows=[];self.error=None;self.done=threading.Event()
        self.limit=10*GIB;self.peak=0
        self.previous_handler=signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1,self._interrupt)
        self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start()

    def _interrupt(self,*_):
        raise MemoryError(self.error or 'Own-process telemetry watchdog')

    def _watch(self):
        while not self.done.is_set():
            try:
                sample=dict(**physical_footprint(),wall_s=time.monotonic()-self.start,phase=self.phase)
                self.peak=max(self.peak,sample['physical_footprint_bytes']);self.rows.append(sample)
                if sample['physical_footprint_bytes']>self.limit:raise MemoryError('Physical footprint10GiB')
            except BaseException as exc:
                self.error=f'{type(exc).__name__}: {exc}'
                if not self.done.is_set():os.kill(os.getpid(),signal.SIGUSR1)
                return
            self.done.wait(.5)

    def check(self,phase=None):
        if phase:self.phase=phase
        if self.error:raise MemoryError(self.error)
        elapsed=time.monotonic()-self.start
        if elapsed>self.wall_s:raise TimeoutError('Stage wall cap')
        sample=dict(**memory_sample(),wall_s=elapsed,phase=self.phase)
        self.peak=max(self.peak,sample['physical_footprint_bytes'])
        if sample['physical_footprint_bytes']>self.limit:raise MemoryError('Physical footprint10GiB')
        if sample['driver_bytes']>min(.8*sample['recommended_bytes'],10*GIB):raise MemoryError('Driver cap')
        if admission(self.root,GIB,15*GIB)['status']!='admitted':raise MemoryError('Disk reserve')
        if sum(p.stat().st_size for p in self.artifact.rglob('*') if p.is_file())+160*2**20>GIB:
            raise MemoryError('Artifact1GiB including160MiB reserved for failure state/telemetry')
        return sample

    def clear(self):
        self.check();clear_request_cache();return self.check()

    def close(self):
        self.done.set();self.thread.join(timeout=1)
        signal.signal(signal.SIGUSR1,self.previous_handler)
        return dict(interval_s=.5,peak_sampled_physical_bytes=self.peak,samples=self.rows,error=self.error,
                    scope='Sampled detection,not an operating-system memory quota;between-sample overshoot remains possible')

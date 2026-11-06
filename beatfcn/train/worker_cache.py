"""Rebuild process-local LRU functions after spawn without serializing caches."""
from functools import lru_cache


class WorkerCacheMixin:
    worker_cache_specs = ()

    def __getstate__(self):
        state = self.__dict__.copy()
        for name, _, _ in self.worker_cache_specs:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, method, size in self.worker_cache_specs:
            setattr(self, name, lru_cache(maxsize=size)(getattr(self, method)))

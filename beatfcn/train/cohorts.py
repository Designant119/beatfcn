"""Resident cohort loader, kept as a standalone utility after the P1 probe retired."""
import time


class ResidentCohorts:
    """Load each consecutive batch cohort before issuing its model computations.

    Unlike asynchronous look-ahead, no background song reads/allocations continue
    while a cohort is being consumed. Every sample and its order are preserved.
    The consumer must not prefetch into the next cohort before finishing this one.
    """

    def __init__(self, batches, pool, *, max_songs, max_batches=128, before_load=None, progress=None):
        if max_songs < 1 or max_batches < 1:
            raise ValueError('cohort bounds must be positive')
        self.batches, self.pool = batches, pool
        self.max_songs, self.max_batches = max_songs, max_batches
        self.before_load, self.progress = before_load, progress

    def __iter__(self):
        pending, keys, number = [], set(), 0

        def consume():
            nonlocal number
            started = time.perf_counter()
            if self.before_load:
                self.before_load()
            drained = time.perf_counter()
            self.pool.request(keys)
            try:
                self.pool.wait_idle()
                if self.progress:
                    self.progress({'cohort': number, 'microbatches': len(pending), 'songs': len(keys),
                                   'drain_s': drained - started, 'load_s': time.perf_counter() - drained})
                number += 1
                yield from pending
            finally:
                self.pool.release(keys)

        for item in self.batches:
            needed = set(item[0]['_audio']['keys'])
            if len(needed) > self.max_songs:
                raise ValueError('one batch exceeds the cohort song bound')
            if pending and (len(pending) >= self.max_batches or len(keys | needed) > self.max_songs):
                yield from consume()
                pending, keys = [], set()
            pending.append(item)
            keys.update(needed)
        if pending:
            yield from consume()


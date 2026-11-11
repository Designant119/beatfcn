"""Wall-clock guard for cold preparation phases (kept after the compiled-loss line retired)."""
from contextlib import contextmanager
import signal


@contextmanager
def preparation_deadline(seconds, label='Prepared phase'):
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError(label + ' exceeded its bound')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

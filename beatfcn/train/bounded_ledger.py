"""Local ledger locking has a deadline; stop control never waits on this IO."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time


@contextmanager
def locked_ledger(path,timeout_s=1.):
    path=Path(path);deadline=time.monotonic()+timeout_s
    with path.with_suffix('.lock').open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Budget ledger lock deadline')
                time.sleep(min(.01,max(0,deadline-time.monotonic())))
        try:
            ledger=json.loads(path.read_text());yield ledger
            temporary=path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(ledger,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            temporary.replace(path)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

"""Verified per-run feature copies on local disk; never silently use remote paths."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import shutil
import threading
import time

NAMES = ('L9.npy', 'L16.npy', 'L24.npy', 'S20.npy')


def materialize(inventory, source_root, destination, *, deadline, workers=4, max_bytes=48*2**30):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if not inventory.get('complete'):
        raise ValueError('incomplete feature inventory')
    entries = []
    for sha, song in inventory['songs'].items():
        if not re.fullmatch('[0-9a-f]{64}', sha) or set(song['files']) != set(NAMES):
            raise ValueError('unsupported feature identity')
        for name in NAMES:
            entry = song['files'][name]
            expected = Path('taps/v4_i8_L9-16-24') / sha / name
            if Path(entry['path'].lstrip('/')) != expected or not isinstance(entry['bytes'], int) or entry['bytes'] <= 0:
                raise ValueError('invalid feature path or size')
            # The relative path is built only from the fixed prefix, hash and whitelist.
            # Avoid resolving every remote path component during the timed preparation.
            source = source_root / expected
            if not source.is_relative_to(source_root) or source.is_relative_to(destination):
                raise ValueError('feature source outside original root or inside destination')
            entries.append((sha, name, source, entry['bytes']))
    total = sum(x[3] for x in entries)
    if total != inventory['feature_bytes'] or total > max_bytes:
        raise ValueError('feature byte bound differs')
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('destination must be a new empty directory')
    if shutil.disk_usage(destination).free < total + 2**30:
        raise ValueError('insufficient local disk')
    started = time.perf_counter()
    stopped = threading.Event()

    def check_deadline():
        if stopped.is_set() or time.perf_counter() >= deadline:
            raise TimeoutError('materialization deadline')

    def copy(entry):
        sha, name, source, expected = entry
        check_deadline()
        target = destination / sha / name
        target.parent.mkdir(exist_ok=True)
        begin = time.perf_counter()
        digest = hashlib.sha256()
        count = 0
        try:
            with source.open('rb', buffering=0) as src, target.open('xb', buffering=0) as dst:
                buffer = bytearray(8 << 20)
                view = memoryview(buffer)
                while True:
                    check_deadline()
                    n = src.readinto(buffer)
                    if not n:
                        break
                    count += n
                    if count > expected:
                        raise ValueError('source grew beyond registered size')
                    digest.update(view[:n])
                    sent = 0
                    while sent < n:
                        written = dst.write(view[sent:n])
                        if not written:
                            raise OSError('short local write')
                        sent += written
            if count != expected:
                raise ValueError('source shorter than registered size')
            copied = time.perf_counter()
            verified = hashlib.sha256()
            with target.open('rb', buffering=0) as stream:
                while True:
                    check_deadline()
                    n = stream.readinto(buffer)
                    if not n:
                        break
                    verified.update(view[:n])
            if verified.digest() != digest.digest() or target.stat().st_size != expected:
                raise ValueError('local feature content differs')
            return dict(sha=sha, name=name, bytes=count, sha256=digest.hexdigest(),
                        copy_and_hash_s=copied-begin, verify_s=time.perf_counter()-copied)
        except BaseException:
            stopped.set()
            target.unlink(missing_ok=True)
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(copy, entries))
    return dict(complete=True, files=records, feature_bytes=total, wall_s=time.perf_counter()-started,
                local_root=str(destination), remote_reads_in_training_allowed=False)


def local_paths(destination, inventory, sha, *, mel=False):
    if mel or sha not in inventory['songs']:
        raise ValueError('requested audio outside verified materialization')
    paths = [Path(destination) / sha / name for name in NAMES]
    for path, name in zip(paths, NAMES):
        if not path.is_file() or path.stat().st_size != inventory['songs'][sha]['files'][name]['bytes']:
            raise ValueError('materialized feature missing or changed size')
    return paths

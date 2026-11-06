"""Batch MERT + mel feature extraction for the database songs on a new machine ("compute on the spot").

Same writer as scripts/draft_song.py (prepare_features -> full_database.commit_feature), MERT loaded ONCE,
audio decoded by a thread pool ahead of the GPU. The tape is the clock authority: a decode LONGER than the
tape's duration (newer ffmpeg mp3 tail padding; verified on DEV to sit at the end, head frames identical) is
truncated to the tape's sample count and recorded; a SHORTER decode is skipped and recorded.
Re-scans the audio folder until every tape song has features (audio may still be downloading).

  python -m scripts.batch_extract_features --tape-dir DIR --audio-dir DIR --features-dir DIR \
      --mert-source DIR [--device cuda --decode-workers 6 --log extract.jsonl --until-idle-s 900]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 24000


def expected_samples(duration_us):
    """The sample count n with n * 1e6 // SR == duration_us (receipt convention)."""
    n = math.ceil(int(duration_us) * SR / 1e6)
    while n * 1_000_000 // SR < int(duration_us):
        n += 1
    while n > 0 and (n - 1) * 1_000_000 // SR >= int(duration_us):
        n -= 1
    return n


def main():
    import numpy as np
    from scripts.draft_song import make_extractor, prepare_features
    from beatfcn.ingest.pipeline import decode
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--tape-dir', type=Path, required=True)
    a.add_argument('--audio-dir', type=Path, required=True)
    a.add_argument('--features-dir', type=Path, required=True)
    a.add_argument('--mert-source', type=Path, required=True)
    a.add_argument('--device', default='cuda')
    a.add_argument('--decode-workers', type=int, default=6)
    a.add_argument('--log', type=Path, required=True)
    a.add_argument('--until-idle-s', type=float, default=900., help='stop after this long with nothing new to do')
    a.add_argument('--shard', default='0/1', help='i/n: this process handles songs whose index mod n == i')
    a.add_argument('--prefetch', type=int, default=32, help='decoded songs held ahead of the GPU (memory bound)')
    args = a.parse_args()
    durations = {}
    for f in sorted(args.tape_dir.glob('*-chart.json')):
        c = json.loads(f.read_text())
        durations.setdefault(c['audio_sha256'], int(c['duration_us']))
    si, sn = (int(x) for x in args.shard.split('/'))
    mine = {sha for k, sha in enumerate(sorted(durations)) if k % sn == si}
    extractor, backend = make_extractor(args.device, str(args.mert_source))
    log = open(args.log, 'a')
    state = dict(done=set(), skipped=set())
    for line in args.log.read_text().splitlines() if args.log.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (state['done'] if r.get('res') in ('ok', 'have') else state['skipped']).add(r.get('sha'))

    def decoded(item):
        sha, path = item
        try:
            wave = decode(str(path))
        except Exception as e:  # noqa: BLE001
            return sha, path, None, dict(res='decode_error', err=type(e).__name__)
        n = expected_samples(durations[sha])
        info = dict(decoded=len(wave), expected=n)
        if len(wave) < n:
            return sha, path, None, dict(info, res='short_decode')
        if len(wave) > n:
            info['truncated'] = len(wave) - n
            wave = wave[:n]
        return sha, path, wave, info

    idle_since = time.monotonic()
    while True:
        todo = []
        for p in sorted(args.audio_dir.iterdir()):
            sha = p.name.split('.')[0]
            if sha in mine and sha not in state['done'] and sha not in state['skipped'] \
                    and not (args.features_dir / 'features' / sha / 'receipt.json').is_file():
                todo.append((sha, p))
        if not todo:
            left = len(mine - state['done'])
            if time.monotonic() - idle_since > args.until_idle_s:
                log.write(json.dumps(dict(event='EXTRACT_IDLE_EXIT', remaining=left)) + '\n'); log.flush()
                return
            time.sleep(60)
            continue
        idle_since = time.monotonic()
        def bounded(pool, items):
            # never more than --prefetch decoded songs in memory (Executor.map submits everything at once)
            from collections import deque
            q = deque()
            it = iter(items)
            for item in it:
                q.append(pool.submit(decoded, item))
                if len(q) >= args.prefetch:
                    break
            while q:
                yield q.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    q.append(pool.submit(decoded, nxt))
        with ThreadPoolExecutor(args.decode_workers) as pool:
            for sha, path, wave, info in bounded(pool, todo):
                t0 = time.monotonic()
                if wave is None:
                    state['skipped'].add(sha)
                    log.write(json.dumps(dict(sha=sha, **info)) + '\n'); log.flush()
                    continue
                try:
                    ns = SimpleNamespace(codec='batch', mert_source=str(args.mert_source), mert_device=args.device,
                                         device=args.device)
                    receipt, rec = prepare_features(path, sha, args.features_dir, ns, extractor=extractor,
                                                    decoder=lambda _p, w=wave: w)
                    # prepare_features labels a passed-in extractor 'injected' (test path); this one is the real pinned
                    # MERT loaded once, and the decoder is ffmpeg (+ tail truncation recorded below)
                    receipt['backend'] = backend
                    receipt['draft_song']['extractor']['backend'] = backend
                    receipt['draft_song']['decoder']['injected'] = False
                    receipt['batch_extract'] = dict(backend=backend, **info)
                    (args.features_dir / 'features' / sha / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
                    state['done'].add(sha)
                    log.write(json.dumps(dict(sha=sha, res='ok', s=round(time.monotonic() - t0, 2), **info)) + '\n')
                except Exception as e:  # noqa: BLE001 - one bad song never stops the batch
                    state['skipped'].add(sha)
                    log.write(json.dumps(dict(sha=sha, res='error', err=f'{type(e).__name__}: {str(e)[:200]}', **info)) + '\n')
                log.flush()


if __name__ == '__main__':
    main()

"""Pack the database MERT feature dirs into ~5 GB uncompressed tar shards for bulk export (CPU; no model).

  python -m scripts.pack_features_for_export --src /data/full-database-v1/mert-graph-diagnostic-v2/features \
      --out /data/export/features --shard-gb 5 [--workers 8]
Writes shard-XXX.tar (each holds whole song dirs) + manifest.json (song -> shard, file sizes). Idempotent per shard.
"""
import argparse, json, os, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def dir_size(p):
    return sum(f.stat().st_size for f in p.iterdir() if f.is_file())


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--src', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--shard-gb', type=float, default=5.0)
    a.add_argument('--workers', type=int, default=8)
    args = a.parse_args()
    t = time.time()
    songs = sorted(p for p in args.src.iterdir() if p.is_dir())
    with ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(dir_size, songs))
    print(f'{len(songs)} songs, {sum(sizes)/1e9:.1f} GB, listed in {time.time()-t:.0f}s', flush=True)
    limit, shards, cur, cur_b = args.shard_gb * 1e9, [], [], 0
    for s, b in zip(songs, sizes):
        if cur and cur_b + b > limit:
            shards.append(cur); cur, cur_b = [], 0
        cur.append(s); cur_b += b
    if cur:
        shards.append(cur)
    args.out.mkdir(parents=True, exist_ok=True)

    def pack(i):
        dst = args.out / f'shard-{i:03d}.tar'
        if dst.exists():
            return i, dst.stat().st_size
        tmp = dst.with_suffix('.tar.part')
        with tarfile.open(tmp, 'w') as tar:
            for s in shards[i]:
                tar.add(s, arcname=s.name)
        os.replace(tmp, dst)
        return i, dst.stat().st_size
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, size in ex.map(pack, range(len(shards))):
            done += 1
            print(f'shard {i:03d} {size/1e9:.2f} GB ({done}/{len(shards)}, {time.time()-t:.0f}s)', flush=True)
    manifest = dict(src=str(args.src), shards=[dict(name=f'shard-{i:03d}.tar', songs=[s.name for s in sh]) for i, sh in enumerate(shards)],
                    songs=len(songs), bytes=sum(sizes))
    (args.out / 'manifest.json').write_text(json.dumps(manifest) + '\n')
    print('PACK_DONE', len(shards), 'shards', f'{time.time()-t:.0f}s', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Phira chart pack bulk downloader + verifier (one shard).

Usage:
  python3 download_shard.py --jsonl <path> --start <i> --end <j> \
      --outdir <dir> --manifest <csv> [--workers N] [--max-attempts N] [--only-ids <file>]

- Reads metadata records [start, end) from the JSONL (one JSON per line).
- Downloads each chart zip from `file` URL (HTTP 303 redirect handled by curl -L).
- Saves as {id}.zip under outdir; id is globally unique.
- Verifies each zip: opens with zipfile, non-empty, contains chart json + info yaml + image + audio.
- Retries failed downloads up to max-attempts; final failures are recorded in the manifest,
  never silently dropped. Idempotent: already-present valid zips are skipped.
- Writes a per-shard manifest CSV with one row per record.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

DIVISIONS = {'regular', 'plain', 'troll', 'visual'}


def division_of(rec):
    tags = rec.get('tags')
    if isinstance(tags, list):
        for t in tags:
            if t in DIVISIONS:
                return t
    elif isinstance(tags, str):
        for t in tags.replace(',', ' ').split():
            if t in DIVISIONS:
                return t
    return ''


def verify_zip(path):
    """Return None if OK, else raise RuntimeError with the reason."""
    if not os.path.exists(path):
        raise RuntimeError('file missing')
    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError('zero-byte file')
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as e:
        raise RuntimeError('BadZipFile: %s' % e)
    if not names:
        raise RuntimeError('empty zip (0 entries)')
    low = [n.lower() for n in names]

    def sniff_magic(n):
        try:
            with zipfile.ZipFile(path) as zf:
                data = zf.read(n)[:16]
        except Exception:
            return ''
        h = data.hex()
        if h.startswith('ffd8ff') or n.lower().endswith('.jfif'):
            return 'image'
        if h.startswith(('89504e47', '47494638', '52494646')):
            return 'image'
        if data[:4] == b'OggS' or data[:3] == b'ID3' or data[:4] in (b'fLaC', b'RIFF', b'\x1aE\xdf\xa3'):
            return 'audio'
        s = data.lstrip()
        if s[:1] in (b'{', b'['):
            return 'chart'
        return ''

    problems = []
    if not any(n.endswith(('.json', '.pec', '.mc', '.chart')) for n in low) and \
       not any(sniff_magic(n) == 'chart' for n in names):
        problems.append('missing chart file')
    if not any(n.endswith(('.yml', '.yaml')) for n in low):
        problems.append('missing info .yml/.yaml')
    if not any(n.endswith(('.png', '.jpg', '.jpeg', '.jfif', '.webp', '.gif', '.bmp')) for n in low) and \
       not any(sniff_magic(n) == 'image' for n in names):
        problems.append('missing cover image')
    if not any(n.endswith(('.ogg', '.oga', '.mp3', '.m4a', '.wav', '.flac', '.opus', '.aac', '.aiff')) for n in low) and \
       not any(sniff_magic(n) == 'audio' for n in names):
        problems.append('missing audio')
    if problems:
        raise RuntimeError('; '.join(problems))
    return size


def download_one(rec, outdir, max_attempts):
    rid = str(rec.get('id'))
    url = rec.get('file') or rec.get('file_url') or ''
    out_path = os.path.join(outdir, rid + '.zip')
    if not url:
        return {'id': rid, 'status': 'failed', 'size': 0, 'local_path': '', 'reason': 'no file url in metadata'}
    # Idempotency: skip if an already-valid zip exists.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            verify_zip(out_path)
            return {'id': rid, 'status': 'success', 'size': os.path.getsize(out_path),
                    'local_path': out_path, 'reason': ''}
        except Exception:
            try:
                os.remove(out_path)
            except OSError:
                pass
    last_err = ''
    for attempt in range(1, max_attempts + 1):
        try:
            r = subprocess.run(
                ['curl', '-L', '-sS', '--fail', '--connect-timeout', '15', '--max-time', '900',
                 '--retry', '2', '--retry-delay', '2', '-o', out_path, url],
                capture_output=True, text=True, timeout=1000)
            if r.returncode != 0:
                raise RuntimeError('curl exit=%d %s' % (r.returncode, (r.stderr or r.stdout).strip()[:200]))
            verify_zip(out_path)
            return {'id': rid, 'status': 'success', 'size': os.path.getsize(out_path),
                    'local_path': out_path, 'reason': ''}
        except Exception as e:
            last_err = str(e)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            if attempt < max_attempts:
                time.sleep(2 * attempt)
    return {'id': rid, 'status': 'failed', 'size': 0, 'local_path': '', 'reason': last_err[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--only-ids', default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    with open(args.jsonl, encoding='utf-8') as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if args.end is None:
        args.end = len(recs)
    recs = recs[args.start:args.end]

    if args.only_ids:
        with open(args.only_ids, encoding='utf-8') as f:
            keep = {line.strip() for line in f if line.strip()}
        recs = [r for r in recs if str(r.get('id')) in keep]

    total = len(recs)
    print('[shard] start=%d end=%d records=%d workers=%d' % (args.start, args.end, total, args.workers), flush=True)
    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, r, args.outdir, args.max_attempts): r for r in recs}
        for fut in as_completed(futs):
            res = fut.result()
            results[res['id']] = res
            done += 1
            if done % 25 == 0 or done == total:
                ok = sum(1 for x in results.values() if x['status'] == 'success')
                print('[shard] %d/%d done, %d ok' % (done, total, ok), flush=True)

    with open(args.manifest, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'name', 'level', 'charter', 'division', 'file_url',
                    'local_path', 'size_bytes', 'status', 'reason'])
        for r in recs:
            res = results.get(str(r.get('id')), {})
            w.writerow([r.get('id', ''), r.get('name', ''), r.get('level', ''),
                        r.get('charter', ''), division_of(r), r.get('file', ''),
                        res.get('local_path', ''), res.get('size', 0),
                        res.get('status', 'missing'), res.get('reason', '')])

    ok = sum(1 for x in results.values() if x['status'] == 'success')
    failed = total - ok
    size = sum(x['size'] for x in results.values())
    print('[shard] DONE total=%d success=%d failed=%d bytes=%d' % (total, ok, failed, size), flush=True)
    print('[shard] manifest written: %s' % args.manifest, flush=True)


if __name__ == '__main__':
    main()

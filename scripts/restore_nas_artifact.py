"""Restore a SHA-verified archived artifact; never overwrite a local file."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT=Path(__file__).resolve().parents[1]


def restore(original, output=None):
    original=(ROOT/original).resolve()
    if not original.is_relative_to(ROOT):raise ValueError('Original artifact must be inside repository')
    pointer=json.loads(Path(str(original)+'.nas.json').read_text())
    manifest=json.loads((ROOT/pointer['manifest']).read_text())
    row=next(r for r in manifest['files'] if r['path']==str(original.relative_to(ROOT)))
    if pointer['sha256']!=row['sha256'] or pointer['bytes']!=row['bytes']:
        raise ValueError('Archive pointer identity mismatch')
    if pointer['remote']!=manifest['remote_root']+'/'+row['path'] or pointer['host']!=manifest['host']:
        raise ValueError('Archive location mismatch')
    destination=Path(output).resolve() if output else original
    if destination.exists():raise FileExistsError('Refuse to overwrite '+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(destination)+'.restoring')
    with temporary.open('xb') as stream:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',pointer['host'],
            'cat '+shlex.quote(pointer['remote'])],stdout=stream,check=True)
    h=hashlib.sha256()
    with temporary.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    if temporary.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:
        raise ValueError('Restored bytes do not match; original path untouched')
    temporary.rename(destination)
    return dict(original=str(original.relative_to(ROOT)),destination=str(destination),bytes=row['bytes'],sha256=h.hexdigest(),verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('--output');args=p.parse_args()
    print(json.dumps(restore(args.original,args.output),ensure_ascii=False))

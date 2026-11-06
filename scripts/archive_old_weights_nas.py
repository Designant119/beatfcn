"""Owner-authorized NAS archival: copy, remote SHA check, then local eviction.

Never evicts a terminal checkpoint, active cache, raw result or Council object.
The immutable source list is the previously inspected intermediate inventory.
Every evicted file has a local sidecar with an exact restoring command.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/nas-archive-2026-10-29'
REMOTE = 'bancho-nb2.local'
DEST = '/home/yukoval/archives/beatfcn/2026-10-29-intermediate-weights'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', REMOTE]


def write(name, value):
    path = FOLDER / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    prior = json.loads((FOLDER / 'manifest.json').read_text()) if (FOLDER / 'manifest.json').exists() else None
    if prior is not None and prior['status'] != 'copying':
        raise FileExistsError('Archive beyond copy stage; inspect before recovery')
    inventory = json.loads((ROOT / 'reports/storage-audit-2026-10-29.json').read_text())
    candidates = inventory['candidates']
    FOLDER.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(ROOT).free
    rows = []
    for item in candidates:
        path = ROOT / item['path']
        if not path.is_file() or path.is_symlink():
            raise ValueError('Archive source missing or linked: ' + item['path'])
        if not item['retained_endpoint_paths'] or not all((ROOT / p).is_file() for p in item['retained_endpoint_paths']):
            raise ValueError('Retained final missing: ' + item['path'])
        if item['path'] in item['retained_endpoint_paths']:
            raise ValueError('Refuse terminal eviction')
        stat = path.stat()
        if stat.st_size != item['bytes']:
            raise ValueError('Inventory size changed: ' + item['path'])
        rows.append(dict(path=item['path'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         sha256=sha(path), retained_local_endpoints=item['retained_endpoint_paths']))
    manifest = dict(id='beatfcn-intermediate-weights-2026-10-29', host=REMOTE, remote_root=DEST,
        authorization='Owner: old weights/data may be archived to NAS bancho-nb2.local',
        status='copying', source_inventory='reports/storage-audit-2026-10-29.json',
        files=rows, total_bytes=sum(r['bytes'] for r in rows), free_before=before,
        started_at=datetime.now(timezone.utc).isoformat(), evicted=[])
    if prior is not None:
        if prior['files'] != rows:
            raise ValueError('Resume source identity changed')
        manifest = prior
        manifest['transfer_note'] = 'Initial rsync rejected /home path by NAS receiver; explicit SSH tar stream used. No initial local evictions.'
    write('manifest.json', manifest)
    listing = FOLDER / 'files.txt'
    listing.write_text(''.join(r['path'] + '\n' for r in rows))
    subprocess.run(SSH + [f'mkdir -p {shlex.quote(DEST)}'], check=True)
    # The NAS rsync receiver rejects its /home bind mount. A tar byte stream
    # over the already authenticated SSH channel has no temporary local copy.
    sender = subprocess.Popen(['tar', '-cf', '-', '-C', str(ROOT), '-T', str(listing)], stdout=subprocess.PIPE)
    receiver = subprocess.Popen(SSH + ['tar -xf - -C ' + shlex.quote(DEST)], stdin=sender.stdout)
    sender.stdout.close()
    receiver_code = receiver.wait()
    sender_code = sender.wait()
    if receiver_code or sender_code:
        raise RuntimeError(f'Archive stream failed: sender={sender_code},receiver={receiver_code}')
    subprocess.run(SSH + ['cat > ' + shlex.quote(DEST + '/manifest-source.json')],
        input=(FOLDER / 'manifest.json').read_bytes(), check=True)
    verifier = """import json,hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]);m=json.loads((root/'manifest-source.json').read_text());rows=[]
for r in m['files']:
 p=root/r['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if p.stat().st_size!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('Archive mismatch '+r['path'])
 rows.append(dict(path=r['path'],bytes=r['bytes'],sha256=h.hexdigest()))
out=dict(verified=True,files=rows,total_bytes=sum(r['bytes'] for r in rows))
(root/'verification.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
"""
    checked = subprocess.run(SSH + ['python3 - ' + shlex.quote(DEST)], input=verifier,
                             text=True, capture_output=True, check=True)
    remote = json.loads(checked.stdout)
    if not remote['verified'] or remote['files'] != [{k:r[k] for k in ('path', 'bytes', 'sha256')} for r in rows]:
        raise ValueError('Remote verification receipt does not match source manifest')
    write('remote-verification.json', remote)
    manifest.update(status='verified_remote_evicting', remote_receipt='remote-verification.json')
    write('manifest.json', manifest)
    for r in rows:
        path = ROOT / r['path']
        stat = path.stat()
        if stat.st_size != r['bytes'] or stat.st_mtime_ns != r['mtime_ns'] or sha(path) != r['sha256']:
            raise ValueError('Local source changed after copy: ' + r['path'])
        # A recoverable archive location remains beside the original path.
        pointer = {**r, 'host': REMOTE, 'remote': DEST + '/' + r['path'],
            'manifest': 'reports/nas-archive-2026-10-29/manifest.json',
            'restore': 'ssh ' + shlex.quote(REMOTE) + ' ' + shlex.quote('cat ' + shlex.quote(DEST + '/' + r['path'])) + ' > ' + shlex.quote(str(path))}
        Path(str(path) + '.nas.json').write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + '\n')
        path.unlink()
        manifest['evicted'].append(r['path'])
        write('manifest.json', manifest)
    manifest.update(status='archived_verified_local_evicted', free_after=shutil.disk_usage(ROOT).free,
        completed_at=datetime.now(timezone.utc).isoformat())
    write('manifest.json', manifest)
    subprocess.run(SSH + ['cat > ' + shlex.quote(DEST + '/manifest-final.json')],
        input=(FOLDER / 'manifest.json').read_bytes(), check=True)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files', 'evicted')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Upload the pinned local MERT2 checkpoint to the full-database volume prefix."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'inputs/d96_mert2_source/checkpoint'
REMOTE = 'full-database-v1/mert2-checkpoint'
FILES = (
    ('config.json', CHECKPOINT / 'config.json', 'f2e194895f58be3ddba327255db129ff0e3bee550cc0ecf08e4d22d79ce3bca3'),
    ('model.safetensors', CHECKPOINT / 'model.safetensors', 'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d'),
    ('configuration_mert2.py', ROOT / 'inputs/d96_mert2_source/mert2_pinned/configuration_mert2.py',
     '77b53ec9d7ee31a599d744fb006e812c7eeaf7390deb46e2f460cf8c17b00bd6'),
    ('modeling_mert2.py', ROOT / 'inputs/d96_mert2_source/mert2_pinned/modeling_mert2.py',
     'b1a3174e5649c4b26b0c90d8626f0adacfbbba111a58ed3bb72ad651945a2f5c'),
)
EXPECTED = {name: expected for name, _, expected in FILES}
RECEIPT_LOCAL = ROOT / 'reports/full-database-cloud-v1/mert-checkpoint-upload-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def read_remote_json(volume, remote):
    payload = b''.join([chunk async for chunk in volume.read_file.aio(remote)])
    return json.loads(payload)


async def main():
    if RECEIPT_LOCAL.exists():
        raise FileExistsError('Checkpoint upload receipt already exists')
    paths = {name: path for name, path, _ in FILES}
    for name, expected in EXPECTED.items():
        local = paths[name]
        if not local.is_file() or digest(local) != expected:
            raise ValueError('Local pinned checkpoint identity mismatch: ' + name)
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    if (config.get('hidden_size'), config.get('num_hidden_layers'), config.get('sampling_rate'),
            config.get('inputs_to_logits_ratio')) != (1024, 24, 24000, 960):
        raise ValueError('Pinned MERT2 configuration changed')
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    started = time.monotonic()
    receipt_remote = REMOTE + '/receipt.json'
    status = 'uploaded'
    try:
        existing = await read_remote_json(volume, receipt_remote)
        if existing.get('sha256') != EXPECTED:
            raise ValueError('Remote checkpoint receipt belongs to another pin')
        result = existing
        status = 'already_present'
    except modal.exception.NotFoundError:
        try:
            entries = await volume.listdir.aio(REMOTE)
            if entries:
                raise ValueError('Partial remote checkpoint prefix exists without receipt')
        except modal.exception.NotFoundError:
            pass
        result = dict(schema='beatfcn.full-database-mert-checkpoint.v1', remote_prefix=REMOTE,
            repo='m-a-p/MERT-v2-FullSong',
            revision='7001f7657276238629e61c7746ce70735e0604c0',
            sha256=EXPECTED,
            bytes={name: paths[name].stat().st_size for name in EXPECTED},
            cloud_compute_calls=0)
        async with volume.batch_upload.aio() as upload:
            for name, path in paths.items():
                upload.put_file(str(path), REMOTE + '/' + name)
            upload.put_file(io.BytesIO((json.dumps(result, ensure_ascii=False) + '\n').encode()),
                            receipt_remote)
        result['uploaded_at_unix'] = time.time()
        result['wall_s'] = time.monotonic() - started
        # Batch upload is transactional, but list the exact files before receipting locally.
        entries = await volume.listdir.aio(REMOTE)
        sizes = {Path(entry.path).name: entry.size for entry in entries}
        if any(sizes.get(name) != paths[name].stat().st_size for name in EXPECTED):
            raise ValueError('Remote checkpoint listing differs after upload')
        # Refresh the remote receipt with upload timing while retaining exact identity.
        # The immutable remote receipt already carries exact identity and sizes;
        # upload timing remains in the separate local receipt only.
        result['upload_verified'] = True
    write_json(RECEIPT_LOCAL, dict(status=status, receipt=result,
        local_files={name: str(paths[name]) for name in EXPECTED},
        cloud_compute_calls=0, volume_storage_writes=status != 'already_present'))
    print(json.dumps(json.loads(RECEIPT_LOCAL.read_text()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())

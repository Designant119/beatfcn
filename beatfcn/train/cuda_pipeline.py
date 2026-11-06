"""CUDA-pilot data path: verified raw receipts and single-pass output hashing."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .full_database import SCHEMA, checked_path


def load_raw_index(root, expected_sha256, manifest_sha256):
    path = Path(root) / 'full-database-v1/raw-content.jsonl'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Raw content index differs from the frozen CUDA-pilot contract')
    result = {}
    for line in payload.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        audio = record['audio_sha256']
        if record.get('manifest_sha256') != manifest_sha256 or audio in result:
            raise ValueError('Raw content index has a manifest or duplicate-audio mismatch')
        result[audio] = record
    return result


def decoded_audio_from_receipt(job, raw_index, root, temporary_root):
    """Decode verified bytes without repeating raw SHA or scanning every ZIP member."""
    audio = job['audio_sha256']
    record = raw_index[audio]
    if not record.get('verified') or record.get('payload_sha256') != audio:
        raise ValueError('Audio lacks verified raw content identity: ' + audio)
    source = checked_path(root, record['archive'])
    from beatfcn.ingest.pipeline import decode
    if record['source'] == 'osu':
        original = source
    elif record['source'] == 'phira':
        member = record.get('archive_member')
        if not member:
            raise ValueError('Verified Phira receipt lacks an archive member')
        temporary_root = Path(temporary_root)
        temporary_root.mkdir(parents=True, exist_ok=True)
        original = temporary_root / (audio + '.audio')
        if not original.exists() or original.stat().st_size != record['payload_bytes']:
            partial = temporary_root / (audio + '.partial')
            size = 0
            with zipfile.ZipFile(source, 'r') as archive, archive.open(member) as stream, \
                    partial.open('wb') as target:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    target.write(chunk)
                    size += len(chunk)
            if size != record['payload_bytes']:
                raise ValueError('Extracted Phira member size differs from its verified receipt')
            partial.replace(original)
    else:
        raise ValueError('Unsupported raw source')
    from beatfcn.ingest.pipeline import decode
    wave = decode(str(original))
    if not np.isfinite(wave).all() or not len(wave):
        raise ValueError('Invalid decoded audio')
    return audio, original, wave


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()

    def write(self, value):
        self.hasher.update(value)
        return self.stream.write(value)

    def flush(self):
        return self.stream.flush()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def commit_feature_hashing(audio, original, output_root, temporary_root, features, mel, expected):
    """Commit immutable features while hashing bytes during np.save, not via a second read."""
    folder = Path(output_root) / 'features' / audio
    folder.mkdir(parents=True, exist_ok=True)
    values = {'L12.npy': features['layers'][0], 'L23.npy': features['layers'][1],
        'times_s.npy': features['times_s'], 'intervals_s.npy': features['intervals_s'],
        'mel10_complete.npy': mel['mel']}
    files = {}
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite output ' + name)
        temporary = folder / (name + '.partial')
        with temporary.open('wb') as stream:
            writer = HashingWriter(stream)
            np.save(writer, np.ascontiguousarray(value), allow_pickle=False)
            writer.flush()
        temporary.replace(folder / name)
        files[name] = dict(bytes=(folder / name).stat().st_size,
            sha256=writer.hasher.hexdigest())
    result = dict(schema=SCHEMA, audio_sha256=audio,
        decoded_samples=features['metadata']['real_samples'], frames=expected,
        metadata=features['metadata'], backend='cuda_bf16', files=files,
        optimizer_updates=0, wall_s=None, complete=True,
        write_hash_strategy='single_pass_during_np_save')
    write_json(folder / 'receipt.json', result)
    original = Path(original)
    if original.is_relative_to(Path(temporary_root)):
        original.unlink()
    return result

"""Materialized metadata batches: sampling is finished before GPU training starts."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import time
from pathlib import Path

import torch

from .data import read_bytes


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class MetadataTape:
    """Load a bounded, verified tape into private CPU memory, without worker IPC.

    Tensors retain the exact packed representation consumed by live training.
    The manifest binds the tape to the data selection, policy and batch size.
    """

    def __init__(self, directory, *, rows_sha256, policy, micro_batch, skip_batches=0, progress=None):
        root = Path(directory)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('schema') != 'beatfcn.metadata_tape.v1' or not manifest.get('complete'):
            raise ValueError('metadata tape is incomplete or unsupported')
        if (manifest['rows_sha256'] != rows_sha256 or manifest['policy_sha256'] != policy_digest(policy)
                or manifest['micro_batch'] != micro_batch):
            raise ValueError('metadata tape data, policy or batch size differs from training')
        if sum(part.get('raw_bytes', part['bytes']) for part in manifest['parts']) > 8 * 2**30:
            raise ValueError('metadata tape exceeds the 8 GiB host-memory bound')
        self.manifest, self.batches = manifest, []
        self.load_timing = dict(read_s=0., verify_s=0., decode_s=0., deserialize_s=0.)
        for part in manifest['parts']:
            if Path(part['file']).name != part['file']:
                raise ValueError('metadata tape part must be a filename')
            started = time.perf_counter()
            payload = read_bytes(root / part['file'])
            self.load_timing['read_s'] += time.perf_counter() - started
            started = time.perf_counter()
            if len(payload) != part['bytes'] or hashlib.sha256(payload).hexdigest() != part['sha256']:
                raise ValueError('metadata tape part failed integrity check')
            self.load_timing['verify_s'] += time.perf_counter() - started
            started = time.perf_counter()
            if part.get('codec') == 'gzip':
                with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                    payload = compressed.read(part['raw_bytes'] + 1)
                if len(payload) != part['raw_bytes'] or hashlib.sha256(payload).hexdigest() != part['raw_sha256']:
                    raise ValueError('decoded metadata tape part failed integrity check')
            elif part.get('codec') not in (None, 'raw'):
                raise ValueError('unsupported metadata tape codec')
            self.load_timing['decode_s'] += time.perf_counter() - started
            started = time.perf_counter()
            batches = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            self.load_timing['deserialize_s'] += time.perf_counter() - started
            if len(batches) != part['batches']:
                raise ValueError('metadata tape part count differs')
            if any(len(batch['_audio']['keys']) != micro_batch for batch, _ in batches):
                raise ValueError('metadata tape contains a partial microbatch')
            self.batches.extend(batches)
            if progress:
                progress({'file': part['file'], 'batches': len(self.batches), **self.load_timing})
        if len(self.batches) != manifest['batches'] or not 0 <= skip_batches < len(self.batches):
            raise ValueError('metadata tape length or resume offset is invalid')
        self.skip_batches = skip_batches

    def __iter__(self):
        for batch, skipped in self.batches[self.skip_batches:]:
            yield dict(batch), dict(skipped)


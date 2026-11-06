"""CPU-prepared native events and MERT/mel inputs for bounded CUDA trials.

The prepared index owns all target filtering, partitions, controls and public
references. Runtime loading only slices numeric arrays. Model inputs never use
the hidden target length to choose a query shape; every target batch has 512
slots. Audio padding buckets depend on public song duration only.
"""
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch


KINDS = ('tap', 'hold', 'flick', 'drag')
TARGET_SLOTS = 512
SCHEMA = 'beatfcn.modal-event-data.v1'


def event_array(events):
    """Keep native microseconds and the complete multiset, including duplicates."""
    return np.asarray([(e.onset_us, e.end_us, KINDS.index(e.type), e.lane)
                       for e in events], dtype=np.int64).reshape(-1, 4)


class MertOnlyFeatureStore:
    """No Pupu arrays, PCA scales or Pupu summaries are opened by this loader."""
    def __init__(self, root, records, *, cache_songs=2):
        self.root = Path(root)
        self.records = {r['audio_sha256']: r for r in records}
        self._open = lru_cache(maxsize=cache_songs)(self._read)

    @classmethod
    def from_reports(cls, root, feature_report, mert_report, **kwargs):
        root = Path(root)
        features = json.loads((root / feature_report).read_text())
        mert = json.loads((root / mert_report).read_text())
        if not features['complete'] or features['failures']:
            raise ValueError('Complete admitted mel features required')
        if not mert['complete'] or mert['committed_songs'] != mert['planned_songs']:
            raise ValueError('Complete MERT cache required')
        admitted = {r['audio_sha256']: r for r in features['records'] if r['passed']}
        records = []
        for row in mert['records']:
            audio = row['audio_sha256']
            if audio not in admitted:
                continue
            feature = admitted[audio]
            if row['samples'] != feature['decoded_samples']:
                raise ValueError('MERT and mel sample clocks differ')
            records.append(dict(audio_sha256=audio, decoded_samples=row['samples'],
                duration_us=feature['duration_us'], mert_path=row['path'],
                complete_mel=feature['complete_mel'], mert_files=row['files'],
                mel_sha256=feature['sha256'][feature['complete_mel']]))
        return cls(root, records, **kwargs)

    def _read(self, audio):
        row = self.records[audio]
        folder = self.root / row['mert_path']
        layers = [np.load(folder / f'L{layer}.npy', mmap_mode='r', allow_pickle=False)
                  for layer in (12, 23)]
        times = np.load(folder / 'times_s.npy', allow_pickle=False)
        intervals = np.load(folder / 'intervals_s.npy', allow_pickle=False)
        mel = np.load(self.root / row['complete_mel'], mmap_mode='r', allow_pickle=False)
        samples = row['decoded_samples']
        frames = (samples + 959) // 960 * 4
        if mel.shape != (frames, 128) or intervals.shape != (len(times), 2):
            raise ValueError('Complete mel or native MERT interval shape changed')
        if any(v.shape != (len(times), 1024) for v in layers):
            raise ValueError('MERT layer shape changed')
        # Match MertCanvasFeatures._read exactly; this is only the legacy song
        # summary input. whole_mel below retains every original fine frame.
        bins = np.floor(times / .5).astype(np.int64)
        n = int(bins[-1]) + 1
        summary = np.zeros((n, 1024), np.float32)
        summary_times = np.zeros(n, np.float32)
        for b in range(n):
            lo, hi = np.searchsorted(bins, (b, b + 1))
            if hi <= lo:
                raise ValueError('Gap in MERT summary clock')
            summary[b] = np.asarray(layers[1][lo:hi], np.float32).mean(0)
            summary_times[b] = times[lo:hi].mean()
        indices = np.arange(frames)
        valid = indices * 240 < samples
        mel_times = indices * .01 + .005
        partial = valid & ((indices + 1) * 240 > samples)
        mel_times[partial] = (indices[partial] * 240 + samples) / 48000
        mel_intervals = np.stack((np.minimum(indices * 240, samples),
                                 np.minimum((indices + 1) * 240, samples)), -1) / 24000
        return layers, times, intervals, mel, summary, summary_times, mel_times, valid, mel_intervals

    def window(self, audio, start_us, end_us, *, include_whole=True):
        row = self.records[audio]
        samples = row['decoded_samples']
        if not 0 <= start_us < end_us or end_us * 24000 > samples * 1_000_000:
            raise ValueError('Request outside admitted audio')
        layers, times, intervals, mel, song, st, mt, valid, mi = self._open(audio)
        lo = max(0, start_us // 40000 - 100)
        nominal_hi = start_us // 40000 + 300
        hi = min(len(times), nominal_hi)
        if hi <= lo or end_us > nominal_hi * 40000:
            raise ValueError('Request outside fixed local context')
        mlo, mhi = lo * 4, min(len(mel), nominal_hi * 4)
        # Preserve the legacy arithmetic order for bitwise public-input parity.
        ids = np.arange(mlo, mhi)
        local_mel_times = ids * .01 + .005 - start_us / 1e6
        partial = valid[mlo:mhi] & ((ids + 1) * 240 > samples)
        local_mel_times[partial] = (ids[partial] * 240 + samples) / 48000 - start_us / 1e6
        result = dict(local=np.stack([np.asarray(v[lo:hi], np.float32) for v in layers])[:, :, None, :],
            local_times=(times[lo:hi] - start_us / 1e6).astype(np.float32),
            local_valid=np.ones(hi - lo, bool), song=song,
            song_times=(st - start_us / 1e6).astype(np.float32), song_valid=np.ones(len(st), bool),
            mel=np.ascontiguousarray(mel[mlo:mhi], dtype=np.float32),
            mel_times=local_mel_times.astype(np.float32), mel_valid=valid[mlo:mhi].copy(),
            local_intervals=(intervals[lo:hi] - start_us / 1e6).astype(np.float32))
        if include_whole:
            result.update(whole_mel=np.asarray(mel, dtype=np.float32),
                whole_mel_times=(mt - start_us / 1e6).astype(np.float32),
                whole_mel_valid=valid.copy(),
                whole_mel_intervals=(mi - start_us / 1e6).astype(np.float32))
        return result


class ModalEventDataset(torch.utils.data.Dataset):
    """Prepared rows; target arrays and references are immutable CPU artifacts."""
    def __init__(self, root, manifest, *, split=None, views=None, cache_songs=2):
        self.root = Path(root)
        self.manifest_path = self.root / manifest
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != SCHEMA or not self.manifest.get('complete'):
            raise ValueError('Complete frozen Modal dataset required')
        if self.manifest['target_slots'] != TARGET_SLOTS:
            raise ValueError('Fixed public 512-query contract changed')
        self.features = MertOnlyFeatureStore(self.root, self.manifest['audio_records'], cache_songs=cache_songs)
        allowed_views = None if views is None else set(views)
        self.records = [r for r in self.manifest['records']
                        if (split is None or r['split'] == split)
                        and (allowed_views is None or r['view'] in allowed_views)]
        self.arrays = {name: np.load(self.root / path, mmap_mode='r', allow_pickle=False)
                       for name, path in self.manifest['arrays'].items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        record = self.records[item]
        request = record['request']
        x = self.features.window(record['audio_sha256'], request['start_us'], request['end_us'])
        a, b = record['reference_slice']
        x.update(reference=np.asarray(self.arrays['reference'][a:b]),
            reference_times=np.asarray(self.arrays['reference_times'][a:b]),
            reference_valid=np.ones(b - a, bool),
            controls=np.asarray(self.arrays['controls'][record['array_index']]),
            mode=np.asarray(1, np.int64))
        for key in ('start_us', 'end_us', 'duration_us'):
            x[key] = np.asarray(request[key], np.int64)
        x.update(origin_s=np.asarray(request['start_us'] / 1e6, np.float32),
            end_s=np.asarray(request['end_us'] / 1e6, np.float32),
            duration_s=np.asarray(request['duration_us'] / 1e6, np.float32),
            joint_frame_times=np.arange((request['end_us'] - request['start_us'] + 39999) // 40000,
                                       dtype=np.float32) * .04,
            copy_times_us=np.asarray(request['copy_times'], np.int64),
            known_carry_us=np.asarray(record['known_carry_us'], np.int64).reshape(-1, 4),
            known_regions_us=np.asarray(record['known_regions_us'], np.int64).reshape(-1, 2),
            allowed_types=np.asarray(record['allowed_types'], bool))
        a, b = record['head_layout_slice']
        x.update(request_start_us=x['start_us'], request_end_us=x['end_us'],
            copy_us=x['copy_times_us'],
            head_us=np.asarray(self.arrays['head_offsets'][a:b]) + request['start_us'],
            type_allowed=np.asarray(self.arrays['head_type_allowed'][a:b]))
        a, b = record['target_slice']
        return dict(inputs=x, targets=np.asarray(self.arrays['targets'][a:b]), metadata=record)

    def public_audio_bucket(self, item, quantum=4096):
        record = self.records[item]
        frames = (self.features.records[record['audio_sha256']]['decoded_samples'] + 959) // 960 * 4
        return math.ceil(frames / quantum) * quantum


def collate_event_batch(rows, *, whole_mel_quantum=4096):
    """Pure numeric CPU collation. Hidden N never changes model query slots."""
    if not rows or whole_mel_quantum < 1:
        raise ValueError('Nonempty batch and positive public audio bucket required')
    fixed = {'local': 400, 'mel': 1600, 'song': 1024, 'reference': 64}
    scalar = {'mode', 'start_us', 'end_us', 'duration_us', 'request_start_us', 'request_end_us',
              'origin_s', 'end_s', 'duration_s', 'controls', 'allowed_types'}
    result = {}
    for key in rows[0]['inputs']:
        values = [np.asarray(r['inputs'][key]) for r in rows]
        if key in scalar:
            joined = np.stack(values)
        else:
            axis = 1 if key == 'local' else 0
            length = max(v.shape[axis] for v in values)
            stream = next((s for s in ('whole_mel', 'local', 'mel', 'song', 'reference')
                           if key == s or key.startswith(s + '_')), None)
            if stream == 'whole_mel':
                length = math.ceil(length / whole_mel_quantum) * whole_mel_quantum
            elif stream in fixed:
                length = max(fixed[stream], math.ceil(length / 64) * 64 if stream in ('reference', 'song') else length)
            shape = list(values[0].shape)
            shape[axis] = length
            joined = np.zeros((len(rows), *shape), dtype=values[0].dtype)
            for i, value in enumerate(values):
                index = [slice(None)] * value.ndim
                index[axis] = slice(0, value.shape[axis])
                joined[(i, *index)] = value
        result[key] = torch.from_numpy(np.ascontiguousarray(joined))
    for name in ('copy_times', 'known_carry', 'known_regions'):
        key = name + '_us'
        lengths = [len(r['inputs'][key]) for r in rows]
        result[name + '_valid'] = torch.arange(result[key].shape[1])[None] < torch.tensor(lengths)[:, None]
    result['copy_valid'] = result['copy_times_valid']
    result['head_valid'] = torch.arange(result['head_us'].shape[1])[None] < torch.tensor(
        [len(r['inputs']['head_us']) for r in rows])[:, None]
    targets = np.zeros((len(rows), TARGET_SLOTS, 4), np.int64)
    target_valid = np.zeros((len(rows), TARGET_SLOTS), bool)
    for i, row in enumerate(rows):
        n = len(row['targets'])
        if n > TARGET_SLOTS:
            raise ValueError(f'Explicit target OVERFLOW: {n} > {TARGET_SLOTS}')
        targets[i, :n] = row['targets']
        target_valid[i, :n] = True
    return dict(inputs=result, targets=torch.from_numpy(targets),
                target_valid=torch.from_numpy(target_valid), metadata=[r['metadata'] for r in rows])


class PublicDurationBatchSampler(torch.utils.data.Sampler):
    """Batch by public fine-audio length; no target count or type is inspected."""
    def __init__(self, dataset, batch_size, *, seed=0, shuffle=True, quantum=4096):
        if batch_size < 1 or quantum < 1:
            raise ValueError('Positive batch size and bucket quantum required')
        self.dataset, self.batch_size = dataset, batch_size
        self.seed, self.shuffle = seed, shuffle
        self.buckets = defaultdict(list)
        for i in range(len(dataset)):
            self.buckets[dataset.public_audio_bucket(i, quantum)].append(i)

    def __len__(self):
        return sum(math.ceil(len(v) / self.batch_size) for v in self.buckets.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches = []
        for values in self.buckets.values():
            ids = list(values)
            if self.shuffle:
                rng.shuffle(ids)
            batches.extend(ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

"""Small, read-only local panels over existing audio caches and complete events.

Only a few request windows are cached in host memory. This is an MPS exploration
loader, not a claim that the full-corpus CUDA transport has been benchmarked.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.event_canvas import RhythmEvent


ONSET_KEYS = ('onset_targets', 'onset_valid')


def _collate_onset(key, values, rows, inputs):
    """Pad per-frame onset supervision to its audio stream's padded frame length.

    Rows carry one onset row per frame of the stream the onset head reads (mel
    rows when mel_bins, else local MERT frames); that stream is identified by
    matching every row's frame count. Padding is 0 / False (never supervised).
    """
    lengths = [len(v) for v in values]
    frames = dict(mel=[len(r['inputs']['mel']) for r in rows] if 'mel' in inputs else None,
                  local=[r['inputs']['local'].shape[1] for r in rows] if 'local' in inputs else None)
    stream = next((name for name, counts in frames.items() if counts == lengths), None)
    if stream is None:
        raise ValueError(f'{key} does not align with the mel or local frames of every row')
    length = inputs['mel'].shape[1] if stream == 'mel' else inputs['local'].shape[2]
    joined = np.zeros((len(rows), length, *values[0].shape[1:]), values[0].dtype)
    for i, v in enumerate(values):
        joined[i, :len(v)] = v
    return torch.from_numpy(joined)


def collate(rows, *, input_shapes=None, sequence_buckets=None):
    """Allocate final graph shapes once, keeping every valid value unchanged."""
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    input_shapes = input_shapes or {}
    if set(input_shapes) - {'local', 'mel', 'song', 'reference', 'intensity'}:
        raise ValueError('Unknown fixed input stream')
    batches=[]
    for section in ('inputs','sequence'):
        result={}
        if any(set(row[section]) != set(rows[0][section]) for row in rows):
            raise ValueError('Batch rows have different fields')
        sequence_length = None
        if section == 'sequence':
            sequence_length = max(len(row[section]['targets']) for row in rows)
            if sequence_buckets:
                sequence_length = next((v for v in sequence_buckets
                    if v >= sequence_length), None)
                if sequence_length is None:
                    raise ValueError('Sequence exceeds the largest frozen bucket')
            else:
                sequence_length = ((sequence_length+127)//128)*128
        for key in rows[0][section]:
            values=[r[section][key] for r in rows]
            if section=='sequence' and key in ONSET_KEYS:
                result[key]=_collate_onset(key,values,rows,batches[0])
                continue
            if key in ('mode','controls'):
                joined=np.stack(values)
            elif key=='local':
                actual=max(v.shape[1] for v in values)
                length=input_shapes.get('local',actual)
                if length < actual:
                    raise ValueError('local exceeds the frozen input shape')
                joined=np.zeros((len(rows),values[0].shape[0],length,
                                 *values[0].shape[2:]),values[0].dtype)
                for i,v in enumerate(values):joined[i,:,:len(v[0])]=v
            else:
                length=max(map(len,values))
                if section=='sequence':
                    length=sequence_length
                else:
                    stream=key.removesuffix('_times').removesuffix('_valid')
                    # Charter-timing features ride on the 10ms mel frames.
                    stream='mel' if stream=='beat' else stream
                    fixed=input_shapes.get(stream,length)
                    if fixed < length or fixed < 1:
                        raise ValueError(f'{key} exceeds the frozen input shape')
                    length=fixed
                joined=np.zeros((len(rows),length,*values[0].shape[1:]),values[0].dtype)
                for i,v in enumerate(values):joined[i,:len(v)]=v
            result[key]=torch.from_numpy(joined)
        if section=='sequence':result['allowed'][...,0]|=~result['valid']
        batches.append(result)
    return tuple(batches)


class LocalEventPanel:
    def __init__(self,root,manifest):
        self.root=Path(root)
        self.manifest=json.loads((self.root/manifest).read_text())
        self.records=self.manifest['records']
        self.scales=np.load(self.root/self.manifest['scales'])
        self.sequence=[]
        for row in self.records:
            request=EventRequest(**row['request'])
            tokens=encode_events([RhythmEvent(**e) for e in row['target_events']],request)
            self.sequence.append(teacher_sequence(tokens,request))

    @lru_cache(maxsize=6)
    def _audio(self,sha,c0,duration):
        layers=(9,16,24)
        base=self.root/'inputs/r09_positive_pairs/taps/v4_i8_L9-16-24'/sha
        raw=[np.load(base/f'L{l}.npy',mmap_mode='r') for l in layers]
        lo,hi=max(0,c0//40_000-100),min(len(raw[0]),c0//40_000+300)
        mel=np.load(self.root/'inputs/d65_local/mel10'/(sha+'.npy'),mmap_mode='r')
        if (c0+8_000_000)>min(duration,len(mel)*10_000):
            raise ValueError('Requested target extends beyond existing mel coverage')
        local=np.stack([np.asarray(x[lo:hi],np.float32)*self.scales[f'scale_{l}'] for x,l in zip(raw,layers)])
        summary=self.root/'inputs/d65_local/song_summary'/(sha+'.npz')
        s=np.load(summary)
        return dict(local=local,local_times=(np.arange(lo,hi)*.04+.02-c0/1e6).astype(np.float32),
            local_valid=np.ones(hi-lo,bool),song=s['values'],song_times=(s['times']-c0/1e6).astype(np.float32),
            song_valid=np.ones(len(s['values']),bool),mel=np.ascontiguousarray(mel[lo*4:hi*4],dtype=np.float32),
            mel_times=(np.arange(lo*4,hi*4)*.01+.005-c0/1e6).astype(np.float32),mel_valid=np.ones((hi-lo)*4,bool))

    def row(self,index,*,controls=None):
        r=self.records[index];q=r['request']
        audio=self._audio(r['audio_sha256'],q['start_us'],q['duration_us'])
        ref=np.zeros((1,16),np.float32);ref[0,1]=(q['end_us']-q['start_us'])/1_200_000_000;ref[0,15]=1
        return dict(inputs={**audio,'reference':ref,'reference_times':np.zeros(1,np.float32),
            'reference_valid':np.ones(1,bool),'controls':np.asarray(r['controls'] if controls is None else controls,np.float32),
            'mode':np.asarray(0 if q['mode']=='mania_4k' else 1,np.int64)},sequence=self.sequence[index])

    def batch(self,indices,*,controls=None,device='cpu'):
        inp,seq=collate([self.row(i,controls=None if controls is None else controls[j]) for j,i in enumerate(indices)])
        return ({k:v.to(device) for k,v in inp.items()},{k:v.to(device) for k,v in seq.items()})

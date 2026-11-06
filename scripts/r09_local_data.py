"""Local real-feature access for the isolated R09 comparison and its probes."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from beatfcn.events.canonical import Event
from beatfcn.model import features as F
from beatfcn.model.batch import SongCache, SongInfo, attach_audio, collate_meta, dequant
from beatfcn.train.data import QuantArray
from beatfcn.train.positive_pairs import condition_view

ROOT=Path(__file__).resolve().parents[1]


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''): h.update(chunk)
    return h.hexdigest()


def verify_files(plan):
    for name,expected in plan['input_sha256'].items():
        if file_sha(ROOT/name)!=expected: raise ValueError(f'input identity changed: {name}')


class LocalPairs:
    def __init__(self, plan):
        self.plan=plan
        self.catalog=torch.load(ROOT/plan['catalog'],map_location='cpu',weights_only=False)
        self.pairs={p['pair_id']:p for p in self.catalog['train_pairs']}
        scales=np.load(ROOT/plan['scale_file'])
        self.scales={l:scales[f'scale_{l}'].astype(np.float32) for l in (9,16,24)}
        if any(not np.isfinite(x).all() or (x<=0).any() for x in self.scales.values()):
            raise ValueError('invalid real PCA scales')
        self.cache=OrderedDict()

    def arrays(self, audio):
        if audio in self.cache:
            self.cache.move_to_end(audio)
            return self.cache[audio]
        path=ROOT/self.plan['feature_root']/'taps/v4_i8_L9-16-24'/audio
        # Copy-on-write mappings are writable to Torch but never mutate the disk.
        taps={l:np.load(path/f'L{l}.npy',mmap_mode='c') for l in self.scales}
        summary=np.load(path/'S20.npy',mmap_mode='c')
        if any(a.dtype!=np.int8 or a.shape[1:]!=(8,1024) for a in taps.values()):
            raise ValueError('real feature representation differs')
        lengths={a.shape[0] for a in taps.values()}
        if len(lengths)!=1 or summary.dtype!=np.float16 or summary.shape[1:]!=(8,1024):
            raise ValueError('actual audio/summary shape or type differs')
        if not np.isfinite(summary).all():
            raise ValueError('nonfinite actual song summary')
        value=dict(taps=taps,summary=summary)
        self.cache[audio]=value
        while len(self.cache)>2: self.cache.popitem(last=False)
        return value

    def entry(self, item): return self.pairs[item['pair_id']]['entries'][item['side']]

    def example(self, item):
        return condition_view(self.entry(item)['example'],item['field'])

    def song_info(self, audio):
        arrays=self.arrays(audio)
        return SongInfo(audio,arrays['taps'][9].shape[0],arrays['summary'].shape[0])

    def batch(self, items, *, initial=False):
        examples=[self.example(item) for item in items]
        if initial:
            examples=[replace(ex,groups=[],right=ex.right[:1].copy()) for ex in examples]
        audio=[self.entry(item)['row']['audio_sha256'] for item in items]
        infos=[self.song_info(sha) for sha in audio]
        bt=collate_meta(examples,infos,(9,16,24),24,1024,1024,True)
        def song(sha):
            arrays=self.arrays(sha)
            return SimpleNamespace(taps={l:torch.from_numpy(a) for l,a in arrays['taps'].items()},
                summary=torch.from_numpy(arrays['summary']),mel10=None)
        return dequant(attach_audio(bt,song,device='cpu',scales=self.scales))

    def generation(self, item, *, field_override=None):
        entry=self.entry(item);row=entry['row'];arrays=self.arrays(row['audio_sha256'])
        ex=self.example(item)
        controls=entry['request']['controls']
        requested={'allowed_types':list(controls['allowed_types'])}
        if item['field']=='all': requested.update(controls)
        else: requested[item['field']]=controls[item['field']]
        if field_override is not None:
            ex=replace(ex,controls=np.asarray(field_override,np.int64).copy())
            from beatfcn.events.controls import BAND_EDGES
            requested={'allowed_types':list(controls['allowed_types'])}
            for name,value in zip(F.CONTROL_FIELDS,ex.controls):
                if value>=0: requested[name]=list(BAND_EDGES[name][value:value+2])
        request={**entry['request'],'controls':requested}
        truth=[Event.from_json(x) for x in entry['truth_events']]
        sample=SimpleNamespace(request=request,hidden=truth,evidence=())
        # mel10 is not read by generate_core; evaluation requiring mel must load
        # actual mel separately and must not treat this wrapper as audio evidence.
        song=SongCache({l:QuantArray(a,self.scales[l]) for l,a in arrays['taps'].items()},
            arrays['summary'],np.empty((0,128),np.float16),round(row['audio_sec']*F.SEC))
        return ex,sample,song,truth,row

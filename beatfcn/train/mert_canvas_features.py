"""Native MERT2 L12/L23 windows with unchanged admitted 10ms mel.

No replication into fictitious frequency bands. Cache clocks and full/stitched
song context remain explicit; chart labels never enter this feature store.
"""
from functools import lru_cache
import json

import numpy as np


class MertCanvasFeatures:
    def __init__(self,root,pupu_store,cache_report):
        self.root=root;self.pupu=pupu_store;self.records=pupu_store.records
        report=json.loads((root/cache_report).read_text())
        if not report['complete'] or report['committed_songs']!=report['planned_songs']:
            raise ValueError('MERT cache incomplete')
        self.mert={r['audio_sha256']:r for r in report['records']}
        self._open=lru_cache(maxsize=2)(self._read)
        self.window=lru_cache(maxsize=6)(self._window)

    def _read(self,audio):
        rec=self.mert[audio];folder=self.root/rec['path']
        layers=[np.load(folder/f'L{l}.npy',mmap_mode='r',allow_pickle=False) for l in (12,23)]
        times=np.load(folder/'times_s.npy',allow_pickle=False)
        bins=np.floor(times/.5).astype(np.int64);n=int(bins[-1])+1
        values=np.zeros((n,1024),np.float32);summary_times=np.zeros(n,np.float32)
        for b in range(n):
            lo,hi=np.searchsorted(bins,(b,b+1))
            if hi<=lo:raise ValueError('Gap in MERT summary clock')
            values[b]=np.asarray(layers[1][lo:hi],np.float32).mean(0)
            summary_times[b]=times[lo:hi].mean()
        return layers,times,values,summary_times

    def _window(self,audio,start_us,end_us):
        layers,times,song,st=self._open(audio)
        shared=self.pupu.window(audio,start_us,end_us)
        samples=self.records[audio]['decoded_samples']
        if self.mert[audio]['samples']!=samples:raise ValueError('Frontend decoded clocks differ')
        lo=max(0,start_us//40000-100);hi=min(len(times),start_us//40000+300)
        if end_us>min(samples/24000*1e6,(start_us//40000+300)*40000):
            raise ValueError('MERT request beyond admitted audio/context')
        local=np.stack([np.asarray(v[lo:hi],np.float32) for v in layers])[:,:,None,:]
        return {**shared,'local':local,'local_times':(times[lo:hi]-start_us/1e6).astype(np.float32),
            'local_valid':np.ones(hi-lo,bool),'song':song,
            'song_times':(st-start_us/1e6).astype(np.float32),'song_valid':np.ones(len(st),bool)}

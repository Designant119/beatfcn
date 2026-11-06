"""Actual complete admitted audio mel summaries, never local-only pooling."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .canonical_probability import probability_batch


def multiscale_flux(mel):
    values=[mel]
    for lag in (1,4,16):
        previous=np.concatenate((np.repeat(mel[:1],lag,axis=0),mel[:-lag]),axis=0) if len(mel)>=lag else np.repeat(mel[:1],len(mel),axis=0)
        values.append(np.maximum(mel-previous,0))
    return np.concatenate(values,-1).astype(np.float32)


def whole_summary(mel,samples):
    frames=(samples+239)//240
    if frames>120000:raise ValueError('Whole-song local resource exceeds1200seconds')
    mel=np.asarray(mel[:frames],np.float32)
    flux=multiscale_flux(mel)
    n=(frames+199)//200
    out=np.zeros((n,2060),np.float32)
    for block in range(n):
        vectors=[];statistics=[]
        for quarter in range(4):
            start=block*200+quarter*50;end=min(start+50,frames)
            if start>=frames:
                vectors.append(np.zeros(512,np.float32));statistics.extend((0.,0.,0.));continue
            v=mel[start:end]
            vectors.append(flux[start:end].mean(0));statistics.extend((float(v.mean()),float(v.std()),float(v.max())))
        out[block]=np.concatenate((*vectors,np.asarray(statistics,np.float32)))
    starts=np.arange(n)*2
    times=((starts+np.minimum(starts+2,samples/24000))/2).astype(np.float32)
    return out,times


class WholeMelStore:
    def __init__(self,root,records,cache):
        self.root=Path(root);self.records=records;self.cache=Path(cache)
        self.open=lru_cache(maxsize=4)(self._open)

    def _open(self,audio):
        path=self.cache/f'{audio}.npz'
        if not path.exists():
            rec=self.records[audio]
            mel=np.load(self.root/rec['complete_mel'],mmap_mode='r',allow_pickle=False)
            summary,times=whole_summary(mel,rec['decoded_samples'])
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,summary=summary,times=times)
        with np.load(path,allow_pickle=False) as data:return data['summary'].copy(),data['times'].copy()

    def batch(self,example,audio,device='mps',*,mask_reference=False,remote_permutation=False):
        inputs,target=probability_batch([example],device='cpu',mask_reference=mask_reference)
        inputs={k:v for k,v in inputs.items() if not k.startswith(('local','song'))}
        mel=inputs['mel'][0].numpy();inputs['mel_flux']=torch.from_numpy(multiscale_flux(mel))[None]
        summary,times=self.open(audio)
        if remote_permutation:
            summary=summary.copy()
            # Keep complete coarse blocks touching local16seconds intact.
            start=example['request'].start_us/1e6
            fixed=(times+1>start-4)&(times-1<start+12)
            indices=np.flatnonzero(~fixed);summary[indices]=summary[indices[::-1]]
        inputs['whole_song']=torch.from_numpy(summary)[None]
        inputs['whole_song_times']=torch.from_numpy(times-example['request'].start_us/1e6)[None]
        inputs['whole_song_valid']=torch.ones(1,len(times),dtype=torch.bool)
        return {k:v.to(device) for k,v in inputs.items()},{k:v.to(device) for k,v in target.items()}

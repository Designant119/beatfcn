"""Frozen, versioned MERT2 extraction with explicit clocks and retained tails.

This is an alternative frontend, not a change to canonical chart timestamps.
Long-song windows are recorded and are not claimed equivalent to a full forward.
"""
from dataclasses import asdict,dataclass
import math
from pathlib import Path
import numpy as np
import torch
from .preprocess import SR, CELL_SAMPLES, HOP, waveform_mel, PREPROCESS_RECIPE

REPO='m-a-p/MERT-v2-FullSong'
REVISION='7001f7657276238629e61c7746ce70735e0604c0'
LOCAL_SOURCE_SHA256={
    'config.json':'f2e194895f58be3ddba327255db129ff0e3bee550cc0ecf08e4d22d79ce3bca3',
    'model.safetensors':'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d',
    'configuration_mert2.py':'77b53ec9d7ee31a599d744fb006e812c7eeaf7390deb46e2f460cf8c17b00bd6',
    'modeling_mert2.py':'b1a3174e5649c4b26b0c90d8626f0adacfbbba111a58ed3bb72ad651945a2f5c',
}


@dataclass(frozen=True)
class Mert2Spec:
    layers: tuple[int,...]=(12,23)
    context_limit_s: int=360
    long_window_s: int=300
    overlap_s: int=100

    def __post_init__(self):
        if not self.layers or len(set(self.layers))!=len(self.layers) or any(not 1<=i<=24 for i in self.layers):
            raise ValueError('MERT2 layers are distinct one-based blocks 1..24')
        if not 0 < self.overlap_s < self.long_window_s <= self.context_limit_s:
            raise ValueError('Invalid long-song context policy')


def windows(samples,spec):
    if samples<1:raise ValueError('Empty audio')
    if samples<=spec.context_limit_s*SR:return [(0,samples)]
    width=spec.long_window_s*SR;stride=(spec.long_window_s-spec.overlap_s)*SR
    if width%CELL_SAMPLES or stride%CELL_SAMPLES:raise ValueError('Window clock must align to 40ms cells')
    chunks=[];start=0
    while True:
        end=min(samples,start+width);chunks.append((start,end))
        if end==samples:return chunks
        start+=stride


def pad_tail(wave,minimum=1025):
    wave=np.asarray(wave,dtype=np.float32)
    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Expected finite nonempty mono audio')
    padded=math.ceil(max(len(wave),minimum)/CELL_SAMPLES)*CELL_SAMPLES
    return np.pad(wave,(0,padded-len(wave)))


def fine_mel(wave):
    """Preserve the final partial 10ms interval; nominal centers stay at +5ms."""
    padded=pad_tail(wave)
    n=math.ceil(len(wave)/HOP)
    # SongPool's direct pinned-file reader requires C-order numeric .npy data.
    values=waveform_mel(padded)[0,0,:n].numpy().astype(np.float16,order='C')
    starts=np.arange(n,dtype=np.int64)*HOP
    return dict(values=values,times_s=(starts+HOP/2)/SR,
                intervals_s=np.stack((starts,np.minimum(starts+HOP,len(wave))),axis=-1)/SR)


def validate_local_source(path,expected_files=None):
    """Require the exact pinned local snapshot, including custom MERT2 code."""
    import hashlib
    folder=Path(path)
    if not folder.is_dir():raise ValueError('Local MERT2 source must be a directory')
    for name,expected in (expected_files or LOCAL_SOURCE_SHA256).items():
        value=folder/name
        if not value.is_file():raise ValueError('Local MERT2 source lacks '+name)
        digest=hashlib.sha256()
        with value.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise ValueError('Local MERT2 source identity changed: '+name)
    return str(folder.resolve())


class FrozenMert2:
    def __init__(self,device='cuda',spec=None,cache_dir=None,local_source=None):
        from transformers import AutoModel
        self.spec=spec or Mert2Spec()
        self.device=device
        self.revision=REVISION
        source=REPO if local_source is None else validate_local_source(local_source)
        model,loading=AutoModel.from_pretrained(source,
            revision=None if local_source is not None else REVISION,
            code_revision=None if local_source is not None else REVISION,
            trust_remote_code=True,attn_implementation='sdpa',cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Pinned checkpoint did not load exactly: '+str(loading))
        self.model=model.to(device).eval().requires_grad_(False)
        c=self.model.config
        if (c.hidden_size,c.num_hidden_layers,c.sampling_rate,c.inputs_to_logits_ratio)!=(1024,24,SR,CELL_SAMPLES):
            raise ValueError('Pinned MERT2 configuration mismatch')

    def _waveform_tensor(self, padded):
        return torch.from_numpy(padded)[None].to(self.device)

    @torch.inference_mode()
    def encode(self,wave):
        wave=np.asarray(wave,dtype=np.float32)
        pad_tail(wave)  # Check the actual signal before deciding windows.
        chunks=windows(len(wave),self.spec)
        kept=[];frame_ids=[];receipts=[]
        for i,(start,end) in enumerate(chunks):
            padded=pad_tail(wave[start:end])
            signal=self._waveform_tensor(padded)
            # Deliberate caller-owned tail zeros produce a final partial cell.
            # A sample-length mask would trim it again in the official forward.
            output=self.model(input_values=signal,output_hidden_states=True)
            features=torch.stack([output.hidden_states[layer-1][0] for layer in self.spec.layers])
            ids=np.arange(features.shape[1])*CELL_SAMPLES+start
            nominal=ids+360  # 15ms nominal receptive-field center; not a label correction.
            left=0 if i==0 else (chunks[i-1][1]+start)//2
            right=len(wave) if i+1==len(chunks) else (end+chunks[i+1][0])//2
            mask=(nominal>=left)&((nominal<right)|(i+1==len(chunks)))&(ids<len(wave))
            keep=torch.from_numpy(np.flatnonzero(mask)).to(self.device)
            values=features.index_select(1,keep).float().cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():raise ValueError('Nonfinite MERT2 output')
            kept.append(values);frame_ids.append(ids[mask])
            receipts.append(dict(start_sample=start,end_sample=end,padded_samples=len(padded),
                                 kept_frames=int(mask.sum()),retain_interval_samples=[left,right]))
        ids=np.concatenate(frame_ids)
        expected=np.arange(math.ceil(len(wave)/CELL_SAMPLES))*CELL_SAMPLES
        if not np.array_equal(ids,expected):raise ValueError('Gap/overlap in stitched feature clock')
        return dict(layers=np.concatenate(kept,axis=1),times_s=(ids+360)/SR,
            intervals_s=np.stack((ids,np.minimum(ids+CELL_SAMPLES,len(wave))),axis=-1)/SR,
            metadata=dict(repo=REPO,revision=REVISION,spec=asdict(self.spec),sample_rate=SR,
                real_samples=len(wave),nominal_first_center_ms=15,measured_acoustic_offset_ms=None,
                full_song_context=len(chunks)==1,windows=receipts,
                encoded_samples=sum(r['padded_samples'] for r in receipts),
                fine_mel_recipe=PREPROCESS_RECIPE))

"""Extend the identified D58 fixed batch with real, clocked 10ms mel evidence."""
from pathlib import Path
import hashlib,json
import numpy as np


def main():
    src=Path('inputs/diagnostic_panels/d58-full-event');dst=Path('inputs/diagnostic_panels/d64-full-event-mel')
    if dst.exists():raise FileExistsError(dst)
    original=json.loads((src/'manifest.json').read_text())
    arrays=dict(np.load(src/'inputs.npz'))
    for row in original['records']:
        i=row['index'];p=Path('inputs/d64_probe_mel')/(row['audio_sha256']+'.npy')
        raw=np.load(p).astype(np.float32)
        if raw.ndim!=2 or raw.shape[1]!=128 or not np.isfinite(raw).all():raise ValueError('invalid actual mel')
        lo,hi=[v*4 for v in row['local_cells']]
        mel=raw[lo:hi]
        if len(mel)!=hi-lo:raise ValueError('Missing local mel coverage')
        times=(np.arange(lo,hi)*.01+.005-row['source_core_start_us']/1e6).astype(np.float32)
        arrays[f'{i}:input:mel']=mel;arrays[f'{i}:input:mel_times']=times
        arrays[f'{i}:input:mel_valid']=np.ones(len(mel),bool)
        row['mel_identity']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            first_center_ms=5,hop_ms=10,local_frames=[lo,hi],
            limitation='Existing cache used only for interior fixed-batch implementation test; original tail repair not inherited.')
    dst.mkdir(parents=True)
    np.savez_compressed(dst/'inputs.npz',**arrays)
    original.update(id='D64-full-event-real-mel-inputs',parent_manifest_sha256=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),
                    tensor_sha256=hashlib.sha256((dst/'inputs.npz').read_bytes()).hexdigest())
    (dst/'manifest.json').write_text(json.dumps(original,indent=2,ensure_ascii=False)+'\n')
    print(dst)

if __name__=='__main__':main()

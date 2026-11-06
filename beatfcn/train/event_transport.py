"""Full-event adapter to the existing tape, pinned ring, and resident song pool.

No chart parsing, sampling, or full-song CPU copies occur in next(). All tensor
shapes and staging banks are fixed before training. This is not a new sampler.
"""
from collections import deque
import time
import torch
from beatfcn.model.batch import pack_batch, unpack_batch
from .pinned import PinnedRing

LARGE_AUDIO={'local','song','mel'}


def pack_event_batch(inputs,sequence,audio,*,condition_contrast=None):
    """CPU preparation only. Audio recipe binds slices to separately cached songs."""
    if not audio.get('keys') or len(audio['keys'])!=len(inputs['mode']):
        raise ValueError('Audio recipe batch mismatch')
    if any(v.device.type!='cpu' for v in [*inputs.values(),*sequence.values()]):
        raise ValueError('Materialize event tape before GPU allocation')
    valid=sequence['valid']
    if bool((valid[:,1:] & ~valid[:,:-1]).any()) or bool((valid.sum(-1)==0).any()):
        raise ValueError('Independent nonempty requests must be right padded')
    if not bool((sequence['allowed'].gather(-1,sequence['targets'][...,None]).squeeze(-1)|~valid).all()):
        raise ValueError('A real target violates prefix-only grammar')
    if condition_contrast is not None:
        from .event_conditions import validate_pair_metadata
        rows,opposing=condition_contrast['pair_rows'],condition_contrast['opposing_controls']
        validate_pair_metadata(inputs,rows,opposing)
        inputs={**inputs,'condition_pair_rows':rows,'opposing_controls':opposing}
    bt={f'input:{k}':v for k,v in inputs.items() if k not in LARGE_AUDIO}
    bt.update({f'sequence:{k}':v for k,v in sequence.items()})
    bt['_event_shapes']={k:tuple(inputs[k].shape) for k in LARGE_AUDIO if k in inputs}
    bt['_audio']={**audio,'ref_reads':[]}
    return pack_batch(bt)


def split_event_batch(batch):
    return ({k[6:]:v for k,v in batch.items() if k.startswith('input:')},
            {k[9:]:v for k,v in batch.items() if k.startswith('sequence:')})


class EventTransport:
    def __init__(self,tape,pool,*,device='cuda',prefetch=8,slots=2,max_audio_buffer_bytes=8*2**30,
                 scales=None):
        if prefetch<1 or slots<2:raise ValueError('At least two device banks and positive prefetch required')
        self.device=torch.device(device);self.pool=pool;self.scales=scales or {}
        self.prefetch=prefetch;self.slots=slots;self.index=0;self.queue=deque();self.closed=False
        batches=tape.batches[tape.skip_batches:]
        if not batches:raise ValueError('Empty event tape')
        self.raw=iter(batches)
        largest={};shapes={}
        for packed,_ in batches:
            for name,buf in packed['_packed'].items():
                if name not in largest or buf.numel()>largest[name].numel():largest[name]=buf
            signature=tuple(sorted(packed['_event_shapes'].items()))
            shapes[signature]=packed['_event_shapes']
        audio_bytes=slots*sum(4*torch.Size(shape).numel() for bucket in shapes.values() for shape in bucket.values())
        if audio_bytes>max_audio_buffer_bytes:raise ValueError('Shape buckets exceed fixed audio buffer budget')
        self.ring=PinnedRing(depth=slots,pin=self.device.type=='cuda',copy_method='bytes')
        self.ring.preallocate(largest)
        self.metadata=[{k:torch.empty_like(v,device=self.device) for k,v in largest.items()} for _ in range(slots)]
        self.audio={sig:[{k:torch.empty(shape,dtype=torch.float32,device=self.device) for k,shape in shape_map.items()}
                        for _ in range(slots)] for sig,shape_map in shapes.items()}
        self.stats=dict(batches=0,loader_wait_s=0.,stage_s=0.,attach_s=0.,audio_buffer_bytes=audio_bytes)
        self.exhausted=False

    def _fill(self):
        while len(self.queue)<self.prefetch and not self.exhausted:
            try:batch,skipped=next(self.raw)
            except StopIteration:self.exhausted=True;break
            self.pool.request(batch['_audio']['keys'])
            self.queue.append((batch,skipped))

    @staticmethod
    def _copy_slice(source,start,out,scale=None):
        end=start+out.shape[0]
        if start<0 or end>source.shape[0]:raise ValueError('Audio recipe exceeds retained feature coverage')
        src=source[start:end]
        if src.ndim==2 and out.ndim==3:src=src[:,None,:]
        if src.shape!=out.shape:raise ValueError('Cache/target audio shape mismatch')
        out.copy_(src)
        if scale is not None:out.mul_(scale)

    def __iter__(self):return self

    def __next__(self):
        if self.closed:raise StopIteration
        t=time.perf_counter();self._fill()
        if not self.queue:self.close();raise StopIteration
        packed,skipped=self.queue.popleft();recipe=packed['_audio']
        slot=self.index%self.slots;self.index+=1
        try:
            # The source features have already been requested several steps ahead.
            songs=[self.pool.get(k) for k in recipe['keys']]
            self.stats['loader_wait_s']+=time.perf_counter()-t
            t=time.perf_counter()
            staged,host_slot=self.ring.take(packed['_packed'])
            on_device={}
            for key,buf in staged.items():
                dst=self.metadata[slot][key][:buf.numel()]
                dst.copy_(buf,non_blocking=self.device.type=='cuda');on_device[key]=dst
            if self.device.type=='cuda':self.ring.done(host_slot,torch.cuda.current_stream(self.device))
            inputs,seq=split_event_batch(unpack_batch({**packed,'_packed':on_device}))
            self.stats['stage_s']+=time.perf_counter()-t;t=time.perf_counter()
            sig=tuple(sorted(packed['_event_shapes'].items()));buffers=self.audio[sig][slot]
            for value in buffers.values():value.zero_()
            for i,song in enumerate(songs):
                lo,n=recipe['local_slices'][i]
                for j,layer in enumerate(recipe['layers']):
                    scale=self.scales.get(layer)
                    if song.taps[layer].dtype==torch.int8 and scale is None:raise ValueError('Missing int8 frontend scale')
                    self._copy_slice(song.taps[layer],lo,buffers['local'][i,j,:n],scale)
                ss,ns=recipe['song_slices'][i]
                self._copy_slice(song.summary,ss,buffers['song'][i,:ns])
                if 'mel' in buffers:
                    ms,nm=recipe['mel_slices'][i]
                    self._copy_slice(song.mel10,ms,buffers['mel'][i,:nm])
            inputs.update(buffers)
            self.stats['attach_s']+=time.perf_counter()-t;self.stats['batches']+=1
            return inputs,seq,skipped
        finally:
            self.pool.release(recipe['keys'])

    def close(self):
        if self.closed:return
        for packed,_ in self.queue:self.pool.release(packed['_audio']['keys'])
        self.queue.clear();self.closed=True

    def telemetry(self):
        return {**self.stats,'staging':dict(self.ring.stats),'song_pool':self.pool.telemetry()}

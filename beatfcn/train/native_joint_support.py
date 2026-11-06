"""Target-only native multiset codec and request-only head support for C10.

This is representation preparation, not a generator or permutation-marginal
likelihood. Existing event grammar validates native serialization separately.
"""
from dataclasses import dataclass

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events


KINDS=('tap','hold','flick','drag')


@dataclass(frozen=True)
class NativeMember:
    cell: int
    rank: int
    address: int
    kind: int
    delta_ms: int


@dataclass(frozen=True)
class NativeMultiset:
    counts: np.ndarray
    members: tuple[NativeMember,...]


@dataclass(frozen=True)
class NativeJointSupport:
    cell_us: int=40000
    max_cells: int=200
    max_count: int=64
    fractional_copies: int=16

    def __post_init__(self):
        if self.cell_us!=40000 or min(self.max_cells,self.max_count,self.fractional_copies)<1:
            raise ValueError('Invalid fixed support contract')

    def addresses(self,request):
        if request.mode!='phira':raise ValueError('This experimental codec is Phira-only')
        cells=(request.end_us-request.start_us+self.cell_us-1)//self.cell_us
        if cells>self.max_cells:raise ValueError('Request interval exceeds core capacity')
        if request.duration_us>1_200_000_000:raise ValueError('Song duration exceeds declared native clock')
        values=np.full((cells,40+self.fractional_copies),-1,np.int64)
        valid=np.zeros(values.shape,bool)
        for c in range(cells):
            lo=request.start_us+c*self.cell_us;hi=min(lo+self.cell_us,request.end_us)
            first_ms=(lo+999)//1000
            native=(first_ms+np.arange(40,dtype=np.int64))*1000
            allowed=(native>=lo)&(native<hi)
            values[c,:40]=native;valid[c,:40]=allowed
            copies=sorted(t for t in request.copy_times if lo<=t<hi and t%1000)
            if len(copies)>self.fractional_copies:raise ValueError('Fractional COPY address overflow')
            values[c,40:40+len(copies)]=copies;valid[c,40:40+len(copies)]=True
        return values,valid

    def encode(self,events,request):
        events=sorted(events,key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))
        encode_events(events,request)  # Includes native time, permissions, tokens and multiplicity.
        values,valid=self.addresses(request);counts=np.zeros(len(values),np.int64);members=[]
        for e in events:
            c=(e.onset_us-request.start_us)//self.cell_us
            matches=np.flatnonzero(valid[c] & (values[c]==e.onset_us))
            if len(matches)!=1:raise ValueError('Native head has no unique authorized address')
            rank=int(counts[c])
            if rank>=self.max_count:raise ValueError('Per-cell event multiplicity overflow')
            delta=e.end_us//1000-e.onset_us//1000 if e.type=='hold' else 0
            members.append(NativeMember(int(c),rank,int(matches[0]),KINDS.index(e.type),int(delta)))
            counts[c]+=1
        return NativeMultiset(counts,tuple(members))

    def decode(self,encoded,request):
        values,valid=self.addresses(request);counts=np.asarray(encoded.counts)
        if counts.shape!=(len(values),) or counts.dtype.kind not in 'iu':raise ValueError('Invalid integer count vector')
        if (counts<0).any() or (counts>self.max_count).any():raise ValueError('Count overflow or invalid decision')
        if int(counts.sum())!=len(encoded.members) or len(encoded.members)>request.max_events:
            raise ValueError('Member/count contract mismatch')
        events=[];seen=set()
        for m in encoded.members:
            if any(type(v) is not int for v in (m.cell,m.rank,m.address,m.kind,m.delta_ms)):
                raise ValueError('Native fields must be exact integer decisions')
            if not 0<=m.cell<len(values) or not 0<=m.rank<counts[m.cell] or (m.cell,m.rank) in seen:
                raise ValueError('Invalid or duplicated member rank')
            if not 0<=m.address<values.shape[1] or not valid[m.cell,m.address] or not 0<=m.kind<4:
                raise ValueError('Illegal head/type decision')
            seen.add((m.cell,m.rank));head=int(values[m.cell,m.address]);kind=KINDS[m.kind]
            if kind=='hold':
                if m.delta_ms<1:raise ValueError('Hold must have a positive native endpoint delta')
                end=(head//1000+m.delta_ms)*1000
            else:
                if m.delta_ms:raise ValueError('Non-hold has a duration decision')
                end=head
            events.append(RhythmEvent(head,end,kind,-1))
        events.sort();encode_events(events,request)
        return events

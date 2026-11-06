"""Isolated exact-us cell/member support, including authorized fractional tails.

The historical token grammar remains unchanged. Its native-tail subdomain is
checked separately; the new field tape has an explicit version and exact ints.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.native_joint_support import NativeJointSupport,KINDS


@dataclass(frozen=True)
class ExactCellMember:
    cell:int
    address:int
    kind:int
    end_us:int


class ExactCellSupport:
    def __init__(self):self.heads=NativeJointSupport()

    def addresses(self,request):return self.heads.addresses(request)

    def encode(self,events,request):
        values,allowed=self.addresses(request);counts=np.zeros(len(values),np.int64);members=[]
        if len(events)>min(256,request.max_events):raise ValueError('Core event capacity overflow')
        for e in sorted(events,key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us)):
            if e.lane!=-1 or e.type not in KINDS or not request.start_us<=e.onset_us<request.end_us:raise ValueError('Invalid exact event')
            if request.allowed_types is not None and e.type not in request.allowed_types:raise ValueError('Disallowed kind')
            c=(e.onset_us-request.start_us)//40000;choices=np.flatnonzero(allowed[c]&(values[c]==e.onset_us))
            if len(choices)!=1:raise ValueError('Unauthorized or aliased exact head')
            if e.type=='hold':
                if not e.onset_us<e.end_us<=request.duration_us:raise ValueError('Illegal exact endpoint')
                if e.end_us%1000 and e.end_us not in request.copy_times:raise ValueError('Unauthorized fractional tail')
            elif e.end_us!=e.onset_us:raise ValueError('Duration on non-hold')
            counts[c]+=1
            if counts[c]>64:raise ValueError('Cell count overflow')
            members.append(ExactCellMember(int(c),int(choices[0]),KINDS.index(e.type),int(e.end_us)))
        if 1+3*len(members)>request.max_tokens:raise ValueError('Exact field-tape resource overflow')
        return counts,tuple(members)

    def decode(self,counts,members,request):
        values,allowed=self.addresses(request);events=[];actual=np.zeros(len(values),np.int64)
        for m in members:
            if not 0<=m.cell<len(values) or not 0<=m.address<values.shape[1] or not allowed[m.cell,m.address] or not 0<=m.kind<4:
                raise ValueError('Invalid exact cell member')
            h=int(values[m.cell,m.address]);events.append(RhythmEvent(h,m.end_us,KINDS[m.kind]));actual[m.cell]+=1
        if not np.array_equal(actual,counts):raise ValueError('Count/member mismatch')
        checked,canonical=self.encode(events,request)
        if Counter(canonical)!=Counter(members) or not np.array_equal(checked,counts):raise ValueError('Exact support mismatch')
        return sorted(events)


class ExactEndpointTree:
    """Request-only endpoint tree: relative 2s block,40ms cell,exact-us leaves."""
    def __init__(self,request,head_us,minimum_end_us=None):
        self.request=request;self.head=int(head_us);self.base=self.head//1000*1000
        self.lo=max(self.head+1,int(minimum_end_us or 0));self.hi=request.duration_us
        self.copies=tuple(sorted({u for u in request.copy_times if self.lo<=u<=self.hi and u%1000}))

    def leaves(self,block,cell):
        base=self.base+int(block)*2000000+int(cell)*40000
        first=max((self.lo+999)//1000*1000,base)
        stop=min(self.hi,base+39999)
        native=list(range(first,stop+1,1000)) if first<=stop else []
        return tuple(sorted(set(native)|{u for u in self.copies if base<=u<base+40000}))

    def blocks(self):
        if self.lo>self.hi:return ()
        first=(self.lo+999)//1000*1000;last=self.hi//1000*1000
        native=set(range((first-self.base)//2000000,(last-self.base)//2000000+1)) if first<=last else set()
        return tuple(sorted(native|{(u-self.base)//2000000 for u in self.copies}))

    def cells(self,block):
        base=self.base+int(block)*2000000
        first=max((self.lo+999)//1000*1000,base);last=min(self.hi//1000*1000,base+1999000)
        native=set(range((first-base)//40000,(last-base)//40000+1)) if first<=last else set()
        copies={(u-base)//40000 for u in self.copies if base<=u<base+2000000}
        return tuple(sorted(native|copies))

    def path(self,end_us):
        if not self.lo<=end_us<=self.hi or (end_us%1000 and end_us not in self.copies):raise ValueError('Endpoint outside exact tree')
        delta=end_us-self.base;b=delta//2000000;c=(delta%2000000)//40000
        leaves=self.leaves(b,c)
        if end_us not in leaves:raise ValueError('Unrepresented exact endpoint')
        return int(b),int(c),leaves.index(end_us)

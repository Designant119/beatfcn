"""Fixed training-only diagnostic priors, never a generation quota or repair.

Duration bins are (0,1ms],(1,2ms],...,(2^20ms,1200s]. A bucket
with any legal endpoint keeps its learned mass; within it native-ms and
authorized fractional COPY leaves are equiprobable. Empty legal buckets have
zero conditional mass. This explicitly defines conditioning, not beat snapping.
"""
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import math


KINDS=('tap','hold','flick','drag')
DURATION_UPPER_US=tuple([1000*(2**k) for k in range(21)]+[1_200_000_000])


def duration_bucket(delta_us):
    if not isinstance(delta_us,int) or not 0<delta_us<=DURATION_UPPER_US[-1]:
        raise ValueError('Duration outside fixed diagnostic support')
    return bisect_left(DURATION_UPPER_US,delta_us)


def fit_priors(cells,unique_events,global_frequency):
    """cells are canonical groups; unique_events retain original multiplicity.

    Caller deduplicates by chart/target array identity, never by event fields.
    Separate identical chart events therefore remain separate observations.
    """
    frequency=[float(n) for n in global_frequency]
    if len(frequency)!=4 or any(not math.isfinite(n) or n<=0 for n in frequency):raise ValueError('Strictly positive train frequency required')
    total=sum(frequency);global_prior=[n/total for n in frequency];counts={}
    for members in cells:
        previous=-1
        for rank,e in enumerate(sorted(members,key=lambda a:(a['onset_us'],KINDS.index(a['type']),a['end_us']))):
            key=f'{rank}:{previous}';values=counts.setdefault(key,[0]*4);kind=KINDS.index(e['type']);values[kind]+=1;previous=kind
    durations=[.5]*len(DURATION_UPPER_US);holds=0
    for e in unique_events:
        if e['type']=='hold':durations[duration_bucket(e['end_us']-e['onset_us'])]+=1;holds+=1
    return dict(global_prior=global_prior,history_counts=counts,history_concentration=4.,duration_counts=durations,
        duration_upper_us=list(DURATION_UPPER_US),duration_pseudocount=.5,hold_observations=holds,
        duration_conditioning='Renormalize bucket mass over nonempty legal buckets;uniform over legal native/COPY leaves within each bucket')


def type_probability(prior,rank,previous_kind,allowed,*,history=True):
    if len(allowed)!=4 or not any(allowed):raise ValueError('Empty type support')
    base=prior['global_prior'];values=prior['history_counts'].get(f'{rank}:{previous_kind}',[0]*4) if history else [0]*4
    alpha=prior['history_concentration'];result=[(float(n)+alpha*g) if legal else 0. for n,g,legal in zip(values,base,allowed)]
    total=sum(result);return [p/total for p in result]


def legal_duration_counts(head_us,song_end_us,copy_times=(),minimum_end_us=None):
    low=max(head_us+1,minimum_end_us or 0);high=song_end_us
    copies=sorted({int(t) for t in copy_times if t%1000 and low<=t<=high})
    result=[];lower=0
    for upper in DURATION_UPPER_US:
        lo=max(low,head_us+lower+1);hi=min(high,head_us+upper)
        native=max(0,hi//1000-(lo+999)//1000+1)
        fractional=max(0,bisect_right(copies,hi)-bisect_left(copies,lo)) if lo<=hi else 0
        result.append(native+fractional);lower=upper
    return result


def duration_probability(prior,head_us,end_us,song_end_us,copy_times=(),minimum_end_us=None):
    low=max(head_us+1,minimum_end_us or 0)
    if not low<=end_us<=song_end_us or (end_us%1000 and end_us not in copy_times):return 0.
    bucket=duration_bucket(end_us-head_us);numbers=legal_duration_counts(head_us,song_end_us,copy_times,minimum_end_us)
    if not numbers[bucket]:return 0.
    mass=sum(w for w,n in zip(prior['duration_counts'],numbers) if n)
    return prior['duration_counts'][bucket]/mass/numbers[bucket]

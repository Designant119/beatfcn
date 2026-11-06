"""Owner v2: all native objects trigger density, only drag objects are removed."""
import numpy as np

from beatfcn.events.canonical import TYPES
from .drag_density_filter import DragFilteredPanel, filtered_request_row


RULE = dict(id='all-events-1s-gt12-drag-only-v2', window_us=1_000_000,
    maximum_events_exclusive=12, predicate='count(ALL event onsets in [t,t+1000000)) > 12',
    event_unit='Every original object counts once at onset;hold end is not a second event',
    interval='Half-open;exactly 12 does not qualify;13 or more qualifies',
    duplicate_policy='Count all native objects including simultaneous exact duplicates',
    removal='Only drag objects in union of qualifying windows;other types unchanged;no chart removal',
    detection_scope='Entire original chart once before request slicing;no iterative recomputation')


def dense_event_drag_mask(onsets, kinds, *, window_us=1_000_000, maximum_events_exclusive=12):
    values=np.asarray(onsets,np.int64);kinds=np.asarray(kinds)
    if values.shape!=kinds.shape or values.ndim!=1 or window_us<=0 or maximum_events_exclusive<1:
        raise ValueError('Invalid strict event-density contract')
    order=np.argsort(values,kind='stable');times=values[order]
    delta=np.zeros(len(times)+1,np.int64);windows=[];right=0
    for left in range(len(times)):
        # One window per distinct onset;the first duplicate includes ALL objects.
        if left and times[left]==times[left-1]:continue
        right=max(right,left)
        while right<len(times) and int(times[right])-int(times[left])<window_us:right+=1
        if right-left>maximum_events_exclusive:
            delta[left]+=1;delta[right]-=1
            windows.append((int(times[left]),int(times[left])+window_us,right-left))
    dense=np.zeros(len(times),bool);dense[order]=np.cumsum(delta[:-1])>0
    removed=dense & (kinds==TYPES.index('drag'))
    return removed,windows


class EventDensityFilteredPanel(DragFilteredPanel):
    """Reuse lossless original-partition adapter; replace only detection law."""
    def _read(self, chart_index):
        chart=self.base.charts[chart_index]
        with np.load(self.base.root/chart['target_path'],allow_pickle=False) as archive:
            arrays={k:archive[k].copy() for k in archive.files}
        removed,windows=dense_event_drag_mask(arrays['onset_us'],arrays['kind'])
        return arrays,removed,windows

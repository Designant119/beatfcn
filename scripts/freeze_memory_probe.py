"""Select actual future exposure groups with the longest edited teacher targets."""
from dataclasses import replace
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import encode_events
from beatfcn.train.counterfactual_density import edit_events_to_multiplier
from beatfcn.train.local_request_panel import event_request


def lengths(job):
    chart, offset = job
    if chart['split'] != 'train':
        return []
    target = Path('/tmp/600m-repair-target-dir/native-v2-d94-580-mert')/Path(chart['target_path']).name
    if hashlib.sha256(target.read_bytes()).hexdigest() != chart['target_sha256']:
        raise ValueError('Target hash mismatch')
    result=[]
    with np.load(target,allow_pickle=False)as z:
        for local,row in enumerate(chart['requests']):
            index=offset+local;sl=slice(row['target_start'],row['target_stop'])
            events=[RhythmEvent(int(h),int(t),TYPES[int(k)],int(l))for h,t,k,l in
                    zip(*(z[key][sl]for key in ('onset_us','end_us','kind','lane')))]
            request=replace(event_request(row['request']),known_carry=())
            for rate in (None,0.,.5,1.5,2.):
                edited=events if rate is None else edit_events_to_multiplier(events,
                    request.start_us,request.end_us,rate,request.mode,request.duration_us,seed=index)[0]
                result.append((index,rate,len(encode_events(edited,request))))
    return result


def main():
    tape=json.load(open('/tmp/600m-repair-tape.json'))
    exposure=json.load(open('/tmp/600m-repair-exposure.json'))
    jobs=[];offset=0;durations={}
    for chart in tape['charts']:
        jobs.append((chart,offset))
        for index in range(offset,offset+len(chart['requests'])):durations[index]=chart['duration_us']
        offset+=len(chart['requests'])
    counts={}
    with Pool(8)as pool:
        for part in pool.imap_unordered(lengths,jobs):
            counts.update({(index,rate):length for index,rate,length in part})
    top=[]
    for first in range(0,len(exposure['steps']),2):
        group=exposure['steps'][first:first+2]
        pairs=[(i,rate)for step in group for i,rate in zip(step['indices'],step['rates'])]
        if len(pairs)!=8:raise ValueError('Partial physical batch')
        maximum=max(counts[pair]for pair in pairs)
        song=max(durations[i]for i,_ in pairs)
        top.append(dict(start_exposure=first+1,end_exposure=first+2,
                        max_tokens=maximum,padded_tokens=((maximum+127)//128)*128,
                        max_song_duration_us=song))
    chosen=[]
    longest_song=0
    for row in sorted(top,key=lambda r:(-r['padded_tokens'],-r['max_song_duration_us'],r['start_exposure'])):
        if row['max_song_duration_us'] > longest_song:
            chosen.append(row)
            longest_song=row['max_song_duration_us']
    result=dict(scope='all actual paired exposure groups; no synthetic target padding',
        selection='Pareto frontier of padded target length and longest song duration',
        tape_sha256=hashlib.sha256(Path('/tmp/600m-repair-tape.json').read_bytes()).hexdigest(),
        exposure_sha256=hashlib.sha256(Path('/tmp/600m-repair-exposure.json').read_bytes()).hexdigest(),
        variants=len(counts),groups=len(top),cases=chosen)
    path=Path('reports/600m-util-resume-20261101/memory-probe.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

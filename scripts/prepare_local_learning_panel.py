"""Freeze a small song-disjoint exploration panel without claiming 300 songs."""
from collections import Counter,defaultdict
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from beatfcn.events.canonical import Event,quantize_us
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest,EventState,encode_events
from beatfcn.train.event_conditions import encode_controls,training_labels

ROOT=Path(__file__).resolve().parents[1]


def pick_diverse(items,count,rng,key):
    buckets=defaultdict(list)
    for item in items:buckets[key(item)].append(item)
    names=sorted(buckets);rng.shuffle(names)
    for entries in buckets.values():rng.shuffle(entries)
    result=[]
    while len(result)<count:
        changed=False
        for name in names:
            if buckets[name]:result.append(buckets[name].pop());changed=True
            if len(result)==count:break
        if not changed:raise ValueError('Insufficient distinct panel records')
    return result


def main():
    source=ROOT/'inputs/r09_positive_pairs/catalog.pt'
    c=torch.load(source,map_location='cpu',weights_only=False);rng=random.Random(20261025)
    selected=[]
    for mode,n in [('phira',12),('mania_4k',4)]:
        pairs=pick_diverse([p for p in c['train_pairs'] if p['mode']==mode],n,rng,lambda p:p['audio_sha256'])
        selected.extend(('train',e) for p in pairs for e in p['entries'])
    for mode,n in [('phira',6),('mania_4k',2)]:
        selected.extend(('val',e) for e in pick_diverse([e for e in c['validation_candidates'] if e['row']['mode']==mode],n,rng,lambda e:e['row']['audio_sha256']))
    alltrain=[e for p in c['train_pairs'] for e in p['entries']]
    for key in ['song_group','audio_sha256']:
        if {e['row'][key] for e in alltrain}&{e['row'][key] for e in c['validation_candidates']}:
            raise ValueError('Source catalog leaks songs across split')
    scales_path='inputs/r08_local_real_probe/pca/i8_9-16-24.npz';scales=np.load(ROOT/scales_path)
    target=ROOT/'inputs/d65_local';summary=target/'song_summary';summary.mkdir(parents=True,exist_ok=True)
    records=[];files={source,ROOT/scales_path,Path(__file__).resolve()}
    for split,entry in selected:
        row=entry['row'];sha=row['audio_sha256'];mode=row['mode'];req=entry['request']
        if row['split']!=split:raise ValueError('Split mismatch')
        scope=[(r['start_us'],r['end_us']) for r in req['regions'] if r['state']=='editable']
        if len(scope)!=1 or req['observed_events'] or req['references']:raise ValueError('Expected blank single-region request')
        events=[Event.from_json(e) for e in entry['truth_events']]
        if any(e.get('fake') for e in entry['truth_events']):raise ValueError('Fake target')
        events=[replace(e,onset_us=quantize_us(e.onset_us),end_us=quantize_us(e.end_us)) for e in events]
        bands,raw=training_labels(mode,events,scope[0])
        request=EventRequest(mode,*scope[0],req['audio_duration_us'],allowed_types=tuple(req['controls']['allowed_types']))
        rhythm=[RhythmEvent.from_event(e,mode) for e in events];tokens=encode_events(rhythm,request)
        replay=EventState(request)
        for t in tokens:replay.apply(t)
        if Counter(rhythm)!=Counter(replay.events) or not replay.done:raise ValueError('Target codec loss')
        mel=ROOT/'inputs/d65_local/mel10'/(sha+'.npy');m=np.load(mel,mmap_mode='r');files.add(mel)
        if scope[0][1]>len(m)*10_000:raise ValueError('Core crosses legacy audio cache tail')
        base=ROOT/'inputs/r09_positive_pairs/taps/v4_i8_L9-16-24'/sha
        files.update(base/f'L{l}.npy' for l in [9,16,24])
        sp=summary/(sha+'.npz')
        if not sp.exists():
            a=np.load(base/'L24.npy',mmap_mode='r');bins=((np.arange(len(a))*40_000+20_000)//500_000).astype(int)
            counts=np.bincount(bins);values=np.zeros((len(counts),1024),np.float32)
            for block in range(len(counts)):
                lo,hi=np.searchsorted(bins,(block,block+1));values[block]=np.asarray(a[lo:hi],np.float32).mean((0,1))*scales['scale_24']
            times=np.bincount(bins,weights=np.arange(len(a))*.04+.02)/counts
            np.savez(sp,values=values,times=times.astype(np.float32))
        files.add(sp)
        records.append(dict(split=split,chart_id=row['chart_id'],audio_sha256=sha,song_group=row['song_group'],
            request=asdict(request),target_events=[asdict(e) for e in rhythm],tokens=len(tokens),
            controls=encode_controls(mode,bands if split=='train' else {},scope[0],origin_us=scope[0][0]).tolist(),
            bands=bands if split=='train' else None,
            condition_source='training target interval bands' if split=='train' else 'unspecified product request; no hidden answer attributes',
            native_content_readmission=False,offset_decision=row.get('offset_decision')))
    report=dict(id='d65-local-panel-v1',scales=scales_path,records=records,complete=True,
        scope='Legacy diagnostic subset with disjoint songs. Not 300-song evidence or newly admitted corpus.',
        per_split={split:dict(requests=sum(r['split']==split for r in records),
             songs=len({r['song_group'] for r in records if r['split']==split}),
             modes=dict(Counter(r['request']['mode'] for r in records if r['split']==split))) for split in ['train','val']},
        inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    out=target/'panel.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['per_split']))


if __name__=='__main__':main()

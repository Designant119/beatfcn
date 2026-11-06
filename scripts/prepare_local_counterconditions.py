"""Freeze negative request labels without changing D68's positive exposure."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.events.controls import raw_controls, union
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.countercondition import validated_countercondition
from beatfcn.train.event_conditions import CONTROL_FIELDS, in_band
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    output=ROOT/'inputs/d69_counterconditions/exposure-v1.json'
    if output.exists():
        raise FileExistsError(output)
    plan_path=ROOT/'bench/rhythm-v2/d68-local-baseline-v3.json'
    plan=json.loads(plan_path.read_text())
    for p,h in plan['inputs_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Source changed: '+p)
    panel=LocalRequestPanel(ROOT,plan['tape'],plan['features'])
    exposure=json.loads((ROOT/plan['exposure']).read_text())
    support=defaultdict(set)
    for step in exposure['steps']:
        for index,specified in zip(step['indices'],step['specified_condition_fields']):
            r=panel.records[index]
            if r['split']!='train':raise ValueError('Training exposure touches validation')
            for i in specified:
                name=CONTROL_FIELDS[i]
                support[(r['mode'],name,tuple(r['bands'][name]))].add(r['component'])
    prepared=[];skipped=Counter();counts=Counter();components=defaultdict(set)
    for step in exposure['steps']:
        negatives=[]
        for slot,(index,specified) in enumerate(zip(step['indices'],step['specified_condition_fields'])):
            r=panel.records[index]
            reason=None
            if step['effective_update']<=2:reason='common_parent_updates'
            elif len(specified)!=1:reason='unchanged_non_single_field_view'
            if reason:
                negatives.append(None);skipped[reason]+=1;continue
            field=CONTROL_FIELDS[specified[0]];scope=(r['request']['start_us'],r['request']['end_us'])
            carry=[RhythmEvent(**e) for e in r['request']['known_carry']]
            events=panel.targets(index)+carry
            actual=raw_controls(r['mode'],canonical(events,r['mode']),[scope])[field]
            positive=r['bands'][field]
            if not in_band(field,actual,positive):raise ValueError('Prepared true label disagrees with real events')
            carry_fraction=sum(max(0,min(scope[1],b)-max(scope[0],a)) for a,b in
                               union((e.onset_us,e.end_us) for e in carry))/(scope[1]-scope[0])
            no_free_lane=r['mode']=='mania_4k' and len({e.lane for e in carry if e.end_us>=scope[1]})==4
            candidates=[]
            for (mode,name,band),songs in sorted(support.items()):
                band=list(band)
                if mode!=r['mode'] or name!=field or len(songs)<10 or in_band(field,actual,band):continue
                if no_free_lane:continue
                if field=='hold_occupancy_ratio' and band[1]<=carry_fraction and not (band[1]==carry_fraction==1.):continue
                candidates.append(band)
            if not candidates:
                negatives.append(None);skipped['no_supported_feasible_opposite']+=1;continue
            identity=f"D69-v1:{exposure['pool_identity']}:{step['effective_update']}:{slot}"
            choice=int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)%len(candidates)
            negative=candidates[choice]
            correct=np.asarray(r['controls'],np.float32).copy()
            correct[[i for i in range(len(correct)) if i not in specified],:3]=0
            opposing=validated_countercondition(r['mode'],field,actual,positive,negative,scope,correct)
            negatives.append(dict(index=index,chart_id=r['chart_id'],component=r['component'],mode=r['mode'],
                field=field,actual=actual,positive=positive,negative=negative,scope_us=list(scope),
                negative_support_songs=len(support[(r['mode'],field,tuple(negative))]),
                carry_occupancy_fraction=carry_fraction,opposing_controls=opposing.tolist()))
            counts[r['mode']+'/'+field]+=1;components[r['mode']+'/'+field].add(r['component'])
        prepared.append(dict(effective_update=step['effective_update'],indices=step['indices'],negatives=negatives))
    sources=[str(plan_path.relative_to(ROOT)),plan['tape'],plan['exposure'],'scripts/prepare_local_counterconditions.py','beatfcn/train/countercondition.py']
    result=dict(id='D69-counterconditions-v1',complete=True,exposure=plan['exposure'],exposure_sha256=sha(ROOT/plan['exposure']),
        pool_identity=exposure['pool_identity'],steps=prepared,support_threshold_songs=10,
        counts=dict(counts),songs_by_field={k:len(v) for k,v in components.items()},skipped=dict(skipped),
        inputs_sha256={p:sha(ROOT/p) for p in sources},
        scope='Same real targets and same positive exposure; supported contradictory single-field labels, not invented positive chart pairs.')
    write_json(output,result)
    summary={k:v for k,v in result.items() if k!='steps'}
    summary['maximum_countercondition_tokens']=max(panel.records[n['index']]['tokens'] for s in prepared for n in s['negatives'] if n)
    write_json(ROOT/'reports/d69_countercondition_support_v1.json',summary)
    print(json.dumps({k:summary[k] for k in ['counts','songs_by_field','skipped','maximum_countercondition_tokens']}))


if __name__=='__main__':run()

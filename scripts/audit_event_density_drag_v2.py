"""New strict 1s/>12 ALL events rule;old audit and data remain immutable."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.train.event_density_drag_filter import EventDensityFilteredPanel,RULE
from scripts.audit_drag_density_data import stats,view_targets
from scripts.run_local_evolution import load_data


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/data-drag-density-v2'
REPORT=ROOT/'reports/local-integration-v1/data-drag-density-audit-v2.json'


def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    if REPORT.exists():raise FileExistsError('Preserve new audit')
    ART.mkdir(parents=True,exist_ok=True);started=time.monotonic();panel,_=load_data();filtered=EventDensityFilteredPanel(panel)
    parent=json.loads((ROOT/'bench/rhythm-v2/c17-masked-native-events-v1.json').read_text())
    previous=json.loads((ROOT/'bench/rhythm-v2/data-drag-density-probe-v1.json').read_text())
    report=dict(id='data-drag-density-v2',rule=RULE,groups={},per_request={},charts=[],cloud_calls=0,optimizer_updates=0,confirmation_scored=False)
    groups={k:[dict(index=i,view='draft') for i in parent['indices'][k]] for k in ('train64','dev','empty_dev')}
    groups['actual_1600_train_views']=[s for b in parent['sample_stream'] for s in b]
    groups['matched_probe_256_views']=[dict(index=i,view='draft') for b in previous['stream'] for i in b]
    for name,specs in groups.items():
        before=[];after=[];rows=[];affected=0;removed=0
        for spec in specs:
            if time.monotonic()-started>300:raise TimeoutError('Audit300s')
            i=spec['index'];r=panel.records[i];request,events=view_targets(panel,spec);request2,events2=view_targets(filtered,spec)
            if Counter(e for e in events if e.type!='drag')!=Counter(e for e in events2 if e.type!='drag'):raise ValueError('Non-drag changed')
            before.append((r,request,events));after.append((r,request2,events2));n=len(events)-len(events2);affected+=bool(n);removed+=n
            rows.append(dict(index=i,view=spec['view'],before=len(events),after=len(events2),removed=n))
        report['groups'][name]=dict(before=stats(before),after=stats(after),affected_requests=affected,removed_events=removed)
        report['per_request'][name]=rows
    excluded={panel.records[i]['component'] for i in parent['indices']['confirm']};manifest={};source={};total=Counter();nwindows=0
    for ci,chart in enumerate(panel.charts):
        if chart['component'] in excluded:continue
        p=panel.root/chart['target_path']
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        if h!=chart['target_sha256']:raise ValueError('Original target drift')
        source[str(p.relative_to(ROOT))]=h;arrays,mask,windows=filtered._arrays(ci);n=int(mask.sum());nwindows+=len(windows)
        total.update(TYPES[int(k)] for k in arrays['kind'])
        manifest[chart['chart_id']]=dict(target_path=chart['target_path'],target_sha256=h,removed_array_indices=np.flatnonzero(mask).tolist(),qualifying_windows=windows)
        report['charts'].append(dict(chart_id=chart['chart_id'],component=chart['component'],split=chart['split'],events_before=len(mask),events_removed=n,events_after=len(mask)-n,qualifying_1s_windows=len(windows)))
    report['staged_corpus']=dict(charts=len(manifest),affected_charts=sum(c['events_removed']>0 for c in report['charts']),
        removed_events=sum(c['events_removed'] for c in report['charts']),types_before=dict(total),qualifying_event_anchored_1s_windows=nwindows,
        window_counting='Unique event-onset anchors,overlapping;not independent requests or disjoint segments',excluded_confirm_components=len(excluded))
    write(ART/'removed-event-indices.json',dict(rule=RULE,charts=manifest,source_sha256=source))
    selected=sorted(set(i for name in ('train64','dev','empty_dev') for i in parent['indices'][name]))
    write(ART/'request-removals.json',dict(rule=RULE,requests=[filtered.removal_record(i) for i in selected]))
    write(ROOT/'bench/rhythm-v2/high-density-drag-policy-v2.json',dict(**RULE,owner_override='2026-10-30 replaces250ms/8drag',
        raw_data_mutated=False,whole_chart_removal=False,filter_predictions=False,
        implementation='beatfcn/train/event_density_drag_filter.py',pipeline_entry='filtered_request_row(EventDensityFilteredPanel, original_public_partition_row)',
        removal_manifest='artifacts/data-drag-density-v2/removed-event-indices.json'))
    report['wall_s']=time.monotonic()-started;write(REPORT,report)
    print(json.dumps(dict(status='completed',wall_s=report['wall_s'],groups={k:dict(before=v['before']['events'],after=v['after']['events'],
        removed=v['removed_events'],affected=v['affected_requests'],types_after={t:v['after']['types'][t]['events'] for t in TYPES}) for k,v in report['groups'].items()},staged=report['staged_corpus']),ensure_ascii=False))


if __name__=='__main__':main()

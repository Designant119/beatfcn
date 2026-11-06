"""Audit original pool, actual masked training exposure and filtered evaluation."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.train.drag_density_filter import DragFilteredPanel, RULE, dense_drag_mask
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.reference_completion import partition
from scripts.run_local_evolution import load_data


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/data-drag-density-v1'
REPORT = ROOT / 'reports/local-integration-v1/data-drag-density-audit-v1.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def quantile(values):
    return dict(zip(('min','p25','median','p75','p90','p99','max'),
        map(float, np.quantile(values,[0,.25,.5,.75,.9,.99,1])))) if values else None


def view_targets(panel, spec):
    base = panel.base if isinstance(panel, DragFilteredPanel) else panel
    events = base.targets(spec['index'])
    request = event_request(panel.records[spec['index']]['request'])
    if spec['view'] in ('middle','random_groups','continuation'):
        request, _, events, _ = partition(events, request, spec['view'],
            f"{panel.records[spec['index']]['chart_id']}:{request.start_us}", 20261027)
    if isinstance(panel, DragFilteredPanel):events=panel.filter_events(spec['index'], events)
    return request, list(events)


def stats(rows):
    counts=Counter();components=defaultdict(Counter);charts=defaultdict(Counter)
    durations=[];gaps=[];rates=[];drag_rates=[];peak=[];duplicate=0;invalid=0;empty=0;high=0
    for row, request, events in rows:
        cc=Counter(e.type for e in events);counts.update(cc)
        for kind,n in cc.items():
            components[kind][row['component']]+=n;charts[kind][row['chart_id']]+=n
        duplicate+=sum(n-1 for n in Counter(events).values())
        invalid+=sum(e.end_us<=e.onset_us for e in events if e.type=='hold')
        durations.extend((e.end_us-e.onset_us)/1000 for e in events if e.type=='hold')
        drag=sorted(e.onset_us for e in events if e.type=='drag')
        gaps.extend(np.diff(drag).astype(float).tolist())
        sec=(request.end_us-request.start_us)/1e6
        rates.append(len(events)/sec);drag_rates.append(len(drag)/sec)
        empty+=not events;high+=len(events)/sec>=12
        values=np.array(sorted(e.onset_us for e in events),np.int64)
        maxima=int(max((np.searchsorted(values,values+250_000,side='left')-np.arange(len(values))).tolist() or [0]))
        peak.append(maxima)
    total=sum(counts.values())
    types={k:dict(events=counts[k],share=counts[k]/max(total,1),components=len(components[k]),charts=len(charts[k]),
        top_component_share=max(components[k].values(),default=0)/max(counts[k],1),
        top_chart_share=max(charts[k].values(),default=0)/max(counts[k],1)) for k in TYPES}
    return dict(requests=len(rows),events=total,types=types,empty_requests=int(empty),
        high_density_requests_ge12_events_per_s=int(high),events_per_s=quantile(rates),drag_per_s=quantile(drag_rates),
        peak_objects_per250ms=quantile(peak),hold_duration_ms=quantile(durations),drag_adjacent_gap_ms=quantile([v/1000 for v in gaps]),
        extra_exact_duplicate_objects=int(duplicate),nonpositive_hold_errors=int(invalid))


def main():
    if REPORT.exists():raise FileExistsError('Preserve original audit')
    ART.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    panel,_=load_data();filtered=DragFilteredPanel(panel)
    plan=json.loads((ROOT/'bench/rhythm-v2/c17-masked-native-events-v1.json').read_text())
    report=dict(id='data-drag-density-v1',rule=RULE,rule_chosen_before_data_numbers=True,
        cloud_calls=0,optimizer_updates=0,confirmation_scored=False,groups={},per_request={},charts=[])
    selected={i for name in ('train64','dev','empty_dev') for i in plan['indices'][name]}
    for name in ('train64','dev','empty_dev','actual_1600_train_views'):
        specs=[dict(index=i,view='draft') for i in plan['indices'][name]] if name!='actual_1600_train_views' else [s for batch in plan['sample_stream'] for s in batch]
        before=[];after=[];changed=0;removed=0;ids=[]
        for spec in specs:
            if time.monotonic()-started>300:raise TimeoutError('Audit300s')
            i=spec['index'];r=panel.records[i]
            request,events=view_targets(panel,spec);request2,events2=view_targets(filtered,spec)
            before.append((r,request,events));after.append((r,request2,events2))
            if Counter(e for e in events if e.type!='drag')!=Counter(e for e in events2 if e.type!='drag'):
                raise ValueError('Non-drag hidden supervision changed')
            diff=len(events)-len(events2);changed+=bool(diff);removed+=diff
            ids.append(dict(index=i,view=spec['view'],removed=diff,before=len(events),after=len(events2)))
        report['groups'][name]=dict(before=stats(before),after=stats(after),affected_requests=changed,removed_events=removed,
            counting_scope='Exposure-weighted views' if name.startswith('actual') else 'Frozen request windows;not full-unique corpus')
        report['per_request'][name]=ids
    # Examine the staged target corpus without unsealing any confirm component.
    excluded={panel.records[i]['component'] for i in plan['indices']['confirm']}
    corpus=Counter();changed_charts=0;sensitivity=Counter();removed_indices={};input_identity={}
    for ci,chart in enumerate(panel.charts):
        if chart['component'] in excluded:continue
        path=panel.root/chart['target_path']
        with path.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
        if h!=chart['target_sha256']:raise ValueError('Raw targets drift')
        input_identity[str(path.relative_to(ROOT))]=h
        values,mask,windows=filtered._arrays(ci)
        counts=Counter(TYPES[int(k)] for k in values['kind'])
        corpus.update(counts);n=int(mask.sum());changed_charts+=bool(n)
        for threshold in (6,8,12):
            mm,_=dense_drag_mask(values['onset_us'],values['kind'],minimum_drag_objects=threshold)
            sensitivity[str(threshold)]+=int(mm.sum())
        removed_indices[chart['chart_id']]=dict(target_path=chart['target_path'],target_sha256=h,
            removed_array_indices=np.flatnonzero(mask).tolist(),qualifying_windows=windows)
        report['charts'].append(dict(chart_id=chart['chart_id'],component=chart['component'],split=chart['split'],
            stratum=chart['stratum'],events_before=len(mask),events_removed=n,types_before=dict(counts),
            events_after=len(mask)-n,selected_for_frozen_pool=any(panel.records[i]['chart_index']==ci for i in selected)))
    report['staged_corpus']=dict(charts=len(report['charts']),affected_charts=changed_charts,types_before=dict(corpus),
        events_removed=sensitivity['8'],threshold_sensitivity_not_model_selection=dict(sensitivity),
        excluded_confirm_components=len(excluded),excluded_confirm_chart_count=sum(c['component'] in excluded for c in panel.charts),
        scope='Staged tape,not all charts admitted/exposed;confirmation components excluded')
    write(ART/'removed-event-indices.json',dict(rule=RULE,charts=removed_indices,source_sha256=input_identity))
    write(ART/'request-removals.json',dict(rule=RULE,requests=[filtered.removal_record(i) for i in sorted(selected)]))
    write(ROOT/'bench/rhythm-v2/high-density-drag-policy-v1.json',dict(**RULE,owner_override='2026-10-30 explicit local high-density drag removal',
        raw_data_mutated=False,whole_chart_removal=False,filter_predictions=False,
        implementation='beatfcn/train/drag_density_filter.py',removal_manifest='artifacts/data-drag-density-v1/removed-event-indices.json',
        evaluation='Both arms cross-scored on same original and same filtered targets;never filter model outputs'))
    report['wall_s']=time.monotonic()-started
    write(REPORT,report)
    print(json.dumps(dict(status='completed',wall_s=report['wall_s'],groups={k:dict(before=v['before']['events'],after=v['after']['events'],
        affected=v['affected_requests'],removed=v['removed_events'],types_before={t:a['events'] for t,a in v['before']['types'].items()},
        types_after={t:a['events'] for t,a in v['after']['types'].items()}) for k,v in report['groups'].items()},corpus=report['staged_corpus']),ensure_ascii=False))


if __name__=='__main__':main()

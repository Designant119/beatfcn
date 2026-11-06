"""Recompute frozen terminal comparisons and behavior; no model generation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.review_frontend_canvas_comparison import metrics
from scripts.run_local_portfolio_scale import write


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/local-evolution-v1'
ART=ROOT/'artifacts/local-evolution-v1'


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()


def raw(arm,task='draft'):
    return [read(f'artifacts/local-evolution-v1/{arm}/{task}-sample{s}.json') for s in range(4 if task!='greedy' else 1)] if arm=='ar_interaction' else [read(f'artifacts/local-evolution-v1/{arm}/quality-1000-{task}.json')]


def behavior(samples):
    rows=[r for sample in samples for r in sample];intervals=[];medians=[];phases=[]
    types=Counter();true_types=Counter();windows=[]
    for r in rows:
        events=r['prediction']['events'];heads=sorted({e['onset_us'] for e in events});diff=np.diff(heads)/1000
        intervals.extend(diff.tolist())
        if len(diff):medians.append(float(np.median(diff)))
        phases.extend((e['onset_us']-r['request']['start_us'])%40000//1000 for e in events)
        kinds=Counter(e['type'] for e in events);types.update(kinds);true_types.update(e['type'] for e in r['target'])
        windows.append(dict(index=r['index'],count=len(events),true_count=len(r['target']),types=dict(kinds),
            unique_heads=len(heads),positive_interval_median_ms=float(np.median(diff)) if len(diff) else None,
            fraction_lt10ms=float(np.mean(diff<10)) if len(diff) else None))
    pooled={}
    for kind in ('tap','hold','flick','drag'):
        total={k:sum(r['scores']['per_type'][kind][k] for r in rows) for k in ('tp','pred','true')}
        total['f1']=2*total['tp']/max(total['pred']+total['true'],1);pooled[kind]=total
    return dict(attempts=len(rows),complete=sum(r['prediction']['complete'] for r in rows),nonempty=sum(bool(r['prediction']['events']) for r in rows),
        predicted_events=sum(types.values()),target_events=sum(true_types.values()),per_type_pooled=pooled,
        positive_interval_median_ms=float(np.median(intervals)) if intervals else None,
        median_of_window_interval_medians_ms=float(np.median(medians)) if medians else None,
        positive_interval_lt10ms=float(np.mean(np.asarray(intervals)<10)) if intervals else None,
        native_phase_histogram_ms=dict(sorted(Counter(map(int,phases)).items())),windows=windows)


def difference(a,b):
    keys=('full_note_f1','onset10','onset50','density_ratio')
    return dict(gains={k:a[k]-b[k] for k in keys},component_wins={k:sum(a['components'][c][k]>v[k] for c,v in b['components'].items()) for k in ('full_note_f1','onset10')})


def main():
    reports={a:read(f'reports/local-evolution-v1/{a}.json') for a in ('ar_interaction','ssd_full','ssd_reset')}
    plan=read('bench/rhythm-v2/local-evolution-v1.json');identities={};summaries={}
    for arm,r in reports.items():
        assert r['health_updates']==100 and r['quality_updates']==1000 and r['health_passed'] and not r['failures']
        weight=f'artifacts/local-evolution-v1/{arm}/quality-1000.pt'
        assert sha(weight)==r['final_weight_sha256'] and r['plan_sha256']==sha('bench/rhythm-v2/local-evolution-v1.json')
        for source,expected in plan['sources'].items():assert sha(source)==expected,source
        primary=raw(arm);m=r['evaluations']['draft'];recomputed=[metrics(v) for v in primary]
        for field in ('full_note_f1','onset10','onset50','density_ratio','complete','attempted'):
            assert abs(float(np.mean([v[field] for v in recomputed]))-m[field])<1e-12,(arm,field)
        identities[arm]=dict(report_sha256=sha(f'reports/local-evolution-v1/{arm}.json'),weight_sha256=sha(weight))
        health={phase:dict(loss=float(np.mean([v['loss'] for v in r[phase+'_health']])),
            parts={k:float(np.mean([v['parts'][k] for v in r[phase+'_health']])) for k in r[phase+'_health'][0]['parts']}) for phase in ('initial','final')}
        summaries[arm]=dict(health=health,primary={k:m[k] for k in ('full_note_f1','onset10','onset50','shift_null10','density_ratio')},
            behavior=behavior(primary),reference_full_gain=r['evaluations']['completion']['full_note_f1']-r['evaluations']['reference_masked']['full_note_f1'],
            empty_events_each_draw=[sum(len(v['prediction']['events']) for v in sample) for sample in raw(arm,'empty')],
            wall_s={k:r[k] for k in ('health_wall_s','quality_wall_s','evaluation_wall_s')})
        if arm=='ar_interaction':summaries[arm]['greedy_behavior']=behavior(raw(arm,'greedy'))
    old=read('reports/local_grouped_chart_v2.json')['evaluations']['draft']
    ar=difference(reports['ar_interaction']['evaluations']['draft'],old)
    ar['predeclared_gate']=dict(onset_gain=ar['gains']['onset10']>=.02,
        full_noninferiority=ar['gains']['full_note_f1']>=-.01,components=ar['component_wins']['onset10']>=10,
        count=.5<=reports['ar_interaction']['evaluations']['draft']['density_ratio']<=1.75)
    ssd=difference(reports['ssd_full']['evaluations']['draft'],reports['ssd_reset']['evaluations']['draft'])
    ssd['predeclared_gate']=dict(onset_gain=ssd['gains']['onset10']>=.02,
        full_noninferiority=ssd['gains']['full_note_f1']>=-.01,components=ssd['component_wins']['onset10']>=10,
        counts=all(.5<=r['evaluations']['draft']['density_ratio']<=1.75 for a,r in reports.items() if a.startswith('ssd_')))
    old_rows=read('artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000-draft.json')
    replay=raw('ssd_full')[0]
    exact=sum(a['prediction']['events']==b['prediction']['events'] and a['index']==b['index'] for a,b in zip(old_rows,replay))
    target_pairs=all(a['index']==b['index'] and a['request']==b['request'] and a['target']==b['target'] for a,b in zip(replay,raw('ssd_reset')[0]))
    result=dict(status='raw_receipts_verified_not_mechanism_closure',identities=identities,summaries=summaries,
        ar_repair_vs_old=ar,ssd_full_minus_reset=ssd,ssd_historical_exact_replay=dict(exact_event_streams=exact,total=32),
        ssd_request_target_pairs_identical=target_pairs,original_quality_endpoints_unchanged=True,optimizer_updates=0,cloud_compute_calls=0,
        limits='One seed per treatment, reused development set; no full-song arrangement/human/edit/playtest claim. Timing is observed MPS wall, not matched-time quality or CUDA speed. Counts use all fixed draws, not best draw.')
    write(FOLDER/'terminal-summary.json',result)
    print(json.dumps(dict(ar=ar,ssd=ssd,ssd_replay=exact,target_pairs=target_pairs),ensure_ascii=False))


if __name__=='__main__':main()

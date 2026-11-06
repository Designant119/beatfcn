"""Recover a late JSON-summary failure from all frozen draws, no inference."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from scripts.review_frontend_canvas_comparison import metrics
from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/c07-count-ancestral-audit-v1'
OUT=ROOT/'reports/local-evolution-v2/count-ancestral-recovered.json'
KINDS=('tap','hold','flick','drag')


def count_row(row):
    a,b=Counter(),Counter();start=row['request']['start_us']
    for events,dest in [(row['prediction']['events'],a),(row['target'],b)]:
        for e in events:
            if e['type']!='tap':dest[((e['onset_us']-start)//40000,e['type'])]+=1
    return dict(index=row['index'],pred=sum(a.values()),true=sum(b.values()),
                intersection=sum((a&b).values()),l1=sum((a-b).values())+sum((b-a).values()))


def f1(rows):
    return 2*sum(r['intersection'] for r in rows)/max(sum(r['pred']+r['true'] for r in rows),1)


def main():
    moment=time.monotonic()
    if OUT.exists():raise FileExistsError('Keep original recovery')
    plan=json.loads((ROOT/'bench/rhythm-v2/c07-count-ancestral-audit-v1.json').read_text())
    original=ROOT/'reports/local-evolution-v2/count-ancestral.json';r=json.loads(original.read_text())
    assert r['neural_forward_calls']==148 and len(r['exports'])==148
    for item in r['exports']:assert sha(ROOT/item['path'])==item['sha256']
    summary={};count={};identities={};attempts=complete=0
    for name in ('old','new'):
        summary[name]={};count[name]={}
        for condition in ('real','null'):
            summary[name][condition]=[];count[name][condition]=[]
            for draw in range(4):
                tasks={};count_table=[]
                for task in plan['views']:
                    path=ART/name/condition/f'draw{draw}-{task}.json';rows=json.loads(path.read_text());identities[str(path.relative_to(ROOT))]=sha(path)
                    tasks[task]=metrics(rows);attempts+=len(rows);complete+=sum(row['prediction']['complete'] for row in rows)
                    if task=='draft':count_table=[count_row(row) for row in rows if row['index'] in plan['indices']['dev'][::2]]
                summary[name][condition].append(tasks);count[name][condition].append(count_table)
    assert attempts==1184 and complete==attempts
    gates={};compact={}
    for name,baseline_path in [('old','reports/local-evolution-v1/ssd_reset.json'),('new','reports/local-evolution-v2/unified-count.json')]:
        baseline=json.loads((ROOT/baseline_path).read_text())['evaluations']['draft']
        real=float(np.mean([f1(x) for x in count[name]['real']]));null=float(np.mean([f1(x) for x in count[name]['null']]))
        wins=int(sum(np.mean([f1([v[i]]) for v in count[name]['real']])>np.mean([f1([v[i]]) for v in count[name]['null']]) for i in range(16)))
        q=summary[name]['real'];empty=float(np.mean([sum(v['empty']['predicted_types'].values()) for v in q]))
        draft={key:float(np.mean([v['draft'][key] for v in q])) for key in ('full_note_f1','onset10','onset50','density_ratio')}
        gates[name]=dict(non_tap_gain=real-{'old':18/327,'new':20/305}[name]>=.03,
            versus_null=real-null>=.02,components=wins>=9,empty=empty<=20,
            full_noninferiority=draft['full_note_f1']>=baseline['full_note_f1']-.01,
            onset_noninferiority=draft['onset10']>=baseline['onset10']-.01,all_complete=True)
        compact[name]=dict(non_tap_f1=real,null_f1=null,component_wins=wins,empty_mean=empty,draft=draft,
            null_onset10=float(np.mean([v['draft']['onset10'] for v in summary[name]['null']])),passed=all(gates[name].values()))
    stat=(ART/'run.log').stat();frozen=(ART/'source/frozen-plan.json').stat()
    result=dict(id=plan['id'],status='terminal_pending_mechanism_review',
        original_execution_status='Late summary JSON serialization failed after1184complete predictions; original report/log retained.',
        recovery=dict(optimizer_updates=0,neural_forward_calls=0,new_draws=0,wall_s=time.monotonic()-moment,
            script_sha256=sha(Path(__file__)),original_report_sha256=sha(original),log_sha256=sha(ART/'run.log')),
        optimizer_updates=0,neural_forward_calls=148,cloud_compute_calls=0,
        plan_sha256=sha(ROOT/'bench/rhythm-v2/c07-count-ancestral-audit-v1.json'),
        identity_replays=148,attempted=attempts,complete=complete,failures=[],
        summary=summary,count_diagnostics=count,mechanism_gate=gates,compact=compact,prediction_sha256=identities,
        timing=dict(log_file_lifecycle_s=stat.st_mtime-stat.st_birthtime,
                    sampling_after_frozen_plan_s=stat.st_mtime-frozen.st_mtime,
                    limitation='Filesystem wall-time observations, not recovered monotonic stage counters.'),
        boundary='32window component macro and16window count diagnostics separate; four draws equally weighted; probability-shift null preserves probability histograms, not realized N.')
    write(OUT,result);print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),compact=compact,gates=gates,timing=result['timing']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Training-only prior fit and fixed existing-TT trace audit, no model import."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from beatfcn.train.native_diagnostic_priors import KINDS,fit_priors,type_probability,duration_probability


ROOT=Path(__file__).resolve().parents[1];ID='c14-training-prior-audit-v1a'
PLAN=ROOT/f'bench/rhythm-v2/{ID}.json';ART=ROOT/f'artifacts/{ID}'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads((ROOT/path).read_text())


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def prepare():
    if PLAN.exists():raise FileExistsError('Preserve prior audit registration')
    parent=read('bench/rhythm-v2/c13-native-quality-v1.json')
    frontend=read('bench/rhythm-v2/local-frontend-comparison-v2.json');tape_path=frontend['tape']
    if sha(ROOT/tape_path)!=frontend['source_sha256'][tape_path]:raise ValueError('Frozen metadata identity')
    files=['scripts/audit_training_diagnostic_priors.py','beatfcn/train/native_diagnostic_priors.py','tests/test_native_diagnostic_priors.py']
    write(PLAN,dict(id=ID,authority='Owner local evolution;C14-S-R01 training-only diagnostic priors;not new neural experiment',
        optimizer_updates=0,model_calls=0,cloud_calls=0,wall_s=60,parent_plan_sha256=sha(ROOT/'bench/rhythm-v2/c13-native-quality-v1.json'),
        tape_path=tape_path,tape_sha256=frontend['source_sha256'][tape_path],
        prior_attempt='artifacts/c14-training-prior-audit-v1/report.json',
        train_indices=parent['indices']['train64'],history_alpha=4.,duration_bucket_pseudocount=.5,max_duration_us=1200000000,
        deduplication='Original chart_id/target array index once;duplicates at different array indices retained;cell grouping only at original phase0',
        source_sha256={n:sha(ROOT/n) for n in files},
        existing_tt={n:sha(ROOT/f'artifacts/c13-native-quality-v1/{n}.json') for n in ('draft','completion','reference_masked')},
        qualification='No free-generation or intervention claim;B/Z/T not executed;no thresholds selected from these results'))


def score_trace(rows,prior):
    result=[]
    for r in rows:
        previous={};model=[];global_nll=[];history_nll=[];tail_model=[];tail_prior=[];forced=0;failures=[]
        for s in r['teacher_steps']:
            cell=s['cell'];rank=s['rank'];event=s['event'];kind=KINDS.index(event['type']);prev=previous.get(cell)
            previous_kind=KINDS.index(prev['type']) if prev else -1
            if sum(s['kind_allowed'])==1:forced+=1
            else:
                g=type_probability(prior,rank,previous_kind,s['kind_allowed'],history=False)
                h=type_probability(prior,rank,previous_kind,s['kind_allowed'])
                p=s['type_probability'][kind]
                if p<=0 or g[kind]<=0 or h[kind]<=0:raise ValueError('Type probability/support failure')
                model.append(-math.log(p));global_nll.append(-math.log(g[kind]));history_nll.append(-math.log(h[kind]))
            if kind==1:
                minimum=prev['end_us'] if prev and prev['onset_us']==event['onset_us'] and prev['type']=='hold' else None
                p=duration_probability(prior,event['onset_us'],event['end_us'],r['request']['duration_us'],r['request']['copy_times'],minimum)
                if p<=0:raise ValueError('Duration prior support failure')
                tail_prior.append(-math.log(p))
            previous[cell]=event
        tail_model=r['teacher_scores']['endpoint_nll']
        if len(tail_model)!=len(tail_prior):raise ValueError('Hold decision coverage')
        result.append(dict(index=r['index'],component=r['component'],type_decisions=len(model),forced=forced,type_model_nll_sum=sum(model),
            type_global_prior_nll_sum=sum(global_nll),type_history_prior_nll_sum=sum(history_nll),hold_decisions=len(tail_model),
            endpoint_model_nll_sum=sum(tail_model),endpoint_duration_prior_nll_sum=sum(tail_prior),support_failures=failures))
    n=sum(r['type_decisions'] for r in result);nh=sum(r['hold_decisions'] for r in result)
    return dict(rows=result,type_decisions=n,hold_decisions=nh,
        type_model_nll=sum(r['type_model_nll_sum'] for r in result)/n,
        type_global_prior_nll=sum(r['type_global_prior_nll_sum'] for r in result)/n,
        type_history_prior_nll=sum(r['type_history_prior_nll_sum'] for r in result)/n,
        endpoint_model_nll=sum(r['endpoint_model_nll_sum'] for r in result)/nh if nh else None,
        endpoint_duration_prior_nll=sum(r['endpoint_duration_prior_nll_sum'] for r in result)/nh if nh else None)


def run():
    if (ART/'report.json').exists():raise FileExistsError('No automatic audit rerun')
    start=time.monotonic();plan=json.loads(PLAN.read_text());parent=read('bench/rhythm-v2/c13-native-quality-v1.json')
    if sha(ROOT/'bench/rhythm-v2/c13-native-quality-v1.json')!=plan['parent_plan_sha256']:raise ValueError('Parent plan drift')
    for n,h in plan['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Prior helper drift')
    tape_path=plan['tape_path']
    if sha(ROOT/tape_path)!=plan['tape_sha256']:raise ValueError('Tape drift')
    tape=read(tape_path);records=[dict(r,chart_index=i) for i,c in enumerate(tape['charts']) for r in c['requests']]
    bychart=defaultdict(list)
    for index in plan['train_indices']:
        row=records[index];chart=tape['charts'][row['chart_index']]
        if chart['split']!='train' or chart['mode']!='phira' or row['request']['start_us']%40000 or row['request']['duration_us']>plan['max_duration_us']:
            raise ValueError('Train-only phase/mode/duration contract')
        bychart[row['chart_index']].append(row)
    cells=defaultdict(dict);events={};identity={tape_path:sha(ROOT/tape_path)};request_events=0
    for chart_index,selected in bychart.items():
        chart=tape['charts'][chart_index];path=chart['target_path']
        if sha(ROOT/path)!=parent['input_payload_sha256'][path]:raise ValueError('Training labels drift')
        identity[path]=sha(ROOT/path)
        with np.load(ROOT/path,allow_pickle=False) as values:
            for row in selected:
                for j in range(row['target_start'],row['target_stop']):
                    e=dict(onset_us=int(values['onset_us'][j]),end_us=int(values['end_us'][j]),type=KINDS[int(values['kind'][j])],lane=int(values['lane'][j]))
                    if e['lane']!=-1:raise ValueError('Prior unexpected spatial labels')
                    key=(chart['chart_id'],j);cell=(chart['chart_id'],e['onset_us']//40000)
                    if key in events and events[key]!=e:raise ValueError('Original event identity collision')
                    events[key]=e;cells[cell][key]=e;request_events+=1
        if time.monotonic()-start>plan['wall_s']:raise TimeoutError('Prior fit wall')
    if any(len(v)>64 for v in cells.values()):raise ValueError('Prior raw cell capacity')
    prior=fit_priors([list(v.values()) for v in cells.values()],list(events.values()),read('artifacts/c13-native-quality-v1/data-identity.json')['prior'])
    write(ART/'prior.json',prior)
    report=dict(status='completed',model_calls=0,optimizer_updates=0,cloud_calls=0,
        train_windows=len(plan['train_indices']),train_charts=len(bychart),train_cells=len(cells),original_event_occurrences=request_events,
        unique_original_event_indices=len(events),identity=identity,prior_sha256=sha(ART/'prior.json'),tasks={})
    for name,h in plan['existing_tt'].items():
        path=ROOT/f'artifacts/c13-native-quality-v1/{name}.json'
        if sha(path)!=h:raise ValueError('Existing TT trace drift')
        report['tasks'][name]=score_trace(json.loads(path.read_text()),prior)
        if time.monotonic()-start>plan['wall_s']:raise TimeoutError('Prior audit wall')
    report.update(wall_s=time.monotonic()-start,scope='Train-only fixed statistical priors scored against existing TT traces;no B/Z/T or SELF/new model inference')
    write(ART/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('identity','tasks')}))
    for name,r in report['tasks'].items():print(name,{k:v for k,v in r.items() if k!='rows'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();globals()[args.action]()

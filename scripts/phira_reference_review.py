"""Additional held-out Phira A/B reference panel; inference only, shared budget."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import hashlib,json,time
import modal
from beatfcn.cloud.capability_review_app import image, data, CHECKPOINTS, REPO, write

app=modal.App('beatfcn-phira-reference-review')
OUTPUT='/data/evolution/capability-evolution-20261024/phira-reference-review-v1'

@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),timeout=700,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':data})
def evaluate():
 import torch
 from collections import defaultdict
 from beatfcn.cloud.evolution_app import load_model
 from beatfcn.train.data import Corpus,selected_for_training
 from beatfcn.events.requests import build_request
 from beatfcn.events.canonical import MODE_TYPES
 from beatfcn.model import features as F
 from beatfcn.model.evaluation import generate_song,score_events,quality_metrics
 from beatfcn.model.benchmark import audio_features,capability_metrics
 from beatfcn.model.input_diagnostics import swap_reference
 torch.set_num_threads(4);started=time.perf_counter();models={};meta={}
 protocol=json.loads(Path('/data/evolution/capability-evolution-20261024/evaluation-review-v1/protocol.json').read_text())
 for label,path in CHECKPOINTS.items():
  m,ck=load_model(path);m.requires_grad_(False);models[label]=m
  fingerprint=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  assert ck['step']==int(label) and fingerprint==protocol['checkpoints'][label]['sha256']
  meta[label]={'path':path,'step':ck['step'],'sha256':fingerprint}
  if label=='3500':corpus=Corpus(**ck['data']);manifest_name=ck['manifest_name']
  del ck
 rows=[json.loads(l) for l in Path('/data/manifest',manifest_name).read_text().splitlines() if l.strip()]
 rows=[r for r in rows if r.get('split')=='val' and r.get('kept') and r['mode']=='phira' and selected_for_training(r)]
 corpus.train_rows(rows);groups=defaultdict(list)
 for r in rows:groups[r['audio_sha256']].append(r)
 candidates=[sorted(g,key=lambda r:r['chart_id']) for _,g in sorted(groups.items()) if len(g)>1]
 result={'schema':'beatfcn.phira_reference_review.v1','checkpoints':meta,'model_source_sha256':protocol['model_source_sha256'],'candidate_audio_groups':len(candidates),'expected_selected_songs':4,'selected':[],'preflight_rejections':[],'records':[],'training_updates':0,'scope':'same-audio alternative human references, fixed spans, observed context, controls and targets; additional held-out panel, reported separately; may overlap main suite'}
 for group in candidates:
  if len(result['selected'])>=4 or time.perf_counter()-started>540:break
  row,donor=group[:2];seed=20261024+len(result['selected'])
  try:
   chart=corpus.chart(row['chart_id']);other=corpus.chart(donor['chart_id']);duration=round(row['audio_sec']*1e6)
   if len({e.location.line_id for e in chart.events+other.events})>64:raise ValueError('union_playable_lines_exceeds_64')
   sample=build_request(chart,kind='reference',seed=seed,audio_id=row['audio_sha256'],audio_duration_us=duration,empty_overlay=None,control_dropout=False)
   if not sample.ok:raise ValueError('model_view_quarantine')
   sample=replace(sample,request={**sample.request,'controls':{'allowed_types':list(MODE_TYPES['phira'])}})
   swapped=swap_reference(sample,other.events)
   # A fixed union mapping is identical under A and B, including all donor IDs.
   line_index=F.line_index_of(chart.events+other.events)
   truth=[e for g in sample.model_view.target_groups for e in g.events]
   # Validate bounds without generation before selecting this panel member.
   for s in (sample,swapped):
    for c0 in range(0,duration,F.CORE_US):
     if any(r['state']=='editable' and r['start_us']<c0+F.CORE_US and c0<r['end_us'] for r in s.request['regions']):
      F.build_core(replace(s,hidden=()),chart.events,c0,duration,line_index=line_index)
   resident=corpus.song(row['audio_sha256'],in_memory=True);af=audio_features(resident.mel10,duration)
  except (ValueError,F.SkipExample,AssertionError) as exc:
   result['preflight_rejections'].append({'chart_id':row['chart_id'],'donor':donor['chart_id'],'reason':str(exc)});continue
  result['selected'].append({'chart_id':row['chart_id'],'donor_chart':donor['chart_id'],'song':row['audio_sha256'],'seed':seed})
  for label in CHECKPOINTS:
   for condition,s in (('baseline',sample),('reference_swap',swapped)):
    begin=time.perf_counter();r={'checkpoint':label,'condition':condition,'song':row['audio_sha256'],'chart_id':row['chart_id'],'donor_chart':donor['chart_id'],'mode':'phira','request_kind':'reference','seed':seed,'request':s.request,'truth_events':[e.to_json() for e in truth],'reference_a':sample.evidence,'reference_b':swapped.evidence,'reference_changed':sample.evidence!=swapped.evidence}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     o=generate_song(models[label],s,resident,line_index=line_index,seed=seed,device='cuda',greedy=True,deadline=min(started+630,time.perf_counter()+90))
    r.update({k:v for k,v in o.items() if k!='events'});r['generated_events']=[e.to_json() for e in o['events']];r['scores']=score_events(o['events'],truth,'phira');r['quality']=quality_metrics(o['events'],truth,s.request['regions'],'phira',resident.summary);r['benchmark']=capability_metrics(o['events'],truth,s.request,af);r['wall_s']=time.perf_counter()-begin
    result['records'].append(r);write(Path(OUTPUT)/'results.json',result);data.commit()
    print(json.dumps({k:r[k] for k in ('checkpoint','condition','chart_id','complete','wall_s')}),flush=True)
 result['wall_s']=time.perf_counter()-started;write(Path(OUTPUT)/'results.json',result);data.commit()
 return {k:v for k,v in result.items() if k!='records'}

@app.local_entrypoint()
def run():
 source=hashlib.sha256()
 for p in sorted((REPO/'beatfcn').rglob('*.py')):
  source.update(str(p.relative_to(REPO)).encode());source.update(p.read_bytes())
 protocol=json.loads((REPO/'reports/capability_review_raw/evaluation-review-v1/protocol.json').read_text())
 if source.hexdigest()!=protocol['model_source_sha256']:raise ValueError('model source changed since main review')
 ledger_path=REPO/'reports/capability_review_budget.json';ledger=json.loads(ledger_path.read_text());key='phira_reference';reserve=ledger['evaluation_rate_usd_s']*820
 if key in ledger['stages']:raise ValueError('already dispatched')
 if ledger['prior_cost_upper_usd']+ledger['other_allowance_usd']+sum(s['reserved_usd'] for s in ledger['stages'].values())+reserve>10:raise ValueError('budget exhausted')
 stage={'reserved_usd':reserve,'status':'dispatched','started_at':time.time(),'app_id':app.app_id};ledger['stages'][key]=stage;write(ledger_path,ledger)
 try:
  result=evaluate.remote();stage.update(status='complete',result=result);print(json.dumps(result),flush=True)
 except BaseException as exc:
  stage.update(status='failed',error=f'{type(exc).__name__}:{exc}');raise
 finally:stage['ended_at']=time.time();write(ledger_path,ledger)

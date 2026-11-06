"""Paired positive-reference intervention with random negative rows omitted in both arms."""
from dataclasses import replace
import hashlib,json,time
from pathlib import Path
import modal
from beatfcn.cloud.capability_review_app import image,data,CHECKPOINTS,REPO,write

app=modal.App('beatfcn-reference-positive-review')
OUTPUT='/data/evolution/capability-evolution-20261024/reference-positive-review-v1'

@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),timeout=700,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':data})
def evaluate():
 import torch
 from beatfcn.cloud.evolution_app import load_model
 from beatfcn.train.data import Corpus,selected_for_training
 from beatfcn.events.requests import build_request
 from beatfcn.events.canonical import MODE_TYPES,ModelView
 from beatfcn.model import features as F
 from beatfcn.model.evaluation import generate_song,score_events,quality_metrics
 from beatfcn.model.benchmark import audio_features,capability_metrics
 from beatfcn.model.input_diagnostics import swap_reference
 torch.set_num_threads(4);started=time.perf_counter();models={}
 original=Path('/data/evolution/capability-evolution-20261024/evaluation-review-v1')
 protocol=json.loads((original/'protocol.json').read_text())
 for label,path in CHECKPOINTS.items():
  m,ck=load_model(path);m.requires_grad_(False);models[label]=m
  assert ck['step']==int(label) and hashlib.sha256(Path(path).read_bytes()).hexdigest()==protocol['checkpoints'][label]['sha256']
  if label=='3500':corpus=Corpus(**ck['data']);manifest_name=ck['manifest_name']
  del ck
 rows=[json.loads(l) for l in Path('/data/manifest',manifest_name).read_text().splitlines() if l.strip()]
 rows=[r for r in rows if r.get('split')=='val' and r.get('kept') and selected_for_training(r)]
 corpus.train_rows(rows);by_id={r['chart_id']:r for r in rows}
 original_records=[json.loads(p.read_text()) for p in sorted((original/'records').glob('*.json'))]
 cohort=[{'chart_id':r['chart_id'],'donor_chart':r['donor_chart'],'seed':r['seed']} for r in original_records if r['checkpoint']=='4500' and r['condition']=='reference_swap' and r['mode']=='mania_4k' and r.get('complete')]
 extra=json.loads(Path('/data/evolution/capability-evolution-20261024/phira-reference-review-v1/results.json').read_text())
 def content(refs):
  return [[(e['onset_us'],e['end_us'],e['type'],e['location']) for e in ref['selected']] for ref in refs]
 for r in extra['records']:
  if r['checkpoint']=='4500' and r['condition']=='reference_swap' and content(r['reference_a'])!=content(r['reference_b']):
   cohort.append({'chart_id':r['chart_id'],'donor_chart':r['donor_chart'],'seed':r['seed']})
 assert len(cohort)==5
 result={'schema':'beatfcn.reference_positive_review.v1','model_source_sha256':protocol['model_source_sha256'],'checkpoints':protocol['checkpoints'],'cohort':cohort,'records':[],'training_updates':0,'scope':'Follow-up chosen for input contrast, not model success: four mania pairs and the one nonduplicate Phira pair. Both arms omit all sampled unselected evidence rows; this is a diagnostic, not the original complete-reference evaluation distribution.'}
 for item in cohort:
  if time.perf_counter()-started>540:break
  row=by_id[item['chart_id']];chart=corpus.chart(item['chart_id']);donor=corpus.chart(item['donor_chart']);duration=round(row['audio_sec']*1e6)
  sample=build_request(chart,kind='reference',seed=item['seed'],audio_id=row['audio_sha256'],audio_duration_us=duration,empty_overlay=None,control_dropout=False)
  assert sample.ok
  sample=replace(sample,request={**sample.request,'controls':{'allowed_types':list(MODE_TYPES[row['mode']])}})
  swapped=swap_reference(sample,donor.events)
  def positives(s):
   return replace(s,evidence=tuple({**ref,'unselected_us':[],'unselected_source':[]} for ref in s.evidence))
  a,b=positives(sample),positives(swapped)
  line_index=F.line_index_of(chart.events+donor.events) if row['mode']=='phira' else F.line_index_of(chart.events)
  positive_changed=content(a.evidence)!=content(b.evidence)
  assert positive_changed
  for s in (a,b):
   for c0 in range(0,duration,F.CORE_US):
    if any(r['state']=='editable' and r['start_us']<c0+F.CORE_US and c0<r['end_us'] for r in s.request['regions']):
     F.build_core(replace(s,hidden=(),model_view=ModelView(row['mode'],(),(),())),chart.events,c0,duration,line_index=line_index)
  resident=corpus.song(row['audio_sha256'],in_memory=True);af=audio_features(resident.mel10,duration)
  truth=[e for g in a.model_view.target_groups for e in g.events]
  for label in CHECKPOINTS:
   for condition,s in (('baseline',a),('reference_swap',b)):
    begin=time.perf_counter();r={**item,'checkpoint':label,'condition':condition,'song':row['audio_sha256'],'mode':row['mode'],'request_kind':'reference','request':s.request,'truth_events':[e.to_json() for e in truth],'reference_a':a.evidence,'reference_b':b.evidence,'reference_changed':positive_changed}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     out=generate_song(models[label],s,resident,line_index=line_index,seed=item['seed'],device='cuda',greedy=True,deadline=min(started+630,time.perf_counter()+90))
    r.update({k:v for k,v in out.items() if k!='events'});r['generated_events']=[e.to_json() for e in out['events']];r['scores']=score_events(out['events'],truth,row['mode']);r['quality']=quality_metrics(out['events'],truth,s.request['regions'],row['mode'],resident.summary);r['benchmark']=capability_metrics(out['events'],truth,s.request,af);r['wall_s']=time.perf_counter()-begin
    result['records'].append(r);write(Path(OUTPUT)/'results.json',result);data.commit();print(json.dumps({k:r[k] for k in ('checkpoint','condition','chart_id','complete','wall_s')}),flush=True)
 result['wall_s']=time.perf_counter()-started;write(Path(OUTPUT)/'results.json',result);data.commit()
 return {'records':len(result['records']),'expected':20,'wall_s':result['wall_s'],'training_updates':0}

@app.local_entrypoint()
def run():
 source=hashlib.sha256()
 for p in sorted((REPO/'beatfcn').rglob('*.py')):source.update(str(p.relative_to(REPO)).encode());source.update(p.read_bytes())
 protocol=json.loads((REPO/'reports/capability_review_raw/evaluation-review-v1/protocol.json').read_text())
 assert source.hexdigest()==protocol['model_source_sha256']
 path=REPO/'reports/capability_review_budget.json';ledger=json.loads(path.read_text());key='reference_positive';reserve=ledger['evaluation_rate_usd_s']*820
 assert key not in ledger['stages']
 assert ledger['prior_cost_upper_usd']+ledger['other_allowance_usd']+sum(s['reserved_usd'] for s in ledger['stages'].values())+reserve<=ledger['approved_total_usd']
 stage={'reserved_usd':reserve,'status':'dispatched','started_at':time.time(),'app_id':app.app_id};ledger['stages'][key]=stage;write(path,ledger)
 try:
  result=evaluate.remote();stage.update(status='complete',result=result);print(json.dumps(result),flush=True)
 except BaseException as exc:stage.update(status='failed',error=f'{type(exc).__name__}:{exc}');raise
 finally:stage['ended_at']=time.time();write(path,ledger)

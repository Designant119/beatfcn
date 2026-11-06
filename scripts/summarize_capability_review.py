"""Summarize saved inference records, preserving missing/failed denominators."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from beatfcn.model.benchmark import scorecard

FIELDS={
 'onset_f1_20ms':('scores','onset_20ms','f1'),
 'note_f1_20ms':('scores','note_20ms','f1'),
 'hold_joint_f1_20ms':('scores','hold_joint_20ms','f1'),
 'density_ratio':('scores','density_ratio'),
 'matched_error_median_ms':('quality','onset_error_50ms','median_ms'),
 'matched_error_p90_ms':('quality','onset_error_50ms','p90_ms'),
 'match_coverage_50ms':('quality','onset_error_50ms','match_rate'),
 'audio_anchor_20ms':('benchmark','generated','anchor_20ms_precision'),
 'audio_anchor_shift_excess':('benchmark','generated','anchor_20ms_shift_excess'),
 'density_flux_spearman':('benchmark','generated','density_flux_spearman'),
 'rapid_same_lane_fraction':('benchmark','generated','rapid_same_lane_fraction'),
 'lane_bigram_entropy':('benchmark','generated','lane_bigram_entropy'),
}
def val(r,path):
 for k in path:
  if not isinstance(r,dict): return None
  r=r.get(k)
 return r

def summarize(rs):
 out={'attempted':len(rs),'completed':sum(bool(r.get('complete')) for r in rs),
      'legal_completed':sum(bool(r.get('complete')) and not r.get('legality') for r in rs),
      'empty_completed':sum(bool(r.get('complete')) and not r.get('generated_events') for r in rs),
      'empty_target_completed':sum(bool(r.get('complete')) and not r.get('truth_events') for r in rs),
      'empty_target_silent':sum(bool(r.get('complete')) and not r.get('truth_events') and not r.get('generated_events') for r in rs),
      'scored':sum('scores' in r for r in rs),
      'failures':dict(Counter(r['failure']['reason'] for r in rs if r.get('failure')))}
 for name,path in FIELDS.items():
  xs=[val(r,path) for r in rs if r.get('complete')]
  xs=[x for x in xs if x is not None and np.isfinite(x)]
  out[name]={'mean_complete_only':float(np.mean(xs)) if xs else None,'n':len(xs)}
 xs=[r['scores']['note_20ms']['f1'] for r in rs if r.get('complete') and r.get('truth_events') and 'scores' in r]
 out['nonempty_target_note_f1_20ms']={'mean_complete_only':float(np.mean(xs)) if xs else None,'n':len(xs)}
 for label,denominator in (('precision','pred'),('recall','true')):
  xs=[]
  for r in rs:
   if not r.get('complete') or 'scores' not in r:continue
   s=r['scores']['note_20ms']
   if s[denominator]:xs.append(s['tp']/s[denominator])
   elif label=='precision' and s['true']:xs.append(0.)
  out[f'note_{label}_20ms']={'mean_complete_only':float(np.mean(xs)) if xs else None,'n':len(xs)}
 return out

def paired(a,b):
 lookup={(r['song'],r['request_kind']):r for r in a}
 pairs=[(lookup[(r['song'],r['request_kind'])],r) for r in b if (r['song'],r['request_kind']) in lookup]
 result={'pairs':len(pairs),'both_complete':sum(x.get('complete') and y.get('complete') for x,y in pairs),
         'output_changed':sum(x.get('generated_events')!=y.get('generated_events') for x,y in pairs if 'generated_events' in x and 'generated_events' in y)}
 for name,path in FIELDS.items():
  bysong=defaultdict(list)
  for x,y in pairs:
   u,v=val(x,path),val(y,path)
   if x.get('complete') and y.get('complete') and u is not None and v is not None:
    bysong[x['song']].append(v-u)
  values=np.array([np.mean(v) for v in bysong.values()])
  ci=None
  if len(values)>=2:
   draws=np.random.default_rng(20261024).choice(values,size=(5000,len(values)),replace=True).mean(1)
   ci=np.quantile(draws,[.025,.975]).tolist()
  result[name]={'delta_b_minus_a':float(values.mean()) if len(values) else None,'song_bootstrap_95':ci,'independent_songs':len(values)}
 return result

def style(events):
 if not events: return None
 times=Counter(e['onset_us'] for e in events)
 iois=np.diff(sorted(times))/1000
 out={'hold_fraction':sum(e['type']=='hold' for e in events)/len(events),
      'drag_fraction':sum(e['type']=='drag' for e in events)/len(events),
      'chord_fraction':sum(n>1 for n in times.values())/len(times),
      'median_ioi_ms':float(np.median(iois)) if len(iois) else None}
 return out

def main():
 directory=Path(sys.argv[1]); records=[json.loads(p.read_text()) for p in sorted((directory/'records').glob('*.json'))]
 protocol=json.loads((directory/'protocol.json').read_text())
 result={'protocol':protocol,'summarized_at':datetime.now(timezone.utc).isoformat(),'records_received':len(records),'groups':{},'paired':{},'reference_responses':[]}
 for label in ('3500','4500'):
  for cond in ('baseline','history_removed','reference_swap','audio_mismatch'):
   for mode in ('mania_4k','phira'):
    rs=[r for r in records if r['checkpoint']==label and r['condition']==cond and r['mode']==mode]
    result['groups'][f'{label}/{cond}/{mode}']=summarize(rs)
    if cond=='baseline':
     for kind in ('from_scratch','middle_holes','reference'):
      result['groups'][f'{label}/{cond}/{mode}/{kind}']=summarize([r for r in rs if r['request_kind']==kind])
    if cond!='baseline':
     base=[r for r in records if r['checkpoint']==label and r['condition']=='baseline' and r['mode']==mode]
     result['paired'][f'{label}/{cond}/{mode}']=paired(base,rs)
  base=[r for r in records if r['checkpoint']==label and r['condition']=='baseline']
  # Fill every missing manifest request so scorecards cannot silently shrink denominators.
  manifest=json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())
  keys={(r['chart_id'],r['request_kind']) for r in base}
  missing=[{'chart_id':q['chart_id'],'song':q['song'],'mode':q['mode'],'request_kind':q['kind'],'stratum':q['stratum'],
            'complete':False,'legality':[],'failure':{'reason':'not_executed'}} for q in manifest['requests'] if (q['chart_id'],q['kind']) not in keys]
  result.setdefault('scorecards',{})[label]=scorecard(base+missing)
  result.setdefault('coverage',{})[label]={'expected':120,'attempted':len(base),'completed':sum(r['complete'] for r in base),'missing':len(missing)}
 for mode in ('mania_4k','phira'):
  for kind in ('from_scratch','middle_holes','reference'):
   a=[r for r in records if r['condition']=='baseline' and r['checkpoint']=='3500' and r['mode']==mode and r['request_kind']==kind]
   b=[r for r in records if r['condition']=='baseline' and r['checkpoint']=='4500' and r['mode']==mode and r['request_kind']==kind]
   result['paired'][f'4500_minus_3500/{mode}/{kind}']=paired(a,b)
 for r in records:
  if r['condition']!='reference_swap' or not r.get('complete'):continue
  original=next((x for x in records if x['condition']=='baseline' and x['checkpoint']==r['checkpoint'] and x['song']==r['song'] and x['request_kind']=='reference' and x['complete']),None)
  if original is None:continue
  a=style([e for ref in r['reference_a'] for e in ref['selected']]); b=style([e for ref in r['reference_b'] for e in ref['selected']])
  ga=style(original['generated_events']);gb=style(r['generated_events'])
  entry={'song':r['song'],'mode':r['mode'],'checkpoint':r['checkpoint'],'reference_changed':r['reference_changed'],'features':{}}
  if all(v is not None for v in (a,b,ga,gb)):
   for f in a:
    if all(v[f] is not None for v in (a,b,ga,gb)):
     dr=b[f]-a[f];dg=gb[f]-ga[f]
     entry['features'][f]={'reference_delta':dr,'generated_delta':dg,'direction_agrees':dr*dg>0 if abs(dr)>1e-8 else None}
  result['reference_responses'].append(entry)
 result['limitations']=['paired changes use complete pairs only; completion and failures reported separately','8-song intervention panel is diagnostic, not population-level proof','history removal tests added adapters only','reference A/B retains original observed context and controls','audio donors matched by duration, not tempo; resampled latent features are an input-dependence stress test','Phira spatial placement is not evaluated']
 out=ROOT/'reports/capability_review_summary.json';out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
 print(json.dumps({'coverage':result['coverage'],'records_received':len(records),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()

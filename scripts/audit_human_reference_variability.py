"""D13 local human-chart metric calibration; never changes acceptance gates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from beatfcn.model import features as F
from beatfcn.model.evaluation import groups_to_events
from beatfcn.model.edit_utility import edit_utility
from beatfcn.train.condition_pairs import blank_context_excerpt
from beatfcn.train.data import chart_from_canon, model_shift_us

ROOT = Path(__file__).resolve().parents[1]


def summarize(records):
    audio = defaultdict(list)
    for row in records:
        audio[row['audio_sha256']].append(row)
    return dict(windows=len(records), independent_audio=len(audio),
        audio_macro_full_note_f1=mean(mean(r['full_note_f1'] for r in rows) for rows in audio.values()) if audio else None,
        audio_macro_timing_f1=mean(mean(r['timing_f1'] for r in rows) for rows in audio.values()) if audio else None,
        audio_macro_first_match_rate=mean(mean(r['first_match_20ms'] for r in rows) for rows in audio.values()) if audio else None,
        window_median_first_difference_ms=median(r['first_difference_ms'] for r in records) if records else None,
        first_mismatches=sum(not r['first_match_20ms'] for r in records))


def main():
    plan_path=ROOT/'bench/rhythm-v2/d13-human-reference-variability-v1.json'
    plan=json.loads(plan_path.read_text())
    support_path=ROOT/'reports/d09_real_pair_support.json'
    selection_path=ROOT/'bench/rhythm-v2/d09-real-pair-support-v1.json'
    for path,key in [(support_path,'support_report_sha256'),(selection_path,'request_selection_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plan[key]:raise ValueError('frozen source changed')
    source=json.loads(support_path.read_text());selection=json.loads(selection_path.read_text())
    metadata={r['chart_id']:r for group in selection['groups'] for r in group['rows']}
    charts={}
    records=[]
    for row in source['records']:
        views=[]
        for chart_id in row['chart_ids']:
            info=metadata[chart_id]
            if chart_id not in charts:
                path=ROOT/selection['output_cache']/info['source']/f'{info["native_id"]}.json'
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=source['canonical_sha256'][chart_id]:raise ValueError('canonical source changed')
                charts[chart_id]=chart_from_canon(json.loads(raw),model_shift_us(info,True))
            chart=charts[chart_id];duration=round(info['audio_sec']*F.SEC)
            request=blank_context_excerpt(chart,row['c0_us'],duration,row['audio_sha256'],20261025)
            ex=F.build_core(request,chart.events,row['c0_us'],duration,stage_a_arm='legacy')
            rhythm=[(g['t_rel_us'],sorted((n['type'],n['lane'] if row['mode']=='mania_4k' else -1,n.get('end_rel_us',0))
                    for n in g['notes'])) for g in ex.groups]
            item=next(x for x in row['items'] if x['chart_id']==chart_id)
            if hashlib.sha256(json.dumps(rhythm).encode()).hexdigest()!=item['rhythm_target_sha256']:
                raise ValueError('original quantized target changed')
            views.append(groups_to_events(ex.groups,row['c0_us'],row['mode']))
        scores=edit_utility(views[0],views[1],row['mode'])
        f1=lambda c:2*c['matched']/(c['pred']+c['true']) if c['pred']+c['true'] else 1.
        first_difference=abs(min(e.onset_us for e in views[0])-min(e.onset_us for e in views[1]))/1000
        records.append(dict(mode=row['mode'],audio_sha256=row['audio_sha256'],chart_ids=row['chart_ids'],
            c0_us=row['c0_us'],rhythm_targets_equal=row['rhythm_targets_equal'],changed_fields=row['changed_fields'],
            full_note_f1=f1(scores['full_note']),timing_f1=f1(scores['timing_point']),
            first_difference_ms=first_difference,first_match_20ms=first_difference<=20,
            note_counts=[len(v) for v in views]))
    if len(records)!=248:raise ValueError('missing accepted D09 windows')
    summaries={}
    for mode in ('mania_4k','phira'):
        rows=[r for r in records if r['mode']==mode]
        for label,predicate in [('all',lambda r:True),('different_rhythm',lambda r:not r['rhythm_targets_equal']),
            ('same_condition_bands',lambda r:not r['changed_fields']),
            ('different_rhythm_same_bands',lambda r:not r['rhythm_targets_equal'] and not r['changed_fields']),
            ('different_rhythm_at_most_one_band_diff',lambda r:not r['rhythm_targets_equal'] and len(r['changed_fields'])<=1)]:
            summaries[f'{mode}/{label}']=summarize([r for r in rows if predicate(r)])
    result=dict(schema='beatfcn.human_reference_variability.v1',complete=True,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),summary=summaries,records=records,
        limitation=plan['limitation'],acceptance_changed=False,model_quality_promotion=False,
        training_updates=0,new_remote_compute=False)
    path=ROOT/'reports/d13_human_reference_variability.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()

"""Apply D90's unchanged pre-registered whole-song gates to the two actual1000 arms."""
import csv
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.local_song_rollout import canonical
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def numerical_gates(control, candidate, limits):
    intervals = candidate['tap_positive_intervals'] > 0
    absolute = intervals and candidate['tap_short_fraction'] <= limits['tap_under10ms_fraction_absolute_alternative']
    relative = intervals and control['tap_positive_intervals'] > 0 and (
        control['tap_short_fraction']-candidate['tap_short_fraction'] >= limits['tap_under10ms_fraction_reduction'])
    return dict(
        complete_legal=candidate['complete_legal_songs'] == limits['full_songs'],
        nonempty=candidate['nonempty_songs'] >= limits['minimum_nonempty_songs'],
        macro_gain=candidate['macro_f1']-control['macro_f1'] >= limits['note_f1_macro_gain_over_matched_control'],
        macro_floor=candidate['macro_f1'] >= limits['note_f1_macro_floor'],
        per_song=candidate['songs_not_worse'] >= limits['minimum_songs_not_worse'],
        tap_spacing=bool(absolute or relative),
        nontap_types=all(candidate['type_matches'].get(t,0) >= limits['minimum_real_20ms_matches_per_nontap_type']
                         for t in ('hold','flick','drag')),
        ez_hd=all(t in control['native_tag_macro_f1'] and t in candidate['native_tag_macro_f1'] and
                  candidate['native_tag_macro_f1'][t] >= control['native_tag_macro_f1'][t]-limits['ez_hd_macro_max_regression']
                  for t in ('EZ','HD')),
        reference_rest=candidate['reference_rest_mean'] <= control['reference_rest_mean']+limits['reference_rest_mean_extra_notes_max'])


def events(path):
    with path.open() as stream:
        return [RhythmEvent(onset_us=int(r['onset_us']),end_us=int(r['end_us']),type=r['type'],lane=int(r['lane']))
                for r in csv.DictReader(stream)]


def type_match_count(kind, scores):
    return scores['hold_joint_20ms' if kind=='hold' else 'note_20ms']['tp']


def main():
    output=ROOT/'reports/d90_quality_comparison_1000_v1.json'
    if output.exists():raise FileExistsError(output)
    sources={}; reports={}; summaries={}; raw={}
    for arm in ('control','candidate'):
        paths=([f'bench/rhythm-v2/d87-sequence-{arm}-v1.json',
                f'reports/d87_{arm}_review_1000_v1.json',f'reports/d87_{arm}_stage_1000_v1.json']
               if arm == 'control' else ['bench/rhythm-v2/d90-relative-horizon-v1.json',
                   'reports/d90_relative_review_1000_v1.json','reports/d90_relative_stage_1000_v1.json'])
        plan,review,stage=(json.loads((ROOT/p).read_text())for p in paths)
        train=json.loads((ROOT/plan['output']).read_text())
        if not train['complete'] or train['updates'] != 1000 or train.get('error') or train['plan_sha256'] != sha(ROOT/paths[0]):
            raise ValueError('Require terminal1000 from each frozen arm')
        if review['checkpoint_sha256'] != train['checkpoint_sha256'] or stage['checkpoint_sha256'] != train['checkpoint_sha256']:
            raise ValueError('Quality evidence uses another checkpoint')
        if not stage.get('complete') or stage.get('error') or len(review['records']) !=20:
            raise ValueError('Incomplete whole-song or stage evidence')
        for p,expected in {**plan['inputs_sha256'],**plan['code_sha256']}.items():
            actual = ('inputs/code_baselines/d87-closed-v1/scripts/run_local_sequence_quality.py'
                      if arm == 'control' and p == 'scripts/run_local_sequence_quality.py' else p)
            if sha(ROOT/actual)!=expected:raise ValueError('Frozen source changed: '+p)
            sources[actual] = expected
        sources.update({p:sha(ROOT/p)for p in paths+[plan['output']]})
        records=review['records'];raw[arm]={}
        targets={r['chart_id']:r for r in json.loads((ROOT/plan['tape']).read_text())['charts']}
        if len({r['chart_id']for r in records}) !=20:raise ValueError('Duplicate evaluation song')
        intervals=[];matches={t:0 for t in ('hold','flick','drag')};tags={}
        type_records=[]
        for r in records:
            folder=ROOT/Path(r['preview']).parent
            pp,hh=folder/'model-timeline.csv',folder/'human-timeline.csv'
            for p in (pp,hh):sources[str(p.relative_to(ROOT))]=sha(p)
            pred,human=events(pp),events(hh)
            target=targets[r['chart_id']]
            if sha(ROOT/target['target_path'])!=target['target_sha256']:raise ValueError('Human target changed')
            with np.load(ROOT/target['target_path'],allow_pickle=False) as archive:
                original_human=[RhythmEvent(int(h),int(t),TYPES[int(k)],int(l))for h,t,k,l in
                    zip(*(archive[key]for key in ('onset_us','end_us','kind','lane')),strict=True)]
            if sorted(human)!=sorted(original_human):raise ValueError('Export changed human reference')
            generated=ROOT/r['source_generation']
            if sha(generated)!=r['source_generation_sha256']:raise ValueError('Raw generation changed')
            original=json.loads(generated.read_text())
            if sorted(pred)!=sorted(RhythmEvent(**e)for e in original['events']):raise ValueError('Export changed generated events')
            if len(pred)!=r['notes']['model'] or len(human)!=r['notes']['human']:raise ValueError('Export counts differ')
            # Recheck full-note score so exports cannot silently alter hold endpoints.
            p,h=canonical(pred,'phira'),canonical(human,'phira')
            score=score_events(p,h,'phira')['edit_utility']['full_note']
            full=2*score['matched']/(score['pred']+score['true'])
            if abs(full-r['full_note_f1'])>1e-12:raise ValueError('Full-note score differs from immutable review')
            gaps=np.diff(sorted({e.onset_us for e in pred if e.type=='tap'}));intervals.extend(gaps.tolist())
            per_type={}
            for kind in matches:
                per_type[kind]=score_events([e for e in p if e.type==kind],[e for e in h if e.type==kind],'phira')
                matches[kind]+=type_match_count(kind,per_type[kind])
            type_records.append(dict(chart_id=r['chart_id'],scores=per_type))
            tags.setdefault(r['stratum'].split('/')[0],[]).append(full)
            raw[arm][r['chart_id']]=dict(f1=full,audio=r['audio_sha256'],stratum=r['stratum'])
        rests=[r for r in stage['rows'] if r['role']=='reference_rest']
        if len(rests)!=4:raise ValueError('Reference-rest panel differs')
        summaries[arm]=dict(macro_f1=float(np.mean([r['full_note_f1']for r in records])),
            complete_legal_songs=sum(r['complete'] and not r['legality']for r in records),
            nonempty_songs=sum(r['notes']['model']>0 for r in records),
            predicted_notes=sum(r['notes']['model']for r in records),
            human_notes=sum(r['notes']['human']for r in records),
            tap_positive_intervals=len(intervals),tap_short_fraction=float(np.mean(np.array(intervals)<10000))if intervals else None,
            type_matches=matches,native_tag_macro_f1={t:float(np.mean(v))for t,v in tags.items()},
            reference_rest_mean=stage['rest_mean_predicted_notes'],type_scores=type_records)
        reports[arm]=(plan,review,stage)
    if raw['control'].keys()!=raw['candidate'].keys():raise ValueError('Evaluation panels differ')
    differences=[]
    for chart_id,a in raw['control'].items():
        b=raw['candidate'][chart_id]
        if (a['audio'],a['stratum'])!=(b['audio'],b['stratum']):raise ValueError('Song/stratum mismatch')
        differences.append(dict(chart_id=chart_id,stratum=a['stratum'],control=a['f1'],candidate=b['f1'],delta=b['f1']-a['f1']))
    summaries['candidate']['songs_not_worse']=sum(r['delta']>=0 for r in differences)
    limits=reports['candidate'][0]['quality_gates']
    if limits!=reports['control'][0]['quality_gates']:raise ValueError('Comparison gates differ')
    gates=numerical_gates(summaries['control'],summaries['candidate'],limits)
    for p in ['scripts/compare_local_horizon_quality.py','tests/test_sequence_quality_gates.py']:
        sources[p]=sha(ROOT/p)
    result=dict(id='D90-quality-comparison-1000-v1',complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
                limits=limits,summary=summaries,song_differences=differences,gates=gates,
                numerical_gates_passed=all(gates.values()),quality_passed=False,converged_candidates=0,
                pending='Actual20-song human review, conditions, audio counterfactual, reference and convergence cannot be inferred from numerical gates.',
                source_sha256=sources)
    write_json(output,result)
    print(json.dumps(dict(gates=gates,numerical_gates_passed=all(gates.values()))))


if __name__=='__main__':main()

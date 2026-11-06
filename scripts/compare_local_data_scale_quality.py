"""Apply unchanged whole-song gates to D94 and matched control, including raw multiplicity diagnostics."""
import ast
from collections import Counter
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.local_song_rollout import canonical
from scripts.run_local_song_baseline import sha, write_json
from scripts.compare_local_sequence_quality import numerical_gates, events, type_match_count

ROOT = Path(__file__).resolve().parents[1]


def verify_sources(plan, arm, root=ROOT):
    """Verify old execution from its archive, keeping scoring semantics fixed."""
    for name, expected in plan['inputs_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    allowed_historical_changes = {
        'beatfcn/events/phira.py', 'beatfcn/train/data.py',
        'beatfcn/train/local_canonical_audit.py', 'scripts/run_local_sequence_quality.py',
    }
    differences = []
    for name, expected in plan['code_sha256'].items():
        if arm == 'control':
            archive = root/plan['source_snapshot']/name
            if not archive.is_file() or sha(archive) != expected:
                raise ValueError('Historical source archive changed: '+name)
        if sha(root/name) == expected:
            continue
        if arm != 'control':
            raise ValueError('Candidate source changed: '+name)
        if name == 'beatfcn/events/canonical.py':
            # Only native admission changed after D87. All scoring types and
            # helpers must still have identical executable definitions.
            def scoring_ast(path):
                module = ast.parse(path.read_text())
                module.body = [node for node in module.body
                               if not isinstance(node, ast.FunctionDef) or node.name != 'build_model_view']
                return ast.dump(module, include_attributes=False)
            if scoring_ast(archive) != scoring_ast(root/name):
                raise ValueError('Historical canonical scoring definitions changed')
        elif name not in allowed_historical_changes:
            raise ValueError('Historical execution/scoring dependency changed: '+name)
        differences.append(name)
    return dict(source_snapshot=plan.get('source_snapshot'),
                historical_non_scoring_changes=differences,
                archived_code_verified=arm == 'control', live_candidate_verified=arm != 'control')


def main():
    output=ROOT/'reports/d94_quality_comparison_1000_v1.json'
    if output.exists():raise FileExistsError(output)
    sources={}; reports={}; summaries={}; raw={}; source_validation={}
    for arm in ('control','candidate'):
        paths=([f'bench/rhythm-v2/d87-sequence-{arm}-v1.json',
                f'reports/d87_{arm}_review_1000_v1.json',f'reports/d87_{arm}_stage_1000_v1.json']
               if arm=='control' else ['bench/rhythm-v2/d94-data-scale-quality-v1.json',
                'reports/d94_review_1000_v1.json','reports/d94_stage_1000_v1.json'])
        plan,review,stage=(json.loads((ROOT/p).read_text())for p in paths)
        train=json.loads((ROOT/plan['output']).read_text())
        if not train['complete'] or train['updates'] != 1000 or train.get('error') or train['plan_sha256'] != sha(ROOT/paths[0]):
            raise ValueError('Require terminal1000 from each frozen arm')
        if review['checkpoint_sha256'] != train['checkpoint_sha256'] or stage['checkpoint_sha256'] != train['checkpoint_sha256']:
            raise ValueError('Quality evidence uses another checkpoint')
        if not stage.get('complete') or stage.get('error') or len(review['records']) !=20:
            raise ValueError('Incomplete whole-song or stage evidence')
        source_validation[arm] = verify_sources(plan, arm)
        sources.update({p:sha(ROOT/p)for p in paths+[plan['output']]})
        records=review['records'];raw[arm]={}
        targets={r['chart_id']:r for r in json.loads((ROOT/plan['tape']).read_text())['charts']}
        if len({r['chart_id']for r in records}) !=20:raise ValueError('Duplicate evaluation song')
        intervals=[];matches={t:0 for t in ('hold','flick','drag')};tags={}
        type_records=[]; multiplicity_records=[]; predicted_types=Counter()
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
            predicted_types.update(e.type for e in pred)
            multiplicities={}
            for label,ev in [('model',pred),('human',human)]:
                heads=Counter(e.onset_us for e in ev)
                taps=Counter(e.onset_us for e in ev if e.type=='tap')
                multiplicities[label]=dict(max_notes_same_timestamp=max(heads.values(),default=0),
                    repeated_onset_objects=sum(n-1 for n in heads.values()),
                    tap_zero_adjacent_pairs=sum(n-1 for n in taps.values()),
                    max_taps_same_timestamp=max(taps.values(),default=0))
            multiplicity_records.append(dict(chart_id=r['chart_id'],**multiplicities))
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
            reference_rest_mean=stage['rest_mean_predicted_notes'],type_scores=type_records,
            predicted_types=dict(predicted_types),multiplicity=multiplicity_records,
            tap_zero_adjacent_pairs=sum(r['model']['tap_zero_adjacent_pairs'] for r in multiplicity_records),
            max_notes_same_timestamp=max(r['model']['max_notes_same_timestamp'] for r in multiplicity_records))
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
    for p in ['scripts/compare_local_data_scale_quality.py','scripts/compare_local_sequence_quality.py','tests/test_sequence_quality_gates.py']:
        sources[p]=sha(ROOT/p)
    result=dict(id='D94-quality-comparison-1000-v1',complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
                limits=limits,summary=summaries,song_differences=differences,gates=gates,
                numerical_gates_passed=all(gates.values()),quality_passed=False,converged_candidates=0,
                pending='Actual20-song human review, conditions, audio counterfactual, reference and convergence cannot be inferred from numerical gates.',
                source_sha256=sources, source_validation=source_validation)
    write_json(output,result)
    print(json.dumps(dict(gates=gates,numerical_gates_passed=all(gates.values()))))


if __name__=='__main__':main()

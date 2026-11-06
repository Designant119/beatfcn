"""Disposable real single-field positive-pair fit; never promotes fitted weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from beatfcn.events import controls as C
from beatfcn.model import features as F
from beatfcn.model.evaluation import groups_to_events,legality_issues,score_events
from beatfcn.model.p1 import P1Config,P1Model
from beatfcn.model.sample import generate_core,core_context
from beatfcn.train.probe import metrics_json,capture_decisions,assess_memorisation,assess_generation
from r09_local_data import LocalPairs,ROOT,file_sha,verify_files


def aggregate(core_rows, step):
    names=set().union(*(row['heads'] for row in core_rows))
    heads={}
    for name in sorted(names):
        values=[row['heads'][name] for row in core_rows if name in row['heads']]
        count=sum(v['count'] for v in values);correct=sum(v['correct'] for v in values)
        heads[name]=dict(loss=sum(v['loss'] for v in values)/len(core_rows),count=count,
            correct=correct,accuracy=correct/count if count else None)
    return dict(step=step,loss=sum(r['loss'] for r in core_rows)/len(core_rows),
        illegal=sum(r['illegal'] for r in core_rows),heads=heads,per_core=core_rows)


def full_f1(scores):
    row=scores['edit_utility']['full_note'];n=row['pred']+row['true']
    return 2*row['matched']/n if n else 1.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    path=parser.parse_args().plan;plan=json.loads(path.read_text());verify_files(plan)
    feasible=json.loads((ROOT/plan['feasibility_report']).read_text())
    if not feasible['passed'] or not feasible['complete']: raise ValueError('real CPU feasibility did not pass')
    torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1);torch.manual_seed(plan['seed'])
    started=time.perf_counter();data=LocalPairs(plan)
    items=[i for pair in plan['probe_pairs'] for i in pair]
    result=dict(schema='beatfcn.r09_single_field_fit.v1',complete=False,probe_passed=False,
        plan_sha256=file_sha(path),optimizer_updates=0,checkpoint_saved=False,model_promotion=False,
        cloud_compute_started=False,history=[],gradients=[],generation=[],pair_gates=[],
        scope='Four fixed training cores; individual hold-head and drag requests only. Not held-out quality.')
    def save(phase):
        result.update(phase=phase,wall_s=time.perf_counter()-started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
        destination=ROOT/plan['result'];temporary=destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(destination)
        print(json.dumps(dict(phase=phase,updates=result['optimizer_updates'],wall_s=result['wall_s'],
            loss=result['history'][-1]['loss'] if result['history'] else None)),flush=True)
    save('start')
    try:
        ck=torch.load(ROOT/plan['checkpoint_file'],map_location='cpu',weights_only=False)
        model=P1Model(P1Config(**ck['config'])).eval();model.load_state_dict(ck['model'],strict=True);del ck
        batches=[data.batch([item]) for item in items]
        def measure(step):
            rows=[]
            for item,bt in zip(items,batches):
                with torch.no_grad(),capture_decisions() as decisions:
                    loss,metrics=model(bt)
                rows.append(dict(item=item,**metrics_json(loss,metrics),decisions=decisions))
            return aggregate(rows,step)
        def decode(item,variant,step,other):
            override=None
            if variant=='removed': override=np.full(len(F.CONTROL_FIELDS),-1,np.int64)
            elif variant=='swapped': override=data.example(other).controls
            ex,sample,song,truth,row=data.generation(item,field_override=override)
            holds,runs,heads=core_context(sample,ex,{})
            diagnostics={};t0=time.perf_counter()
            with torch.no_grad():
                groups=generate_core(model,ex,song,holds,runs,heads,device='cpu',greedy=True,seed=plan['seed'],
                    diagnostics=diagnostics,deadline=min(started+plan['timeout_s']-5,t0+plan['generation_core_timeout_s']))
            events=groups_to_events(groups,ex.c0_us,row['mode'],placement=False)
            issues=legality_issues(sample.request,events,placement=False)
            _,raw=C.derive_controls(row['mode'],events,runs)
            requested=sample.request['controls'].get(item['field'])
            score=score_events(events,truth,row['mode'])
            other_truth=data.generation(other)[3]
            return dict(step=step,variant=variant,item=item,chart_id=row['chart_id'],mode=row['mode'],c0_us=ex.c0_us,
                complete=bool(diagnostics.get('terminated_by_end')),legal=not issues,legality=issues,
                diagnostics=diagnostics,scores=score,other_target_full_note_f1=full_f1(score_events(events,other_truth,row['mode'])),
                generated_events=[e.to_json() for e in events],truth_events=[e.to_json() for e in truth],
                predicted_types=dict(Counter(e.type for e in events)),truth_types=dict(Counter(e.type for e in truth)),
                requested_band=requested,achieved_value=raw.get(item['field']),
                in_band=C.band_of(item['field'],raw.get(item['field']))==requested if requested is not None else None,
                wall_s=time.perf_counter()-t0)
        result['history'].append(measure(0))
        baseline=sum(row['loss'] for row in feasible['core_metrics'])/len(items)
        if abs(result['history'][0]['loss']-baseline)>1e-6: raise ValueError('real initial loss differs from feasibility')
        for pair in plan['probe_pairs']:
            for side,item in enumerate(pair): result['generation'].append(decode(item,'conditioned',0,pair[1-side]))
        save('initial_measured')
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=0.)
        for step in range(1,plan['steps']+1):
            if time.perf_counter()-started>plan['fit_deadline_s']: raise TimeoutError('frozen fit deadline')
            optimizer.zero_grad(set_to_none=True)
            for bt in batches:
                loss,metrics=model(bt)
                if not torch.isfinite(loss) or int(metrics['_illegal']): raise ValueError('invalid loss/target')
                (loss/len(batches)).backward()
            condition_grad=float(model.ctrl_e.weight.grad.abs().sum())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True)
            result['gradients'].append(dict(step=step,finite=bool(torch.isfinite(norm)),
                norm_before_clip=float(norm),condition_embedding_gradient_l1=condition_grad))
            optimizer.step();result['optimizer_updates']=step
            if step in plan['logging_steps']:
                result['history'].append(measure(step));save('fitting')
        result['supervised_gate']=assess_memorisation(result['history'],result['gradients'],plan['supervised_gate'],plan['steps'])
        confusion=np.sum([r['decisions']['type']['confusion_truth_rows_pred_columns']
            for r in result['history'][-1]['per_core']],axis=0)
        per_type={F.TYPES[i]:float(confusion[i,i]/confusion[i].sum()) for i in range(4) if confusion[i].sum()}
        result['type_memorisation_gate']=dict(passed=all(v>=.95 for v in per_type.values()),recall=per_type)
        model.requires_grad_(False)
        for pair in plan['probe_pairs']:
            final={}
            for side,item in enumerate(pair):
                for variant in ('conditioned','removed','swapped'):
                    row=decode(item,variant,plan['steps'],pair[1-side]);result['generation'].append(row)
                    final[(side,variant)]=row;save('fitted_example_decoded')
            margins=[full_f1(final[(s,'conditioned')]['scores'])-final[(s,'conditioned')]['other_target_full_note_f1'] for s in (0,1)]
            removed_equal=final[(0,'removed')]['generated_events']==final[(1,'removed')]['generated_events']
            swaps_equal=all(final[(s,'swapped')]['generated_events']==final[(1-s,'conditioned')]['generated_events'] for s in (0,1))
            in_band=all(final[(s,'conditioned')]['in_band'] for s in (0,1))
            legal_complete=all(r['legal'] and r['complete'] for r in final.values())
            passed=removed_equal and swaps_equal and in_band and legal_complete and min(margins)>=plan['matching_target_margin_at_least']
            result['pair_gates'].append(dict(pair_id=pair[0]['pair_id'],field=pair[0]['field'],mode=pair[0]['mode'],
                removed_identical=removed_equal,swapped_matches_other=swaps_equal,all_requested_bands_met=in_band,
                complete_legal=legal_complete,target_f1_margins=margins,passed=passed))
        correct=[r for r in result['generation'] if r['step']==plan['steps'] and r['variant']=='conditioned']
        result['generation_gate']=assess_generation(correct,len(items),plan['generation_f1_at_least'])
        result['binding_gate']=dict(kind='generation_swap',passed=all(r['passed'] for r in result['pair_gates']),
            passed_fields=[f"{r['mode']}:{r['field']}" for r in result['pair_gates'] if r['passed']],total_fields=len(result['pair_gates']),
            scope='Single-field control on memorised real pairs, not a generalisation benchmark.')
        result.update(complete=True,probe_passed=all(result[k]['passed'] for k in
            ('supervised_gate','type_memorisation_gate','generation_gate','binding_gate')))
        save('complete')
        if not result['probe_passed']: raise SystemExit('R09 necessary real single-field probe failed')
    except BaseException as exc:
        if result['complete']: raise
        result['error']=f'{type(exc).__name__}: {exc}';save('failed');raise


if __name__=='__main__': main()

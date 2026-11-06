"""Fixed-denominator free outcomes first, conditional diagnostics afterwards.

This is an evaluation scheduler. It never changes the model, training, native
decode, or event repair policy. Incomplete predictions remain in every primary
denominator, while raw partial events and their diagnostic scores are retained.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from .field_integrity import coverage,coverage_summary,cpu
from .native_cell_chain import canonical_head_mask
from .exact_cell_support import ExactEndpointTree
from .native_quality_metrics import teacher_scores,summarize


ORDER=('draft','empty','audio_zero','audio_shift','audio_foreign',
       'completion','reference_masked','density_low','density_high')
KINDS=('tap','hold','flick','drag')


def zero_scores(raw):
    result=deepcopy(raw);result['full_note_f1']=0.;result['onset_f1']=0.
    for name,value in result.items():
        if name.startswith('full_note_f1'):result[name]=0.
        elif isinstance(value,dict) and 'tp' in value:
            value.update(tp=0,f1=0.)
    for value in result['per_type'].values():value.update(tp=0,f1=0.)
    return result


def score_outcome(engine,events,targets,complete):
    raw=engine.score(events,targets)
    return (raw if complete else zero_scores(raw)),raw


def outcome_tasks(engine,plan):
    tasks=engine.specs(plan)
    for name in ('audio_zero','audio_shift','audio_foreign'):
        tasks[name]=[(i,'draft',None,False) for i in plan['indices']['dev']]
    if set(tasks)!=set(ORDER) or sum(map(len,tasks.values()))!=170:
        raise ValueError('Expected fixed 170 free tasks')
    return {name:tasks[name] for name in ORDER}


def core_coverage(inputs,source=None,*,shift=0.,scale=1.):
    source=inputs if source is None else source
    query=cpu(inputs['joint_frame_times'])[0]+np.float32(.02)
    result={}
    for name,step in (('local',.04),('mel',.01),('song',.5)):
        result[name]=coverage_summary(coverage(cpu(source[name+'_times']),cpu(source[name+'_valid']),query*scale,step,shift))
    result['scope']='Core-centre clock range only;source validity/content/receptive field are different questions'
    return result


def interpretable_pair(baseline,control,minimum=.95):
    if not baseline['prediction']['complete'] or not control['prediction']['complete']:return False
    original=baseline.get('control',{}).get('core_coverage',{})
    changed=control.get('control',{}).get('core_coverage',{})
    return all(original.get(k,{}).get('fraction',0)>=minimum and changed.get(k,{}).get('fraction',0)>=minimum for k in ('local','mel','song'))


def acoustic_pairs(draft,control,minimum=.95):
    keyed={r['index']:r for r in control};groups={}
    for r in draft:
        alternative=keyed[r['index']]
        groups.setdefault(str(r['component']),[]).append((r,alternative))
    rows={}
    for component,pairs in groups.items():
        covered=all(interpretable_pair(a,b,minimum) for a,b in pairs)
        rows[component]=dict(requests=len(pairs),common_core_covered=covered,
            complete=all(a['prediction']['complete'] and b['prediction']['complete'] for a,b in pairs),
            delta_onset10=float(np.mean([(a['scores']['onset_10ms']['f1'] or 0)-(b['scores']['onset_10ms']['f1'] or 0) for a,b in pairs])))
    selected=[r for r in rows.values() if r['common_core_covered']]
    return dict(components=rows,complete=all(r['complete'] for r in rows.values()),
        common_covered_components=len(selected),positive_covered_components=sum(r['delta_onset10']>0 for r in selected),
        covered_delta_onset10=float(np.mean([r['delta_onset10'] for r in selected])) if selected else None,
        full_denominator_components=len(rows),minimum_core_fraction=minimum,
        scope='All-component primary remains unchanged;covered pairs are an explicit sensitivity result')


def teacher_record(engine,steps,target,request,output,parts,prior):
    # Aggregate exact log-softmax factors come from the training law. Serialized
    # probabilities are for per-decision diagnostics and may underflow in FP32.
    cp=engine.array(output['count'][0].softmax(-1))
    result=teacher_scores(steps,target,request,cp,prior)
    counts=torch.as_tensor(target['counts'],device=output['count'].device,dtype=torch.long)
    exact={k:float(v) for k,v in parts.items()}
    exact['count']=float(F.cross_entropy(output['count'][0],counts,reduction='sum'))
    if not all(math.isfinite(v) for v in exact.values()):raise FloatingPointError('Exact teacher factor NLL')
    expected=sum(map(len,target['members']))
    if len(steps)!=expected or len({(s['cell'],s['rank']) for s in steps})!=expected:raise ValueError('Teacher member coverage')
    previous={};heads=[];underflow=[]
    if cp.shape[0]!=len(counts):raise ValueError('Teacher count support length')
    for cell,number in enumerate(target['counts']):
        if cp[cell,number]==0:underflow.append(dict(field='count',cell=cell,status='fp32_probability_underflow'))
    for s in steps:
        cell=s['cell'];truth=target['members'][cell][s['rank']];p=np.asarray(s['head_probability'])
        mask=canonical_head_mask(target['heads'][cell],target['allowed'][cell],previous.get(cell),request)
        if len(p)!=len(mask):raise ValueError('Teacher head support length')
        supported=bool(mask[truth.address]);prob=float(p[truth.address]);status='finite' if prob>0 else 'fp32_probability_underflow'
        if not supported:status='structural_support_failure'
        heads.append(dict(cell=cell,rank=s['rank'],type=KINDS[truth.kind],status=status,
            nll=-math.log(prob) if supported and prob>0 else None,loss_semantics='infinite' if not supported else 'exact aggregate uses log_softmax',
            legal_addresses=int(mask.sum()),target_address=truth.address,allowed=mask.tolist()))
        if not supported or prob==0:underflow.append(dict(field='head',cell=cell,rank=s['rank'],status=status))
        if not s['kind_allowed'][truth.kind]:underflow.append(dict(field='type',cell=cell,rank=s['rank'],status='structural_support_failure'))
        elif s['type_probability'][truth.kind]==0:underflow.append(dict(field='type',cell=cell,rank=s['rank'],status='fp32_probability_underflow'))
        if truth.kind==1:
            prev=previous.get(cell);head=s['event']['onset_us']
            minimum=prev.end_us if prev and prev.onset_us==head and prev.type=='hold' else None
            path=ExactEndpointTree(request,head,minimum).path(truth.end_us)
            if len(s['tail'])!=3:raise ValueError('Teacher endpoint support layers')
            for level,trace in enumerate(s['tail']):
                if len(trace['probability'])!=len(trace['labels']):raise ValueError('Teacher endpoint support length')
                if path[level] not in trace['labels']:underflow.append(dict(field='endpoint',cell=cell,rank=s['rank'],status='structural_support_failure',level=level))
                elif trace['probability'][trace['labels'].index(path[level])]==0:underflow.append(dict(field='endpoint',cell=cell,rank=s['rank'],status='fp32_probability_underflow',level=level))
        previous[cell]=RhythmEvent(**s['event'])
    result.update(count_nll=exact['count'],exact_factor_nll=exact,exact_joint_nll=sum(exact.values()),public_cell_denominator=len(counts),
        exact_joint_nll_per_cell=sum(exact.values())/len(counts),head_decisions=heads,numerical_or_support_failures=underflow,
        scope='All target events/counts under true canonical history;not free-event quality or unique-machine-head evidence')
    return result


def case_inputs(engine,plan,panel,row,name,index,view,bands,masked):
    rr=engine.cell_row(panel,row(index,view,0,bands),mask_reference=masked)
    x,t=engine.cell_batch(rr,'mps');control=dict(core_coverage=core_coverage(x))
    if name=='audio_zero':x=engine.zero_audio(x);control['scope']='All3featurestreams zero;not waveform silence'
    elif name=='audio_shift':
        control['core_coverage']=core_coverage(x,shift=.733);x,cov=engine.audio_variant(x,x,shift=.733);control.update(shift_s=.733,coverage=cov)
    elif name=='audio_foreign':
        dev=plan['indices']['dev'];at=dev.index(index)
        donor=next(dev[(at+j)%len(dev)] for j in range(1,len(dev)) if panel.records[dev[(at+j)%len(dev)]]['component']!=panel.records[index]['component'])
        dr=engine.cell_row(panel,row(donor,'draft'));dx,_=engine.cell_batch(dr,'mps')
        scale=(dr['request'].end_us-dr['request'].start_us)/(rr['request'].end_us-rr['request'].start_us)
        control['core_coverage']=core_coverage(x,dx,scale=scale)
        x,cov=engine.audio_variant(x,dx,scale=scale);control.update(donor=donor,donor_component=panel.records[donor]['component'],scale=scale,coverage=cov)
    return rr,x,t,control


def summarize_teacher(records):
    out={}
    for kind in KINDS:
        groups={}
        for r in records:
            if r.get('teacher_status')!='completed':continue
            values=r['teacher_scores']['type_binary'][kind]
            groups.setdefault(str(r['component']),[]).extend(values)
        out[kind]={c:dict(decisions=len(v),positive=sum(a['positive'] for a in v),negative=sum(not a['positive'] for a in v),
            gain=float(np.mean([a['prior_nll']-a['nll'] for a in v])) if v else None,
            positive_gain=float(np.mean([a['prior_nll']-a['nll'] for a in v if a['positive']])) if any(a['positive'] for a in v) else None,
            negative_gain=float(np.mean([a['prior_nll']-a['nll'] for a in v if not a['positive']])) if any(not a['positive'] for a in v) else None) for c,v in groups.items()}
    return out


def finalize(engine,plan,groups,report,p2):
    summaries={name:summarize(rows) for name,rows in groups.items()}
    s=summaries['draft'];draft=groups['draft'];g=plan['research_gate'];type_checks={}
    teacher_complete=all(r.get('teacher_status')=='completed' for r in draft)
    for kind in ('hold','flick','drag'):
        null=[]
        for j in range(len(plan['shifts_us'])):
            values=[r['output_null'][j]['types'][kind] for r in draft]
            null.append(2*sum(v['tp'] for v in values)/max(sum(v['pred']+v['true'] for v in values),1))
        binary=[b for r in draft if r.get('teacher_status')=='completed' for b in r['teacher_scores']['type_binary'][kind]]
        gain=float(np.mean([b['prior_nll']-b['nll'] for b in binary])) if binary else None
        tail=[a-b for r in draft if r.get('teacher_status')=='completed' for a,b in zip(r['teacher_scores']['endpoint_prior_nll'],r['teacher_scores']['endpoint_nll'])]
        endpoint=float(np.mean(tail)) if tail else None
        proper_valid=teacher_complete and not any(r['teacher_scores']['numerical_or_support_failures'] for r in draft if r.get('teacher_status')=='completed')
        evidence=proper_valid and ((gain is not None and gain>=g['proper_gain']) or (kind=='hold' and endpoint is not None and endpoint>=g['proper_gain']))
        typ=s['types'][kind]
        type_checks[kind]=dict(**typ,null_micro_f1_mean=float(np.mean(null)),binary_proper_gain=gain,binary_decisions=len(binary),
            positive_decisions=sum(b['positive'] for b in binary),negative_decisions=sum(not b['positive'] for b in binary),
            endpoint_path_gain=endpoint if kind=='hold' else None,
            passed=bool(typ['f1']>=g['one_non_tap_f1'] and typ['tp']>=g['non_tap_tp'] and typ['matched_components']>=g['non_tap_components'] and typ['f1']-np.mean(null)>=g['non_tap_null_gain'] and evidence))
    pairs={n:acoustic_pairs(draft,groups[n]) for n in ('audio_shift','audio_foreign')}
    ratio=s['pred']/max(s['true'],1)
    checks=dict(complete=s['complete']==32,empty=summaries['empty']['pred']==0 and summaries['empty']['complete']==2,
        count=g['count_ratio'][0]<=ratio<=g['count_ratio'][1],full=s['full50_component_macro']>=g['full50'],
        null=s['onset10_component_macro']-s['null10_component_macro']>=g['null10_gain'],
        positive_components=sum(c['onset10']>c['null10'] for c in s['components'].values())>=g['positive_components'],
        shift=s['onset10_component_macro']-summaries['audio_shift']['onset10_component_macro']>=g['shifted10_gain'],
        foreign=s['onset10_component_macro']-summaries['audio_foreign']['onset10_component_macro']>=g['foreign10_gain'],
        acoustic_controls=all(p['complete'] and p['common_covered_components']>=10 and p['positive_covered_components']>=10 for p in pairs.values()),
        non_tap=any(v['passed'] for v in type_checks.values()),teacher_complete=teacher_complete,
        p2=len(p2)==8 and all('failure' not in p for p in p2),
        all_free_complete=all(r['prediction']['complete'] for rows in groups.values() for r in rows),
        all_teacher_complete=all(r.get('teacher_status')=='completed' for rows in groups.values() for r in rows))
    report.update(evaluation=summaries,p2=p2,research_checks={k:bool(v) for k,v in checks.items()},research_qualified=bool(all(checks.values())),
        non_tap_evidence=type_checks,count_ratio=ratio,acoustic_pairs=pairs,teacher_component_scores=summarize_teacher(draft),
        confirmation_scored=False,scaling_authorized=False,all_free_outcomes=sum(map(len,groups.values())),
        quality_gate_status='Separate historical stricter gate;not implied by research qualification')
    baseline=plan.get('comparison_baseline')
    if baseline:
        q=plan['quality_gate'];gains={k:s['types'][k]['f1']-baseline['per_type'][k]['f1'] for k in ('hold','flick','drag')}
        strict=dict(full=s['full50_component_macro']>=q['full50'],
            two_non_tap_gains=sum(v>=q['two_non_tap_gains'] for v in gains.values())>=2,
            third_non_tap_retained=min(gains.values())>=-q['third_non_tap_max_drop'],
            count=q['count_ratio'][0]<=ratio<=q['count_ratio'][1],
            null=s['onset10_component_macro']-s['null10_component_macro']>=q['null_gain'],
            positive_components=sum(c['onset10']>c['null10'] for c in s['components'].values())>=q['positive_components'],
            shift=s['onset10_component_macro']-summaries['audio_shift']['onset10_component_macro']>=q['input_shift_gain'],
            foreign=s['onset10_component_macro']-summaries['audio_foreign']['onset10_component_macro']>=q['foreign_gain'],
            controls_interpretable=checks['acoustic_controls'],all_free_complete=checks['all_free_complete'],empty=checks['empty'])
        report.update(quality_gate_status='passed' if all(strict.values()) else 'not_passed',
            quality_gate_checks={k:bool(v) for k,v in strict.items()},non_tap_vs_baseline=gains,
            baseline_comparison_scope='Whole recipe against frozen strongest SSD metrics;different seed/objective/optimizer,not isolated architecture causality')
    engine.write(engine.ART/'evaluation-summary.json',report)


def metadata_outcomes(engine,plan,panel,row):
    tasks=outcome_tasks(engine,plan);groups={k:[] for k in tasks}
    # This is executed before registration/training. Discard each feature row;
    # only native labels, requests, references and initial failure rows persist.
    for name,items in tasks.items():
        for index,view,bands,masked in items:
            rr=engine.cell_row(panel,row(index,view,0,bands),mask_reference=masked)
            scores,raw=score_outcome(engine,[],rr['targets'],False)
            null=engine.output_null([],rr['targets'],rr['request'],plan['shifts_us'])
            groups[name].append(dict(index=index,component=panel.records[index]['component'],view=view,bands=bands,masked=masked,
                request=asdict(rr['request']),target=[asdict(e) for e in rr['targets']],visible=[asdict(e) for e in rr['visible']],
                known_regions=list(rr.get('known_regions',())),provenance=rr.get('provenance'),status='not_attempted',
                prediction=dict(complete=False,failure='Not attempted',events=[]),scores=scores,raw_partial_scores=raw,
                output_null=null,output_null_onset10=[0.]*len(null),control={}))
            del rr
    return groups


@torch.no_grad()
def evaluate(engine,model,plan,panel,row,guard,report):
    report.update(updates=1000,model_training_updates=1000,diagnostic_optimizer_updates=0)
    prior=engine.read(str((engine.ART/'data-identity.json').relative_to(engine.ROOT)))['prior']
    tasks=outcome_tasks(engine,plan);deadline=guard.start+guard.wall_s;p2=[]
    path=engine.ROOT/plan['expected_free_metadata']
    if engine.sha(path)!=plan['expected_free_metadata_sha256']:raise ValueError('Expected free metadata drift')
    groups=engine.read(plan['expected_free_metadata'])
    for name,records in groups.items():engine.write(engine.ART/f'{name}.json',records)
    engine.write(engine.ART/'expected-outcomes.json',[dict(task=k,index=r['index'],component=r['component'],view=r['view'],target_count=len(r['target'])) for k,rows in groups.items() for r in rows])
    def persist():
        for name,records in groups.items():engine.write(engine.ART/f'{name}.json',records)
        engine.write(engine.ART/'free-progress.json',dict(expected=170,attempted=sum(r['status']!='not_attempted' for rows in groups.values() for r in rows),
            completed=sum(r['prediction']['complete'] for rows in groups.values() for r in rows)))
    try:
        for name,items in tasks.items():
            for rec,(index,view,bands,masked) in zip(groups[name],items):
                guard.check('free_'+name);rec['status']='attempted'
                rr,x,t,control=case_inputs(engine,plan,panel,row,name,index,view,bands,masked)
                out=model(x);pred=engine.prediction(model,rr,x,t,guard,deadline,out);events=[RhythmEvent(**e) for e in pred['events']]
                scores,raw=score_outcome(engine,events,rr['targets'],pred['complete']);null=engine.output_null(events,rr['targets'],rr['request'],plan['shifts_us'])
                if not pred['complete']:
                    for n in null:
                        n.update(onset10=0.,full50=0.)
                        for value in n['types'].values():value.update(tp=0,f1=0.)
                cp=engine.array(out['count'][0].softmax(-1))
                rec.update(status='completed' if pred['complete'] else 'decode_failed',prediction=pred,scores=scores,raw_partial_scores=raw,
                    control=control,input_sha256=engine.input_digest(x),count_probability=cp.tolist(),count_target=t['counts'].tolist(),
                    count_diagnostics=dict(expected_without_overflow=float((cp[:,:65]*np.arange(65)[None]).sum()),overflow_probability_sum=float(cp[:,65].sum()),overflow_argmax_cells=int((cp.argmax(-1)==65).sum())),
                    output_null=null,output_null_onset10=[n['onset10'] for n in null],endpoint_min_cost=engine.endpoint_diagnostic(events,rr['targets']),
                    merged_visible_plus_generated=[asdict(e) for e in sorted([*rr['visible'],*events])],merged_scope='External multiset merge;no dedup;unknown-only primary score')
                engine.write(engine.ART/f'{name}.json',groups[name]);del x,t,out,rr;guard.clear()
            persist();print('free',name,len(items),flush=True)
        report['free_phase_complete']=True;persist()
        # Conditional scores cannot preempt persistence of any free outcome.
        for name,items in tasks.items():
            for rec,(index,view,bands,masked) in zip(groups[name],items):
                guard.check('teacher_'+name);rr,x,t,_=case_inputs(engine,plan,panel,row,name,index,view,bands,masked);out=model(x)
                try:
                    _,parts,steps,_=engine.run_members(model,out,x,rr['request'],t,t['counts'],teacher=True,trace=True,deadline=deadline)
                    rec.update(teacher_status='completed',teacher_steps=steps,teacher_scores=teacher_record(engine,steps,t,rr['request'],out,parts,prior))
                    del parts,steps
                except ValueError as exc:
                    rec.update(teacher_status='support_or_protocol_failed',teacher_failure=str(exc),joint_nll_semantics='infinite_or_protocol_failure;not removed from denominator')
                engine.write(engine.ART/f'{name}.json',groups[name]);del x,t,out,rr;guard.clear()
        for index in plan['p2_indices']:
            baseline=next(r for r in groups['draft'] if r['index']==index)
            if not baseline['prediction']['complete']:
                p2.append(dict(index=index,failure='Incomplete baseline;fixed index retained'));continue
            guard.check('p2');rr=engine.cell_row(panel,row(index,'draft'));x,t=engine.cell_batch(rr,'mps');out=model(x)
            p2.append(engine.field_panel(model,rr,x,t,baseline['prediction'],out,guard,'quality-p2',False))
            engine.write(engine.ART/'p2-progress.json',p2);del x,t,out,rr;guard.clear()
        report['status']='terminal_pending_mechanism_review'
    except BaseException as exc:
        report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
        for rows in groups.values():
            for r in rows:
                if r['status']=='attempted':r.update(status='execution_failed',failure=report['failure'])
        missing={r['index'] for r in p2}
        p2.extend(dict(index=i,failure='Not attempted after evaluation stop') for i in plan['p2_indices'] if i not in missing)
    finally:
        persist();finalize(engine,plan,groups,report,p2)

"""Frozen, serial requirement screen shared by the training loop and checkpoint CLI."""
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import encode_events, teacher_sequence
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import event_loss
from beatfcn.train.checkpoint_resume import file_sha
from beatfcn.train.counterfactual_density import real_density_curve
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request, request_inputs
from beatfcn.train.requirement_diagnostics import diagnose, summarize_requirements

ACOUSTIC = ('local', 'mel', 'song')


def write_result(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def case_specs(protocol):
    cases=[dict(index=s['index'],condition='free')for s in protocol['samples']]
    for anchor in protocol['anchors']:
        cases.extend(dict(index=anchor['index'],condition=kind,foreign_index=anchor['foreign_index'])
            for kind in ('audio_zero','local_roll','audio_foreign'))
        cases.extend(dict(index=anchor['index'],condition=kind)
            for kind in ('density_zero','density_low','density_high','density_shuffle'))
    return cases


def validate_panel(panel, protocol):
    if protocol['schema']!='requirement-screen-v2':raise ValueError('Wrong requirement protocol version')
    seen=set()
    for sample in protocol['samples']:
        row=panel.records[sample['index']]
        for key in ('chart_id','audio_sha256','component','mode','split','request'):
            if row[key]!=sample[key]:raise ValueError('Frozen sample changed: '+key)
        if row['split']!='val' or panel.charts[row['chart_index']]['target_sha256']!=sample['target_sha256']:
            raise ValueError('Requirement screen must use frozen held-out targets')
        if sample['index']in seen:raise ValueError('Duplicate frozen sample')
        seen.add(sample['index'])
    for anchor in protocol['anchors']:
        if anchor['index']not in seen or anchor['foreign_index']not in seen:raise ValueError('Unfrozen counterfactual source')
        a,b=[panel.records[anchor[k]]for k in ('index','foreign_index')]
        if a['audio_sha256']==b['audio_sha256']or a['mode']!=b['mode']:
            raise ValueError('Foreign audio requires another held-out song in the same mode')


def audio_intervention(inputs, condition, *, foreign=None, roll_s=2.):
    result={key:np.array(value,copy=True)for key,value in inputs.items()}
    if condition=='audio_zero':
        for stream in ACOUSTIC:result[stream].fill(0)
    elif condition=='audio_foreign':
        if foreign is None:raise ValueError('Foreign features missing')
        for stream in ACOUSTIC:
            for key in (stream,stream+'_times',stream+'_valid'):
                result[key]=np.array(foreign[key],copy=True)
    elif condition=='local_roll':
        for stream,axis in (('local',1),('mel',0)):
            valid=np.flatnonzero(result[stream+'_valid'])
            times=result[stream+'_times'][valid]
            if len(valid)>1:
                shift=int(round(roll_s/float(np.median(np.diff(times)))))
                data=np.take(result[stream],valid,axis=axis)
                index=[slice(None)]*data.ndim;index[axis]=valid
                result[stream][tuple(index)]=np.roll(data,shift,axis=axis)
    else:raise ValueError('Unknown audio intervention')
    return result


def prepare_case(panel, protocol, spec):
    sample=next(s for s in protocol['samples']if s['index']==spec['index'])
    row=panel.records[spec['index']];request=event_request(row['request']);events=panel.targets(spec['index'])
    from beatfcn.train.counterfactual_exposure import unconditional_inputs
    # Unconditional controls exactly as training sees a fully dropped sample.
    inputs,_=request_inputs(panel.features,row['audio_sha256'],request,
        unconditional_inputs(request.mode,request,render_cell_us=protocol['cell_us'])['controls'])
    curve=None;kind=spec['condition'];provenance={}
    if kind.startswith('density_'):
        baseline=real_density_curve(events,request.start_us,request.end_us)
        factor={'density_zero':0.,'density_low':.5,'density_high':2.,'density_shuffle':2.}[kind]
        values=[s['density']*factor for s in baseline]
        if kind=='density_shuffle':values=list(reversed(values))
        curve=[dict(start_us=s['start_us'],end_us=s['end_us'],density=value)
               for s,value in zip(baseline,values)]
        provenance=dict(condition_source='reference-derived density as explicit caller request; not exact event supervision',
            multiplier=factor,shuffled=kind=='density_shuffle')
    inputs.update(grid_density_curve(curve or [],request.start_us,request.end_us,cell_us=protocol['cell_us'],scale=16.))
    if kind in ('audio_zero','local_roll','audio_foreign'):
        foreign=None
        if kind=='audio_foreign':
            donor=panel.records[spec['foreign_index']];dreq=event_request(donor['request'])
            foreign=panel.features.window(donor['audio_sha256'],dreq.start_us,dreq.end_us)
            provenance=dict(foreign_audio_sha256=donor['audio_sha256'],foreign_index=spec['foreign_index'])
        inputs=audio_intervention(inputs,kind,foreign=foreign,roll_s=protocol['local_roll_s'])
    return dict(**sample,condition=kind,case_id=f'{sample["index"]}:{kind}',inputs=inputs,
        request_object=request,truth=[asdict(e)for e in events],requested_curve=curve,
        condition_provenance=provenance)


def _teacher_ce(model, case, device):
    request=case['request_object']
    from beatfcn.model.event_canvas import RhythmEvent
    sequence=teacher_sequence(encode_events([RhythmEvent(**e)for e in case['truth']],request),request)
    inp,seq=collate([dict(inputs=case['inputs'],sequence=sequence)])
    inp={k:v.to(device)for k,v in inp.items()};seq={k:v.to(device)for k,v in seq.items()}
    return float(event_loss(model(inp,seq),seq))


def run_requirement_screen(model,panel,protocol,*,checkpoint,plan_sha256,output,device,
                           max_wall_s=150.,max_tokens=2048,batch_size=8,compiled=False):
    if max_wall_s<=0 or max_tokens<1 or not 1<=batch_size<=8:
        raise ValueError('Invalid bounded evaluation resources')
    validate_panel(panel,protocol);device=torch.device(device);start=time.monotonic();deadline=start+max_wall_s
    specs=case_specs(protocol);expected=[f'{s["index"]}:{s["condition"]}'for s in specs]
    source_files=[__file__,str(Path(__file__).with_name('requirement_diagnostics.py'))]
    import beatfcn.model.full_event_generate as generator
    import beatfcn.train.density_curve as density
    source_files += [generator.__file__,density.__file__]
    result=dict(schema='requirement-screen-results-v2',protocol_sha256=hashlib.sha256(
        json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        protocol_file_sha256=protocol.get('_file_sha256'),checkpoint=checkpoint,
        plan_sha256=plan_sha256,config=asdict(model.c),code_sha256={Path(p).name:file_sha(p)for p in source_files},
        expected_cases=expected,records=[],all_cases_measured=False,
        inference=dict(batch_size=batch_size,max_tokens=max_tokens,max_wall_s=max_wall_s,
            kv_cache=True,greedy=True,postprocessing='none'),
        unmeasured=['whole-song structure','native play/edit','human preference','non-density controls'])
    output=Path(output)
    if output.exists():raise FileExistsError('Evaluation identity already exists; never overwrite')
    write_result(output,result)
    flags=[(module,module.training)for module in model.modules()]
    force_eager=torch.compiler.set_stance('force_eager')if compiled else nullcontext()
    amp=torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()
    cuda_devices=[device.index or 0]if device.type=='cuda'else []
    try:
        with torch.random.fork_rng(devices=cuda_devices),force_eager,torch.no_grad(),amp:
            model.eval()
            for first in range(0,len(specs),batch_size):
                if time.monotonic()>=deadline:break
                group=[prepare_case(panel,protocol,s)for s in specs[first:first+batch_size]]
                for case in group:
                    case['teacher_joint_ce']=(_teacher_ce(model,case,device)
                                              if case['condition']=='free' and time.monotonic()<deadline else None)
                if time.monotonic()>=deadline:break
                rows=[]
                for case in group:
                    dummy=teacher_sequence(encode_events([],case['request_object']),case['request_object'])
                    rows.append(dict(inputs=case['inputs'],sequence=dummy))
                inp,_=collate(rows);inp={k:v.to(device)for k,v in inp.items()}
                groups_left=(len(specs)-first+batch_size-1)//batch_size
                group_deadline=min(deadline,time.monotonic()+max(0.,deadline-time.monotonic())/groups_left)
                predictions=generate_batch(model,inp,[c['request_object']for c in group],max_tokens,
                    deadline=group_deadline)
                for case,prediction in zip(group,predictions):
                    case.pop('inputs');case.pop('request_object');case['prediction']=prediction
                    case['diagnostics']=diagnose(prediction,case['truth'],case['request'],
                        requested_curve=case['requested_curve'],reference_applicable=not case['condition'].startswith('density_'))
                    result['records'].append(case)
                result['summary']=summarize_requirements(result['records'],protocol)
                result['elapsed_s']=time.monotonic()-start;write_result(output,result)
                print(json.dumps(dict(phase='requirement_eval_progress',optimizer_updates=checkpoint['optimizer_updates'],
                    measured=len(result['records']),expected=len(specs))),flush=True)
                del inp,rows,group,predictions
    finally:
        for module,training in flags:module.training=training
    measured={r['case_id']for r in result['records']}
    result.update(all_cases_measured=len(measured)==len(expected),unattempted_cases=[c for c in expected if c not in measured],
        elapsed_s=time.monotonic()-start,summary=summarize_requirements(result['records'],protocol))
    write_result(output,result)
    return result

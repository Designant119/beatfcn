"""Budgeted real-input audit and 100-step memorisation prerequisite.

Writes diagnostic JSON only. Does not save or promote an updated checkpoint.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import source_digest, save

STAGE='fixed-batch-health-v1'
FIXTURE=training.REPO/'bench/rhythm-v2/fixed-batch-probe-v1.json'
app=modal.App('beatfcn-fixed-batch-health')
data=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
image=(training.image.add_local_file(__file__,'/probe/fixed_batch_probe_app.py')
       .add_local_file(str(FIXTURE),'/probe/fixture.json'))
RESOURCES=dict(gpu='L40S',cpu=4,memory_gib=16,timeout_s=360,startup_allowance_s=120,retries=0,max_containers=1)


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),volumes={'/data':data},
              timeout=360,retries=0,max_containers=1,scaledown_window=2)
def probe(expected_source,expected_script,expected_fixture):
    import io
    import numpy as np
    import torch
    import beatfcn
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.events.canonical import Event
    from beatfcn.events.requests import build_request,assert_no_leak
    from beatfcn.model import features as F
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config,P1Model
    from beatfcn.model.sample import generate_core,core_context
    from beatfcn.model.evaluation import groups_to_events,legality_issues,score_events
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.probe import tensor_summary,metrics_json,capture_decisions,assess_memorisation,assess_generation
    started=time.perf_counter()
    torch.set_num_threads(4)
    if source_digest(Path(beatfcn.__file__).parent)!=expected_source:raise ValueError('source differs from reservation')
    if hashlib.sha256(Path('/probe/fixed_batch_probe_app.py').read_bytes()).hexdigest()!=expected_script:raise ValueError('script changed')
    payload=Path('/probe/fixture.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_fixture:raise ValueError('fixture changed')
    fixture=json.loads(payload)
    output=Path('/data/evolution/capability-evolution-20261024')/STAGE
    if output.exists():raise ValueError('output exists; inspect original handle instead of retrying')
    output.mkdir(parents=True)
    report=dict(stage=STAGE,complete=False,source_sha256=expected_source,script_sha256=expected_script,
                fixture_sha256=expected_fixture,protocol=fixture,history=[],gradients=[],generation=[],
                per_core_teacher_forced=[],data_health={},optimizer_updates=0,checkpoint_saved=False,
                scope='Fixed training examples; memorisation diagnostic only, not benchmark generalisation.')
    def record():
        report['function_wall_s']=time.perf_counter()-started
        save(output/'result.json',report);data.commit()
    def progress(event,**values):
        print(json.dumps(dict(event=event,elapsed_s=round(time.perf_counter()-started,3),**values)),flush=True)
    record()
    monitor=GpuUtil()
    try:
        ck_bytes=Path(fixture['checkpoint']['path']).read_bytes()
        if hashlib.sha256(ck_bytes).hexdigest()!=fixture['checkpoint']['sha256']:raise ValueError('checkpoint changed')
        ck=torch.load(io.BytesIO(ck_bytes),map_location='cpu',weights_only=False)
        if ck['step']!=3500:raise ValueError('checkpoint step differs')
        cfg=P1Config(**ck['config'])
        if cfg.offset_sigma_ms or cfg.lane_history_groups or cfg.selection:raise ValueError('unexpected baseline interventions')
        model=P1Model(cfg).cuda().eval();model.load_state_dict(ck['model'])
        corpus=Corpus(**ck['data']);corpus.train_rows([r['row'] for r in fixture['rows']])
        report['model_config']=ck['config'];report['corpus_config']=ck['data']
        del ck,ck_bytes
        torch.manual_seed(fixture['probe']['seed']);np.random.seed(fixture['probe']['seed'])
        examples=[];songs=[];samples=[];truths=[];alignment=[]
        for item in fixture['rows']:
            row=item['row'];cid=row['chart_id'];duration=item['duration_us'];c0=item['c0_us']
            canonical=Path('/data/canon')/row['source']/f'{row["native_id"]}.json'
            raw=canonical.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['canon_sha256']:raise ValueError('canonical identity differs')
            if json.loads(raw)['audio_sha256']!=row['audio_sha256']:raise ValueError('audio/label identity mismatch')
            chart=corpus.chart(cid)
            sample=build_request(chart,kind=item['kind'],seed=item['seed'],audio_id=row['audio_sha256'],
                                 audio_duration_us=duration,control_dropout=False,empty_overlay=None)
            assert_no_leak(sample.request,sample.hidden)
            ex=F.build_core(sample,chart.events,c0,duration,stage_a_arm='legacy')
            groups=json.loads(json.dumps(ex.groups,default=lambda x:x.tolist()))
            if groups!=item['groups'] or sample.request['controls']!=item['controls']:raise ValueError('reconstruction differs')
            bad=F.check_targets(ex)
            if bad:raise ValueError(f'illegal target: {bad}')
            song=corpus.song(row['audio_sha256'],in_memory=True)
            truth=[e for g in sample.model_view.target_groups for e in g.events if c0<=e.onset_us<c0+F.CORE_US]
            if not truth or not song.taps or any(not np.isfinite(song.taps[l].scale).all() for l in cfg.taps):
                raise ValueError('empty truth or invalid quantization scale')
            alignment.append(dict(chart_id=cid,mode=row['mode'],c0_us=c0,types=dict(Counter(e.type for e in truth)),
                                  groups=len(ex.groups),target_bounds_ok=True,request_scope='whole song',
                                  core_target_scope_us=[c0,c0+F.CORE_US],
                                  tap_shapes={str(l):list(song.taps[l].shape) for l in cfg.taps},
                                  mel_shape=list(song.mel10.shape),summary_shape=list(song.summary.shape),
                                  song_duration_us=song.duration_us,declared_duration_us=duration,
                                  source_clock_shift_ms=row['offset_decision'],
                                  target_audio_hash=row['audio_sha256']))
            examples.append(ex);songs.append(song);samples.append(sample);truths.append(truth)
        raw_batch=collate(examples,songs,cfg.taps,cfg.top_tap,mel=True)
        bt=to_device(dict(raw_batch),torch.device('cuda'))
        report['data_health']=dict(alignment=alignment,tensors={k:tensor_summary(v) for k,v in bt.items() if torch.is_tensor(v)},
                                  condition_scope_note='Global labels are intentional baseline inputs; local differences are not relabelled in this probe.')
        report['data_health']['passed']=all(x['nan']==x['posinf']==x['neginf']==0 for x in report['data_health']['tensors'].values())
        if not report['data_health']['passed']:raise ValueError('nonfinite real model input')
        record();progress('data_health_passed',cores=len(examples),types=[x['types'] for x in alignment])

        def teacher(batch,step,**identity):
            with torch.no_grad(),capture_decisions() as decisions,torch.autocast('cuda',dtype=torch.bfloat16):
                loss,met=model(dict(batch))
            return dict(step=step,**identity,**metrics_json(loss,met),decisions=decisions)

        def detailed(step):
            for item,ex,song,sample,truth in zip(fixture['rows'],examples,songs,samples,truths,strict=True):
                if time.perf_counter()-started>290:raise TimeoutError('leave reserve for saving diagnostic evidence')
                ident=dict(chart_id=item['row']['chart_id'],mode=item['row']['mode'],c0_us=item['c0_us'])
                one=to_device(collate([ex],[song],cfg.taps,cfg.top_tap),torch.device('cuda'))
                report['per_core_teacher_forced'].append(teacher(one,step,**ident));del one
                observed=[Event.from_json(e) for e in sample.request['observed_events']]
                holds,runs,heads=core_context(sample,ex,F.visible_line_index(observed,sample.evidence))
                diag={};begin=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=generate_core(model,ex,song,holds,runs,heads,device='cuda',greedy=True,
                        seed=item['seed'],diagnostics=diag,deadline=min(started+320,begin+20))
                events=groups_to_events(generated,item['c0_us'],item['row']['mode'],placement=False)
                issues=legality_issues(sample.request,events,placement=False)
                complete=not diag.get('deadline_reached') and not diag.get('group_limit_reached')
                report['generation'].append(dict(step=step,**ident,complete=complete,legal=not issues,legality=issues,
                    diagnostics=diag,predicted_types=dict(Counter(e.type for e in events)),
                    truth_types=dict(Counter(e.type for e in truth)),scores=score_events(events,truth,item['row']['mode']),
                    generated_events=[e.to_json() for e in events],truth_events=[e.to_json() for e in truth],
                    context='correct earlier training history; no target notes within this core supplied',wall_s=time.perf_counter()-begin))
                progress('generated',step=step,chart_id=ident['chart_id'],types=report['generation'][-1]['predicted_types'])
                record()

        initial=teacher(bt,0);report['history'].append(initial)
        if initial['illegal']:raise ValueError('masked targets are illegal')
        progress('initial_teacher_forced',loss=initial['loss']);record();detailed(0)
        opt=torch.optim.AdamW(model.parameters(),lr=fixture['probe']['lr'],weight_decay=0.)
        for step in range(1,fixture['probe']['steps']+1):
            if time.perf_counter()-started>270:raise TimeoutError('probe wall-clock reserve before final evaluation')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,met=model(dict(bt))
            if not bool(torch.isfinite(loss)) or int(met['_illegal']):raise ValueError('nonfinite loss or illegal targets')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            gradient=dict(step=step,finite=bool(torch.isfinite(norm)),norm_before_clip=float(norm))
            if step in (1,100):
                by_module={};missing=[]
                for name,p in model.named_parameters():
                    if p.grad is None:missing.append(name);continue
                    key=name.split('.')[0]
                    by_module[key]=by_module.get(key,0.)+float(p.grad.float().square().sum())
                gradient.update(module_l2_after_clip={k:v**.5 for k,v in by_module.items()},missing_parameters=missing)
            report['gradients'].append(gradient);opt.step();report['optimizer_updates']=step
            if step in fixture['probe']['logging_steps']:
                measured=teacher(bt,step);report['history'].append(measured)
                progress('overfit',step=step,loss=measured['loss'],accuracy={k:v['accuracy'] for k,v in measured['heads'].items()
                         if k in ('time','offset','type','lane','end_block','end_cell','end_off')});record()
        detailed(fixture['probe']['steps'])
        report['supervised_gate']=assess_memorisation(report['history'],report['gradients'],
                                                   fixture['probe']['supervised_gate'],fixture['probe']['steps'])
        final=[r for r in report['generation'] if r['step']==fixture['probe']['steps']]
        report['generation_gate']=assess_generation(final,len(examples),fixture['probe']['generation_gate']['full_note_f1_at_least'])
        report['complete']=True
        report['probe_passed']=report['supervised_gate']['passed'] and report['generation_gate']['passed']
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        report['gpu']=monitor.take()
        record()
        return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';record();raise
    finally:
        monitor.close()


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve,update_stage
    root=training.REPO;ledger=root/'reports/rhythm_v2_budget.json'
    sha=source_digest(root/'beatfcn');script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_sha=hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    reserve(ledger,STAGE,RESOURCES,json.loads((root/'reports/rhythm_v2_rates.json').read_text()),
            json.loads((root/'reports/rhythm_v2_connection_status.json').read_text()),app_id=app.app_id,source_sha256=sha)
    try:
        call=probe.spawn(sha,script_sha,fixture_sha)
        update_stage(ledger,STAGE,status='running',function_call_id=call.object_id,
                     script_sha256=script_sha,fixture_sha256=fixture_sha)
        result=call.get();save(root/f'reports/{STAGE}.json',result)
        update_stage(ledger,STAGE,status='complete' if result['complete'] else 'incomplete',
                     probe_passed=result.get('probe_passed',False),result_path=f'/data/evolution/capability-evolution-20261024/{STAGE}/result.json',
                     function_wall_s=result['function_wall_s'],optimizer_updates=result['optimizer_updates'],ended_at=time.time())
        print(json.dumps({k:result[k] for k in ('complete','probe_passed','supervised_gate','generation_gate','function_wall_s')}))
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}',observed_at=time.time())
        raise

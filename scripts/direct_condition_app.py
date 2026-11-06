"""R05: direct condition residuals, frozen matched data and bounded experiment stages."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.condition_training_app import VARIANT
from beatfcn.cloud.rhythm_diagnostic_app import save, source_digest

REPO=training.REPO
PLAN=REPO/'bench/rhythm-v2/r05-direct-condition-v1.json'
LEGACY=REPO/'inputs/code_baselines/r05-direct-condition/beatfcn/model/p1.py'
ROOT='/data/evolution/capability-evolution-20261024/direct-condition-v1'
OLD='/data/evolution/capability-evolution-20261024/window-conditions-v1'
STAGES={phase:f'r05-direct-condition-{phase}' for phase in ('preflight','train','evaluate')}
app=modal.App('beatfcn-direct-condition-r05')
data=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
image=(training.image.add_local_file(__file__,'/audit/direct_condition_app.py')
       .add_local_file(str(PLAN),'/audit/r05-plan.json')
       .add_local_file(str(LEGACY),'/audit/r05_legacy_p1.py'))


def contract(expected_source,expected_script,expected_plan):
    import beatfcn
    if source_digest(Path(beatfcn.__file__).parent)!=expected_source:raise ValueError('source differs')
    for path,sha in [('/audit/direct_condition_app.py',expected_script),('/audit/r05-plan.json',expected_plan)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('experiment identity differs')
    plan=json.loads(Path('/audit/r05-plan.json').read_text())
    if hashlib.sha256(Path('/audit/r05_legacy_p1.py').read_bytes()).hexdigest()!=plan['legacy_p1_sha256']:raise ValueError('legacy model differs')
    for file,key in [('tape/manifest.json','tape_manifest_sha256'),('panel.json','panel_sha256'),('probe-batch.pt','probe_batch_sha256')]:
        if hashlib.sha256((Path(OLD)/file).read_bytes()).hexdigest()!=plan[key]:raise ValueError('data identity differs')
    return plan


def load_checkpoint(spec):
    import torch
    raw = Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('checkpoint identity differs')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)

def compare_gradients(left, right):
    """Report every parameter before making a strict equivalence decision."""
    import torch
    if left.keys() != right.keys():
        raise ValueError('parameter names changed')
    rows = {}
    for name, a in left.items():
        b = right[name]
        if a is None or b is None:
            rows[name] = dict(passed=a is None and b is None, gradient_present=a is not None,
                              compared_gradient_present=b is not None)
            continue
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        delta = (a-b).float()
        rows[name] = dict(passed=finite and bool(torch.allclose(a,b,atol=1e-6,rtol=1e-6)),
            finite=finite, maximum_absolute_difference=float(delta.abs().max()) if finite else None,
            relative_l2_difference=float(delta.norm()/a.float().norm().clamp_min(1e-30)) if finite else None,
            elements_outside_tolerance=int((delta.abs() > 1e-6+1e-6*b.abs()).sum()) if finite else None)
    return dict(passed=all(r['passed'] for r in rows.values()),
                failed_parameters=[n for n,r in rows.items() if not r['passed']], parameters=rows)

def compare_snapshots(left, right):
    result = compare_gradients(left[1],right[1])
    loss_difference = abs(left[0]['loss']-right[0]['loss']) if left[0]['loss'] is not None and right[0]['loss'] is not None else None
    result.update(left=left[0],right=right[0],loss_difference=loss_difference,
                  rng_equal=left[0]['rng_sha256']==right[0]['rng_sha256'])
    result['passed'] = result['passed'] and loss_difference is not None and loss_difference<=1e-6 and result['rng_equal']
    return result

def control_replays(model, corpus, plan):
    import torch
    from beatfcn.model import features as F, sample
    from beatfcn.model.evaluation import groups_to_events
    from beatfcn.train.condition_pairs import blank_context_excerpt
    from beatfcn.train.control_panel import requested_view
    panel = json.loads((Path(OLD) / 'panel.json').read_text())
    old = json.loads((Path(OLD) / plan['control_benchmark']).read_text())
    wanted = {r['case_id']: r for r in old['records'] if r['variant'] == 'unspecified_conditions'}
    corpus.train_rows([c['row'] for c in panel['cases']]); checks = []
    model.eval().requires_grad_(False)
    for case in panel['cases']:
        row = case['row']; c0 = case['c0_us']
        canonical = Path('/data/canon') / row['source'] / f'{row["native_id"]}.json'
        if hashlib.sha256(canonical.read_bytes()).hexdigest() != case['canon_sha256']:
            raise ValueError('control replay canonical changed')
        chart = corpus.chart(row['chart_id'])
        request = blank_context_excerpt(chart, c0, case['duration_us'], row['audio_sha256'], 20261025)
        ex = requested_view(F.build_core(request, chart.events, c0, case['duration_us'], stage_a_arm='legacy'))
        song = corpus.song(row['audio_sha256'], in_memory=True); diagnostics = {}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            groups = sample.generate_core(model, ex, song, [], [(c0, c0+F.CORE_US)], [], device='cuda',
                greedy=True, seed=20261025, diagnostics=diagnostics, deadline=time.perf_counter()+15)
        events = [e.to_json() for e in groups_to_events(groups, c0, row['mode'], placement=False)]
        equal = events == wanted[case['case_id']]['generated_events']
        if not equal or not diagnostics.get('terminated_by_end'):
            raise ValueError('historical control output changed; paired comparator cannot be reused')
        checks.append(dict(case_id=case['case_id'], equal=equal, complete=True))
        del song
    return checks


def load_warm_state(model,state):
    missing,unexpected=model.load_state_dict(state,strict=False)
    expected={'condition_residual'} if model.c.direct_conditions else set()
    if set(missing)!=expected or unexpected:raise ValueError(f'incompatible checkpoint: {missing}, {unexpected}')


def snapshot(model_class,cfg,state,bt,seed):
    import torch
    model=model_class(cfg).cuda().train();load_warm_state(model,state)
    torch.manual_seed(seed)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model(dict(bt))
    loss.backward()
    summary=dict(loss=float(loss.detach()) if bool(torch.isfinite(loss)) else None,
        rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
    grads={n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in model.named_parameters()}
    new_gradient=grads.pop('condition_residual',None)
    return (summary,grads),new_gradient


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),volumes={'/data':data},
              timeout=360,retries=0,max_containers=1,scaledown_window=2)
def preflight(expected_source,expected_script,expected_plan):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from beatfcn.model import features as F
    from beatfcn.model.batch import unpack_batch,attach_audio,collate
    from beatfcn.model.p1 import P1Config,P1Model
    from beatfcn.model.information_diagnostics import prefix_states,decisions,response
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.probe import tensor_summary,metrics_json,capture_decisions,assess_memorisation
    from beatfcn.train.window_controls import window_control_view
    from beatfcn.train.control_panel import requested_view,REQUEST_BANDS
    from beatfcn.train.condition_pairs import blank_context_excerpt
    torch.set_num_threads(4);started=time.perf_counter();plan=contract(expected_source,expected_script,expected_plan)
    root=Path(ROOT)
    if root.exists():raise ValueError('R05 exists; inspect original handles, do not repeat')
    root.mkdir(parents=True);output=root/'preflight.json'
    report=dict(complete=False,probe_passed=False,source_sha256=expected_source,script_sha256=expected_script,
        plan_sha256=expected_plan,optimizer_updates=0,checkpoint_saved=False,history=[],gradients=[],
        torch_version=torch.__version__,cuda_version=torch.version.cuda)
    def record():
        report['function_wall_s']=time.perf_counter()-started;save(output,report);data.commit()
    record()
    try:
        ck=load_checkpoint(plan['initial_checkpoint']);cfg=replace(P1Config(**ck['config']),**VARIANT)
        corpus=Corpus(**ck['data']);manifest=json.loads((Path(OLD)/'tape/manifest.json').read_text())
        corpus.train_rows(manifest['probe_rows'])
        bt=unpack_batch(torch.load(Path(OLD)/'probe-batch.pt',map_location='cpu',weights_only=False))
        for i,(row,sha) in enumerate(zip(manifest['probe_rows'],manifest['probe_canon_sha256'],strict=True)):
            raw=(Path('/data/canon')/row['source']/f'{row["native_id"]}.json').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=sha or json.loads(raw)['audio_sha256']!=bt['_audio']['keys'][i]:
                raise ValueError('actual batch label/audio identity changed')
        songs={}
        for sha in set(bt['_audio']['keys']):
            taps,summary,_=corpus.load_numpy(sha,mel=False)
            songs[sha]=SimpleNamespace(taps={k:torch.from_numpy(v) for k,v in taps.items()},summary=torch.from_numpy(summary),mel10=None)
        bt=to_device(attach_audio(bt,songs.__getitem__,device='cpu',scales=corpus.scale),torch.device('cuda'))
        del songs,taps,summary
        original=bt;bt=window_control_view(bt,plan['window_control_policy'])
        if not bt.get('_density_cf') or not torch.equal(bt['controls'],bt['full_window_controls']):
            raise ValueError('probe must use the registered all-condition window view')
        if any(original[k] is not bt[k] for k in original if k!='controls'):raise ValueError('data view changed more than conditions')
        stats={k:tensor_summary(v) for k,v in bt.items() if torch.is_tensor(v)}
        report['data_health']=dict(tensors=stats,label_audio_identity_verified=True,all_condition_view_verified=True,
            passed=all(not(v['nan'] or v['posinf'] or v['neginf']) for v in stats.values()))
        if not report['data_health']['passed']:raise ValueError('nonfinite actual input')
        name='beatfcn.model._r05_legacy';spec=importlib.util.spec_from_file_location(name,'/audit/r05_legacy_p1.py')
        legacy=importlib.util.module_from_spec(spec);sys.modules[name]=legacy;spec.loader.exec_module(legacy)
        settings=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with sdpa_kernel(SDPBackend.MATH):
                old,_=snapshot(legacy.P1Model,cfg,ck['model'],bt,plan['seed'])
                off,_=snapshot(P1Model,cfg,ck['model'],bt,plan['seed'])
                on,new_gradient=snapshot(P1Model,replace(cfg,direct_conditions=True),ck['model'],bt,plan['seed'])
            report['equivalence']=dict(default_off=compare_snapshots(old,off),zero_enabled_common=compare_snapshots(old,on))
            report['new_parameter_gradient']=dict(finite=bool(torch.isfinite(new_gradient).all()),
                outer_norm=float(new_gradient[...,:cfg.d].norm()),inner_norm=float(new_gradient[...,cfg.d:].norm()))
            del old,off,on,new_gradient
        finally:
            torch.use_deterministic_algorithms(settings[0],warn_only=settings[1])
            torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=settings[2:4]
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=settings[4:]
        record()
        if not all(x['passed'] for x in report['equivalence'].values()):raise ValueError('real common loss/gradient equivalence failed')
        g=report['new_parameter_gradient']
        if not g['finite'] or min(g['outer_norm'],g['inner_norm'])<=0:raise ValueError('condition gradient path missing')
        control=load_checkpoint(plan['control_checkpoint']);report['control_replays']={}
        for enabled in (False,True):
            model=P1Model(replace(P1Config(**control['config']),direct_conditions=enabled)).cuda()
            load_warm_state(model,control['model'])
            report['control_replays'][str(enabled)]=control_replays(model,corpus,plan);del model
        del control;record()
        model=P1Model(replace(cfg,direct_conditions=True)).cuda().eval();load_warm_state(model,ck['model']);del ck
        torch.manual_seed(plan['seed'])
        def measure(step):
            with torch.no_grad(),capture_decisions() as observed,torch.autocast('cuda',dtype=torch.bfloat16):
                loss,metrics=model(dict(bt))
            return dict(step=step,**metrics_json(loss,metrics),decisions=observed)
        report['history'].append(measure(0));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        for step in range(1,101):
            if time.perf_counter()-started>310:raise TimeoutError('preflight time reserve')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model(dict(bt))
            if not bool(torch.isfinite(loss)) or int(metrics['_illegal']):raise ValueError('nonfinite loss or illegal target')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            report['gradients'].append(dict(step=step,finite=bool(torch.isfinite(norm)),norm_before_clip=float(norm)))
            opt.step();report['optimizer_updates']=step
            if step in (1,10,25,50,75,100):
                report['history'].append(measure(step));record()
                print(json.dumps(dict(event='r05_probe',step=step,loss=report['history'][-1]['loss'])),flush=True)
        report['supervision_gate']=assess_memorisation(report['history'],report['gradients'],plan['probe_gate'],100)
        report['learned_residual_norm']=float(model.condition_residual.detach().float().norm())
        report['cached_parity']=[];panel=json.loads((Path(OLD)/'panel.json').read_text())
        # A nonzero learned adapter must be applied identically by the cached decoder.
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for mode in ('mania_4k','phira'):
                case=next(c for c in panel['cases'] if c['row']['mode']==mode);row=case['row'];c0=case['c0_us']
                chart=corpus.chart(row['chart_id']);request=blank_context_excerpt(chart,c0,case['duration_us'],row['audio_sha256'],20261025)
                ex=requested_view(F.build_core(request,chart.events,c0,case['duration_us'],stage_a_arm='legacy'),
                    'hold_head_ratio',REQUEST_BANDS['hold_head_ratio'][2])
                song=corpus.song(row['audio_sha256'],in_memory=True)
                full=to_device(collate([ex],[song],model.c.taps,model.c.top_tap),torch.device('cuda'))
                h,pa,_=model.encode(full)
                for index,prefix,hp,pp in prefix_states(model,ex,song,[0,4],device='cuda'):
                    diffs=response(decisions(model,full,h[:,index:index+1],pa,ex.groups[index],index),
                        decisions(model,prefix,hp,pp,ex.groups[index],index))
                    delta=float((hp-h[:,index:index+1]).abs().max())
                    passed=delta<=1e-4 and all(x['maximum_absolute_difference']<=1e-4 for x in diffs.values())
                    report['cached_parity'].append(dict(case_id=case['case_id'],prefix_groups=index,hidden_max_absolute=delta,
                        probability_response=diffs,passed=passed))
                del song,full,h,pa
        report.update(complete=True,probe_passed=report['supervision_gate']['passed'] and report['learned_residual_norm']>0
            and all(x['passed'] for x in report['cached_parity']),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        record();return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';record();raise


@app.function(image=image,gpu='L40S',cpu=(12,12),memory=(49152,49152),volumes={'/data':data},
              timeout=1050,retries=0,max_containers=1,scaledown_window=2)
def train(expected_source,expected_script,expected_plan):
    import torch
    torch.set_num_threads(4);plan=contract(expected_source,expected_script,expected_plan)
    before=json.loads((Path(ROOT)/'preflight.json').read_text())
    if not before['complete'] or not before['probe_passed'] or before['source_sha256']!=expected_source or before['plan_sha256']!=expected_plan:
        raise ValueError('matching actual preflight has not passed')
    run=plan['run_id']
    if (Path('/data/runs')/run).exists():raise ValueError('R05 run exists; inspect original handles')
    manifest=json.loads((Path(OLD)/'tape/manifest.json').read_text())
    result=training.train.local(run_id=run,steps=8000,stop_at=4500,round_updates=1000,
        initialize_from=plan['initial_checkpoint']['path'],source_sha256=expected_source,
        micro_batch=8,accumulation=4,workers=8,val_cores=40,val_songs=8,lr=.0002,warmup=500,validate_every=250,
        taps_key='v4_i8_L9-16-24',taps=(9,16,24),pca='i8_9-16-24',tap_dim=1024,
        pool_songs=40,pool_per_worker=1,prefetch=16,loaders=4,graph=False,
        min_util=70.,check_after_s=480.,max_data_wait=.8,check_data_after_s=480.,max_wall_s=1000.,
        manifest_name='charts.offset-v2-3d99ff6bd5e7.jsonl',variant={**VARIANT,'direct_conditions':True},
        training_policy=plan['training_policy'],prep=manifest['prep_key'],review_start_step=3500,
        data_seed_step=3500,tape_start_step=3500,metadata_tape=plan['tape'],
        window_control_policy=plan['window_control_policy'],trace_after_s=(),trace_updates=2,trace_detailed=False)
    path=Path('/data/runs')/run/'last.pt'
    result.update(source_sha256=expected_source,plan_sha256=expected_plan,
        complete=result['round_effective_updates']==1000 and not result['aborted'],
        checkpoint=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.exists() else None)
    save(Path(ROOT)/'training.json',result);data.commit();return result


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),volumes={'/data':data},
              timeout=360,retries=0,max_containers=1,scaledown_window=2)
def evaluate(expected_source,expected_script,expected_plan):
    import torch
    from beatfcn.cloud.window_condition_app import benchmark
    from beatfcn.model.p1 import P1Config,P1Model
    from beatfcn.train.data import Corpus
    started=time.perf_counter();torch.set_num_threads(4)
    plan=contract(expected_source,expected_script,expected_plan)
    trained=json.loads((Path(ROOT)/'training.json').read_text())
    if not trained['complete'] or trained['source_sha256']!=expected_source or trained['plan_sha256']!=expected_plan:
        raise ValueError('same-source 1000-update training incomplete')
    output=Path(ROOT)/'benchmark.json'
    if output.exists():raise ValueError('R05 benchmark exists; inspect original handle')
    ck=load_checkpoint(trained['checkpoint']);model=P1Model(P1Config(**ck['config'])).cuda().eval()
    model.load_state_dict(ck['model']);corpus=Corpus(**ck['data']);del ck
    panel=json.loads((Path(OLD)/'panel.json').read_text())
    result=benchmark(model,corpus,panel,output,started+320)
    result.update(source_sha256=expected_source,plan_sha256=expected_plan,function_wall_s=time.perf_counter()-started,
        compared_control=plan['control_checkpoint'],product_promotion=False)
    save(output,result);data.commit();return result


@app.local_entrypoint()
def main(phase:str='preflight'):
    raise RuntimeError('R05 rejected by the frozen quality gates; entrypoint disabled. Reproduction source is archived in inputs/code_baselines/r05-direct-condition-rejected/. Do not restart this failed experiment.')

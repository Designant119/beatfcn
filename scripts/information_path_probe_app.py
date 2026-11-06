"""D08: frozen-weight information-path diagnosis, never a training or quality run."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import modal

from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import save, source_digest

STAGE = 'information-path-diagnostic-v1'
PLAN = training.REPO / 'bench/rhythm-v2/information-path-diagnostic-v1.json'
REMOTE = '/data/evolution/capability-evolution-20261024'
RESOURCES = dict(gpu='L40S', cpu=4, memory_gib=16, timeout_s=360, startup_allowance_s=120,
                 retries=0, max_containers=1)
app = modal.App('beatfcn-information-path-diagnostic')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (training.image.add_local_file(__file__, '/probe/information_path_probe_app.py')
         .add_local_file(str(PLAN), '/probe/plan.json'))


@app.function(image=image, gpu='L40S', cpu=(4,4), memory=(16384,16384), volumes={'/data':data},
              timeout=360, retries=0, max_containers=1, scaledown_window=2)
def probe(expected_source, expected_script, expected_plan):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import beatfcn
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from beatfcn.events.controls import MODE_FIELDS
    from beatfcn.model import features as F, sample
    from beatfcn.model.batch import collate
    from beatfcn.model.evaluation import groups_to_events
    from beatfcn.model.information_diagnostics import decisions, prefix_states, response
    from beatfcn.model.input_diagnostics import mismatch_audio
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.condition_pairs import blank_context_excerpt
    from beatfcn.train.control_panel import REQUEST_BANDS, requested_view
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.probe import tensor_summary

    started=time.perf_counter(); torch.set_num_threads(4)
    for path,expected in ((Path('/probe/information_path_probe_app.py'),expected_script),
                          (Path('/probe/plan.json'),expected_plan)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('registered file identity changed')
    if source_digest(Path(beatfcn.__file__).parent)!=expected_source: raise ValueError('source identity changed')
    plan=json.loads(Path('/probe/plan.json').read_text())
    root=Path(REMOTE)/'window-conditions-v1'; output=Path(REMOTE)/STAGE
    if output.exists(): raise ValueError('diagnostic output exists; inspect original call instead of restarting')
    panel_raw=(root/'panel.json').read_bytes()
    if hashlib.sha256(panel_raw).hexdigest()!=plan['panel_sha256']: raise ValueError('panel identity changed')
    panel=json.loads(panel_raw);spec=plan['model']
    payload=Path(spec['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=spec['sha256']: raise ValueError('checkpoint identity changed')
    ck=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    model=P1Model(P1Config(**ck['config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model']);corpus=Corpus(**ck['data']);del ck,payload
    corpus.train_rows([c['row'] for c in panel['cases']])
    saved=json.loads((root/spec['saved_benchmark']).read_text())
    wanted={r['case_id']:r for r in saved['records'] if r['variant']=='unspecified_conditions'}
    output.mkdir(parents=True)
    report=dict(stage=STAGE,complete=False,optimizer_updates=0,checkpoint_saved=False,diagnostic_only=True,
        source_sha256=expected_source,script_sha256=expected_script,plan_sha256=expected_plan,protocol=plan,
        records=[],parity=[],data_health=[],replays=[],torch_version=torch.__version__)

    def record():
        report['function_wall_s']=time.perf_counter()-started
        save(output/'result.json',report);data.commit()

    def deadline():
        if time.perf_counter()-started>310: raise TimeoutError('bounded diagnostic; preserve partial results')

    def batch(ex,song):
        return to_device(collate([ex],[song],model.c.taps,model.c.top_tap),torch.device('cuda'))

    def measure(bt,ex,case_id,variant,**extra):
        deadline()
        with torch.no_grad(): h,pa,_=model.encode(bt)
        found={}
        for row in plan['prefix_rows']:
            item=decisions(model,bt,h[:,row:row+1],pa,ex.groups[row],row)
            report['records'].append(dict(case_id=case_id,prefix_groups=row,variant=variant,
                mode=F.MODES[ex.mode],**extra,**item))
            found[row]=item
        return h,pa,found

    record()
    try:
        # Match the saved production decoder before switching to diagnostic numerics.
        for case in panel['cases']:
            deadline();row=case['row'];c0=case['c0_us']
            canon=Path('/data/canon')/row['source']/f'{row["native_id"]}.json'
            if hashlib.sha256(canon.read_bytes()).hexdigest()!=case['canon_sha256']: raise ValueError('canon changed')
            chart=corpus.chart(row['chart_id'])
            req=blank_context_excerpt(chart,c0,case['duration_us'],row['audio_sha256'],20261025)
            ex=requested_view(F.build_core(req,chart.events,c0,case['duration_us'],stage_a_arm='legacy'))
            truth=[e.to_json() for g in req.model_view.target_groups for e in g.events]
            if F.check_targets(ex) or truth!=wanted[case['case_id']]['truth_events']: raise ValueError('target identity changed')
            if min(len(ex.groups),len(ex.right)-1)<=max(plan['prefix_rows']): raise ValueError('prefix lacks a next group')
            song=corpus.song(row['audio_sha256'],in_memory=True)
            diagnostic={}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                groups=sample.generate_core(model,ex,song,[],[(c0,c0+F.CORE_US)],[],device='cuda',greedy=True,
                    seed=20261025,diagnostics=diagnostic,deadline=min(started+310,time.perf_counter()+15))
            serial=[e.to_json() for e in groups_to_events(groups,c0,row['mode'],placement=False)]
            if not diagnostic.get('terminated_by_end') or serial!=wanted[case['case_id']]['generated_events']:
                raise ValueError('production free replay mismatch')
            report['replays'].append(dict(case_id=case['case_id'],equal=True,generated_events=serial))
            del song
        record()
        # No mixed precision: parity differences must not be caused by bf16 GEMM shapes.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with sdpa_kernel(SDPBackend.MATH):
            for case in panel['cases']:
                deadline();row=case['row'];c0=case['c0_us'];chart=corpus.chart(row['chart_id'])
                req=blank_context_excerpt(chart,c0,case['duration_us'],row['audio_sha256'],20261025)
                ex=requested_view(F.build_core(req,chart.events,c0,case['duration_us'],stage_a_arm='legacy'))
                song=corpus.song(row['audio_sha256'],in_memory=True);bt=batch(ex,song)
                stats={k:tensor_summary(v) for k,v in bt.items() if torch.is_tensor(v)}
                if any(s['nan'] or s['posinf'] or s['neginf'] for s in stats.values()): raise ValueError('nonfinite actual input')
                if not bool((bt['controls']==-1).all()): raise ValueError('factual conditions are not missing')
                report['data_health'].append(dict(case_id=case['case_id'],tensors=stats,controls_missing=True,
                    label_audio_identity_verified=True,truth_events=wanted[case['case_id']]['truth_events']))
                h,pa,base=measure(bt,ex,case['case_id'],'base')
                for index,prefix,hp,pp in prefix_states(model,ex,song,plan['prefix_rows'],device='cuda'):
                    delta=float((hp-h[:,index:index+1]).abs().max())
                    prob=decisions(model,prefix,hp,pp,ex.groups[index],index)
                    diffs=response(base[index],prob)
                    passed=delta<=plan['parity_max_absolute'] and all(
                        d['maximum_absolute_difference']<=plan['parity_max_absolute'] for d in diffs.values())
                    report['parity'].append(dict(case_id=case['case_id'],prefix_groups=index,
                        hidden_max_absolute=delta,probability_response=diffs,passed=passed))
                    if not passed: raise ValueError('teacher/generation mismatch; stop sensitivity audit and diagnose')
                del h,pa
                for field in MODE_FIELDS[row['mode']]:
                    for level in (0,2):
                        view=requested_view(ex,field,REQUEST_BANDS[field][level])
                        varied={**bt,'controls':torch.from_numpy(view.controls)[None].to('cuda')}
                        measure(varied,ex,case['case_id'],'control',field=field,level=level,
                                requested=REQUEST_BANDS[field][level])
                peers=[c for c in panel['cases'] if c['row']['mode']==row['mode']]
                donor_case=peers[(next(i for i,c in enumerate(peers) if c['case_id']==case['case_id'])+1)%len(peers)]
                donor_sha=donor_case['row']['audio_sha256']
                if donor_sha==row['audio_sha256']: raise ValueError('donor audio is not different')
                donor=corpus.song(donor_sha,in_memory=True);swapped=mismatch_audio(song,donor)
                altered=batch(ex,swapped)
                for variant,keys in [('local_audio_mismatch',('audio',)),('primary_audio_mismatch',('audio','song_cells'))]:
                    if any(k not in bt or altered[k].shape!=bt[k].shape for k in keys): raise ValueError('audio intervention shape changed')
                    varied={**bt,**{k:altered[k] for k in keys}}
                    measure(varied,ex,case['case_id'],variant,donor_audio_sha256=donor_sha,
                        donor_case_id=donor_case['case_id'],changed_tensor_keys=list(keys),
                        donor_duration_us=donor.duration_us,target_duration_us=song.duration_us,
                        artificial_feature_resampling=True)
                print(json.dumps(dict(event='information_path',case_id=case['case_id'],
                    parity_passed=True,records=len(report['records']))),flush=True)
                del bt,altered,varied,song,donor,swapped
                record()
        report['complete']=(len(report['records'])==plan['expected_records'] and len(report['parity'])==16
            and len(report['data_health'])==8 and len(report['replays'])==8)
        report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        record();return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';record();raise


@app.local_entrypoint()
def main():
    from beatfcn.train.campaign_budget import reserve, update_stage
    root=training.REPO;ledger=root/'reports/rhythm_v2_budget.json'
    identities=dict(source_sha256=source_digest(root/'beatfcn'),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    registration=json.loads((root/'reports/d08_registration.json').read_text())
    if any(registration[k]!=v for k,v in identities.items()): raise ValueError('D08 differs from registration')
    reserve(ledger,STAGE,RESOURCES,json.loads((root/'reports/rhythm_v2_rates.json').read_text()),
        json.loads((root/'reports/rhythm_v2_connection_status.json').read_text()),
        app_id=app.app_id,source_sha256=identities['source_sha256'])
    try:
        call=probe.spawn(*identities.values())
        update_stage(ledger,STAGE,status='running',function_call_id=call.object_id,
            script_sha256=identities['script_sha256'],plan_sha256=identities['plan_sha256'])
        result=call.get();save(root/f'reports/{STAGE}.json',result)
        update_stage(ledger,STAGE,status='complete' if result['complete'] else 'incomplete',
            result_path=f'{REMOTE}/{STAGE}/result.json',optimizer_updates=0,
            function_wall_s=result['function_wall_s'],ended_at=time.time())
        print(json.dumps({k:result[k] for k in ('complete','optimizer_updates','function_wall_s')}))
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}',observed_at=time.time())
        raise

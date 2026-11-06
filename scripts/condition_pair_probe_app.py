"""One bounded condition-identifiability diagnostic; never saves trained weights."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import modal
from beatfcn.cloud import train_app as training
from beatfcn.cloud.rhythm_diagnostic_app import source_digest,save

STAGE='condition-pair-health-v1'
FIXTURE=training.REPO/'bench/rhythm-v2/condition-pair-probe-v1.json'
app=modal.App('beatfcn-condition-pair-health')
data=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
image=(training.image.add_local_file(__file__,'/probe/condition_pair_probe_app.py')
       .add_local_file(str(FIXTURE),'/probe/fixture.json'))
RESOURCES=dict(gpu='L40S',cpu=4,memory_gib=16,timeout_s=360,startup_allowance_s=120,retries=0,max_containers=1)


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(16384,16384),volumes={'/data':data},
              timeout=360,retries=0,max_containers=1,scaledown_window=2)
def probe(expected_source,expected_script,expected_fixture):
    from dataclasses import replace
    import io
    import numpy as np
    import torch
    import beatfcn
    from beatfcn.cloud.gpu_util import GpuUtil
    from beatfcn.events import controls as C
    from beatfcn.model import features as F
    from beatfcn.model.batch import collate,SongInfo
    from beatfcn.model.p1 import P1Config,P1Model
    from beatfcn.model.sample import generate_core
    from beatfcn.model.evaluation import groups_to_events,legality_issues,score_events
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.condition_pairs import blank_context_excerpt,assert_same_generation_inputs,without_conditions
    from beatfcn.train.probe import tensor_summary,metrics_json,capture_decisions,assess_memorisation,assess_generation
    started=time.perf_counter();torch.set_num_threads(4)
    if source_digest(Path(beatfcn.__file__).parent)!=expected_source:raise ValueError('source differs')
    if hashlib.sha256(Path('/probe/condition_pair_probe_app.py').read_bytes()).hexdigest()!=expected_script:raise ValueError('script differs')
    payload=Path('/probe/fixture.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_fixture:raise ValueError('fixture differs')
    fixture=json.loads(payload);protocol=fixture['probe']
    output=Path('/data/evolution/capability-evolution-20261024')/STAGE
    if output.exists():raise ValueError('output exists; inspect prior handle instead of retrying')
    output.mkdir(parents=True)
    report=dict(stage=STAGE,complete=False,probe_passed=False,source_sha256=expected_source,script_sha256=expected_script,
        fixture_sha256=expected_fixture,protocol=fixture,optimizer_updates=0,checkpoint_saved=False,
        history=[],gradients=[],generation=[],per_core_teacher_forced=[],input_isolation=[],data_health={})
    def record():
        report['function_wall_s']=time.perf_counter()-started
        save(output/'result.json',report);data.commit()
    def progress(event,**values):
        print(json.dumps(dict(event=event,elapsed_s=round(time.perf_counter()-started,3),**values)),flush=True)
    record();monitor=GpuUtil()
    try:
        payload=Path(fixture['checkpoint']['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=fixture['checkpoint']['sha256']:raise ValueError('checkpoint differs')
        ck=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        if ck['step']!=3500:raise ValueError('checkpoint step differs')
        cfg=P1Config(**ck['config'])
        if cfg.offset_sigma_ms or cfg.lane_history_groups or cfg.selection:raise ValueError('baseline intervention differs')
        model=P1Model(cfg).cuda().eval();model.load_state_dict(ck['model'])
        corpus=Corpus(**ck['data'])
        corpus.train_rows([item['row'] for pair in fixture['pairs'] for item in pair['items']])
        report['model_config']=ck['config'];report['corpus_config']=ck['data'];del ck,payload
        torch.manual_seed(fixture['seed']);np.random.seed(fixture['seed'])
        cache={};pairs=[];examples=[];songs=[]
        for pair in fixture['pairs']:
            entries=[]
            for item in pair['items']:
                row=item['row'];sha=row['audio_sha256'];cid=row['chart_id']
                raw=(Path('/data/canon')/row['source']/f'{row["native_id"]}.json').read_bytes()
                if hashlib.sha256(raw).hexdigest()!=item['canon_sha256'] or json.loads(raw)['audio_sha256']!=sha:
                    raise ValueError('canonical/audio identity differs')
                chart=corpus.chart(cid)
                sample=blank_context_excerpt(chart,item['c0_us'],item['duration_us'],sha,fixture['seed'])
                ex=F.build_core(sample,chart.events,item['c0_us'],item['duration_us'],stage_a_arm='legacy')
                if F.check_targets(ex):raise ValueError('illegal targets')
                if json.loads(json.dumps(ex.groups,default=lambda v:v.tolist()))!=item['groups'] or sample.request['controls']!=item['controls']:
                    raise ValueError('fixture reconstruction differs')
                if sha not in cache:cache[sha]=corpus.song(sha,in_memory=True)
                song=cache[sha]
                if any(not np.isfinite(song.taps[l].scale).all() for l in cfg.taps):raise ValueError('nonfinite scale')
                truth=[e for g in sample.model_view.target_groups for e in g.events]
                entries.append(dict(item=item,sample=sample,ex=ex,song=song,truth=truth))
                examples.append(ex);songs.append(song)
            isolation=assert_same_generation_inputs(entries[0]['ex'],entries[1]['ex'],SongInfo.of(entries[0]['song'],0,cfg.taps[0]),
                                                    cfg.taps,cfg.top_tap,cfg.tap_dim,cfg.summary_dim)
            report['input_isolation'].append(dict(pair_id=pair['pair_id'],**isolation))
            pairs.append((pair['pair_id'],entries))
        bt=to_device(collate(examples,songs,cfg.taps,cfg.top_tap,mel=True),torch.device('cuda'))
        stats={k:tensor_summary(v) for k,v in bt.items() if torch.is_tensor(v)}
        report['data_health']=dict(tensors=stats,passed=all(v['nan']==v['posinf']==v['neginf']==0 for v in stats.values()))
        if not report['data_health']['passed']:raise ValueError('nonfinite input')
        progress('data_health_passed',cores=len(examples),isolated_pairs=len(pairs));record()

        def teacher(batch,step,**identity):
            with torch.no_grad(),capture_decisions() as decisions,torch.autocast('cuda',dtype=torch.bfloat16):
                loss,met=model(dict(batch))
            return dict(step=step,**identity,**metrics_json(loss,met),decisions=decisions)

        def detailed(step,variants):
            for pair_id,entries in pairs:
                for index,entry in enumerate(entries):
                    item=entry['item'];ex=entry['ex'];song=entry['song'];sample=entry['sample'];truth=entry['truth']
                    ident=dict(pair_id=pair_id,chart_id=item['row']['chart_id'],mode=item['row']['mode'],c0_us=item['c0_us'])
                    one=to_device(collate([ex],[song],cfg.taps,cfg.top_tap),torch.device('cuda'))
                    report['per_core_teacher_forced'].append(teacher(one,step,**ident));del one
                    for variant in variants:
                        if time.perf_counter()-started>290:raise TimeoutError('save reserve reached')
                        current=ex if variant=='conditioned' else (without_conditions(ex) if variant=='removed' else
                                                                  replace(ex,controls=entries[1-index]['ex'].controls.copy()))
                        diag={};begin=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            generated=generate_core(model,current,song,[],[(item['c0_us'],item['c0_us']+F.CORE_US)],[],
                                device='cuda',greedy=True,seed=fixture['seed'],diagnostics=diag,deadline=min(started+320,begin+15))
                        events=groups_to_events(generated,item['c0_us'],item['row']['mode'],placement=False)
                        issues=legality_issues(sample.request,events,placement=False)
                        actual=C.raw_controls(item['row']['mode'],events,[(item['c0_us'],item['c0_us']+F.CORE_US)])
                        target_index=index if variant!='swapped' else 1-index
                        requested=entries[target_index]['sample'].request['controls'] if variant!='removed' else {}
                        bands={f:dict(requested=requested.get(f),achieved=actual.get(f),
                            in_band=C.band_of(f,actual.get(f))==requested[f] if f in requested else None)
                            for f in C.MODE_FIELDS[item['row']['mode']]}
                        row=dict(step=step,variant=variant,**ident,complete=bool(diag.get('terminated_by_end')),
                            legal=not issues,legality=issues,diagnostics=diag,
                            scores=score_events(events,truth,item['row']['mode']),
                            alternative_scores=score_events(events,entries[1-index]['truth'],item['row']['mode']),
                            requested_controls=requested,achieved_controls=bands,predicted_types=dict(Counter(e.type for e in events)),
                            truth_types=dict(Counter(e.type for e in truth)),generated_events=[e.to_json() for e in events],
                            truth_events=[e.to_json() for e in truth],wall_s=time.perf_counter()-begin)
                        report['generation'].append(row)
                        progress('generated',step=step,variant=variant,chart_id=ident['chart_id'],types=row['predicted_types']);record()

        report['history'].append(teacher(bt,0));record();detailed(0,['conditioned'])
        opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        for step in range(1,protocol['steps']+1):
            if time.perf_counter()-started>230:raise TimeoutError('leave reserve for condition audit')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,met=model(dict(bt))
            if not bool(torch.isfinite(loss)) or int(met['_illegal']):raise ValueError('nonfinite loss or illegal targets')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['clip'],error_if_nonfinite=True)
            grad=dict(step=step,finite=bool(torch.isfinite(norm)),norm_before_clip=float(norm))
            if step in (1,100):
                grad['condition_embedding_gradient_l2']=float(model.ctrl_e.weight.grad.float().norm())
            report['gradients'].append(grad);opt.step();report['optimizer_updates']=step
            if step in protocol['logging_steps']:
                measured=teacher(bt,step);report['history'].append(measured)
                progress('overfit',step=step,loss=measured['loss'],type_accuracy=measured['heads']['type']['accuracy']);record()
        detailed(protocol['steps'],['conditioned','removed','swapped'])
        report['supervised_gate']=assess_memorisation(report['history'],report['gradients'],protocol['supervised_gate'],protocol['steps'])
        final=[r for r in report['generation'] if r['step']==protocol['steps']]
        report['generation_gate']=assess_generation([r for r in final if r['variant']=='conditioned'],len(examples),
                                                    protocol['generation_gate']['full_note_f1_at_least'])
        def full_f1(score):
            m=score['edit_utility']['full_note'];n=m['pred']+m['true']
            return 2*m['matched']/n if n else 1.
        pair_gates=[]
        for pair_id,entries in pairs:
            def get(index,variant):
                return next(r for r in final if r['pair_id']==pair_id and r['chart_id']==entries[index]['item']['row']['chart_id'] and r['variant']==variant)
            margins=[full_f1(get(i,'conditioned')['scores'])-full_f1(get(i,'conditioned')['alternative_scores']) for i in (0,1)]
            removed_equal=get(0,'removed')['generated_events']==get(1,'removed')['generated_events']
            swaps_equal=all(get(i,'swapped')['generated_events']==get(1-i,'conditioned')['generated_events'] for i in (0,1))
            in_band=all(v['in_band'] is True for i in (0,1) for v in get(i,'conditioned')['achieved_controls'].values())
            complete_legal=all(r['complete'] and r['legal'] for r in final if r['pair_id']==pair_id)
            passed=removed_equal and swaps_equal and in_band and complete_legal and min(margins)>=protocol['generation_gate']['matching_target_margin_at_least']
            pair_gates.append(dict(pair_id=pair_id,matching_target_f1_margins=margins,removed_controls_identical=removed_equal,
                                   swapped_matches_other=swaps_equal,all_requested_bands_met=in_band,complete_legal=complete_legal,passed=passed))
        report['condition_gate']=dict(passed=all(r['passed'] for r in pair_gates),pairs=pair_gates,
            scope='Joint vector on memorised excerpts; not independent per-field control or held-out quality.')
        report['probe_passed']=all(report[k]['passed'] for k in ('supervised_gate','generation_gate','condition_gate'))
        report['complete']=True;report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30;report['gpu']=monitor.take();record()
        return report
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';record();raise
    finally:monitor.close()


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
            probe_passed=result.get('probe_passed',False),function_wall_s=result['function_wall_s'],
            result_path=f'/data/evolution/capability-evolution-20261024/{STAGE}/result.json',
            optimizer_updates=result['optimizer_updates'],ended_at=time.time())
        print(json.dumps({k:result[k] for k in ('complete','probe_passed','supervised_gate','generation_gate','condition_gate','function_wall_s')}))
    except BaseException as exc:
        update_stage(ledger,STAGE,status='observation_error',error=f'{type(exc).__name__}: {exc}',observed_at=time.time());raise

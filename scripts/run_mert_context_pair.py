"""Fresh paired SSD training: full versus32s extraction, batch4/50-25-25.

Same architecture, initialization, all4000 supervised requests, optimizer,
mel, layer selection, clocks and decision rule. No dynamic prefix replay.
The comparison isolates extraction context within this complete training
recipe; it does not separately identify batch-size or sample-mixture effects.
"""
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.direct_song_canvas import DirectSongCanvas
from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1
from beatfcn.train.mert_canvas_features import MertCanvasFeatures
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_group_diffusion_health import optimizer_for,rate
from scripts.run_local_portfolio_scale import row_factory,weight_digest,sha,write
from scripts import run_local_portfolio_data_scale as evaluator
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-mert-context-pair-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_mert_context_pair_v1.json';ART=ROOT/f'artifacts/{IDENT}'
MODEL_CLASS=DirectSongCanvas
MODEL_CONFIG_OVERRIDE=None
SEED_OVERRIDE=None
SINGLE_FRONTEND=False
EXTRA_SOURCES=()
SINGLE_HYPOTHESIS=None


def fresh(plan,device='mps'):
    torch.manual_seed(plan['seed'])
    return MODEL_CLASS(FullEventConfig(**plan['model']),density_reference=plan['density_reference']).to(device)


def checkpoint(model,opt,arm,stage,step):
    path=ART/arm/f'{stage}-{step:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),stage=stage,updates=step,
        config=asdict(model.c),plan_sha256=sha(PLAN),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),path)
    return path


@torch.no_grad()
def diagnostic(model,batches):
    model.eval();result=[]
    for e,inp,tgt in batches:
        loss,parts=canvas_loss(model(inp),tgt,inp)
        result.append(dict(index=e['index'],loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    return result


def structure_error(records):
    """Eight-second chart-structure proxy; not full-song or emotion evidence."""
    by_component={}
    for r in records:
        start=r['request']['start_us'];end=r['request']['end_us'];bins=(end-start+1_999_999)//2_000_000
        values=[]
        for events in (r['target'],r['prediction']['events']):
            x=np.zeros((bins,4),np.int64)
            for e in events:
                b=(e['onset_us']-start)//2_000_000
                if not 0<=b<bins:raise ValueError('Structure metric event outside request')
                x[b,('tap','hold','flick','drag').index(e['type'])]+=1
            values.append(x)
        target,pred=values
        error=dict(count_l1=float(np.abs(target.sum(-1)-pred.sum(-1)).sum()/max(int(target.sum()),1)),
            typed_count_l1=float(np.abs(target-pred).sum()/max(int(target.sum()),1)))
        by_component.setdefault(r['component'],[]).append(error)
    component={c:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for c,rows in by_component.items()}
    return dict(component=component,**{k:float(np.mean([r[k] for r in component.values()])) for k in ('count_l1','typed_count_l1')})


@torch.no_grad()
def global_controls(model,plan,row,panel,arm,deadline):
    model.eval();result={};dev=plan['indices']['dev']
    for variant in ('song_removed','song_swapped'):
        records=[]
        for j,index in enumerate(dev):
            if time.monotonic()>deadline:raise TimeoutError('Global control wall')
            e=row(index,'draft');inp,tgt=relative_batch([e],'mps');donor=dev[(j+2)%len(dev)]
            if variant=='song_removed':changed=torch.zeros_like(inp['song'])
            else:
                other,_=relative_batch([row(donor,'draft')],'mps');changed=other['song']
                if panel.records[donor]['component']==panel.records[index]['component']:raise ValueError('Global donor must differ')
            altered={**inp,'song':changed};out=model(altered);mask=inp['canvas_valid'][...,None].expand_as(tgt['count']).contiguous()
            count_nll=float(torch.nn.functional.cross_entropy(out['count'][mask],tgt['count'][mask]))
            prediction=generate_relative(model,altered,e['request'],deadline=deadline);events=[RhythmEvent(**v) for v in prediction['events']]
            start=e['request'].start_us;width=e['request'].end_us-start
            shifts=[match_f1([start+(v.onset_us-start+d)%width for v in events],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
            records.append(dict(index=index,component=panel.records[index]['component'],view='draft',request=asdict(e['request']),
                prediction=prediction,target=[asdict(v) for v in e['targets']],scores=score(events,e['targets']),shifted_onset10=shifts,count_nll=count_nll,
                donor_index=donor if variant=='song_swapped' else None))
            write(ART/arm/f'{variant}.json',records)
        result[variant]=dict(metrics=metrics(records),structure=structure_error(records),count_nll=float(np.mean([r['count_nll'] for r in records])))
    return result


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve paired context experiment')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('No silent CPU fallback')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    cachepath=ROOT/'reports/local_mert_context_cache_v1.json';cache=json.loads(cachepath.read_text())
    if not cache['complete'] or cache['committed_songs']!=cache['planned_songs']:raise ValueError('Need complete32second cache')
    for record in cache['records']:
        for name,info in record['files'].items():
            if sha(ROOT/record['path']/name)!=info['sha256']:raise ValueError('Short-context cache changed')
    parent=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    ssd=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json').read_text())
    comparison_path=ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json'
    comparison=json.loads(comparison_path.read_text());panel,full=setup(comparison)
    cohort_path=ROOT/'bench/rhythm-v2/local-portfolio-cohort-v1.json';cohort=json.loads(cohort_path.read_text())
    needed=parent['indices']['train64']+parent['indices']['dev']+parent['indices']['empty_dev']
    audios={panel.records[i]['audio_sha256'] for i in needed}
    input_hashes={};target_paths=set()
    for audio in sorted(audios):
        rec=cohort['mert_cache_identities'][audio]
        if rec['path']!=full.mert[audio]['path']:raise ValueError('Original full-cache route changed')
        for name,item in rec['files'].items():
            path=ROOT/rec['path']/name;actual=sha(path)
            if actual!=item['sha256']:raise ValueError('Full-cache payload drift')
            input_hashes[str(path.relative_to(ROOT))]=actual
        raw=full.pupu.records[audio];path=ROOT/raw['complete_mel'];actual=sha(path)
        if actual!=raw['sha256'][raw['complete_mel']]:raise ValueError('Shared10msmel drift')
        input_hashes[str(path.relative_to(ROOT))]=actual
    for i in needed:
        chart=panel.charts[panel.records[i]['chart_index']];target_paths.add(chart['target_path'])
    for name in sorted(target_paths):
        actual=sha(ROOT/name)
        if actual!=cohort['target_sha256'][name]:raise ValueError('Frozen native target drift')
        input_hashes[name]=actual
    stream=[];views=('draft','single_condition','all_conditions','draft','random_groups','middle','continuation','continuation')
    flat=[v for batch in parent['sample_stream']['train64'][:2000] for v in batch]
    for step in range(1000):
        stream.append([dict(index=flat[4*step+j]['index'],view=views[(4*step+j)%8],field_index=step) for j in range(4)])
    plan=dict(id=IDENT,seed=SEED_OVERRIDE or 20261064,model=MODEL_CONFIG_OVERRIDE or ssd['model'],density_reference=ssd['density_reference'],architecture=MODEL_CLASS.__name__,
        indices=parent['indices'],health_indices=parent['health_indices'],empty_health_index=parent['empty_health_index'],
        sample_stream=stream,quality_steps=1000,health_steps=100,effective_batch=4,
        learning_rate=.0003,minimum_learning_rate=.00003,warmup=100,weight_decay=.01,clip=1.,
        health_wall_s=1200,quality_wall_s=6000,evaluation_wall_s=1200,max_parameters=12_000_000,
        arms=['full'] if SINGLE_FRONTEND else ['full','short32'],cache_report_sha256=sha(cachepath),shifts_us=parent['shifts_us'],
        full_cache_report_sha256=sha(ROOT/comparison['mert_cache_report']),cohort_sha256=sha(cohort_path),comparison_sha256=sha(comparison_path),input_payload_sha256=input_hashes,
        mixture=dict(draft=2000,completion=1000,continuation=1000),
        strategy=dict(count='Existing SSD canvas composite loss',readout='Original expected_tail_rank',
            frontend='Same frozen MERT weights L12/L23, bf16 inference/fp16 cache; identical10msmel',
            extraction='Originalfull(300sstitchedonlywhereoriginal) versus32soverlap16',
            downstream='Same whole-song pooled summary in both arms',dynamic_generated_prefix_replay=False),
        hypothesis='Full pretrained extraction context improves cross-component placement and complete-event decoding over32s extraction under matched capacity and supervision.',
        matched_supervision_sha256=__import__('hashlib').sha256(json.dumps(stream,sort_keys=True).encode()).hexdigest(),
        full_context_support_gate=dict(full_note_gain=.015,onset10_gain=.015,min_component_wins=10,structure_count_error_reduction=.10),
        structure_metric='Component macro normalizedL1 of generated versus target event counts in fixed2secondbins. Type-count trajectory also reported. Only an8second structure proxy, not full-song arrangement/emotion.',
        local_screen=dict(min_nonempty=24,count_ratio=[.5,1.75],onset10=.20,onset10_minus_shift=.05,positive_components=12,full_note_f1=.20),
        interpretation='Single-seed local paired screening, not emotion/playability/scale proof. Different from old batch2 mixture; no claim of isolated batch or mixture improvement.100health is fit/numeric/legality only.',
        confirmation_scored=False,cloud_compute_calls=0,
        source_sha256={n:sha(ROOT/n) for n in ('scripts/run_mert_context_pair.py','scripts/run_local_portfolio_data_scale.py','beatfcn/model/local_mamba_canvas.py',
            'beatfcn/model/local_typed_canvas.py','beatfcn/model/direct_song_canvas.py','beatfcn/model/fullstack_mamba.py','beatfcn/model/mamba2_reference.py',
            'beatfcn/model/full_event_model.py','beatfcn/model/local_bundle.py','beatfcn/train/mert_canvas_features.py','beatfcn/train/relative_canvas.py',
            'beatfcn/train/local_typed_canvas.py','beatfcn/train/local_bundle.py','beatfcn/train/canonical_probability.py')+EXTRA_SOURCES})
    plan['global_routing']='Parameter-identical physical read of independently encoded whole-song stream at every query; full andshort arms both use it. Original concatenation-only control has a separate same-seed/budget contract.'
    if SINGLE_FRONTEND:
        plan.update(hypothesis=SINGLE_HYPOTHESIS or 'Complete10.14M attention control under the actual batch4/50-25-25 warmup/WD recipe. Original100step F1 health failure is retained; this100step fitting/numeric/legal admission precedes fresh1000 quality.',
            interpretation='Standalone full-context candidate; not the paired FullSong/short-context test or an isolated width/depth causal study.',
            global_routing='Original model routing unchanged; see the explicit architecture class and paired-control review.')
    write(PLAN,plan)
    for n in plan['source_sha256']:
        dest=ART/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    short=MertCanvasFeatures(ROOT,full.pupu,str(cachepath.relative_to(ROOT)))
    report=dict(id=IDENT,status='running',plan_sha256=sha(PLAN),arms={},failures=[],confirmation_scored=False,cloud_compute_calls=0)
    write(REPORT,report);evaluator.ART=ART;evaluator.LIVE['panel']=panel
    def stop(*_):raise KeyboardInterrupt('External context pair stop')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        for arm,store in ([('full',full)] if SINGLE_FRONTEND else [('full',full),('short32',short)]):
            original_row=row_factory(panel,store)
            def row(*args,**kwargs):return explicit_row(original_row(*args,**kwargs))
            model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
            r=dict(status='health_running',health_updates=0,quality_updates=0,health_weights_used=False,initial_weight_sha256=weight_digest(model),
                parameters=sum(p.numel() for p in model.parameters()),curves=[],exposure=dict(windows=0,events=0,views={},types={}))
            if r['parameters']>plan['max_parameters']:raise ValueError('Parameter cap')
            if report['arms'] and r['initial_weight_sha256']!=report['arms']['full']['initial_weight_sha256']:raise ValueError('Paired initialization mismatch')
            report['arms'][arm]=r;write(REPORT,report);started=time.monotonic()
            specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['empty_health_index'],'draft')]
            batches=[]
            for index,view in specs:
                e=row(index,view);inp,tgt=relative_batch([e],'mps');batches.append((e,inp,tgt))
            r['input_audit']=[dict(index=e['index'],tensors={k:dict(shape=list(v.shape),finite=bool(torch.isfinite(v).all()),
                minimum=float(v.min()),maximum=float(v.max()),mean=float(v.float().mean()),std=float(v.float().std()))
                for k,v in inp.items() if k in ('local','mel','song','reference','controls','local_times','mel_times','song_times')})
                for e,inp,tgt in batches]
            r['initial_diagnostic']=diagnostic(model,batches)
            # Exact real-input forward and gradient parity, same untouched gate.
            cpu=fresh(plan,'cpu').eval();e,inp,tgt=batches[0];ci,ct=({k:v.cpu() for k,v in d.items()} for d in (inp,tgt))
            model.eval();ca,ma=cpu(ci),model(inp)
            forward=max(float((ca[k]-ma[k].cpu()).abs().max()) for k in ca)
            cl,_=canvas_loss(ca,ct,ci);ml,_=canvas_loss(ma,tgt,inp);cl.backward();ml.backward();cp=dict(cpu.named_parameters());gradient=0.
            for n,p in model.named_parameters():
                if (p.grad is None)!=(cp[n].grad is None):raise ValueError('Gradient presence parity')
                if p.grad is not None:gradient=max(gradient,float((p.grad.cpu()-cp[n].grad).abs().max()))
            r['parity']=dict(forward=forward,gradient=gradient)
            if max(forward,gradient)>1e-4:raise ValueError('CPU/MPS parity')
            del cpu,ca,ma,cl,ml,cp
            for step in range(1,101):
                if time.monotonic()-started>plan['health_wall_s']:raise TimeoutError('Pair health wall')
                model.train();opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=rate(plan,step)
                for e,inp,tgt in batches:
                    loss,_=canvas_loss(model(inp),tgt,inp);(loss/len(batches)).backward()
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();r['health_updates']=step
            r['final_diagnostic']=diagnostic(model,batches)
            predictions=[]
            for e,inp,tgt in batches:
                prediction=generate_relative(model,inp,e['request'],deadline=started+plan['health_wall_s'])
                predictions.append(dict(index=e['index'],prediction=prediction,target=[asdict(v) for v in e['targets']]))
            write(ART/arm/'health-predictions.json',predictions)
            a=r['initial_diagnostic'];b=r['final_diagnostic']
            gates=dict(fit=np.mean([v['loss'] for v in b])<=.9*np.mean([v['loss'] for v in a]),complete=all(v['prediction']['complete'] for v in predictions))
            for k in ('offset','tail'):
                if k in a[0]['parts']:gates[k]=np.mean([v['parts'][k] for v in b])<=1.05*np.mean([v['parts'][k] for v in a])
            r.update(health_gates={k:bool(v) for k,v in gates.items()},health_wall_s=time.monotonic()-started)
            checkpoint(model,opt,arm,'health',100)
            if not all(gates.values()):r['status']='health_failed';write(REPORT,report);continue
            del model,opt,batches;gc.collect();torch.mps.empty_cache()
            model=fresh(plan);torch.mps.manual_seed(plan['seed']);opt=optimizer_for(model,plan)
            if weight_digest(model)!=r['initial_weight_sha256']:raise ValueError('Health weight reuse')
            r['status']='quality_running';write(REPORT,report);started=time.monotonic()
            actual_supervision=hashlib.sha256()
            for step in range(1,1001):
                if time.monotonic()-started>plan['quality_wall_s']:raise TimeoutError('Pair quality wall')
                model.train();opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=rate(plan,step)
                total=0.
                for spec in plan['sample_stream'][step-1]:
                    e=row(spec['index'],spec['view'],spec['field_index']);inp,tgt=relative_batch([e],'mps')
                    actual_supervision.update(json.dumps(dict(index=e['index'],view=e['view'],request=asdict(e['request']),
                        target=[asdict(v) for v in e['targets']],visible=[asdict(v) for v in e.get('visible',())]),sort_keys=True).encode())
                    for name in ('mel','mel_times','mel_valid','controls','reference','reference_times','reference_valid'):
                        value=np.asarray(e['inputs'][name]);actual_supervision.update(str((name,value.shape,str(value.dtype))).encode());actual_supervision.update(value.tobytes())
                    loss,_=canvas_loss(model(inp),tgt,inp);(loss/4).backward();total+=float(loss.detach())/4
                    exposure=r['exposure'];exposure['windows']+=1;exposure['events']+=len(e['targets'])
                    exposure['views'][e['view']]=exposure['views'].get(e['view'],0)+1
                    for event in e['targets']:exposure['types'][event.type]=exposure['types'].get(event.type,0)+1
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();r['quality_updates']=step
                if step%25==0 or step==1:
                    torch.mps.synchronize()
                    point=dict(step=step,loss=total,gradient=grad,wall_s=time.monotonic()-started,
                        memory=dict(tensor_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory(),recommended_bytes=torch.mps.recommended_max_memory()))
                    r['curves'].append(point);write(REPORT,report);print(json.dumps(dict(arm=arm,**point)),flush=True)
                if step%250==0:checkpoint(model,opt,arm,'quality',step)
            weight=checkpoint(model,opt,arm,'quality',1000);r.update(status='evaluating',final_weight_sha256=sha(weight),training_wall_s=time.monotonic()-started,
                actual_supervision_and_common_mel_sha256=actual_supervision.hexdigest());write(REPORT,report)
            moment=time.monotonic();r['evaluations']=evaluator.evaluate({arm:model},plan,row,'quality-1000',min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']),full=True)[arm]
            r['structure']=structure_error(json.loads((ART/arm/'quality-1000-draft.json').read_text()))
            r['global_controls']=global_controls(model,plan,row,panel,arm,min(started+plan['quality_wall_s'],moment+plan['evaluation_wall_s']))
            d=r['evaluations']['draft'];g=plan['local_screen'];gates=dict(complete=d['complete']==d['attempted'],nonempty=d['nonempty']>=g['min_nonempty'],count_ratio=g['count_ratio'][0]<=d['density_ratio']<=g['count_ratio'][1],onset10=d['onset10']>=g['onset10'],onset_null=d['onset10']-d['shift_null10']>=g['onset10_minus_shift'],components=sum(v['onset10']>v['shift_null10'] for v in d['components'].values())>=g['positive_components'],full_f1=d['full_note_f1']>=g['full_note_f1'])
            r.update(status='completed',local_screen=gates,local_screen_passed=all(gates.values()),evaluation_wall_s=time.monotonic()-moment,total_quality_wall_s=time.monotonic()-started);write(REPORT,report)
            del model,opt;gc.collect();torch.mps.empty_cache()
        arms=report['arms']
        if 'short32' in plan['arms'] and all(arms[k]['status']=='completed' for k in plan['arms']):
            if arms['full']['actual_supervision_and_common_mel_sha256']!=arms['short32']['actual_supervision_and_common_mel_sha256']:
                raise ValueError('Actual paired supervision/common mel changed')
            report['actual_supervision_and_common_mel_exact']=True
            a,b=[arms[k]['evaluations']['draft'] for k in plan['arms']]
            gain=dict(full_note_f1=a['full_note_f1']-b['full_note_f1'],onset10=a['onset10']-b['onset10'],
                component_wins=sum(a['components'][c]['full_note_f1']>b['components'][c]['full_note_f1'] for c in a['components']))
            first,second=[arms[k]['structure']['count_l1'] for k in plan['arms']]
            gain['structure_count_error_reduction']=(second-first)/max(second,1e-12)
            g=plan['full_context_support_gate'];report['full_context_comparison']=dict(gains=gain,supported=gain['full_note_f1']>=g['full_note_gain'] and gain['onset10']>=g['onset10_gain'] and gain['component_wins']>=g['min_component_wins'] and gain['structure_count_error_reduction']>=g['structure_count_error_reduction'])
        report['status']='completed';write(REPORT,report)
    except BaseException as exc:
        report['status']='stopped';report['failures'].append(f'{type(exc).__name__}: {exc}');write(REPORT,report);raise


if __name__=='__main__':main()

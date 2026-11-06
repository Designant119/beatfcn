"""C16 shared-total paired experiment: preparation, throwaway fit, fresh A/C, eval."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.shared_total_inventory import SharedTotalInventory as TypedInventory, count_log_marginals, count_nll, count_modes, total_log_probs
from beatfcn.train.managed_mps import ManagedMPS
from beatfcn.train.typed_inventory import inventory_row, inventory_batch, run_typed_members, SUPPORT
from beatfcn.train.shared_total_inventory import inventory_loss, decode_inventory, apply_update
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.native_quality_metrics import summarize
from beatfcn.train.native_trial_evaluation import metadata_outcomes, outcome_tasks, case_inputs, score_outcome, acoustic_pairs
from scripts import run_native_cell_recovery as original
from scripts.check_experiment_storage import admission, GIB
from scripts.run_joint_native_set import audit_tensors
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha, write, weight_digest
from scripts.run_native_cell_chain import closure


ROOT=Path(__file__).resolve().parents[1]
ID='c16-shared-total-inventory-v1';ART=ROOT/f'artifacts/{ID}';PLAN=ROOT/f'bench/rhythm-v2/{ID}.json'
REPORT=ROOT/f'reports/local-integration-v1/{ID}.json'
ENGINE=SimpleNamespace(cell_row=inventory_row,cell_batch=inventory_batch,specs=original.specs,score=original.score,
    output_null=original.output_null,zero_audio=original.zero_audio,audio_variant=original.audio_variant)


def read(path):return json.loads(Path(path).read_text())


def publish(report):
    write(REPORT,report)
    path=ROOT/'reports/local-integration-v1/state.json';state=read(path)
    state.update(status=report['status'],candidate=ID,report=str(REPORT.relative_to(ROOT)),
        progress_file=f'artifacts/{ID}/{report.get("active_stage","preparation")}.json',
        training_process_running=report['status'].endswith('_running'),confirmation_scored=False)
    write(path,state)


def runtime():
    if torch.__version__!='2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise ValueError('Isolated torch2.14/no silent fallback required')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)


def fresh(plan):
    if sha(ROOT/plan['initial_state_path'])!=plan['initial_state_sha256']:raise ValueError('Initial tensor file drift')
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    model=TypedInventory(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    model.load_state_dict(torch.load(ROOT/plan['initial_state_path'],map_location='cpu',weights_only=True))
    if weight_digest(model)!=plan['initial_weight_digest']:raise ValueError('Initial tensor digest')
    return model.to('mps')


def prepare():
    if PLAN.exists() or (ART/'preparation-registration.json').exists():raise FileExistsError('No preparation retry')
    ART.mkdir(parents=True,exist_ok=True);started=time.monotonic();runtime()
    write(ART/'preparation-registration.json',dict(id=ID,wall_s=600,decision_sha256=sha(ROOT/'reports/council-inventory-c16/parent-final-decision.md'),
        seed=20261064,optimizer_updates=0,cloud_calls=0,source_sha256={p:sha(ROOT/p) for p in closure('scripts/run_shared_total_inventory.py')}))
    parent_path=ROOT/'bench/rhythm-v2/c13-native-quality-v1.json';parent=read(parent_path)
    if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Storage admission')
    for name,h in {**parent['source_sha256'],**parent['runtime_records'],**parent['input_payload_sha256']}.items():
        if sha(ROOT/name)!=h:raise ValueError('Parent/cache/runtime drift '+name)
    panel,row=load_data();frequencies=np.full((4,65),.5,np.float64);total_frequency=np.full(65,.5,np.float64);seen={};audit=[]
    for subset in ('train64','dev','empty_dev'):
        for index in parent['indices'][subset]:
            if time.monotonic()-started>600:raise TimeoutError('Preparation600s')
            record=panel.records[index];request=event_request(record['request']);events=panel.targets(index)
            counts,members=SUPPORT.encode(events,request);roundtrip=SUPPORT.decode(counts,members,request)
            if Counter(roundtrip)!=Counter(events):raise ValueError('Native multiset roundtrip')
            typed=np.zeros((len(counts),4),np.int64)
            for m in members:typed[m.cell,m.kind]+=1
            if request.start_us%40000:raise ValueError('Registered phase0 prior expected')
            if subset=='train64':
                for c,values in enumerate(typed):
                    key=(record['chart_id'],request.start_us+c*40000)
                    if key in seen and seen[key]!=tuple(values):raise ValueError('Overlapping prior cell label mismatch')
                    if key not in seen:
                        seen[key]=tuple(map(int,values))
                        for k,n in enumerate(values):frequencies[k,n]+=1
                        total_frequency[int(sum(values))]+=1
            audit.append(dict(subset=subset,index=index,events=len(events),typed_count_sha256=__import__('hashlib').sha256(typed.tobytes()).hexdigest(),public_carry=len(request.known_carry)))
    prior=np.log(frequencies/frequencies.sum(-1,keepdims=True))
    total_prior=np.r_[np.log(total_frequency/total_frequency.sum()),-8.]
    random.seed(parent['seed']);np.random.seed(parent['seed']);torch.manual_seed(parent['seed'])
    model=TypedInventory(FullEventConfig(**parent['model']),density_reference=parent['density_reference'])
    old_initial=ROOT/'artifacts/c15-typed-inventory-v1/common-initial-state.pt'
    old_state=torch.load(old_initial,map_location='cpu',weights_only=True)
    same_body=[name for name,value in model.state_dict().items() if name not in
               ('typed_count.weight','typed_count.bias','total_count.weight','total_count.bias')]
    if any(not torch.equal(model.state_dict()[name],old_state[name]) for name in same_body):raise ValueError('Shared untrained initializer body drift')
    del old_state
    with torch.no_grad():
        model.typed_count.bias.copy_(torch.tensor(prior.reshape(-1),dtype=torch.float32))
        model.total_count.bias.copy_(torch.tensor(total_prior,dtype=torch.float32))
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>12_000_000:raise ValueError('12M parameter cap')
    initial=ART/'common-initial-state.pt';torch.save(model.state_dict(),initial)
    digest=weight_digest(model);del model
    p=dict(id=ID,seed=parent['seed'],model=parent['model'],density_reference=parent['density_reference'],parameters=parameters,
        parent_plan=str(parent_path.relative_to(ROOT)),parent_plan_sha256=sha(parent_path),
        indices=parent['indices'],sample_stream=parent['sample_stream'][:400],fit_stream=parent['sample_stream'][:2],
        shifts_us=parent['shifts_us'],initial_state_path=str(initial.relative_to(ROOT)),initial_state_sha256=sha(initial),initial_weight_digest=digest,
        input_payload_sha256=parent['input_payload_sha256'],runtime_records=parent['runtime_records'],runtime_target=parent['runtime_target'],
        prior_potentials=prior.tolist(),total_prior_scores=total_prior.tolist(),total_prior_histogram=total_frequency.tolist(),prior_cell_count=len(seen),prior_origin='Train416 phase0 base requests;chart/absolute-cell identity dedup only',
        shared_body_initializer=dict(path=str(old_initial.relative_to(ROOT)),sha256=sha(old_initial),equal_tensors=len(same_body),scope='Comparison with original untrained tensor only;no quality or health warmstart'),
        fit_updates=100,quality_updates=400,effective_batch=4,max_optimizer_updates=1002,
        wall_s=dict(preparation=600,health_total=1200,quality_per_arm=3000,evaluation=1200,total=9000),
        source_decision='reports/council-inventory-c16/parent-final-decision.md',decision_sha256=sha(ROOT/'reports/council-inventory-c16/parent-final-decision.md'),
        public_carry='Preserved and added as authorized positive reference;masked keeps publiccarry;A/C identical metadata',
        objective='TotalcategoricalNLL+conditionaltypedcompositionNLL+unchangednativewithin-typehead/endNLL;publiccellnormalization',
        decision='TotalmodeincludingOmega→exactconditionalcompositionmode→within-typegreedy;notchartjointMAP',
        fit_gate=dict(loss_ratio=.5,positive_vector_accuracy=.9,zero_nll_nondecreasing=False,head_and_tail_nll_decrease=True),
        research_gate=dict(joint_gain=.02,typed_positive_gain=.02,positive_components=3,full50=.269192,onset10=.249404,count_ratio=[.6,1.4],empty_max=13,
            type_tp=dict(hold=5,flick=5,drag=16,tap=225),type_components=dict(hold=2,flick=3,drag=6),non_tap_gain=5,acoustic_components=10,acoustic_gain=.10),
        primary_dimension='Architecture:generativeprobabilityfactorization;fullcandidatewithcorrespondingdecision',
        predecessors=['c15-typed-inventory-v1'],shared_failure_modes=['BOS_type_learning','canonical_wrong_prefix','sparse_count_decision','public_carry'],
        mechanism_changes=dict(
            BOS_type_learning=dict(change='Musicconditionedtypedcounts',prediction='Freepositivetypesandproperbeatmatchedcontrol',discriminating_control='Train-timeallthreeaudiozeroA/Cpair'),
            canonical_wrong_prefix=dict(change='Separatewithin-typecanonicalchains;nofixedrankselftraining',prediction='Cross-typehistorynolongerrequired',discriminating_control='Support/multisetactualtests;remainingwithin-typeexposureexplicit'),
            sparse_count_decision=dict(change='Independentnormalizedtotalplusconditionaltypedmultiset;train-onlyactualtotalhistogramprior',prediction='Restoreoccurrencewithoutrequiringoneuncertaintypebeforetotaldecision;typedproperandfreestillrequired',discriminating_control='CountproperA/C/staticpriorplusfree/empty'),
            public_carry=dict(change='Independentpubliccarrypreservingreferenceadapter',prediction='Allcallerholdsmultisetretained',discriminating_control='Duplicatemasked/unmaskedcarryactualtestandmetadataaudit')),
        pretraining_probe_receipts=[dict(status='completed',evidence='reports/council-inventory-c16/shared-total-tests.log',scope='Newnormalizedlaw/marginals/gradientCPU-MPSandnativeCOPYmemberregression;notactuald256admission'),
            dict(status='completed',evidence='reports/council-inventory-c16/shared-total-MPS-member-tests.log',scope='ActualnewclassMPSmember/nativeCOPY/crosswindow/shortgapgradientregression')],
        execution_ready=True,execution_ready_scope='Probeentryonly;formalrequiresactual100healthgates',cloud_calls=0,scaling_authorized=False,confirmation_scored=False)
    metadata=metadata_outcomes(ENGINE,p,panel,row);old=read(ROOT/parent['expected_free_metadata'])
    for task,records in metadata.items():
        reference={r['index']:r for r in old[task]}
        for r in records:
            if r['target']!=reference[r['index']]['target']:raise ValueError('Fixed evaluation target identity')
    # Validate the actual frozen training views, not only phase0 base windows.
    view_audit=[];checked=set()
    for batch in p['sample_stream']:
        for spec in batch:
            key=(spec['index'],spec['view'],spec.get('field_index',0))
            if key in checked:continue
            if time.monotonic()-started>600:raise TimeoutError('Preparation600s')
            rr=inventory_row(panel,row(*key));x,target=inventory_batch(rr,'cpu')
            count,members=SUPPORT.encode(rr['targets'],rr['request'])
            if Counter(SUPPORT.decode(count,members,rr['request']))!=Counter(rr['targets']):raise ValueError('Actual view multiset roundtrip')
            view_audit.append(dict(index=key[0],view=key[1],field_index=key[2],events=len(rr['targets']),public_input_sha256=original.input_digest(x)))
            checked.add(key);del rr,x,target
    write(ART/'training-view-support.json',dict(status='passed',views=view_audit,source='Frozen400x4stream;no neural forward',optimizer_updates=0))
    write(ART/'expected-free-metadata.json',metadata);p.update(expected_free_metadata=str((ART/'expected-free-metadata.json').relative_to(ROOT)),expected_free_metadata_sha256=sha(ART/'expected-free-metadata.json'))
    p['source_sha256']={name:sha(ROOT/name) for name in set(closure('scripts/run_shared_total_inventory.py'))|{'tests/test_shared_total_inventory.py'}}
    write(PLAN,p)
    for name in p['source_sha256']:
        target=ART/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write(ART/'source/frozen-plan.json',p)
    write(ART/'preparation.json',dict(status='completed',wall_s=time.monotonic()-started,parameters=parameters,prior_cells=len(seen),targets=audit,actual_training_views=len(checked),optimizer_updates=0,cloud_calls=0))
    print(json.dumps(dict(status='prepared',parameters=parameters,wall_s=time.monotonic()-started)))


def row_inputs(panel,row,s,arm):
    rr=inventory_row(panel,row(s['index'],s['view'],s.get('field_index',0)))
    x,t=inventory_batch(rr,'mps')
    return rr,original.zero_audio(x) if arm=='C' else x,t


def actual_admission(model,opt_mps,plan,panel,row,guard,report):
    model.eval()
    twin=TypedInventory(FullEventConfig(**plan['model']),density_reference=plan['density_reference'])
    twin.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});twin.eval()
    cases=[s for batch in plan['fit_stream'] for s in batch]
    report.update(initial_digest=weight_digest(model),cpu_mps={},requests=[],branch_norms={})
    cpu_values=[];cpu_gradients={};diffs=dict(typed_scores=0.,total_scores=0.,x=0.,loss=0.,gradient=0.)
    opt_cpu=torch.optim.AdamW(twin.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    for backend,net,opt in [('cpu',twin,opt_cpu),('mps',model,opt_mps)]:
        opt.zero_grad(set_to_none=True)
        for i,s in enumerate(cases):
            guard.check('admission_'+backend)
            rr,x,t=row_inputs(panel,row,s,'A')
            # Changing hidden supervision cannot change public inputs for this fixed request.
            hidden={**rr,'targets':[]};blind,_=inventory_batch(hidden,'mps')
            if any(not torch.equal(x[k],blind[k]) for k in x):raise ValueError('Hidden target changed public input')
            del blind
            if backend=='cpu':x={k:v.cpu() for k,v in x.items()}
            out=net(x);loss,detail=inventory_loss(net,out,x,rr['request'],t)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Admission loss')
            if backend=='cpu':
                cpu_values.append({**{k:out[k].detach().cpu() for k in ('typed_scores','total_scores','x')},'loss':float(loss.detach())})
                report['requests'].append(dict(index=s['index'],view=s['view'],field_index=s.get('field_index',0),
                    request=asdict(rr['request']),inputs=audit_tensors(x),target_events=len(rr['targets']),
                    holds=sum(e.type=='hold' for e in rr['targets']),
                    cross_window_holds=sum(e.type=='hold' and e.end_us>rr['request'].end_us for e in rr['targets']),
                    maximum_hold_us=max([e.end_us-e.onset_us for e in rr['targets'] if e.type=='hold'] or [0]),
                    hidden_input_unchanged=True))
            else:
                for k in ('typed_scores','total_scores','x'):diffs[k]=max(diffs[k],float((out[k].detach().cpu()-cpu_values[i][k]).abs().max()))
                diffs['loss']=max(diffs['loss'],abs(float(loss.detach().cpu())-cpu_values[i]['loss']))
            (loss/len(cases)).backward()
            del rr,x,t,out,loss,detail;guard.clear()
        norms={name:float(p.grad.norm().cpu()) if p.grad is not None else None for name,p in net.named_parameters()
               if name in ('typed_count.weight','total_count.weight','member_initial.weight','endpoint_key.0.weight')}
        report['branch_norms'][backend]=norms
        if any(v is None or not math.isfinite(v) or v<=0 for v in norms.values()):raise FloatingPointError('Admission gradient branch')
        if backend=='cpu':cpu_gradients={name:p.grad.detach().clone() if p.grad is not None else None for name,p in net.named_parameters()}
        if backend=='mps':
            for name,q in model.named_parameters():
                p=cpu_gradients[name]
                if (p is None)!=(q.grad is None):raise ValueError('Admission gradient support mismatch')
                if p is not None:
                    diffs['gradient']=max(diffs['gradient'],float((p-q.grad.cpu()).abs().max()))
            if not all(math.isfinite(v) and v<=1e-4 for v in diffs.values()):raise ValueError('Actual d256 CPU/MPS mismatch '+str(diffs))
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.))
        if not math.isfinite(norm):raise FloatingPointError('Admission gradient')
        apply_update(opt,report)
        report.setdefault('applied_backend_updates',{})[backend]=1
        original.finite_optimizer(net,opt)
    report.update(status='admission_passed',cpu_mps=diffs,hidden_input_checks=16,
                  health_weights_discarded=True,scope='8 fixed requests on each backend;one discardable AdamW update each')
    del twin,opt_cpu,cpu_values,cpu_gradients;guard.clear()


@torch.no_grad()
def fit_measure(model,plan,panel,row,arm,guard,decode=False):
    result=dict(loss=0.,count=0.,head=0.,tail=0.,zero_nll=0.,zero_cells=0,positive_cells=0,correct_positive_vectors=0,requests=[])
    model.eval()
    for batch in plan['fit_stream']:
        for s in batch:
            guard.check('fit_measure');rr,x,t=row_inputs(panel,row,s,arm);out=model(x);loss,detail=inventory_loss(model,out,x,rr['request'],t)
            labels=torch.tensor(t['counts'][None],device='mps');values=count_nll(out['typed_scores'],out['total_scores'],x['typed_allowed'],labels)[0]
            counts,failures=count_modes(out['typed_scores'],out['total_scores'],x['typed_allowed']);positive=t['counts'].sum(-1)>0
            result['zero_nll']+=float(values[torch.tensor(~positive,device='mps')].sum().cpu());result['zero_cells']+=int((~positive).sum())
            result['positive_cells']+=int(positive.sum());result['correct_positive_vectors']+=int(((counts==t['counts']).all(-1)&positive).sum())
            for k in ('loss','count','head','tail'):result[k]+=float(loss.cpu()) if k=='loss' else float(detail[k].cpu())
            item=dict(index=s['index'],view=s['view'],targets=len(rr['targets']),holds=detail['holds'],overflow_cells=failures)
            if decode:item['prediction']=decode_inventory(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},output=out,deadline=guard.start+guard.wall_s)
            result['requests'].append(item);del rr,x,t,out,loss,detail,values;guard.clear()
    for k in ('loss','count','head','tail'):result[k]/=8
    result['zero_nll']/=max(result['zero_cells'],1);result['positive_vector_accuracy']=result['correct_positive_vectors']/max(result['positive_cells'],1)
    return result


def train_stage(model,opt,plan,panel,row,guard,report,arm,health):
    stream=plan['fit_stream'] if health else plan['sample_stream'];steps=100 if health else 400
    if health and arm=='A':
        admission_receipt=read(ART/'admission.json')
        if admission_receipt['status']!='admission_passed':raise ValueError('Actual admission receipt required')
    if health:report['before']=fit_measure(model,plan,panel,row,arm,guard)
    samples=[];exposure=Counter()
    for step in range(1,steps+1):
        model.train();opt.zero_grad(set_to_none=True);values=[]
        for micro,s in enumerate(stream[(step-1)%len(stream)]):
            report['accumulation_phase']=micro;guard.check('training_forward');rr,x,t=row_inputs(panel,row,s,arm)
            if step==1:report.setdefault('input_probe',[]).append(dict(index=s['index'],view=s['view'],inputs=audit_tensors(x),request=asdict(rr['request'])))
            out=model(x);loss,detail=inventory_loss(model,out,x,rr['request'],t)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Inventory loss')
            (loss/4).backward();values.append(float(loss.detach().cpu()));exposure.update(e.type for e in rr['targets'])
            del rr,x,t,out,loss,detail;guard.clear()
        if step==1:
            branches={name:dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),norm=float(p.grad.norm().cpu()) if p.grad is not None else None)
                      for name,p in model.named_parameters() if name in ('typed_count.weight','total_count.weight','member_initial.weight','endpoint_key.0.weight')}
            report['first_gradient_branches']=branches
            if any(not v['finite'] or v['norm']<=0 for v in branches.values()):raise FloatingPointError('Missing/nonfinite supervised branch gradient')
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(gradient):raise FloatingPointError('Inventory gradient')
        rate=3e-4 if health else (3e-4*step/40 if step<=40 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-40)/360)))
        for group in opt.param_groups:group['lr']=rate
        report['accumulation_phase']='optimizer';guard.check('optimizer');apply_update(opt,report);original.finite_optimizer(model,opt)
        report.update(exposure=dict(exposure));sample=guard.clear();samples.append(sample)
        if step==1 or step%10==0:
            report.setdefault('curve',[]).append(dict(step=step,loss=float(np.mean(values)),gradient=gradient,lr=rate,**sample));write(ART/f'{report["stage"]}.json',report)
            print(report['stage'],step,round(float(np.mean(values)),6),round(sample['wall_s'],2),flush=True)
        if not health and step<=64 and time.monotonic()-guard.start>900:raise TimeoutError('Quality first64 wall')
        if not health and step==64:
            growth=float(np.median([v['physical_footprint_bytes'] for v in samples[32:]])-np.median([v['physical_footprint_bytes'] for v in samples[:32]]))
            median=float(np.median([v['physical_footprint_bytes'] for v in samples[32:]]))
            report['prefix64']=dict(growth_bytes=growth,median_bytes=median,passed=growth<=256*2**20 and median<=5*GIB)
            if not report['prefix64']['passed']:raise MemoryError('Quality first64 stability')
    if health:
        report['after']=fit_measure(model,plan,panel,row,arm,guard,decode=True)
        a,b=report['after'],report['before'];structural=all(r['prediction']['complete'] for r in a['requests'])
        checks=dict(finite=all(math.isfinite(a[k]) for k in ('loss','count','head','tail','zero_nll')),free_structural=structural)
        if arm=='A':checks.update(loss=a['loss']<=.5*b['loss'],vectors=a['positive_vector_accuracy']>=.9,zero=a['zero_nll']<=b['zero_nll'],head=a['head']<b['head'],tail=a['tail']<b['tail'])
        report.update(checks=checks,status='health_passed' if all(checks.values()) else 'health_failed_pending_mechanism_review',health_weights_discarded=True)
    else:
        checkpoint=ART/f'quality-{arm}-0400.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=400,plan_sha256=sha(PLAN),seed=plan['seed'],complete=True),checkpoint)
        report.update(status='quality_completed',checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),weight_digest=weight_digest(model))


def load_quality(plan,arm):
    r=read(ART/f'quality-{arm}.json');path=ROOT/r['checkpoint']
    if sha(path)!=r['checkpoint_sha256']:raise ValueError('Quality checkpoint drift')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['updates']!=400 or state['plan_sha256']!=sha(PLAN):raise ValueError('Quality checkpoint identity')
    model=fresh(plan);model.load_state_dict(state['model']);del state
    return model.eval()


@torch.no_grad()
def evaluate(plan,panel,row,guard,report):
    expected=read(ROOT/plan['expected_free_metadata']);all_groups={}
    deadline=guard.start+guard.wall_s
    for arm in ('A','C'):
        groups=deepcopy(expected);all_groups[arm]=groups
        for task,records in groups.items():write(ART/f'{arm}-{task}.json',records)
    # All340 free outcomes precede any teacher/proper pass.
    for arm in ('A','C'):
        model=load_quality(plan,arm);before=weight_digest(model)
        for task,records in all_groups[arm].items():
            for rec in records:
                guard.check('free_'+arm+'_'+task);rec['status']='attempted'
                rr,x,t,control=case_inputs(ENGINE,plan,panel,row,task,rec['index'],rec['view'],tuple(rec['bands']) if rec['bands'] else None,rec['masked'])
                if arm=='C':x=original.zero_audio(x)
                model.query_coverage={}
                out=model(x);pred=decode_inventory(model,x,rr['request'],{k:t[k] for k in ('heads','allowed')},output=out,deadline=deadline)
                events=[RhythmEvent(**e) for e in pred['events']];scores,partial=score_outcome(ENGINE,events,rr['targets'],pred['complete'])
                null=original.output_null(events,rr['targets'],rr['request'],plan['shifts_us'])
                if not pred['complete']:
                    for v in null:v.update(onset10=0.,full50=0.)
                rec.update(status='completed' if pred['complete'] else 'decode_failed',prediction=pred,scores=scores,raw_partial_scores=partial,control=control,
                    output_null=null,output_null_onset10=[v['onset10'] for v in null],input_sha256=original.input_digest(x),query_coverage_free=deepcopy(model.query_coverage),query_coverage_scope='Actualmembercandidatequeries;headincludesmaskedcandidates;rangeonlynotcontent/receptivefield')
                write(ART/f'{arm}-{task}.json',records);del rr,x,t,out,pred,events;guard.clear()
            print('free',arm,task,len(records),flush=True)
        if weight_digest(model)!=before:raise ValueError('Evaluation mutated weights')
        del model;guard.clear()
    for arm in ('A','C'):
        model=load_quality(plan,arm)
        for task,records in all_groups[arm].items():
            for rec in records:
                guard.check('proper_'+arm+'_'+task)
                rr,x,t,_=case_inputs(ENGINE,plan,panel,row,task,rec['index'],rec['view'],tuple(rec['bands']) if rec['bands'] else None,rec['masked'])
                if arm=='C':x=original.zero_audio(x)
                model.query_coverage={}
                out=model(x);labels=torch.tensor(t['counts'][None],device='mps')
                nll=count_nll(out['typed_scores'],out['total_scores'],x['typed_allowed'],labels)[0]
                total_nll=-total_log_probs(out['total_scores'],x['typed_allowed']).gather(-1,labels.sum(-1)[...,None]).squeeze(-1)[0]
                marginal=count_log_marginals(out['typed_scores'],out['total_scores'],x['typed_allowed'])[0]
                actual=-marginal.gather(-1,labels[0,...,None]).squeeze(-1)
                prior=torch.tensor(plan['prior_potentials'],device='mps',dtype=torch.float32)[None,None].expand(1,len(t['counts']),4,65)
                po=torch.tensor(plan['total_prior_scores'],device='mps',dtype=torch.float32)[None,None].expand(1,len(t['counts']),66)
                prior_nll=count_nll(prior,po,x['typed_allowed'],labels)[0]
                _,parts,_,holds=run_typed_members(model,out,x,rr['request'],t,t['counts'],teacher=True,trace=False,deadline=deadline)
                rawpath=ART/f'probability-{arm}-{task}-{rec["index"]}.npz'
                np.savez_compressed(rawpath,scores=out['typed_scores'][0].cpu().numpy(),total_scores=out['total_scores'][0].cpu().numpy(),allowed=x['typed_allowed'][0].cpu().numpy(),target=t['counts'])
                rec.update(teacher_status='completed',count_target=t['counts'].tolist(),typed_allowed=x['typed_allowed'][0].cpu().tolist(),
                    joint_count_nll=nll.cpu().tolist(),total_count_nll=total_nll.cpu().tolist(),conditional_composition_nll=(nll-total_nll).cpu().tolist(),prior_joint_count_nll=prior_nll.cpu().tolist(),type_marginal_nll=actual.cpu().tolist(),query_coverage_teacher=deepcopy(model.query_coverage),
                    teacher_head_nll=float(parts['head'].cpu()),teacher_tail_nll=float(parts['tail'].cpu()),teacher_holds=holds,
                    probability_path=str(rawpath.relative_to(ROOT)),probability_sha256=sha(rawpath))
                write(ART/f'{arm}-{task}.json',records);del rr,x,t,out,labels,nll,total_nll,marginal,actual,prior,po,prior_nll,parts;guard.clear()
            print('proper',arm,task,len(records),flush=True)
        del model;guard.clear()
    report['summary']={a:{task:summarize(records) for task,records in groups.items()} for a,groups in all_groups.items()}
    report['research']=research_decision(plan,all_groups,report['summary'])
    report.update(status='terminal_pending_mechanism_review',free_completed=340,conditional_completed=340,research_qualified=False,scaling_authorized=False)


def research_decision(plan,groups,summary):
    def component_average(records,field):
        by=defaultdict(list)
        for r in records:by[r['component']].append(float(np.mean(r[field])))
        return float(np.mean([np.mean(v) for v in by.values()]))
    a=groups['A']['draft'];c=groups['C']['draft'];sa=summary['A']['draft'];sc=summary['C']['draft'];g=plan['research_gate']
    joint=component_average(a,'joint_count_nll');control=component_average(c,'joint_count_nll');prior=component_average(a,'prior_joint_count_nll');types={}
    for k,kind in enumerate(('tap','hold','flick','drag')):
        strata={}
        for arm,rows in [('A',a),('C',c)]:
            bins={'positive':[],'zero':[]};components=defaultdict(list)
            for r in rows:
                values=np.array(r['type_marginal_nll'])[:,k];counts=np.array(r['count_target'])[:,k];allowed=np.array(r['typed_allowed'])[:,k]
                for label,mask in [('positive',(counts>0)&allowed),('zero',(counts==0)&allowed)]:bins[label].extend(values[mask].tolist())
                components[r['component']].extend(values[(counts>0)&allowed].tolist())
            strata[arm]=dict(positive=float(np.mean(bins['positive'])) if bins['positive'] else None,zero=float(np.mean(bins['zero'])) if bins['zero'] else None,
                positive_support=len(bins['positive']),zero_support=len(bins['zero']),components={n:float(np.mean(v)) for n,v in components.items() if v})
        av,cv=strata['A'],strata['C'];improved=sum(cv['components'].get(comp,-math.inf)>value for comp,value in av['components'].items())
        passed=av['positive'] is not None and cv['positive'] is not None and cv['positive']-av['positive']>=.02 and av['zero']<=cv['zero'] and improved>=3
        types[kind]=dict(strata=strata,positive_improved_components=improved,passed=passed)
    pairs={task:acoustic_pairs(groups['A']['draft'],groups['A'][task]) for task in ('audio_shift','audio_foreign')}
    checks=dict(complete=all(r['prediction']['complete'] for tasks in groups.values() for records in tasks.values() for r in records),
        joint=control-joint>=.02 and prior-joint>=.02,typed=all(types[k]['passed'] for k in ('hold','flick','drag')),
        full=sa['full50_component_macro']>=g['full50'],onset=sa['onset10_component_macro']>=g['onset10'],
        count=.6<=sa['pred']/max(sa['true'],1)<=1.4,empty=summary['A']['empty']['pred']<=13,
        non_tap=all(sa['types'][k]['tp']>=g['type_tp'][k] and sa['types'][k]['matched_components']>=g['type_components'][k] for k in ('hold','flick','drag')),
        tap=sa['types']['tap']['tp']>=225,
        matched_control=sum(sa['types'][k]['tp']-sc['types'][k]['tp'] for k in ('hold','flick','drag'))>=5 and sa['types']['hold']['tp']>sc['types']['hold']['tp'],
        acoustic=all(v['common_covered_components']>=10 and v['positive_covered_components']>=10 and v['covered_delta_onset10'] is not None and v['covered_delta_onset10']>=.10 for v in pairs.values()),
        null=sa['onset10_component_macro']-sa['null10_component_macro']>=.10)
    return dict(checks={k:bool(v) for k,v in checks.items()},supported=all(checks.values()),joint_nll_A=joint,joint_nll_C=control,joint_nll_prior=prior,typed=types,acoustic=pairs,
        scope='Exploratorywholecandidate;A/Crawinputlearningeffect;factorizationanddecisionnotisolated;core/fullcontextcoveragequalificationsseparate;no product/scaling')


def worker(stage,cap):
    runtime();plan=read(PLAN)
    for name,h in plan['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('New source drift '+name)
    guard=ManagedMPS(ROOT,ART,cap);report=dict(id=ID,stage=stage,status='running',updates=0,optimizer_updates=0,cloud_calls=0)
    model=opt=None
    def stop(*_):raise TimeoutError('Registered stage cap/interruption')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(max(1,int(cap)-2))
    try:
        panel,row=load_data()
        if stage=='evaluation':evaluate(plan,panel,row,guard,report)
        elif stage=='admission':
            model=fresh(plan)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            actual_admission(model,opt,plan,panel,row,guard,report)
        else:
            health=stage.startswith('health');arm=stage[-1]
            if not health and any(read(ART/f'health-{a}.json')['status']!='health_passed' for a in ('A','C')):raise ValueError('Both actual health receipts required')
            model=fresh(plan);report['initial_digest']=weight_digest(model)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            train_stage(model,opt,plan,panel,row,guard,report,arm,health)
    except BaseException as exc:
        report.update(status='execution_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc),phase=guard.phase))
        if model is not None:
            path=ART/f'{stage}-failure-state.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,updates=report['updates'],accumulation_phase=report.get('accumulation_phase'),plan_sha256=sha(PLAN)),path)
            report.update(failure_checkpoint=str(path.relative_to(ROOT)),failure_checkpoint_sha256=sha(path))
    finally:
        signal.alarm(0);report.update(telemetry=guard.close(),wall_s=time.monotonic()-guard.start)
        write(ART/f'{stage}.json',report)
    print(json.dumps({k:report.get(k) for k in ('stage','status','updates','wall_s','failure')}),flush=True)


def run():
    if REPORT.exists():raise FileExistsError('No automatic experiment retry')
    plan=read(PLAN);prep=read(ART/'preparation.json');started=time.monotonic();health_wall=0
    env={**os.environ,'PYTHONPATH':str(ROOT/plan['runtime_target'])+os.pathsep+str(ROOT)}
    report=dict(id=ID,status='registered',stages={},preparation_wall_s=prep['wall_s'],optimizer_updates=0,cloud_calls=0,cloud_spend_usd=0,confirmation_scored=False,scaling_authorized=False)
    try:
        for stage in ('admission','health-A','health-C','quality-A','quality-C','evaluation'):
            remaining=9000-prep['wall_s']-(time.monotonic()-started)
            cap=min(600-prep['wall_s'] if stage=='admission' else 1200-health_wall if stage.startswith('health') else 1200 if stage=='evaluation' else 3000,remaining)
            if cap<=2:raise TimeoutError('No remaining registered wall budget')
            if admission(ROOT,GIB,15*GIB)['status']!='admitted':raise MemoryError('Storage admission')
            report.update(status=stage+'_running',active_stage=stage);publish(report)
            with (ART/f'{stage}.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.run_shared_total_inventory','worker',stage,str(int(cap))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cap,check=True)
            r=read(ART/f'{stage}.json');report['stages'][stage]={k:r.get(k) for k in ('status','updates','wall_s','failure')}
            report['optimizer_updates']+=r.get('updates',0)
            if stage.startswith('health'):health_wall+=r['wall_s']
            expected='admission_passed' if stage=='admission' else 'health_passed' if stage.startswith('health') else 'terminal_pending_mechanism_review' if stage=='evaluation' else 'quality_completed'
            if r['status']!=expected:report.update(status=r['status'],failed_stage=stage);break
        else:report.update(status='terminal_pending_mechanism_review',research=read(ART/'evaluation.json')['research'])
    except BaseException as exc:report.update(status='supervisor_failed_pending_mechanism_review',failure=dict(type=type(exc).__name__,error=str(exc)))
    finally:report['wall_s']=time.monotonic()-started+prep['wall_s'];publish(report)
    print(json.dumps({k:report.get(k) for k in ('status','failed_stage','optimizer_updates','wall_s','failure')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);parser.add_argument('stage',nargs='?');parser.add_argument('cap',nargs='?',type=int);args=parser.parse_args()
    if args.action=='worker':worker(args.stage,args.cap)
    else:globals()[args.action]()

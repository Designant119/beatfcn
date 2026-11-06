"""C08 single intervention, using the frozen C07 training/evaluation driver.

Only process-local dependency bindings change: output paths, model constructor,
preflight and extra diagnostics. C07 source, schedule and weights stay untouched.
"""
from dataclasses import asdict
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import AnchorMarkChain, KINDS, mark_loss
from beatfcn.model.no_mark_history import NoMarkHistory
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.anchor_marks import anchor_inputs, teacher_example
from beatfcn.train.mark_prefix import prefix_outputs
from beatfcn.train.mark_trace import sampled_trace, greedy_marks
from scripts import run_c07_anchor_marks as driver
from scripts.run_local_portfolio_scale import sha, write, weight_digest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'bench/rhythm-v2/c08-no-mark-history-v1.json'
ART = ROOT/'artifacts/c08-no-mark-history-v1'
FOLDER = ROOT/'reports/local-evolution-v4'
REPORT = FOLDER/'no-mark-history.json'
OLD = ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1'
ORIGINAL_LOAD = driver.load_data
ORIGINAL_EVALUATE = driver.evaluate
ORIGINAL_VALIDATE = driver.validate_successor
STARTED = None


def prepare():
    if PLAN.exists(): raise FileExistsError('Frozen candidate already exists')
    p = driver.read('bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json')
    extras = ['scripts/run_c08_no_mark_history.py','beatfcn/model/no_mark_history.py',
        'beatfcn/train/mark_trace.py','beatfcn/train/mark_prefix.py','tests/test_no_mark_history.py',
        'beatfcn/model/full_event_codec.py','beatfcn/model/sample.py','beatfcn/model/event_canvas.py',
        'beatfcn/train/reference_completion.py','beatfcn/train/event_conditions.py',
        'beatfcn/model/local_song_rollout.py','scripts/validate_mechanism_reviews.py']
    p.update(id='c08-no-mark-history-v1',predecessors=['c07-anchor-conditioned-mark-chain-v1'],
        single_intervention='Architecture: gate only non-BOS discrete previous-type embedding to zero in both training and all inference; acoustic GRU recurrence, weights/shapes/init/supervision unchanged.',
        shared_failure_modes=['teacher_generated_type_history_shift'],
        mechanism_changes=dict(teacher_generated_type_history_shift=dict(
            change='Fresh retrain at g=0; no discrete previous-type information, retain BOS/reference/acoustic hidden recurrence.',
            prediction='Oracle-free proper score and T1 type quality improve beyond prior and fixed B, with local/all-audio negative-control gaps.',
            discriminating_control='Gate1 exact B event replay; gate0 label invariance, oracle free versus machine free and local-roll/full-foreign audio.')),
        parent_deviation='Choose Sol S03 / Grok independent ALT-NOTYPE before self-prefix recovery. Zero feedback, not constant learned BOS at all positions. Keep original C07 driver by explicit process-local dependency bindings.',
        execution_ready=False,pretraining_probe_receipts=[],
        gate=dict(I=dict(nll_max=1.059535,prior_components=9,B_free_gain=.10,audio_gain=.05,audio_components=9),
            Q=dict(typed50=.448334,components=9,full50=.382781,non_tap=.185210),
            M=dict(typed50=.312469,components=9,full50=.255021,non_tap=.089422),
            default=dict(typed50=.323897,full50=.306075)),
        model={**p['model'],'feedback_gate':0,'expected_parameters':613895},
        frozen_predecessor=dict(plan='bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json',
            weight=str((OLD/'quality-1000.pt').relative_to(ROOT)),weight_sha256=sha(OLD/'quality-1000.pt'),
            data_identity_sha256=sha(OLD/'data-identity.json'),
            initial_weight_sha256='52071f8f94c2d7745a555ac7c81f933d93cc508bdb63aae31d6c932959688316'),
        extra_evaluation='True-anchor four-T1 actual self-prefix probabilities under real/local_roll/song_foreign/audio_foreign; fixed greedy machine/oracle; preserve all original views and donor failures. Node strata, per-type scores and analysis-only50ms anchor matching in terminal analysis.',
        dimension='Architecture',cloud_compute_calls=0)
    p['sources'] = {f:sha(ROOT/f) for f in sorted(set(p['sources'])|set(extras))}
    p['predecessor_event_sha256'] = {str(f.relative_to(ROOT)):sha(f) for track in ('real','oracle')
        for f in (OLD/'evaluation'/track).glob('draw*-draft.json')}
    write(PLAN,p)
    for f in p['sources']:
        dest=ART/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    shutil.copy2(PLAN,ART/'source/initial-plan.json')
    write(FOLDER/'state.json',dict(status='prepared',active_candidate=p['id'],plan=str(PLAN.relative_to(ROOT)),
        report=str(REPORT.relative_to(ROOT)),cloud_compute_calls=0,argon='excluded_no_replacement'))


def canonical(events):
    return [asdict(v) for v in sorted(RhythmEvent(**e) for e in events)]


@torch.no_grad()
def free_trace(model, plan, panel, row, condition, deadline):
    components=sorted({panel.records[i]['component'] for i in plan['indices']['dev']})
    donors={c:next(i for i in plan['indices']['dev'] if panel.records[i]['component']==components[(j+1)%len(components)]) for j,c in enumerate(components)}
    result=[]
    for i in plan['indices']['dev']:
        rr=row(i,'draft');x,y,heads=teacher_example(rr);audio=None
        if condition in ('song_foreign','audio_foreign'):
            donor=row(donors[panel.records[i]['component']],'draft')['inputs']
            audio=donor if condition=='audio_foreign' else {**rr['inputs'],'song':donor['song'],'song_valid':donor['song_valid']}
        x=anchor_inputs(rr,heads,audio=audio,roll_audio=condition=='local_roll')
        t=sampled_trace(model,driver.device(x),heads,rr['request'].duration_us,index=i,seeds=plan['seeds'],
            labels=y['kind'][0].numpy(),allowed_types=rr['request'].allowed_types,deadline=deadline)
        result.append(dict(index=i,component=panel.records[i]['component'],condition=condition,**t))
    return result


def load_and_preflight():
    plan=driver.read(str(PLAN.relative_to(ROOT)));panel,row=ORIGINAL_LOAD()
    deadline=STARTED+plan['wall_s']['preflight']
    frozen=plan['frozen_predecessor']
    if sha(ROOT/frozen['weight'])!=frozen['weight_sha256']:raise ValueError('B weight identity drift')
    for f,h in plan['predecessor_event_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('B event identity drift '+f)
    torch.manual_seed(plan['seed']);model=NoMarkHistory(gate=1).to('mps')
    if weight_digest(model)!=frozen['initial_weight_sha256']:raise ValueError('Initial parameters changed')
    model.load_state_dict(torch.load(ROOT/frozen['weight'],map_location='cpu',weights_only=False)['model']);model.eval()
    traces=free_trace(model,plan,panel,row,'real',deadline);replays=0
    for draw in range(4):
        old={v['index']:v for v in driver.read(str((OLD/'evaluation/oracle'/f'draw{draw}-draft.json').relative_to(ROOT)))}
        for t in traces:
            if canonical(t['events'][draw])!=old[t['index']]['prediction']['events']:raise ValueError('Oracle exact replay')
            replays+=1
    old_machine=[{v['index']:v for v in driver.read(str((OLD/'evaluation/real'/f'draw{d}-draft.json').relative_to(ROOT)))} for d in range(4)]
    for i in plan['indices']['dev']:
        rr=row(i,'draft');heads=sorted(v['onset_us'] for v in old_machine[0][i]['prediction']['events'])
        x=anchor_inputs(rr,heads)
        out=sampled_trace(model,driver.device(x),heads,rr['request'].duration_us,index=i,seeds=plan['seeds'],allowed_types=rr['request'].allowed_types,deadline=deadline)
        for d in range(4):
            if canonical(out['events'][d])!=old_machine[d][i]['prediction']['events']:raise ValueError('Machine exact replay')
            replays+=1
    nll=sum(len(v['heads'])*v['nll'] for v in traces)/sum(len(v['heads']) for v in traces)
    write(ART/'B-self-prefix-T1-baseline.json',dict(status='completed',nll=nll,replays=replays,
        traces=traces,optimizer_updates=0,weight_sha256=frozen['weight_sha256']))
    teacher_parity=[]
    for i,v in list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(plan['indices']['dev'][0],'draft'),(plan['indices']['dev'][1],'draft')]:
        x,y,_=teacher_example(row(i,v));dx=driver.device(x);dy=driver.device(y)
        with torch.no_grad():
            direct=model.teacher(dx,dy['kind']);step=prefix_outputs(model,dx,replay_ids=dy['kind'])[0]
            error=max(float((direct[k]-step[k]).abs().max()) if direct[k].numel() else 0 for k in direct)
        if error>1e-4:raise ValueError('B teacher/step mismatch')
        teacher_parity.append(dict(index=i,view=v,error=error))
    del model;gc.collect();torch.mps.empty_cache()
    torch.manual_seed(plan['seed']);candidate=NoMarkHistory().to('mps')
    x,y,_=teacher_example(row(plan['health_indices'][0],'draft'));dx,dy=driver.device(x),driver.device(y)
    a=candidate.teacher(dx,dy['kind']);b=candidate.teacher(dx,(dy['kind']+1)%4)
    invariant=max(float((a[k]-b[k]).abs().max().detach()) for k in a)
    loss,_=mark_loss(candidate,dx,dy);loss.backward()
    embedding_zero=float(candidate.types.weight.grad[:4].abs().max())
    acoustic_grad={k:float(sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)) for k,m in [('projection',candidate.projection),('gru',candidate.gru),('kind',candidate.kind),('tail',candidate.tail_block)]}
    if invariant>1e-6 or embedding_zero!=0 or not all(np.isfinite(v) and v>0 for v in acoustic_grad.values()):raise ValueError('Gate/gradient semantics')
    driver.guard(plan,STARTED,STARTED,'preflight')
    write(FOLDER/'feedback-preflight.json',dict(status='completed',replays=replays,B_T1_free_nll=nll,
        teacher_parity=teacher_parity,label_invariance_max=invariant,non_BOS_embedding_gradient=embedding_zero,
        acoustic_gradients=acoustic_grad,wall_s=time.monotonic()-STARTED,optimizer_updates=0,cloud_calls=0))
    del candidate,a,b,loss;gc.collect();torch.mps.empty_cache()
    return panel,row


def validate_and_identify(plan,known):
    if driver.read(str((ART/'data-identity.json').relative_to(ROOT)))!=driver.read(str((OLD/'data-identity.json').relative_to(ROOT))):
        raise ValueError('Not exactly the same B training data and inputs')
    plan['pretraining_probe_receipts'].append(dict(status='completed',scope='new_implementation',evidence=str((FOLDER/'feedback-preflight.json').relative_to(ROOT))))
    return ORIGINAL_VALIDATE(plan,known)


def evaluate(model,plan,panel,row,prior,deadline):
    result=ORIGINAL_EVALUATE(model,plan,panel,row,prior,deadline)
    paths={}
    for condition in ('real','local_roll','song_foreign','audio_foreign'):
        trace=free_trace(model,plan,panel,row,condition,deadline)
        path=ART/'free-probability'/f'{condition}.json';write(path,trace);paths[condition]=str(path.relative_to(ROOT))
    result['oracle_free_probability_files']=paths
    lookup={v['index']:v for v in driver.read(plan['baseline']+'/quality-1000-draft.json')}
    shifts=driver.read('bench/rhythm-v2/c06-reset-unified-count-law-v1.json')['shifts_us']
    for track in ('machine','oracle'):
        records=[];traces=[]
        for i in plan['indices']['dev']:
            rr=row(i,'draft');heads=teacher_example(rr)[2] if track=='oracle' else sorted(v['onset_us'] for v in lookup[i]['prediction']['events'])
            out=greedy_marks(model,driver.device(anchor_inputs(rr,heads)),heads,rr['request'].duration_us,allowed_types=rr['request'].allowed_types,deadline=deadline)
            records.append(driver.record(out['events'],rr,i,panel.records[i]['component'],'draft',shifts));traces.append(dict(index=i,heads=heads,probabilities=out['probabilities']))
        write(ART/'greedy'/f'{track}-events.json',records);write(ART/'greedy'/f'{track}-trace.json',traces)
    return result


def run():
    global STARTED
    STARTED=time.monotonic()
    driver.PLAN,driver.ART,driver.FOLDER,driver.REPORT=PLAN,ART,FOLDER,REPORT
    driver.AnchorMarkChain=NoMarkHistory
    driver.load_data=load_and_preflight
    driver.validate_successor=validate_and_identify
    driver.evaluate=evaluate
    try:driver.run()
    except BaseException as exc:
        if REPORT.exists():
            r=driver.read(str(REPORT.relative_to(ROOT)))
            if r['status'] not in ('stopped_pending_mechanism_review','health_failed_pending_mechanism_review'):
                r.update(status='preflight_failed_pending_mechanism_review',error=f'{type(exc).__name__}: {exc}',process_wall_s=time.monotonic()-STARTED)
                write(REPORT,r);driver.state(status=r['status'],pid=None)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent MPS fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    def stop(*unused):raise KeyboardInterrupt('External stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    {'prepare':prepare,'run':run}[args.action]()

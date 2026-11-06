"""Zero-update numerical/wiring check, not selection or learning-health approval."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from beatfcn.model.anchor_mark_chain import mark_loss
from beatfcn.train.anchor_marks import teacher_example
from beatfcn.train.mark_prefix import prefix_outputs,self_prefix_loss
from scripts.run_c07_anchor_marks import fresh,device,digest_example
from scripts.run_local_evolution import load_data
from scripts.run_local_portfolio_scale import sha,write,weight_digest


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'bench/rhythm-v2/c08-prefix-implementation-preflight-v1.json'
ART=ROOT/'artifacts/c08-prefix-implementation-preflight-v1'
REPORT=ROOT/'reports/local-evolution-v3/c08-prefix-implementation-preflight.json'


def prepare():
    if PLAN.exists():raise FileExistsError('Frozen probe already exists')
    files=('beatfcn/train/mark_prefix.py','tests/test_mark_prefix.py','scripts/preflight_c08_prefix.py',
        'beatfcn/model/anchor_mark_chain.py','beatfcn/train/anchor_marks.py','scripts/run_c07_anchor_marks.py',
        'beatfcn/model/canvas_diffusion.py','scripts/run_local_evolution.py')
    base=json.loads((ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json').read_text())
    p=dict(id='c08-prefix-implementation-preflight-v1',scope='Implementation preparation before C08 synthesis;not a selected training recipe or a learning-health pass.',
        source_recommendation='C08 Grok NEXT-PREFIX-1;Sol independent review pending',
        predecessor_plan_sha256=sha(ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json'),
        predecessor_initial_sha256='52071f8f94c2d7745a555ac7c81f933d93cc508bdb63aae31d6c932959688316',
        predecessor_data_sha256=sha(ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1/data-identity.json'),
        seed=20261032,optimizer_updates=0,quality_updates=0,cloud_calls=0,wall_s=150,
        tests=['Same initialized613895parameters','CPU/MPS loss/gradient<=1e-4 on realized self-prefixes',
            'Actual458/459teacher-replay logits<=1e-4','BOS then sampled ids;changed id changes next embedding',
            'Stop discrete choice but retain hidden-state BPTT','One current quality input/target exactly matches frozen B tensor digest'],
        sampling='FixedPCG64 SeedSequence([20261032,0,health_index,0]);T1 categorical;no resampling if comparison fails.',
        learning_health='Not executed;must independently cover selected new training path before quality dispatch.',
        sources={f:sha(ROOT/f) for f in files})
    write(PLAN,p)
    for f in files:
        dst=ART/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')


def run():
    if REPORT.exists():raise FileExistsError('No automatic numerical rerun')
    started=time.monotonic();p=json.loads(PLAN.read_text())
    for f,h in p['sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Source changed '+f)
    data_path=ROOT/'artifacts/c07-anchor-conditioned-mark-chain-v1/data-identity.json'
    if sha(data_path)!=p['predecessor_data_sha256']:raise ValueError('Data receipt drift')
    base=json.loads((ROOT/'bench/rhythm-v2/c07-anchor-conditioned-mark-chain-v1.json').read_text());panel,row=load_data()
    cpu=fresh(base,'cpu').eval();gpu=fresh(base,'mps').eval();initial=weight_digest(cpu)
    if initial!=p['predecessor_initial_sha256'] or weight_digest(gpu)!=initial:raise ValueError('Initialization identity')
    index=base['health_indices'][0];x,y,heads=teacher_example(row(index,'draft'))
    u=np.random.default_rng(np.random.SeedSequence([20261032,0,index,0])).random(len(heads))
    co,trace=prefix_outputs(cpu,x,u,trace=True);go,gtrace=prefix_outputs(gpu,device(x),u,trace=True)
    if not torch.equal(trace['sampled_ids'],gtrace['sampled_ids']):raise ValueError('Different stochastic branch in parity control')
    cl,_=self_prefix_loss(cpu,x,y,u);gl,_=self_prefix_loss(gpu,device(x),device(y),u);cl.backward();gl.backward();torch.mps.synchronize()
    loss_error=abs(float(cl.detach())-float(gl.detach()));gradient=0.
    for a,b in zip(cpu.parameters(),gpu.parameters()):
        if (a.grad is None)!=(b.grad is None):raise ValueError('Gradient presence')
        if a.grad is not None:
            if not bool(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()):raise ValueError('Nonfinite gradients')
            gradient=max(gradient,float((a.grad-b.grad.cpu()).abs().max()))
    if max(loss_error,gradient)>1e-4:raise ValueError('Actual self-prefix CPU/MPS parity')
    previous=torch.cat(trace['previous_ids'],1)
    if previous[0,0]!=4 or not torch.equal(previous[:,1:],trace['sampled_ids'][:,:-1]):raise ValueError('Prefix wiring')
    different=(previous[:,1:]!=y['kind'][:,:-1]).nonzero()
    if not len(different):raise ValueError('No realized differing id to test new input path')
    j=int(different[0,1])+1
    embedding_delta=float((trace['previous_embeddings'][j]-cpu.types(y['kind'][:,j-1:j])).abs().max())
    if embedding_delta<=0:raise ValueError('Sampled id did not alter next vector')
    parity=[]
    with torch.no_grad():
        for idx in (458,459):
            q,t,_=teacher_example(row(idx,'draft'));a=cpu.teacher(q,t['kind']);b,_=prefix_outputs(cpu,q,replay_ids=t['kind'])
            error=float((a['kind']-b['kind']).abs().max());parity.append(dict(index=idx,max_logit_error=error))
            if error>1e-4:raise ValueError('Step teacher identity')
    spec=base['sample_stream'][0][0];sx,sy,sh=teacher_example(row(spec['index'],spec['view'],spec['field_index']))
    identity=json.loads(data_path.read_text())['rows'][str((spec['index'],spec['view'],spec['field_index']))]
    measured=digest_example(sx,sy,sh)
    if any(measured[k]!=identity[k] for k in measured):raise ValueError('Current quality example drift')
    memory=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory(),recommended=torch.mps.recommended_max_memory())
    if memory['driver']>min(8*2**30,.7*memory['recommended']):raise MemoryError('Numerical probe cap')
    if time.monotonic()-started>p['wall_s']:raise TimeoutError('Numerical probe cap')
    result=dict(id=p['id'],status='completed',passed=True,scope=p['scope'],optimizer_updates=0,quality_updates=0,cloud_calls=0,
        initial_weight_sha256=initial,parameters=sum(v.numel() for v in cpu.parameters()),
        cpu_loss=float(cl.detach()),mps_loss=float(gl.detach()),loss_error=loss_error,gradient_error=gradient,
        sampled_ids=trace['sampled_ids'].tolist(),previous_ids=previous.tolist(),different_ids=int(len(different)),next_embedding_delta=embedding_delta,
        teacher_identity=parity,quality_example_identity_matches=True,memory=memory,wall_s=time.monotonic()-started,
        plan_sha256=sha(PLAN),learning_health_passed=None,
        objective_note='Only sampled ids stopped;hidden BPTT retained. Supervised self-prefix semigradient surrogate,not full human-sequence maximum likelihood.')
    write(REPORT,result);print(json.dumps({k:v for k,v in result.items() if k not in ('sampled_ids','previous_ids')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));a=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent fallback prohibited')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.5)
    {'prepare':prepare,'run':run}[a.action]()

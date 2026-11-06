"""Evaluate one SHA-bound checkpoint; outer process enforces the full wall bound."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(args):
    import torch
    from beatfcn.model.full_event_model import FullEventConfig,FullEventModel
    from beatfcn.train.checkpoint_resume import file_sha
    from beatfcn.train.training_panel import build_training_panel
    from beatfcn.train.requirement_screen import run_requirement_screen,write_result
    started=time.monotonic();plan=json.loads(args.plan.read_text())
    if file_sha(args.checkpoint)!=args.checkpoint_sha:raise ValueError('Checkpoint SHA changed')
    if file_sha(args.protocol)!=args.protocol_sha:raise ValueError('Protocol SHA changed')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=FullEventConfig(**plan['config'])
    if saved['config']!=asdict(config) or saved['plan_sha256']!=file_sha(args.plan):
        raise ValueError('Checkpoint/config/training-plan identity mismatch')
    if saved['updates']%plan['accum_steps']:raise ValueError('Checkpoint is not an optimizer boundary')
    for key in ('beatfcn/model/full_event_model.py','beatfcn/model/full_event_generate.py'):
        expected=plan['code_sha256'].get(key)
        if expected and file_sha(args.root/key)!=expected:raise ValueError('Model/generator code changed: '+key)
    protocol=json.loads(args.protocol.read_text())
    if file_sha(args.root/plan['tape'])!=protocol['tape_sha256']:raise ValueError('Frozen data changed')
    panel=build_training_panel(args.root,plan)
    selected={panel.records[s['index']]['chart_index']for s in protocol['samples']}
    for index in selected:
        chart=panel.charts[index]
        for path_key,sha_key in [('target_path','target_sha256'),('token_path','token_sha256')]:
            if file_sha(args.root/chart[path_key])!=chart[sha_key]:raise ValueError('Frozen target bytes changed')
    model=FullEventModel(config).to(args.device)
    model.load_state_dict(saved['model'],strict=True)
    step=saved['updates'];del saved
    protocol['_file_sha256']=args.protocol_sha
    budget=args.max_wall_s-(time.monotonic()-started)-5
    if budget<5:raise TimeoutError('No original evaluation time left after identity checks')
    result=run_requirement_screen(model,panel,protocol,checkpoint=dict(path=str(args.checkpoint),sha256=args.checkpoint_sha,
        exposure_step=step,optimizer_updates=step//plan['accum_steps']),plan_sha256=file_sha(args.plan),
        output=args.out,device=args.device,max_wall_s=budget,max_tokens=args.max_tokens,batch_size=args.batch_size)
    print(json.dumps(dict(phase='requirement_checkpoint_complete',output=str(args.out),sha256=file_sha(args.out),summary=result['summary'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','protocol','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint-sha',required=True);p.add_argument('--protocol-sha',required=True)
    p.add_argument('--max-wall-s',type=float,default=150.);p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if not 0<a.max_wall_s<=150 or not 1<=a.batch_size<=8:p.error('Bound must be<=150s, batch size<=8')
    for value in (a.checkpoint_sha,a.protocol_sha):
        if len(value)!=64 or any(c not in'0123456789abcdef'for c in value):p.error('Expected SHA256 hex digest')
    if a.worker:worker(a);return
    process=subprocess.Popen([sys.executable,'-m','scripts.evaluate_requirement_panel',*sys.argv[1:],'--worker'],start_new_session=True)
    try:returncode=process.wait(timeout=a.max_wall_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait();raise SystemExit('Evaluation hard wall bound reached; raw partial file, if present, is not a pass')
    raise SystemExit(returncode)


if __name__=='__main__':main()

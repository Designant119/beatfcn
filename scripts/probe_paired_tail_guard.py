"""Audit a paired tail counterfactual on frozen complete outputs, without updates."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from beatfcn.train.conditional_rl import paired_tail_feasible, tail_prior_feasible
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from beatfcn.train.process_memory import physical_footprint
from scripts.calibrate_conditional_tail import tail_nll
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha,load_checkpoint_model
from scripts.run_output_policy_health import device_rows


def run(plan):
    p=json.loads(Path(plan).read_text());out=Path(p['out_dir']);out.mkdir(exist_ok=False)
    result=dict(completed=False,optimizer_updates=0,plan_sha256=file_sha(plan));resources=[]
    def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
    def check():
        gc.collect();torch.mps.empty_cache();torch.mps.synchronize();mem=physical_footprint();resources.append(mem)
        if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']:raise MemoryError('Footprint cap')
        if time.time()>p['deadline_unix']:raise TimeoutError('Probe deadline')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted':raise RuntimeError('Storage admission')
        for path,digest in p['identities'].items():
            if file_sha(path)!=digest:raise ValueError('Identity changed: '+path)
        torch.set_num_threads(4);model=load_checkpoint_model(p['checkpoint'],'mps');policy=OutputOnlyPolicy(model,p['checkpoint_sha256']);provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        items={r['chart_id']:r for r in p['rows']};limits={r['chart_id']:r['threshold']for r in json.loads(Path(p['calibration']).read_text())}
        inputs={};records=[]
        for case in p['cases']:
            cid=case['chart_id']
            if cid not in inputs:inputs[cid]=provider.build(items[cid],[.3,.4])
            inp,req,_=inputs[cid];events=case['events'];holds=sum(e['type']=='hold'for e in events)
            short=[dict(e,end_us=e['onset_us']+1000)if e['type']=='hold'else e for e in events]
            scores={}
            arms=[('original',events),('short',short)]
            if case['source']=='frozen_baseline':
                long=[dict(e,end_us=req.duration_us//1000*1000)if e['type']=='hold'else e for e in events]
                arms.append(('song_end',long))
            for name,ev in arms:
                try:sequence=provider.sequence(ev,req)
                except ValueError as exc:
                    if name!='song_end':raise
                    scores[name]=dict(grammar_rejected=True,reason=str(exc));continue
                x,sq=device_rows([dict(inputs=inp,sequence=sequence)]);h,logits=policy.features(x,sq)
                value=tail_nll(policy.prior(h),sq)[0];scores[name]=dict(tail_nll=value,grammar_rejected=False)
                del x,sq,h,logits;check()
            a=scores['original']['tail_nll'];b=scores['short']['tail_nll']
            row=dict(chart_id=cid,seed=case['seed'],source=case['source'],holds=holds,scores=scores,
                old_accept=tail_prior_feasible(a,holds,limits[cid]),paired_accept=paired_tail_feasible(a,b,holds,p['minimum_margin']),
                short_accept=paired_tail_feasible(b,b,holds,p['minimum_margin']),margin=None if not holds else b-a)
            if 'song_end'in scores:
                long=scores['song_end'];row['song_end_accept']=False if long['grammar_rejected']else paired_tail_feasible(long['tail_nll'],b,holds,p['minimum_margin'])
            records.append(row)
            with(out/'scores.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
        result.update(completed=True,cases=len(records),original_accepted=sum(r['paired_accept']for r in records),old_original_accepted=sum(r['old_accept']for r in records),short_accepted=sum(r['short_accept']for r in records if r['holds']),long_accepted=sum(r.get('song_end_accept',False)for r in records),long_cases=sum('song_end_accept'in r for r in records),scores_sha256=file_sha(out/'scores.jsonl'))
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        result['ended_unix']=time.time();save('resources.json',resources);save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

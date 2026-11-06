"""Frozen-prior tail calibration on every cached high-hold sample; zero updates."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from beatfcn.model.full_event_codec import FIELDS
from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from beatfcn.train.output_only_policy import OutputOnlyPolicy
from scripts.evolve_surgery import file_sha, load_checkpoint_model
from scripts.run_output_policy_health import device_rows
from scripts.check_experiment_storage import admission
from beatfcn.train.process_memory import physical_footprint


def tail_nll(logits, seq):
    losses = F.cross_entropy(logits.float().masked_fill(~seq['allowed'], -1e9).flatten(0, 1),
        seq['targets'].flatten(), reduction='none').view_as(seq['valid'])
    tail = torch.zeros_like(seq['valid'])
    for name in ('tail_block', 'tail_cell', 'tail_ms'):
        tail |= seq['fields'] == FIELDS.index(name)
    mask = tail & seq['valid'] & (seq['allowed'].sum(-1) > 1)
    return [float(loss[m].mean()) if m.any() else None for loss, m in zip(losses, mask)]


def run(plan):
    p = json.loads(Path(plan).read_text()); out = Path(p['out_dir']); out.mkdir(exist_ok=False)
    result = dict(passed=False, optimizer_updates=0, plan_sha256=file_sha(plan))
    rows = []
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    try:
        disk=admission(out,p['planned_peak_bytes'],p['reserve_bytes']);save('storage.json',disk)
        if disk['status']!='admitted': raise RuntimeError('Storage not admitted')
        for path, digest in p['identities'].items():
            if file_sha(path)!=digest: raise ValueError('Identity changed: '+path)
        torch.set_num_threads(4)
        model=load_checkpoint_model(p['checkpoint'],'mps');policy=OutputOnlyPolicy(model,p['checkpoint_sha256'])
        provider=FrozenLocalInputs(p['root'],p['source_manifest'],model)
        raw={r['chart_id']:r for r in map(json.loads,Path(p['raw_outputs']).read_text().splitlines()) if r['condition']=='native_high_hold'}
        for item in p['rows']:
            inputs,req,_=provider.build(item,[.3,.4]); scores=[]
            for seed,sample in enumerate(raw[item['chart_id']]['samples']):
                if not sample['complete']: raise ValueError('Incomplete calibration source')
                events=sample['events'];short=[dict(e,end_us=e['onset_us']+1000) if e['type']=='hold' else e for e in events]
                arms={}
                for name,ev in [('natural',events),('one_ms',short)]:
                    inp,seq=device_rows([dict(inputs=inputs,sequence=provider.sequence(ev,req))])
                    h,_=policy.features(inp,seq)
                    arms[name]=dict(tail_nll=tail_nll(policy.prior(h),seq)[0],events=ev)
                    del h,inp,seq
                    torch.mps.synchronize()
                    mem=physical_footprint()
                    if mem['physical_footprint_bytes']>p['physical_footprint_cap_bytes']: raise MemoryError('Physical footprint cap')
                    if time.time()>p['deadline_unix']: raise TimeoutError('Absolute calibration deadline')
                scores.append(dict(seed=seed,**arms))
            natural=[s['natural']['tail_nll'] for s in scores]
            short=[s['one_ms']['tail_nll'] for s in scores]
            threshold=max(n for n in natural if n is not None)+p['natural_margin']
            passed=all(v is not None and v>threshold for v in short)
            row=dict(chart_id=item['chart_id'],threshold=threshold,passed=passed,samples=scores)
            rows.append(row);save('calibration.json',rows)
            print(json.dumps(dict(chart_id=item['chart_id'],natural=natural,one_ms=short,threshold=threshold,passed=passed)),flush=True)
        result.update(passed=all(r['passed'] for r in rows),charts=len(rows),expected=len(p['rows']),
                      calibration_sha256=file_sha(out/'calibration.json'))
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}');raise
    finally:
        result['ended_unix']=time.time();save('result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)

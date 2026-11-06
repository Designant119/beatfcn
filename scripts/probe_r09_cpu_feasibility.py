"""Bounded actual-tensor and per-core CPU feasibility before any R09 optimization."""
import argparse
import json
from pathlib import Path
import resource
import time

import torch

from beatfcn.model.p1 import P1Config,P1Model
from beatfcn.train.probe import metrics_json,tensor_summary
from r09_local_data import LocalPairs,verify_files,file_sha,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    path=parser.parse_args().plan;plan=json.loads(path.read_text());verify_files(plan)
    torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(plan['seed'])
    started=time.perf_counter();data=LocalPairs(plan)
    result=dict(complete=False,passed=False,optimizer_updates=0,plan_sha256=file_sha(path),
        data_health=[],core_metrics=[],initial_input_isolation=[],gradients={},cloud_compute_started=False)
    def save(phase):
        result.update(phase=phase,wall_s=time.perf_counter()-started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
        (ROOT/plan['result']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(phase=phase,wall_s=result['wall_s'])),flush=True)
    save('start')
    try:
        ck=torch.load(ROOT/plan['checkpoint_file'],map_location='cpu',weights_only=False)
        model=P1Model(P1Config(**ck['config'])).eval();model.load_state_dict(ck['model'],strict=True)
        del ck
        for pair in plan['probe_pairs']:
            inputs=[data.batch([item],initial=True) for item in pair]
            compared=[]
            if inputs[0].keys()!=inputs[1].keys(): raise ValueError('initial tensor inventories differ')
            for key,left in inputs[0].items():
                right=inputs[1][key]
                if key=='controls': continue
                if torch.is_tensor(left):
                    if not torch.equal(left,right): raise ValueError(f'initial actual tensor differs: {key}')
                    compared.append(key)
                elif left!=right: raise ValueError(f'initial metadata differs: {key}')
            if torch.equal(inputs[0]['controls'],inputs[1]['controls']): raise ValueError('condition contrast absent')
            result['initial_input_isolation'].append(dict(pair_id=pair[0]['pair_id'],
                field=pair[0]['field'],compared_actual_tensors=compared,passed=True))
            del inputs
        elapsed=0.
        for item in [i for pair in plan['probe_pairs'] for i in pair]:
            bt=data.batch([item]);health={k:tensor_summary(v) for k,v in bt.items() if torch.is_tensor(v)}
            if any(v['nan'] or v['posinf'] or v['neginf'] for v in health.values()): raise ValueError('nonfinite real tensor')
            result['data_health'].append(dict(item=item,tensors=health))
            t0=time.perf_counter();loss,metrics=model(bt)
            if not torch.isfinite(loss) or int(metrics['_illegal']): raise ValueError('invalid supervised target/loss')
            (loss/4).backward();seconds=time.perf_counter()-t0;elapsed+=seconds
            result['core_metrics'].append(dict(item=item,forward_backward_s=seconds,**metrics_json(loss,metrics)))
            del bt,loss,metrics
            save('real_core_checked')
        for name,p in model.named_parameters():
            finite=p.grad is None or bool(torch.isfinite(p.grad).all())
            if not finite: raise ValueError(f'nonfinite gradient: {name}')
            result['gradients'][name]=dict(present=p.grad is not None,finite=finite)
        result.update(fixed_four_core_step_s=elapsed,
            projected_100_step_probe_s=elapsed*100*plan['timing_safety_factor']+plan['post_probe_allowance_s'],
            scope='Four actual training cores, single-field positive contrasts. No optimizer step, no CUDA claim.')
        result['passed']=result['projected_100_step_probe_s']<=plan['maximum_projected_fit_s']
        result['complete']=True;save('complete')
        if not result['passed']: raise SystemExit('CPU probe projection exceeds frozen limit')
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}';save('failed');raise


if __name__=='__main__': main()

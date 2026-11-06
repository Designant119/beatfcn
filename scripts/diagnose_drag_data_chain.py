"""One posthoc frozen-state positive/negative control; zero optimizer updates."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_drag_data_probe as probe
from scripts.analyze_drag_data_probe import proper_summary
from scripts.run_local_evolution import load_data
from beatfcn.train.drag_density_filter import DragFilteredPanel
from beatfcn.train.masked_native_events import empty_state
from beatfcn.train.managed_mps import ManagedMPS


def main():
    art=probe.ART;plan_path=art/'posthoc-chain-controls-plan.json'
    if plan_path.exists():raise FileExistsError('Single frozen diagnostic only')
    probe.runtime();plan=probe.read(probe.PLAN);started=time.monotonic()
    registration=dict(id='data-drag-posthoc-chain-controls-v1',selection='All original32draft windows;no chosen outcomes',
        conditions=['true odd-slot events with even slots masked','all-mask with ALL raw audio streams zeroed'],
        held_fixed='Quality128weights/public request/support/prior and target anchors;no training/no new free generation',
        purpose='Positive control for whether true event context helps;audio-zero is OOD dependence check,not emotion or musical understanding',
        wall_s=120,optimizer_updates=0,cloud_calls=0,source_sha256={n:probe.sha(probe.ROOT/n) for n in ['scripts/diagnose_drag_data_chain.py','scripts/run_drag_data_probe.py']},
        weights={a:probe.sha(art/f'{a}-quality-0128.pt') for a in ('raw','filtered')})
    probe.write(plan_path,registration)
    panel,row=load_data();filtered=DragFilteredPanel(panel);guard=ManagedMPS(probe.ROOT,art,120);result={}
    try:
        for arm in ('raw','filtered'):
            net=probe.fresh(plan);net.load_state_dict(torch.load(art/f'{arm}-quality-0128.pt',map_location='cpu',weights_only=False)['model']);net.eval();digest=probe.weight_digest(net)
            records=[]
            with torch.no_grad():
                for i in plan['eval_indices']:
                    guard.check('posthoc-chain-'+arm);rr,x,t=probe.batch(panel,row,filtered,i,arm)
                    selected=np.arange(256)%2==0;teacher={k:v.copy() for k,v in t['state'].items()};teacher['status'][selected]=0
                    for k in ('head','kind','end'):teacher[k][selected]=0
                    real=probe.proper(net,x,rr,t,plan,teacher,selected);guard.clear()
                    zero=probe.proper(net,probe.original.zero_audio(x),rr,t,plan,empty_state(),np.ones(256,bool))
                    records.append(dict(index=i,component=panel.records[i]['component'],true_context=real,audio_zero=zero))
                    probe.write(art/f'{arm}-posthoc-chain-controls.json',records);del rr,x,t,teacher;guard.clear()
            if digest!=probe.weight_digest(net):raise ValueError('Diagnostic changed weights')
            result[arm]=dict(true_context=proper_summary(records,'true_context',plan),audio_zero=proper_summary(records,'audio_zero',plan),weight_digest_unchanged=True)
            del net;torch.mps.empty_cache()
        result.update(status='completed',optimizer_updates=0,new_free_outcomes=0,cloud_calls=0)
    except BaseException as exc:
        result.update(status='failed',failure=dict(type=type(exc).__name__,error=str(exc)));raise
    finally:
        result['runtime']=guard.close();result['wall_s']=time.monotonic()-started;probe.write(probe.ROOT/'reports/local-integration-v1/data-drag-density-posthoc-chain-v1.json',result)
    print(json.dumps({a:dict(true_context_type=v['true_context']['type_nll_micro'],zero_audio_type=v['audio_zero']['type_nll_micro'],
        true_context_head=v['true_context']['head_nll'],true_context_tail=v['true_context']['hold_tail']['model_nll']) for a,v in result.items() if a in ('raw','filtered')}))


if __name__=='__main__':main()

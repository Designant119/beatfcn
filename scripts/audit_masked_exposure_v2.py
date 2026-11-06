"""Exact selected tuple exposure under the frozen mask stream;no NN calls."""
from collections import Counter
import json
from pathlib import Path

from beatfcn.train.event_density_drag_filter import EventDensityFilteredPanel
from beatfcn.train.masked_native_events import KINDS,empty_state,corruption,stable_rng
from scripts.run_local_evolution import load_data

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'artifacts/data-drag-density-probe-v2/masked-exposure-audit.json'
    if path.exists():raise FileExistsError('Preserve exposure audit')
    plan=json.loads((ROOT/'bench/rhythm-v2/data-drag-density-probe-v2.json').read_text());panel,_=load_data();filtered=EventDensityFilteredPanel(panel);result={}
    for arm in ('raw','filtered'):
        selected_counts=Counter();weighted=Counter();allcounts=Counter();holds_batches=0
        for step,batch in enumerate(plan['stream'],1):
            holds=0
            for micro,index in enumerate(batch):
                events=sorted(panel.targets(index) if arm=='raw' else filtered.targets(index),key=lambda e:(e.onset_us,KINDS.index(e.type),e.end_us))
                clean=empty_state();clean['status'][:]=1
                for slot,e in enumerate(events):
                    clean['status'][slot]=2;clean['head'][slot]=e.onset_us;clean['kind'][slot]=KINDS.index(e.type);clean['end'][slot]=e.end_us
                _,selected,rate=corruption(clean,stable_rng(plan['seed'],step,micro,'mask'))
                allcounts.update(e.type for e in events)
                for slot,e in enumerate(events):
                    if selected[slot]:selected_counts[e.type]+=1;weighted[e.type]+=1/rate;holds+=e.type=='hold'
            holds_batches+=holds>0
        result[arm]=dict(total_hidden_targets=dict(allcounts),actually_selected_tuple_targets=dict(selected_counts),
            IPW_weighted_selected_type_counts=dict(weighted),updates_with_selected_hold=holds_batches,
            note='Counts before loss/256;canonical slot assignment changes after deletion,so identical masks need not select the same surviving hold')
    result.update(neural_forwards=0,optimizer_updates=0,cloud_calls=0)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

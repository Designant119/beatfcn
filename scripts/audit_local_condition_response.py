"""Fixed external7field/combination requests on frozen SSD1000, CPU0updates."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.events.controls import BAND_EDGES,MODE_FIELDS
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import CONTROL_FIELDS,training_labels
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write


ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'bench/rhythm-v2/local-condition-response-v1.json'
REPORT=ROOT/'reports/local_condition_response_v1.json';ART=ROOT/'artifacts/local-condition-response-v1'


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve condition audit')
    torch.set_num_threads(2)
    parent_path=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(parent_path.read_text())
    support=json.loads((ROOT/'reports/local_portfolio_condition_support_v1.json').read_text());requests={}
    for field in MODE_FIELDS['phira']:
        bands=sorted(json.loads(s) for s in support['fields'][field]['supported_bands'])
        if len(bands)<2:raise ValueError('Train-only support not sufficient')
        requests[field]=[bands[0],bands[-1]]
    combos=[{'density_notes_per_second':[2.,3.],'hold_head_ratio':requests['hold_head_ratio'][1]},
            {'density_notes_per_second':[6.,8.],'chord_ratio':requests['chord_ratio'][1]}]
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt'
    plan=dict(id='local-condition-response-v1',model='frozenSSD1000',weight_sha256=sha(weight),indices=parent['indices']['dev'][::8],
        single_fields=requests,combinations=combos,support_sha256=sha(ROOT/'reports/local_portfolio_condition_support_v1.json'),
        wall_s=600,optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False,
        interpretation='All request bands frozen from TRAIN support only. Counterfactual hidden-chart F1 is not treated as a unique correctness target. Response and band attainment only.')
    write(PLAN,plan);started=time.monotonic()
    model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval()
    saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(parent_path):raise ValueError('Weight identity')
    model.load_state_dict(saved['model']);panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    specs=[(field,side,{field:band}) for field,bands in requests.items() for side,band in enumerate(bands)]
    specs += [('combination',i,bands) for i,bands in enumerate(combos)]
    records=[];report=dict(status='evaluating',optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False,processed=0)
    write(REPORT,report)
    try:
        for index in plan['indices']:
            for field,side,bands in specs:
                if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Condition response wall')
                e=row(index,'draft');inp,_=relative_batch([e],'cpu')
                for name,band in bands.items():
                    i=CONTROL_FIELDS.index(name);scale=BAND_EDGES[name][-1]
                    inp['controls'][0,i,:3]=torch.tensor([band[0]/scale,band[1]/scale,1.])
                pred=generate_relative(model,inp,e['request'],deadline=started+plan['wall_s'])
                events=[RhythmEvent(**v) for v in pred['events']]
                _,raw=training_labels('phira',canonical(events,'phira'),(e['request'].start_us,e['request'].end_us))
                hits={name:bool(raw[name] is not None and band[0]<=raw[name] and (raw[name]<band[1] or raw[name]==band[1]==1.)) for name,band in bands.items()}
                records.append(dict(index=index,component=panel.records[index]['component'],field=field,side=side,requested=bands,realized=raw,in_band=hits,prediction=pred))
                write(ART/'predictions.json',records);report['processed']=len(records);write(REPORT,report)
        summary={}
        for field in requests:
            pairs=[]
            for index in plan['indices']:
                pair=[v for v in records if v['index']==index and v['field']==field]
                low,high=[v['realized'][field] for v in pair]
                pairs.append(dict(index=index,low=low,high=high,positive_direction=low is not None and high is not None and high>low,both_in_band=all(v['in_band'][field] for v in pair)))
            summary[field]=dict(pairs=pairs,positive=sum(v['positive_direction'] for v in pairs),both_in_band=sum(v['both_in_band'] for v in pairs))
        report.update(status='completed',wall_s=time.monotonic()-started,results=summary,complete=sum(v['prediction']['complete'] for v in records),attempted=len(records))
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}');write(REPORT,report);raise


if __name__=='__main__':main()

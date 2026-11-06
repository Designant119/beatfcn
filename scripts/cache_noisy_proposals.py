"""Freeze and generate real SSD1000 proposals for the noisy-attribute route."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.noisy_proposals import proposal_cache_key
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write


ROOT=Path(__file__).resolve().parents[1];IDENT='local-noisy-proposal-cache-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_noisy_proposal_cache_v1.json';ART=ROOT/'artifacts/local-noisy-proposal-v1'


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve exact proposal cache')
    torch.set_num_threads(2)
    parent_path=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(parent_path.read_text())
    cohort=json.loads((ROOT/'bench/rhythm-v2/local-portfolio-scale-v1.json').read_text())
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt'
    specs=[dict(**s,masked=False,bands=None) for pair in parent['sample_stream'][:1000] for s in pair]
    specs += [dict(index=i,view=v,field_index=0,masked=False,bands=None) for i,v in zip(cohort['health_indices'],('draft','single_condition','random_groups','middle'))]
    specs.append(dict(index=cohort['empty_health_index'],view='draft',field_index=0,masked=False,bands=None))
    specs += [dict(index=i,view='draft',field_index=0,masked=False,bands=None) for i in parent['indices']['dev']+parent['indices']['empty_dev']]
    for index in parent['indices']['dev'][::2]:
        specs += [dict(index=index,view='middle',field_index=0,masked=mask,bands=None) for mask in (False,True)]
    for index in parent['indices']['dev'][::8]:
        specs += [dict(index=index,view='draft',field_index=0,masked=False,bands=band) for band in ((2.,3.),(6.,8.))]
    plan=dict(id=IDENT,checkpoint=str(weight.relative_to(ROOT)),checkpoint_sha256=sha(weight),parent_plan_sha256=sha(parent_path),
        specs=specs,wall_s=1800,device='cpu',optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False,
        role='Real free proposals only. Later refiner may train with disclosed jitter/background augmentation, but validation uses only frozenSSD proposals. No threshold/seed/checkpoint search.',
        source_sha256={n:sha(ROOT/n) for n in ('scripts/cache_noisy_proposals.py','beatfcn/train/noisy_proposals.py','beatfcn/model/local_mamba_canvas.py','beatfcn/train/relative_canvas.py')})
    write(PLAN,plan);started=time.monotonic()
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval();saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(parent_path):raise ValueError('SSD1000 identity')
    model.load_state_dict(saved['model'])
    report=dict(status='generating',optimizer_updates=0,processed=0,unique=0,files={},spec_keys=[],failures=[],cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report)
    try:
        for spec in specs:
            if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Frozen proposal preparation1800s')
            band=tuple(spec['bands']) if spec['bands'] is not None else None
            e=explicit_row(row(spec['index'],spec['view'],spec['field_index'],band));key=proposal_cache_key(e,spec['masked'])
            report['spec_keys'].append(key);report['processed']+=1
            if key not in report['files']:
                inp,_=relative_batch([e],'cpu',mask_reference=spec['masked'])
                pred=generate_relative(model,inp,e['request'],deadline=started+plan['wall_s'])
                path=ART/'proposals'/f'{key}.json'
                write(path,dict(key=key,index=spec['index'],view=spec['view'],request=asdict(e['request']),masked=spec['masked'],prediction=pred))
                report['files'][key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path));report['unique']+=1
                if not pred['complete']:report['failures'].append(dict(key=key,failure=pred['failure']))
            if report['processed']%25==0:
                report['wall_s']=time.monotonic()-started;write(REPORT,report);print(json.dumps(dict(processed=report['processed'],unique=report['unique'],wall_s=report['wall_s'])),flush=True)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}');write(REPORT,report);raise


if __name__=='__main__':main()

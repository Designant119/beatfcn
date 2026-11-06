"""Frozen SSD1000 acoustic-path dependence, CPU zero updates.

Feature content interventions hold clocks, masks, controls and references fixed.
They are OOD sensitivity tests, not claims about training without pretraining.
Actual waveform translation is a separate re-extraction contract.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.model.local_mamba_canvas import MambaCanvas
from beatfcn.model.sample import match_f1
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.relative_canvas import relative_batch
from scripts.audit_mamba_readout import decode
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-ssd-acoustic-routes-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_ssd_acoustic_routes_v1.json';ART=ROOT/f'artifacts/{IDENT}'


@torch.inference_mode()
def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve acoustic audit')
    torch.set_num_threads(2)
    oldpath=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(oldpath.read_text())
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt';dev=parent['indices']['dev']
    variants=['correct','remove_mel','remove_local','remove_song','remove_mert','remove_all','swap_all']
    plan=dict(id=IDENT,checkpoint_sha256=sha(weight),indices=dev,variants=variants,wall_s=900,optimizer_updates=0,device='cpu',
        donors={str(i):dev[(j+2)%len(dev)] for j,i in enumerate(dev)},shifts_us=parent['shifts_us'],
        interventions='Zero selected acoustic content only; swap local/song/mel together using the next frozen component. Preserve request times, masks, metadata, controls, reference and coordinate origin. Incoming holds supplied only when visible.',
        scope='OOD conditional dependence, not a matched trained frontend comparison, real waveform translation, or emotion evidence.',cloud_compute_calls=0)
    write(PLAN,plan);panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval()
    saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(oldpath):raise ValueError('Frozen weights changed')
    model.load_state_dict(saved['model']);del saved
    report=dict(id=IDENT,status='running',optimizer_updates=0,processed=0,cloud_compute_calls=0);write(REPORT,report)
    records={k:[] for k in variants};started=time.monotonic()
    originals={r['index']:r for r in json.loads((ROOT/'artifacts/local-mamba-readout-v1/draft-expected_tail_rank.json').read_text())}
    try:
        for index in dev:
            e=explicit_row(row(index,'draft'));inp,tgt=relative_batch([e],'cpu')
            donor=plan['donors'][str(index)];other,_=relative_batch([explicit_row(row(donor,'draft'))],'cpu')
            if panel.records[index]['component']==panel.records[donor]['component']:raise ValueError('Donor must be another component')
            for variant in variants:
                if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Acoustic audit wall')
                altered=dict(inp)
                keys={'remove_mel':['mel'],'remove_local':['local'],'remove_song':['song'],
                    'remove_mert':['local','song'],'remove_all':['local','song','mel']}.get(variant,[])
                for key in keys:altered[key]=torch.zeros_like(inp[key])
                if variant=='swap_all':altered.update({key:other[key] for key in ('local','song','mel')})
                out={k:v.numpy() for k,v in model(altered).items()}
                pred,detail=decode(out,altered,e['request'],'expected_tail_rank',parent['seed']+17*index)
                if variant=='correct' and [asdict(v) for v in pred]!=originals[index]['prediction']['events']:raise ValueError('Explicit-input baseline differs from frozen reference')
                count=out['count'][0].astype(np.float64);count-=count.max(-1,keepdims=True);prob=np.exp(count);prob/=prob.sum(-1,keepdims=True)
                cp=ART/'count-probabilities'/f'{variant}-{index}.npy';cp.parent.mkdir(parents=True,exist_ok=True);np.save(cp,prob.astype(np.float32))
                start=e['request'].start_us;width=e['request'].end_us-start
                shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
                records[variant].append(dict(index=index,component=panel.records[index]['component'],audio_donor_index=donor if variant=='swap_all' else None,
                    request=asdict(e['request']),prediction=dict(complete=True,events=[asdict(v) for v in pred],count_decision=detail),
                    target=[asdict(v) for v in e['targets']],scores=score(pred,e['targets']),shifted_onset10=shifts))
                write(ART/f'{variant}.json',records[variant])
            report['processed']+=1;write(REPORT,report)
        report.update(status='completed',results={k:metrics(v) for k,v in records.items()},wall_s=time.monotonic()-started,
            checkpoint_unchanged=sha(weight)==plan['checkpoint_sha256']);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',failure=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':main()

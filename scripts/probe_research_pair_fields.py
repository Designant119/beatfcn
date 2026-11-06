"""Same-panel field diagnostics for allocation and auxiliary-learning pairs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from beatfcn.model.local_typed_canvas import canvas_loss
from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.relative_canvas import relative_batch, generate_relative
from scripts.diagnose_whole_song_arrangement import first_components
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_music_research import make_model
from scripts.run_local_portfolio_scale import row_factory, write
from scripts.run_local_typed_canvas import score


ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'reports/local_research_pair_field_diagnostics_v2.json'
    planpath=ROOT/'bench/rhythm-v2/local-research-pair-field-diagnostics-v2.json'
    if target.exists() or planpath.exists():raise FileExistsError('No automatic diagnostic repeat')
    torch.set_num_threads(2)
    plans={kind:json.loads((ROOT/f'bench/rhythm-v2/local-music-{kind}-pair-v1.json').read_text()) for kind in ('depth','ssl')}
    panel,store=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));factory=row_factory(panel,store)
    train=first_components(panel,plans['depth']['indices']['train64'],8);dev=first_components(panel,plans['depth']['indices']['dev'],16)
    plan=dict(id='local-research-pair-field-diagnostics-v2',created_at=datetime.now(timezone.utc).isoformat(),
        train=train,dev=dev,arms={k:v['arms'] for k,v in plans.items()},optimizer_updates=0,device='cpu',threads=2,
        wall_s=900,interpretation='Same draft field-loss panels separate supervised fit/heldout effects from hard count decisions. Counterfactual removes only learned temporal trunk in-memory, keeps inputs/heads/weights. Not a fresh-training depth ablation or new quality endpoint.',cloud_compute_calls=0)
    write(planpath,plan);result=dict(status='running',arms={},optimizer_updates=0,cloud_compute_calls=0);started=time.monotonic()
    try:
        for kind,p in plans.items():
            for arm in p['arms']:
                model=make_model(p,arm,'cpu').eval();path=ROOT/f'artifacts/local-music-{kind}-pair-v1/{arm}/quality-1000.pt'
                saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
                rows=[]
                with torch.no_grad():
                    for split,ids in [('train',train),('dev',dev)]:
                        for i in ids:
                            if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Field diagnostic wall')
                            e=explicit_row(factory(i,'draft'));inp,tgt=relative_batch([e],'cpu');inp={k:v for k,v in inp.items() if not k.startswith(('local','song'))}
                            loss,parts=canvas_loss(model(inp),tgt,inp)
                            row=dict(index=i,split=split,loss=float(loss),parts={k:float(v) for k,v in parts.items()})
                            if split=='dev' and kind=='depth':
                                pred=generate_relative(model,inp,e['request']);from beatfcn.model.event_canvas import RhythmEvent
                                row['scores']=score([RhythmEvent(**v) for v in pred['events']],e['targets'])
                                trunk=model.temporal
                                try:
                                    model.temporal=nn.ModuleList();disabled,dp=canvas_loss(model(inp),tgt,inp)
                                    other=generate_relative(model,inp,e['request'])
                                finally:model.temporal=trunk
                                row['without_temporal']=dict(loss=float(disabled),parts={k:float(v) for k,v in dp.items()},
                                    scores=score([RhythmEvent(**v) for v in other['events']],e['targets']))
                            rows.append(row)
                result['arms'][kind+'-'+arm]=dict(weight_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=rows)
                write(target,result);print(kind,arm,'done',flush=True)
        result.update(status='completed',wall_s=time.monotonic()-started);write(target,result)
    except BaseException as exc:
        result.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(target,result);raise


if __name__=='__main__':main()

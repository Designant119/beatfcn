"""Post-terminal probability preservation, zero updates and no new selection."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.train.canonical_probability import explicit_row
from beatfcn.train.whole_song_mel import WholeMelStore
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write
from scripts.run_whole_song_arrangement import fresh


ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/local-whole-song-arrangement-v1'
REPORT=ROOT/'reports/local_whole_song_probability_archive_v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve probability extraction')
    torch.set_num_threads(2)
    plan_path=ROOT/'bench/rhythm-v2/local-whole-song-arrangement-v1.json'
    plan=json.loads(plan_path.read_text());parent=json.loads((ROOT/'reports/local_whole_song_arrangement_v1.json').read_text())
    weight=ART/'quality-1000.pt'
    if parent['status']!='completed' or parent['final_weight_sha256']!=sha(weight):raise ValueError('Require exact completed1000checkpoint')
    started=time.monotonic();model=fresh(plan,'cpu').eval();saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=1000 or saved['plan_sha256']!=sha(plan_path):raise ValueError('Weight/plan identity')
    model.load_state_dict(saved['model']);panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()))
    row=row_factory(panel,features);store=WholeMelStore(ROOT,panel.features.records,ART/'whole-mel')
    report=dict(status='extracting',checkpoint_sha256=sha(weight),optimizer_updates=0,processed=0,files={},role='Post-terminal recording only; original primary/allseeds/screen unchanged',cloud_compute_calls=0,confirmation_scored=False)
    write(REPORT,report)
    try:
        for remote in (False,True):
            for index in plan['indices']['dev']:
                if time.monotonic()-started>600:raise TimeoutError('Probability recording600s')
                e=explicit_row(row(index,'draft'));inp,_=store.batch(e,panel.records[index]['audio_sha256'],'cpu',remote_permutation=remote)
                with torch.no_grad():out=model(inp)
                path=ART/'probabilities'/f'{"remote" if remote else "real"}-{index}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in out.items()})
                report['files'][str(path.relative_to(ROOT))]=sha(path);report['processed']+=1;write(REPORT,report)
        report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',wall_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}');write(REPORT,report);raise


if __name__=='__main__':main()

"""Zero-update count-distribution versus hard-readout audit, CPU only."""
import argparse
import json

import torch

from scripts.run_local_typed_canvas import PLAN, ART, panel_for, fresh, write
from beatfcn.train.local_bundle import make_row
from beatfcn.train.local_typed_canvas import canvas_batch
from beatfcn.model.local_typed_canvas import canvas_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('step',type=int);args=p.parse_args()
    plan=json.loads(PLAN.read_text())
    saved=torch.load(ART/f'quality-{args.step:04d}.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);model=fresh(plan);model.load_state_dict(saved['model']);model.eval()
    panel=panel_for(plan);rows=[]
    for split,indices in [('train',plan['health_indices']),('val',plan['indices']['val'][:4])]:
        for index in indices:
            row=make_row(panel,index);x,y=canvas_batch([row]);valid=x['canvas_valid'][...,None]
            with torch.no_grad():
                out=model(x);prob=out['count'].softmax(-1);loss,terms=canvas_loss(out,y,x)
                expected=((prob*torch.arange(17)).sum(-1)*valid).sum()
                hard=(out['count'].argmax(-1)*valid).sum()
                nonzero=prob[...,1:].sum(-1)*valid
            rows.append(dict(split=split,index=index,target=len(row['targets']),expected_events=float(expected),
                hard_argmax_events=int(hard),maximum_nonzero_probability=float(nonzero.max()),
                mean_nonzero_probability=float(nonzero.sum()/valid.sum()/4),loss=float(loss),
                parts={k:float(v) for k,v in terms.items()}))
    write(ART/f'count-readout-{args.step:04d}.json',rows)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()

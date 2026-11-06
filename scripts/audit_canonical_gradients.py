"""Zero-update shared-projection gradient diagnosis on the fixed A100state."""
import json
from pathlib import Path
import time

import torch

from beatfcn.model.canonical_event_package import canonical_package_loss
from beatfcn.train.canonical_probability import explicit_row,probability_batch
from scripts.run_canonical_package_experiment import fresh
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write


ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/local_canonical_gradient_audit_v1.json'


def main():
    if REPORT.exists():raise FileExistsError('Preserve gradient diagnosis')
    torch.set_num_threads(2)
    path=ROOT/'bench/rhythm-v2/local-canonical-package-v1.json';plan=json.loads(path.read_text())
    weight=ROOT/'artifacts/local-canonical-package-v1/health-0100.pt';saved=torch.load(weight,map_location='cpu',weights_only=False)
    if saved['updates']!=100 or saved['plan_sha256']!=sha(path):raise ValueError('Exact health100identity')
    model=fresh(plan,'cpu').eval();model.load_state_dict(saved['model'])
    panel,features=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,features)
    selected=['frame_read.weight','mel_projection.weight','local_projection.weight'];parameters=dict(model.named_parameters());pp=[parameters[n] for n in selected]
    report=dict(status='diagnosing',checkpoint_sha256=sha(weight),optimizer_updates=0,selected_projections=selected,
        scope='Selected shared projection gradients, not whole-backbone or free-generation evidence. No optimizer or training ablation.',rows=[],cloud_compute_calls=0)
    write(REPORT,report);started=time.monotonic()
    specs=list(zip(plan['health_indices'],('draft','single_condition','random_groups','middle')))+[(i,'draft') for i in plan['indices']['dev'][:4]]
    for index,view in specs:
        if time.monotonic()-started>600:raise TimeoutError('Gradient diagnosis600s')
        e=explicit_row(row(index,view));inp,tgt=probability_batch([e],'cpu');loss,parts=canonical_package_loss(model,inp,tgt,e['request']);vectors={}
        for name in ('count','onset','tail'):
            gradients=torch.autograd.grad(parts[name],pp,retain_graph=True,allow_unused=True)
            vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(pp,gradients)])
        total=vectors['onset']+vectors['tail'];count=vectors['count'];denominator=float(count.norm()*total.norm())
        report['rows'].append(dict(index=index,view=view,loss=float(loss.detach()),norms={k:float(v.norm()) for k,v in vectors.items()},
            attribute_to_count_norm=float(total.norm()/count.norm().clamp_min(1e-30)),
            count_attribute_cosine=float(count@total)/denominator if denominator else None))
        write(REPORT,report)
    report.update(status='completed',wall_s=time.monotonic()-started);write(REPORT,report)


if __name__=='__main__':main()

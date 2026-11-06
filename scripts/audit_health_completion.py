"""Read-only supplement to C13: full probabilities and historical Adam groups."""
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.train.field_integrity import validate,probability
from scripts.complete_frozen_health import structure
from scripts.run_mps_adamw_admission import fresh
from scripts.run_local_portfolio_scale import sha,write,weight_digest


ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/local-integration-v1/c13-health-probability-supplement.json'


def main():
    if DEST.exists():raise FileExistsError('Preserve original supplement')
    if torch.__version__!='2.14.0':raise ValueError('Use isolated registered runtime')
    torch.set_num_threads(4)
    art=ROOT/'artifacts/c13-frozen-health-completion-v1';health=art/'report.json'
    r=json.loads(health.read_text());plan=json.loads((ROOT/'bench/rhythm-v2/c13-frozen-health-completion-v1.json').read_text())
    parent=json.loads((ROOT/plan['parent_plan']).read_text());checkpoint=ROOT/plan['checkpoint']
    if sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('Checkpoint drift')
    s=torch.load(checkpoint,map_location='cpu',weights_only=True);model=fresh(parent,'cpu')
    audit=structure(model,s,parent,plan)
    if audit['status']!='structure_passed':raise ValueError(audit)
    groups=[]
    for group in s['optimizer']['param_groups']:
        expected=dict(lr=.002,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        for key,value in expected.items():
            if group[key]!=value:raise ValueError('Historical Adam hyperparameter '+key)
        groups.append({key:group[key] for key in expected})
    model.load_state_dict(s['model'],strict=True);digest=weight_digest(model)
    if digest!=r['weight_digest_before'] or digest!=r['weight_digest_after']:raise ValueError('Saved model digest differs')
    certs=[];identities={str(health.relative_to(ROOT)):sha(health),str(checkpoint.relative_to(ROOT)):sha(checkpoint)}
    for row in r['rows']:
        path=art/f'fields-{row["index"]}.json';f=json.loads(path.read_text())
        probabilities=np.asarray(row['count_probability'])
        if len(probabilities)!=len(row['prediction']['count_decision']):raise ValueError('Count length')
        for p in probabilities:probability(p)
        if probabilities.argmax(-1).tolist()!=row['prediction']['count_decision']:raise ValueError('Count mode changed')
        certs.append(dict(index=row['index'],count_cells=len(probabilities),**validate(f,row['prediction'])))
        identities[str(path.relative_to(ROOT))]=sha(path)
    write(DEST,dict(status='passed',health_receipt_sha256=sha(health),checkpoint_sha256=sha(checkpoint),
        identity=identities,optimizer_structure=audit,optimizer_groups=groups,model_digest=digest,
        field_certificates=certs,optimizer_updates=0,model_forward_calls=0,cloud_compute_calls=0,
        scope='Later supplemental assertion on existing raw health outcomes;original C12 failure and health receipt unchanged'))
    print(json.dumps(dict(status='passed',rows=len(certs),members=sum(c['member_count'] for c in certs),
        hold_paths=sum(c['machine_hold_paths'] for c in certs),maximum_diff=max(c['maximum_probability_diff'] for c in certs))))


if __name__=='__main__':main()

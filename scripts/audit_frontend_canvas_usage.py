"""Predeclared frozen-weight removal of pretrained audio, retaining common mel."""
import json
import time

import torch

from scripts.run_local_bundle import ROOT,write,sha
from scripts.run_frontend_canvas_comparison import setup,rows_for,models
from scripts.run_frontend_canvas_quality_v2 import PLAN,ART
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics
from beatfcn.train.relative_canvas import relative_batch,generate_relative
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.sample import match_f1


def main():
    p=json.loads(PLAN.read_text());plan=ROOT/'bench/rhythm-v2/frontend-investment-screen-v1.json'
    if 'secondary_diagnostic' not in json.loads(plan.read_text()):raise ValueError('Unregistered diagnostic')
    output=ART/'frontend-removed.json'
    if output.exists():raise FileExistsError('Preserve usage probe')
    identities={a:sha(ART/a/'quality-0900.pt') for a in ('pupu','mert')}
    write(ART/'frontend-usage-plan.json',dict(optimizer_updates=0,device='cpu',weight_sha256=identities,
        predeclared_screen_sha256=sha(plan),source_sha256=sha(__file__),wall_limit_s=300,
        operation='Set local_valid and song_valid to false together; mel/reference/conditions/native address unchanged; OOD removal diagnostic, not a training ablation'))
    torch.set_num_threads(2);panel,store=setup(p);made=models(p)
    for arm,m in made.items():
        s=torch.load(ART/arm/'quality-0900.pt',map_location='cpu',weights_only=False)
        if s['plan_sha256']!=sha(PLAN):raise ValueError('Checkpoint identity')
        m.load_state_dict(s['model']);m.eval()
    started=time.monotonic();deadline=started+300;rows={a:[] for a in made}
    for index in p['indices']['val']+p['new_val_indices']:
        if time.monotonic()>=deadline:raise TimeoutError('Usage probe wall')
        by_arm=rows_for(panel,store,[dict(index=index,view='draft')])
        for arm,model in made.items():
            row=by_arm[arm][0];x,_=relative_batch([row])
            masked={**x,'local_valid':torch.zeros_like(x['local_valid']),
                    'song_valid':torch.zeros_like(x['song_valid'])}
            pr=generate_relative(model,masked,row['request'],deadline=deadline)
            pred=[RhythmEvent(**e) for e in pr['events']];ss=score(pred,row['targets'])
            start=row['request'].start_us;width=row['request'].end_us-start
            shifts=[match_f1([start+(e.onset_us-start+d)%width for e in pred],
                [e.onset_us for e in row['targets']],10000)[3] for d in p['shifts_us']]
            rows[arm].append(dict(index=index,component=panel.records[index]['component'],
                prediction=pr,target=[dict(onset_us=e.onset_us,end_us=e.end_us,type=e.type,lane=e.lane) for e in row['targets']],
                scores=ss,shifted_onset10=shifts))
    write(output,rows)
    summary={a:metrics(v) for a,v in rows.items()}
    if any(sha(ART/a/'quality-0900.pt')!=s for a,s in identities.items()):raise ValueError('Weight changed')
    write(ART/'frontend-usage-summary.json',dict(optimizer_updates=0,wall_s=time.monotonic()-started,
        pretrained_audio_removed=summary,weights_unchanged=True,interpretation='Inference-time OOD removal; measures reliance, not causal superiority of a frontend or a trained-mel-only comparison.'))
    print(json.dumps({a:{k:v[k] for k in ['full_note_f1','onset10','onset50','density_ratio']} for a,v in summary.items()}),flush=True)


if __name__=='__main__':main()

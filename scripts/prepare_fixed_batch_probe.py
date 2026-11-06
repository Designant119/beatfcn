"""Freeze a tiny real training fixture before any charged diagnostic run."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.batch import MAX_NOTES
from beatfcn.train.data import selected_for_training, dup_only, chart_from_canon, model_shift_us, check_stacks


def main():
    manifest=ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    heldout={r['audio_sha256'] for r in json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows']}
    candidates={mode:[] for mode in ('mania_4k','phira')}
    rejected=[]
    for row in rows:
        if row.get('split')!='train' or not selected_for_training(row) or not (row.get('kept') or dup_only(row)):
            continue
        path=ROOT/'inputs/training_audit_20261023/canon'/row['source']/f'{row["native_id"]}.json'
        if not path.exists() or not 30<=row['audio_sec']<=200:
            continue
        try:
            doc=json.loads(path.read_text())
            assert doc['audio_sha256']==row['audio_sha256'] and row['audio_sha256'] not in heldout
            chart=chart_from_canon(doc,model_shift_us(row,True))
            if dup_only(row):check_stacks(chart.events)
            duration=round(row['audio_sec']*1e6)
            sample=build_request(chart,kind='from_scratch',seed=20261025,audio_id=row['audio_sha256'],
                                 audio_duration_us=duration,control_dropout=False,empty_overlay=None)
            if not sample.ok:raise ValueError('model view quarantined')
            for c0 in range(0,duration-F.CORE_US+1,F.CORE_US):
                events=[e for e in chart.events if c0<=e.onset_us<c0+F.CORE_US]
                types=Counter(e.type for e in events)
                required=('tap','hold') if chart.mode=='mania_4k' else ('tap','hold','drag','flick')
                if not 16<=len(events)<=56 or any(types[t]<2 for t in required):continue
                ex=F.build_core(sample,chart.events,c0,duration,stage_a_arm='legacy')
                if not ex.groups or F.check_targets(ex) or max(len(g['notes']) for g in ex.groups)>MAX_NOTES:continue
                record=dict(row=row,c0_us=c0,seed=20261025,kind='from_scratch',
                            canon_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            groups=ex.groups,types=dict(types),groups_count=len(ex.groups),
                            controls=sample.request['controls'],duration_us=duration)
                # Small enough to memorise quickly, all applicable types retained.
                rank=(abs(len(events)-32),row['audio_sec'],row['chart_id'],c0)
                candidates[chart.mode].append((rank,record))
        except (ValueError,F.SkipExample,AssertionError) as exc:
            rejected.append(dict(chart_id=row['chart_id'],reason=str(exc)))
    selected=[min(candidates[m],key=lambda p:p[0])[1] for m in candidates]
    protocol=dict(schema='beatfcn.fixed_batch_probe.v1',status='frozen_before_dispatch',
        purpose='Real input health and memorisation prerequisite; not validation or checkpoint promotion.',
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),rows=selected,
        selection='Two training audio songs, one per mode; all applicable types, 16–56 notes per core; no withheld benchmark audio.',
        eligible_window_counts={k:len(v) for k,v in candidates.items()},rejected_charts=rejected,
        checkpoint=dict(path='/data/runs/s1-v21/best.pt',sha256='64a7797142fee342491ea269e4a105261f7399be32cfc19650a97395966ed1b4',step=3500),
        probe=dict(steps=100,seed=20261025,lr=.0003,weight_decay=0.,clip=1.,model_mode='eval_with_grad',
                   dropout=False,ssl=False,offset_soft_targets=False,augmentation=False,
                   optimizer='fresh AdamW; diagnostic only, no production optimizer state',
                   logging_steps=[0,1,10,25,50,75,100],
                   supervised_gate=dict(final_loss_at_most=.1,reduction_at_least=.95,
                                        per_head_accuracy_at_least=.95,illegal_targets=0,finite_gradients=True),
                   generation_gate=dict(complete=True,legal=True,full_note_f1_at_least=.9),
                   no_checkpoint_saved=True))
    out=ROOT/'bench/rhythm-v2/fixed-batch-probe-v1.json'
    out.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,default=lambda value:value.tolist())+'\n')
    print(json.dumps(dict(fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                         cores=[dict(chart_id=x['row']['chart_id'],c0_us=x['c0_us'],types=x['types'],
                                     groups=x['groups_count']) for x in selected],
                         candidates=protocol['eligible_window_counts']),ensure_ascii=False))


if __name__=='__main__':main()

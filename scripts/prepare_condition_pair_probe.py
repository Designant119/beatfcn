"""Freeze supported same-audio pairs and check causal isolation before paid work."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from beatfcn.events.validate import validate_request
from beatfcn.audio.memory import SUMMARY_CELLS
from beatfcn.model import features as F
from beatfcn.model.batch import SongInfo
from beatfcn.train.data import chart_from_canon,model_shift_us,selected_for_training,dup_only
from beatfcn.train.condition_pairs import blank_context_excerpt,assert_same_generation_inputs


def main():
    manifest=ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    rows={r['chart_id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    selected=[(['osu:2086346','osu:2115530'],0),(['phira:45917','phira:13363'],64_000_000)]
    heldout={r['audio_sha256'] for r in json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())['rows']}
    pairs=[]
    for ids,c0 in selected:
        items=[];examples=[]
        for cid in ids:
            row=rows[cid];sha=row['audio_sha256'];duration=round(row['audio_sec']*F.SEC)
            if row['split']!='train' or not selected_for_training(row) or not (row.get('kept') or dup_only(row)) or sha in heldout:
                raise ValueError('pair includes unsupported or held-out chart')
            path=ROOT/'inputs/training_audit_20261023/canon'/row['source']/f'{row["native_id"]}.json'
            raw=path.read_bytes();doc=json.loads(raw)
            if doc['audio_sha256']!=sha:raise ValueError('audio identity differs')
            chart=chart_from_canon(doc,model_shift_us(row,True))
            sample=blank_context_excerpt(chart,c0,duration,sha,20261025)
            errors=validate_request(sample.request,scene_line_ids=set())
            if errors:raise ValueError(f'invalid excerpt request: {errors}')
            ex=F.build_core(sample,chart.events,c0,duration,stage_a_arm='legacy')
            if F.check_targets(ex):raise ValueError('illegal excerpt targets')
            examples.append(ex)
            items.append(dict(row=row,c0_us=c0,duration_us=duration,canon_sha256=hashlib.sha256(raw).hexdigest(),
                controls=sample.request['controls'],raw_controls=sample.raw_controls,groups=ex.groups,
                target_types=dict(Counter(e.type for e in sample.hidden)),target_count=len(sample.hidden)))
        if items[0]['row']['audio_sha256']!=items[1]['row']['audio_sha256']:raise ValueError('pair audio differs')
        if items[0]['duration_us']!=items[1]['duration_us']:raise ValueError('pair duration differs')
        # Shape-only metadata is sufficient for local isolation; actual cache checked remotely.
        cells=-(-duration//F.CELL_US)
        info=SongInfo(sha,cells,-(-cells//SUMMARY_CELLS))
        equality=assert_same_generation_inputs(*examples,info,(9,16,24),24,1024,1024)
        pairs.append(dict(pair_id=f'{items[0]["row"]["mode"]}:{c0}',items=items,input_isolation=equality))
    fixture=dict(schema='beatfcn.condition_pair_probe.v1',status='frozen_before_dispatch',
        purpose='Test joint condition-vector memorisation on real alternative charts; not generalisation or individual-field control.',
        seed=20261025,pairs=pairs,checkpoint=json.loads((ROOT/'bench/rhythm-v2/fixed-batch-probe-v1.json').read_text())['checkpoint'],
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        probe=dict(steps=100,lr=.0003,weight_decay=0.,clip=1.,dropout=False,ssl=False,augmentation=False,
            logging_steps=[0,1,10,25,50,75,100],
            supervised_gate=dict(final_loss_at_most=.1,reduction_at_least=.95,per_head_accuracy_at_least=.95),
            generation_gate=dict(full_note_f1_at_least=.9,matching_target_margin_at_least=.1,
                                 removed_controls_identical=True,all_requested_bands_met=True),
            checkpoint_saved=False))
    out=ROOT/'bench/rhythm-v2/condition-pair-probe-v1.json'
    out.write_text(json.dumps(fixture,ensure_ascii=False,indent=2,default=lambda v:v.tolist())+'\n')
    print(json.dumps(dict(fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        pairs=[dict(pair_id=p['pair_id'],changed=p['input_isolation']['changed_fields'],
            items=[dict(chart_id=i['row']['chart_id'],count=i['target_count'],types=i['target_types']) for i in p['items']]) for p in pairs]),ensure_ascii=False))


if __name__=='__main__':main()

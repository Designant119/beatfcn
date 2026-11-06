"""Recompute saved R24 event metrics and action replays before isolation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from beatfcn.model.action_state import ManiaAction,PhiraAction,replay
from scripts.acoustic_probe_metrics import event_metrics

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    target=ROOT/path
    if not target.exists():
        isolation=json.loads((ROOT/'reports/r24_rollback.json').read_text())
        archived=next(x['archive'] for x in isolation['paths'] if x['original']==path)
        target=ROOT/archived
    return sha256(target.read_bytes()).hexdigest()


def key(e):return (e['onset_us'],e['end_us'],e['type'],e['lane'])


def main():
    plan_path='bench/rhythm-v2/r24-state-backbone-paired100-v1.json'
    result_path='reports/r24_state_fit.json'
    plan=json.loads((ROOT/plan_path).read_text());result=json.loads((ROOT/result_path).read_text())
    assert result['plan_sha256']==sha(plan_path)
    for path,digest in plan['input_sha256'].items():assert sha(path)==digest,path
    panel=json.loads((ROOT/plan['manifest']).read_text());rows={r['id']:r for r in panel['rows']}
    evidence=[]
    for name,arm in result['arms'].items():
        assert arm['complete'] and arm['optimizer_updates']==100 and arm['training'][-1]['update']==100
        assert abs(sum(x['loss'] for x in arm['final_teacher']['fields'].values() if x['n'])-arm['final_teacher']['loss'])<1e-6
        for generation in arm['generation']:
            row=rows[generation['panel_id']]
            assert Counter(map(key,row['target_events']))==Counter(map(key,generation['truth']))
            metrics=event_metrics(generation['events'],row['target_events'],row['mode'])
            assert metrics==generation['metrics']
            if row['mode']=='mania_4k':
                actions=[ManiaAction(**a) for a in generation['actions']]
            else:
                actions=[PhiraAction(a['time_us'],tuple(a['starts']),tuple(tuple(p) for p in a['releases'])) for a in generation['actions']]
            decoded=replay(actions,row['mode'])
            assert Counter((e.onset_us,e.end_us,e.type,e.lane) for e in decoded.events)==Counter(map(key,generation['events']))
            true_actions={a['time_us']:a for a in row['target_actions']};pred_actions={a['time_us']:a for a in generation['actions']}
            differences=[dict(time_us=t,truth=true_actions.get(t),prediction=pred_actions.get(t)) for t in sorted(true_actions.keys()|pred_actions.keys()) if true_actions.get(t)!=pred_actions.get(t)]
            wanted=Counter(map(key,row['target_events']));actual=Counter(map(key,generation['events']))
            evidence.append(dict(arm=name,panel_id=row['id'],mode=row['mode'],audio=row['audio'],
                audio_sha256=row['audio_sha256'],source_clip_start_us=row['source_clip_start_us'],
                metrics=metrics,constraint_interventions=generation['constraint_interventions'],
                exact_missing=list((wanted-actual).elements()),exact_extra=list((actual-wanted).elements()),action_differences=differences))
    assert result['optimizer_updates']==200 and result['complete'] and not result['passed']
    audit=dict(verified=True,verified_at_unix=time.time(),frozen_inputs_verified=len(plan['input_sha256']),
        event_metrics_recomputed=4,action_streams_replayed=4,optimizer_updates=200,
        new_cloud_spend_usd=0,model_promotion=False,rows=evidence,
        source_sha256={result_path:sha(result_path),plan_path:sha(plan_path),plan['manifest']:sha(plan['manifest'])},
        limitation='Teacher losses checked for component-sum consistency only; no trained checkpoint or raw logits were saved. No claim of recomputed neural loss.')
    (ROOT/'reports/r24_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(verified=True,event_metrics_recomputed=4,action_streams_replayed=4,inputs_verified=len(plan['input_sha256']))))


if __name__=='__main__':main()

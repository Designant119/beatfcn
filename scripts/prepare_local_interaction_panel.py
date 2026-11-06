"""Freeze broader real pair support independently of any fitted diagnostic weights."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from beatfcn.train.local_condition_pairs import build_pair_catalog, paired_exposure
from beatfcn.train.local_request_panel import LocalRequestPanel
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def run():
    output = ROOT/'inputs/d83_condition_interaction/pairs-v1.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    tape='reports/d67_local_request_tape_v1.json'; features='reports/d67_local_feature_audit_v1.json'
    panel=LocalRequestPanel(ROOT,tape,features)
    pairs,checked=build_pair_catalog(panel)
    steps=paired_exposure(pairs,1000,seed=20261025)
    songs=defaultdict(set);counts=Counter();exposures=Counter();exposure_songs=defaultdict(set)
    for p in pairs:
        key=p['mode']+'/'+p['field'];counts[key]+=1;songs[key].add(p['component'])
    for step in steps:
        p=pairs[step['pair']];key=p['mode']+'/'+p['field'];exposures[key]+=1;exposure_songs[key].add(p['component'])
    sources=[tape,features,'inputs/d67_local_requests/initial100/learning-exposure-1000-v1.json',
        'beatfcn/train/local_condition_pairs.py','scripts/prepare_local_interaction_panel.py',
        'beatfcn/train/event_conditions.py','beatfcn/train/local_request_panel.py','beatfcn/events/controls.py']
    identities={p:h for c in panel.charts for p,h in [(c['target_path'],c['target_sha256']),(c['token_path'],c['token_sha256'])]}
    if any(sha(ROOT/p)!=h for p,h in identities.items()):
        raise ValueError('Canonical target/token identity changed')
    result=dict(id='D83-condition-interaction-panel-v1',complete=True,selection_split='train',
        tape=tape,features=features,seed=20261025,catalog=pairs,steps=steps,checked_request_count=len(checked),
        pairs_by_mode_field=dict(counts),songs_by_mode_field={k:len(v) for k,v in songs.items()},
        auxiliary_pair_exposures_by_mode_field=dict(exposures),exposed_songs_by_mode_field={k:len(v) for k,v in exposure_songs.items()},
        pair_requests_per_update=2,optimizer_updates=0,new_cloud_spend_usd=0,wall_s=time.monotonic()-started,
        inputs_sha256={p:sha(ROOT/p) for p in sources},canonical_identity=identities,
        scope='Original CE exposure remains fixed. This adds one real pair per update with 75:25 Phira/mania and balanced supported fields/songs. Training labels never become hidden validation conditions.')
    write_json(output,result)
    write_json(ROOT/'reports/d83_condition_pair_support_v1.json',{k:v for k,v in result.items() if k not in ('catalog','steps','canonical_identity')})
    print(json.dumps({k:v for k,v in result.items() if k not in ('catalog','steps','canonical_identity','inputs_sha256')},ensure_ascii=False))


if __name__=='__main__':run()

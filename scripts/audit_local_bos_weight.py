"""Exact first-decision CE weights from the frozen exposure and condition masks."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from beatfcn.model.full_event_codec import EventVocabulary
from scripts.run_local_song_baseline import sha, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'reports/d84_bos_supervision_weight_v2.json'
    if output.exists():raise FileExistsError(output)
    tape_path='reports/d67_local_request_tape_v1.json'
    exposure_path='inputs/d67_local_requests/initial100/learning-exposure-1000-v1.json'
    tape=json.loads((ROOT/tape_path).read_text());exposure=json.loads((ROOT/exposure_path).read_text())
    records=[];sources={}
    vocab=EventVocabulary()
    for chart in tape['charts']:
        source=ROOT/chart['token_path']
        if sha(source)!=chart['token_sha256']:raise ValueError('Frozen tokens changed')
        sources[chart['token_path']]=chart['token_sha256'];tokens=np.load(source,mmap_mode='r',allow_pickle=False)
        for row in chart['requests']:
            empty=row['event_count']==0
            if (row['tokens']==1)!=empty or int(tokens[row['token_start']])!=vocab.token('group',0 if empty else 1):
                raise ValueError('Head counts and actual BOS labels disagree')
            records.append(dict(row,mode=chart['mode'],split=chart['split']))
    counts=defaultdict(lambda:dict(requests=0,empty_requests=0,bos_end_weight=0.,bos_new_weight=0.,target_tokens=0))
    for step in exposure['steps']:
        for index,specified in zip(step['indices'],step['specified_condition_fields'],strict=True):
            row=records[index]
            if row['split']!='train':raise ValueError('Exposure crosses split')
            known=any(row['controls'][field][2] for field in specified)
            for key in (row['mode'],row['mode']+('/specified' if known else '/unknown')):
                r=counts[key];r['requests']+=1;r['empty_requests']+=row['event_count']==0;r['target_tokens']+=row['tokens']
                r['bos_end_weight' if row['event_count']==0 else 'bos_new_weight']+=1/row['tokens']
    for r in counts.values():
        r['empty_request_fraction']=r['empty_requests']/r['requests']
        r['weighted_constant_bos_end_optimum']=r['bos_end_weight']/(r['bos_end_weight']+r['bos_new_weight'])
    sources.update({p:sha(ROOT/p) for p in [tape_path,exposure_path,'beatfcn/model/full_event_model.py',
        'scripts/run_local_song_baseline.py','scripts/audit_local_bos_weight.py']})
    report=dict(id='D84-BOS-supervision-weight-audit-v2',complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
        source_sha256=sources,by_mode_and_view=dict(counts),supersedes='v1 retained valid mode totals but incorrectly inspected full labels instead of the actual specified-field masks; per-view counts withdrawn.',
        derivation='Mean valid-token CE per request gives its first token weight1/N. Empty N=1 requests contribute1 to END; nonempty requests contribute1/N to NEW. Summed weights give the isolated constant BOS classifier optimum, not the whole model optimum.',
        limits='Identical CE exposure inD68/D83; cannot alone attribute their difference. Do not remove legitimate empty intervals, label silence as notes or force NEW at inference. No loss/data/architecture change.')
    write_json(output,report);print(json.dumps(report['by_mode_and_view'],indent=2))


if __name__=='__main__':main()

"""D51: source-to-tensor verification of the unchanged, failed-screen D50 batch."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for path,digest in plan['input_sha256'].items():
        assert sha(path)==digest,path
    started=time.time()
    panel=json.loads((ROOT/plan['manifest']).read_text())
    base=(ROOT/plan['manifest']).parent
    original=json.loads((ROOT/'inputs/diagnostic_panels/d48-audio-grid/panel.json').read_text())
    split={r['chart_id']:r for r in original['rows']}
    source_report=json.loads((ROOT/'reports/d50_action_state_batch.json').read_text())
    assert source_report['passed'] is False and source_report['gates']['mania_short_repress'] is False
    for path,digest in source_report['artifacts_sha256'].items():
        assert sha(path)==digest
    x=np.load(base/'inputs.npz',allow_pickle=False)
    y=np.load(base/'supervision.npz',allow_pickle=False)
    assert set(x.files)=={'audio','song_audio','song_times_us','audio_times_us','mode','core_bounds_us','duration_us','condition_values','condition_known','audio_valid'}
    tensor_stats={}
    for name in x.files:
        tensor=torch.from_numpy(x[name].copy())
        number=tensor.to(torch.float64)
        assert torch.isfinite(number).all()
        tensor_stats[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),nan=int(torch.isnan(number).sum()),inf=int(torch.isinf(number).sum()),min=float(number.min()),max=float(number.max()),mean=float(number.mean()),std=float(number.std(unbiased=False)))
    assert not x['condition_known'].any() and not x['condition_values'].any()
    assert np.array_equal(x['core_bounds_us'],np.tile([8_000_000,16_000_000],(12,1)))
    assert np.all(x['duration_us']==32_000_000)
    rows=[]
    for i,row in enumerate(panel['rows']):
        source=split[row['chart_id']]
        assert source['split']=='train' and source['audio_sha256']==row['audio_sha256']
        assert sha(row['canonical'])==row['canonical_sha256']
        assert sha(row['source_features'])==row['source_features_sha256']
        raw=json.loads((ROOT/row['canonical']).read_text())['events']
        c0=row['source_core_start_us']; clip=row['source_clip_start_us']; shift=row['clock_shift_us']
        wanted=[]
        for event in raw:
            a,b=event['onset_us']+shift,event['end_us']+shift
            assert not(event['type']=='hold' and a<c0<=b)
            if c0<=a<c0+8_000_000:
                wanted.append((a-clip,b-clip,event['type'],event['location']['lane'] if row['mode']=='mania_4k' else -1))
        actual=[(e['onset_us'],e['end_us'],e['type'],e['lane']) for e in row['target_events']]
        assert Counter(wanted)==Counter(actual)
        assert all(0<=a<=b<=32_000_000 for a,b,_,_ in actual)
        with np.load(ROOT/row['source_features'],allow_pickle=False) as z:
            full=z['features'].astype(np.float32)
        first=clip//10_000
        assert clip%10_000==0
        assert np.array_equal(x['audio'][i],full[first:first+3200])
        partitions=np.linspace(0,len(full),65,dtype=np.int64)
        expected_summary=np.stack([full[a:b].mean(axis=0,dtype=np.float64) for a,b in zip(partitions,partitions[1:])]).astype(np.float32)
        assert np.array_equal(x['song_audio'][i],expected_summary)
        assert np.array_equal(x['song_times_us'][i],np.array([(a+b-1)*5000+20000-clip for a,b in zip(partitions,partitions[1:])]))
        assert np.array_equal(x['audio_times_us'][i],np.arange(3200)*10000+20000)
        assert np.array_equal(x['audio_valid'][i],x['audio_times_us'][i]<32_000_000)
        times=[a['time_us'] for a in row['target_actions']]
        assert all(a<b for a,b in zip(times,times[1:]))
        assert len(times)<=256 and y['action_valid'][i].sum()==len(times)
        assert np.array_equal(y['time_us'][i,:len(times)],times)
        # Prefix occupancy is derived from original intervals with a strict head
        # inequality, before the boundary's releases and heads. No codec reuse.
        maximum_groups=maximum_holds=joint=pure=0
        for j,(t,prefix) in enumerate(zip(times,row['teacher_forcing_prefix_states'])):
            active=[e for e in actual if e[2]=='hold' and e[0]<t<=e[1]]
            starts=Counter(e[0] for e in active)
            maximum_holds=max(maximum_holds,len(active));maximum_groups=max(maximum_groups,len(starts))
            released=[e for e in actual if e[2]=='hold' and e[1]==t]
            heads=[e for e in actual if e[0]==t]
            joint+=int(bool(released and heads));pure+=int(bool(released and not heads))
            if row['mode']=='mania_4k':
                lanes=[next((e[0] for e in active if e[3]==key),None) for key in range(4)]
                assert prefix['lane_starts']==lanes
                assert np.array_equal(y['prefix_lane_starts_us'][i,j],[-1 if t is None else t for t in lanes])
                assert y['mania_code'][i,j]==row['target_actions'][j]['code']
            else:
                groups=[list(p) for p in sorted(starts.items())]
                assert [list(p) for p in prefix['hold_groups']]==groups
                assert len(groups)<=8
                assert np.array_equal(y['phira_starts'][i,j],row['target_actions'][j]['starts'])
                for k,(a,count) in enumerate(groups):
                    assert np.array_equal(y['prefix_hold_groups'][i,j,k],[a,count])
                    assert y['phira_release_counts'][i,j,k]==sum(e[0]==a for e in released)
            if j==0:
                assert not active and prefix['last_time_us'] is None
            else:
                assert prefix['last_time_us']==times[j-1]
        rows.append(dict(id=row['id'],mode=row['mode'],tier=row['tier'],events=len(actual),actions=len(times),
            maximum_active_holds=maximum_holds,maximum_active_start_groups=maximum_groups,
            release_with_head_boundaries=joint,pure_release_boundaries=pure))
    mania=[r for r in rows if r['mode']=='mania_4k'];phira=[r for r in rows if r['mode']=='phira']
    gates=dict(source_and_tensor_alignment=True, unchanged_d50_failure_preserved=True,
        mania_multiple_held_keys=sum(r['maximum_active_holds']>=2 for r in mania)>=2,
        phira_multiple_start_groups=sum(r['maximum_active_start_groups']>=2 for r in phira)>=2,
        joint_release_attack_each_mode=all(sum(r['release_with_head_boundaries'] for r in group)>=4 for group in [mania,phira]),
        pure_release_each_mode=all(sum(r['pure_release_boundaries'] for r in group)>=4 for group in [mania,phira]),
        complete_mode_type_support=source_report['gates']['mania_type_support'] and source_report['gates']['phira_types'],
        train_only_and_two_per_cell=source_report['gates']['all_train_no_heldout'] and source_report['gates']['two_per_cell'])
    result=dict(id='D51',complete=True,passed=all(gates.values()),gates=gates,rows=rows,input_tensor_statistics=tensor_stats,
        optimizer_updates=0,model_constructed=False,model_promotion=False,new_cloud_spend_usd=0,
        scope='source-to-tensor and representative state-mechanism coverage only; not learned quality',
        plan_sha256=sha(str(args.plan)),wall_s=time.time()-started,source_sha256=plan['input_sha256'])
    destination=ROOT/plan['result']
    assert not destination.exists()
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],gates=gates,wall_s=result['wall_s'])))


if __name__=='__main__':
    main()

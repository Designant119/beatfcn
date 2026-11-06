"""Read-only real-prefix support audit; does not activate a training route."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES
from beatfcn.events.controls import MODE_FIELDS, raw_controls
from beatfcn.train.event_reference import VisibleReference, encode_reference
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def visible_prefix(events, start, lookback_us=16_000_000):
    lower = max(0, start-lookback_us)
    visible = tuple(e for e in events if lower <= e.onset_us < start)
    unseen_carry = [e.event_id for e in events if e.is_hold and e.onset_us < lower and e.end_us > start]
    return lower, visible, unseen_carry


def encoding_digest(encoded):
    h = hashlib.sha256()
    for key, value in sorted(encoded.items()):
        h.update(key.encode()); h.update(str(value.shape).encode()); h.update(value.dtype.str.encode()); h.update(value.tobytes())
    return h.hexdigest()


def signature(events):
    return hashlib.sha256(json.dumps([(e.onset_us,e.end_us,e.type,
        e.location.lane if isinstance(e.location,LaneLoc) else -1) for e in events],separators=(',',':')).encode()).hexdigest()


def run():
    tape_path = 'reports/d67_local_request_tape_v1.json'
    output = ROOT/'reports/d84_reference_support_v1.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    tape = json.loads((ROOT/tape_path).read_text())
    if not tape['complete'] or tape['failures']:
        raise ValueError('Need admitted complete tape')
    source_sha = {tape_path:sha(ROOT/tape_path), 'scripts/audit_local_reference_support.py':sha(Path(__file__))}
    cases, skipped, groups, examples = [], Counter(), defaultdict(list), []
    stats = dict(count=0, sum=0., sumsq=0., minimum=float('inf'), maximum=float('-inf'), nan=0, inf=0, max_rows=0)
    index = 0
    for chart in tape['charts']:
        path = ROOT/chart['target_path']
        if sha(path) != chart['target_sha256']:
            raise ValueError('Frozen chart changed')
        source_sha[chart['target_path']] = chart['target_sha256']
        mode = chart['mode']
        with np.load(path, allow_pickle=False) as a:
            events = [Event(f"{chart['chart_id']}:{i}", int(h),int(t),TYPES[int(k)],
                LaneLoc(int(l)) if mode=='mania_4k' else LineLoc('rhythm','above',0.)) for i,(h,t,k,l) in
                enumerate(zip(*(a[key] for key in ('onset_us','end_us','kind','lane'))))]
        for row in chart['requests']:
            current_index = index; index += 1
            start, end = row['request']['start_us'], row['request']['end_us']
            if start < 16_000_000:
                skipped['insufficient_16s_prefix'] += 1; continue
            lower, visible, unseen = visible_prefix(events, start)
            if unseen:
                skipped['incoming_hold_not_exposed_by_visible_prefix'] += 1; continue
            source = VisibleReference(chart['chart_id'],mode,chart['audio_sha256'],'human_input',((lower,start),),visible)
            encoded, provenance = encode_reference(mode=mode,audio_sha256=chart['audio_sha256'],
                duration_us=chart['duration_us'],origin_us=start,unknown_regions=((start,end),),sources=(source,))
            arr = encoded['reference'].astype(np.float64)
            stats['nan'] += int(np.isnan(arr).sum()); stats['inf'] += int(np.isinf(arr).sum())
            stats['count'] += arr.size; stats['sum'] += float(arr.sum()); stats['sumsq'] += float((arr*arr).sum())
            stats['minimum'] = min(stats['minimum'],float(arr.min())); stats['maximum'] = max(stats['maximum'],float(arr.max()))
            stats['max_rows'] = max(stats['max_rows'],len(arr))
            # Visible hold tails may cross the boundary; no hidden head enters the encoder.
            if any(p['event_id'] is not None and p['interval_us'][0] >= start for p in provenance):
                raise ValueError('Hidden target leaked into reference')
            target = [e for e in events if start <= e.onset_us < end]
            case = dict(index=current_index,chart_id=chart['chart_id'],component=chart['component'],
                audio_sha256=chart['audio_sha256'],mode=mode,split=chart['split'],stratum=chart['stratum'],
                visible_scope_us=[lower,start],target_scope_us=[start,end],visible_notes=len(visible),
                reference_rows=len(arr),reference_sha256=encoding_digest(encoded),target_sha256=signature(target),
                visible_controls=raw_controls(mode,visible,[(lower,start)]),
                target_controls=raw_controls(mode,events,[(start,end)]),
                request_identity=row['request'])
            cases.append(case)
            groups[(mode,chart['split'],chart['audio_sha256'],start,end,json.dumps(row['request'],sort_keys=True))].append(case)
            if len(examples)<12 and not any(e['chart_id']==chart['chart_id'] for e in examples):
                examples.append(dict(chart_id=chart['chart_id'],index=current_index,
                    input_shape=list(encoded['reference'].shape),source_id=source.source_id,
                    visible_scope_us=[lower,start],unknown_scope_us=[start,end],
                    visible_event_count=len(visible),known_cross_boundary_holds=sum(e.is_hold and e.end_us>start for e in visible),
                    provenance_sha256=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()))
    counts = defaultdict(Counter); songs=defaultdict(set); pairs=defaultdict(Counter); pair_songs=defaultdict(set)
    field_support=defaultdict(lambda:defaultdict(lambda:dict(same_direction=set(),opposite_direction=set(),prefix_only=set(),target_only=set())))
    for case in cases:
        key=case['split']+'/'+case['mode'];counts[key]['requests']+=1;counts[key]['known_silent_prefixes']+=case['visible_notes']==0;songs[key].add(case['component'])
    for group in groups.values():
        for a,b in combinations(group,2):
            key=a['split']+'/'+a['mode'];pairs[key]['real_same_audio_pairs']+=1
            if a['reference_sha256']==b['reference_sha256']:
                pairs[key]['same_visible_prefix']+=1;continue
            if a['target_sha256']==b['target_sha256']:
                pairs[key]['same_target_rhythm']+=1;continue
            pairs[key]['distinct_prefix_and_target']+=1;pair_songs[key].add(a['component'])
            for field in MODE_FIELDS[a['mode']]:
                values=[c[group_name][field] for group_name in ('visible_controls','target_controls') for c in (a,b)]
                if any(v is None for v in values):continue
                dp,dt=values[1]-values[0],values[3]-values[2]
                label='same_direction' if dp*dt>0 else 'opposite_direction' if dp*dt<0 else 'prefix_only' if dp else 'target_only' if dt else None
                if label:field_support[key][field][label].add(a['component'])
    mean=stats['sum']/stats['count'];stats.update(mean=mean,std=max(0,stats['sumsq']/stats['count']-mean*mean)**.5)
    report=dict(id='D84-reference-support-audit-v1',complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
        training_path_activated=False,wall_s=time.monotonic()-started,source_sha256=source_sha,
        prior_evidence='D66 encoder mechanical probe only; every completed D68/D83 training request has unknown reference.',
        hypothesis_status='Data support inspection only; no new architecture/loss or candidate selected while D83 is running.',
        permission='Use only real same-chart human heads in the preceding16s; full tails of visible holds are authorized. Exclude incoming holds whose heads are not visible. Hidden continuation never enters reference tensors.',
        total_original_requests=index,skipped=dict(skipped),tensor_stats=stats,examples=examples,
        by_split_mode={key:dict(counts[key],independent_songs=len(songs[key])) for key in sorted(counts)},
        pair_support={key:dict(pairs[key],independent_songs=len(pair_songs[key])) for key in sorted(pairs)},
        style_direction_support={key:{field:{direction:len(ids) for direction,ids in directions.items()} for field,directions in fields.items()} for key,fields in field_support.items()},
        direction_limits='Descriptive song counts; one song may have aligned and opposed windows. Any nonzero delta is counted, not meaningful effect size or causal style evidence. Hidden val attributes are not used to choose evaluation requests.',
        cases=cases,scope='Existing100-song panel only. Do not call repeated windows independent songs. Model reference input shape1 must change in a separately frozen intervention; no current training claim.')
    write_json(output,report)
    print(json.dumps({k:report[k] for k in ('complete','wall_s','total_original_requests','skipped','by_split_mode','pair_support','tensor_stats')}))


if __name__=='__main__':run()

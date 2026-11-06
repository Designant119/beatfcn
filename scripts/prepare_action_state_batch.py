"""D50: deterministic train-only action-state batch from existing audio caches."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.model.action_state import State, ManiaAction, apply_action, encode, mania_digits
from beatfcn.model.event_canvas import RhythmEvent

ROOT = Path(__file__).resolve().parents[1]
TYPES = ('tap', 'hold', 'flick', 'drag')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value):
    a = np.asarray(value)
    f = a.astype(np.float64)
    return dict(shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(f).sum()),
        inf=int(np.isinf(f).sum()), min=float(f.min()) if f.size else None,
        max=float(f.max()) if f.size else None, mean=float(f.mean()) if f.size else None,
        std=float(f.std()) if f.size else None, sha256=hashlib.sha256(a.tobytes()).hexdigest())


def profile(events, mode):
    actions = encode(events, mode)
    state = State(mode)
    counts = Counter(e.type for e in events)
    pure = joint = multiple = max_groups = max_holds = max_count = 0
    for action in actions:
        if isinstance(action, ManiaAction):
            digits = mania_digits(action.code)
            releases = sum(d in (3, 4, 5) for d in digits)
            heads = sum(d in (1, 2, 4, 5) for d in digits)
        else:
            releases = sum(n for _, n in action.releases)
            heads = sum(action.starts)
            max_count = max(max_count, *action.starts)
        state = apply_action(state, action).state
        pure += int(releases > 0 and heads == 0)
        joint += int(releases > 0 and heads > 0)
        max_holds = max(max_holds, state.active_holds)
        max_groups = max(max_groups, len(state.hold_groups))
        multiple += int((state.active_holds if mode == 'mania_4k' else len(state.hold_groups)) > 1)
    short_repress = exact_repress = 0
    if mode == 'mania_4k':
        for lane in range(4):
            ordered = sorted(e for e in events if e.lane == lane)
            for prev, nxt in zip(ordered, ordered[1:]):
                if prev.type == 'hold' and 0 <= nxt.onset_us - prev.end_us <= 20_000:
                    short_repress += 1
                    exact_repress += int(nxt.onset_us == prev.end_us)
    return dict(events=len(events), actions=len(actions), types=dict(counts),
        pure_release_boundaries=pure, release_with_head_boundaries=joint,
        same_key_repress_within_20ms=short_repress, same_key_exact_repress=exact_repress,
        multiple_active_boundaries=multiple, maximum_active_holds=max_holds,
        maximum_active_hold_start_groups=max_groups, maximum_per_type_head_count=max_count)


def score(p, mode):
    n = p['types']
    if mode == 'mania_4k':
        return (int(p['pure_release_boundaries'] > 0), int(p['release_with_head_boundaries'] > 0),
            min(p['same_key_repress_within_20ms'], 4), min(p['multiple_active_boundaries'], 8),
            min(p['release_with_head_boundaries'], 4), min(p['pure_release_boundaries'], 4),
            min(n.get('hold', 0), 16), min(n.get('tap', 0), 16), -abs(p['events'] - 64))
    return (int(p['maximum_active_hold_start_groups'] > 1), min(p['release_with_head_boundaries'], 4),
        len(n), min(n.get('hold', 0), 4), min(n.get('flick', 0), 4), min(n.get('drag', 0), 4),
        min(p['pure_release_boundaries'], 4), min(p['multiple_active_boundaries'], 8), -abs(p['events'] - 64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    report_path = ROOT / plan['result']
    if report_path.exists():
        raise FileExistsError(report_path)
    output = ROOT / plan['output_dir']
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(id='D50', complete=False, plan_sha256=sha(args.plan),
        model_constructed=False, optimizer_updates=0, new_cloud_spend_usd=0,
        audio_model_forward_calls=0, formal_training=False, model_promotion=False,
        rows=[], candidates=[], failures=[])
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    save()
    panel = json.loads((ROOT / plan['source_panel']).read_text())
    cache = json.loads((ROOT / plan['audio_inference']).read_text())
    cached = {x['chart_id']: x for x in cache['songs']}
    train = [x for x in panel['rows'] if x['split'] == 'train']
    held_out = {x['audio_sha256'] for x in panel['rows'] if x['split'] != 'train'}
    cells = Counter((x['mode'], x['tier']) for x in train)
    assert len(train) == 11 and cells['mania_4k', 'low'] == 1
    features, song_features, song_times, labels = [], [], [], []
    for row in train:
        assert row['audio_sha256'] not in held_out
        chart = json.loads((ROOT / row['canonical']).read_text())
        shift = row['frozen_model_shift_us']
        events = sorted(RhythmEvent(e['onset_us'] + shift, e['end_us'] + shift, e['type'],
            e['location']['lane'] if row['mode'] == 'mania_4k' else -1) for e in chart['events'])
        heads = [e.onset_us for e in events]
        holds = [e for e in events if e.type == 'hold']
        duration = round(row['declared_audio_sec'] * 1e6)
        candidates = []
        for start in range(8_000_000, duration - 24_000_000, 100_000):
            if any(e.onset_us < start <= e.end_us for e in holds):
                continue
            chosen = events[bisect_left(heads, start):bisect_left(heads, start + 8_000_000)]
            if not chosen or max(e.end_us for e in chosen) > start + 24_000_000:
                continue
            p = profile(chosen, row['mode'])
            candidates.append((score(p, row['mode']), start, p))
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        report['candidates'].append(dict(chart_id=row['chart_id'], eligible=len(candidates)))
        selected = []
        required = 2 if cells[row['mode'], row['tier']] == 1 else 1
        for rank, start, p in candidates:
            if any(abs(start - old) < 8_000_000 for old in selected):
                continue
            selected.append(start)
            identity = f"d50-{len(report['rows']):02d}"
            clip_start = start - 8_000_000
            target = tuple(replace(e, onset_us=e.onset_us - clip_start, end_us=e.end_us - clip_start)
                for e in events[bisect_left(heads, start):bisect_left(heads, start + 8_000_000)])
            actions = encode(target, row['mode'])
            state = State(row['mode'])
            prefix = []
            for action in actions:
                prefix.append(asdict(state))
                state = apply_action(state, action).state
            assert not state.active_holds
            item = dict(id=identity, chart_id=row['chart_id'], mode=row['mode'], tier=row['tier'],
                split='train', audio=row['audio'], audio_sha256=row['audio_sha256'],
                canonical=row['canonical'], canonical_sha256=row['canonical_sha256'],
                source_features=cached[row['chart_id']]['arrays'],
                source_features_sha256=cached[row['chart_id']]['arrays_sha256'],
                clock_shift_us=shift, source_core_start_us=start, source_clip_start_us=clip_start,
                core_bounds_us=[8_000_000, 16_000_000], clip_duration_us=32_000_000,
                selection_score=list(rank), support=p, empty_initial_state=True,
                target_events=[asdict(e) for e in target], target_actions=[asdict(a) for a in actions],
                teacher_forcing_prefix_states=prefix)
            labels.append(item)
            report['rows'].append({k: v for k, v in item.items() if k not in ['target_events','target_actions','teacher_forcing_prefix_states']})
            source = cached[row['chart_id']]
            assert sha(ROOT / source['arrays']) == source['arrays_sha256']
            with np.load(ROOT / source['arrays'], allow_pickle=False) as arrays:
                audio = arrays['features'].astype(np.float32)
            frame_start = clip_start // 10_000
            assert clip_start % 10_000 == 0
            excerpt = audio[frame_start:frame_start + 3200]
            assert excerpt.shape == (3200, 130)
            features.append(excerpt)
            partitions = np.linspace(0, len(audio), 65, dtype=np.int64)
            song_features.append(np.stack([audio[a:b].mean(axis=0, dtype=np.float64)
                for a, b in zip(partitions, partitions[1:])]).astype(np.float32))
            song_times.append(np.array([(a + b - 1) * 5000 + 20_000 - clip_start
                for a, b in zip(partitions, partitions[1:])], dtype=np.int64))
            print(json.dumps(dict(id=identity, source=row['chart_id'], start_us=start, support=p)), flush=True)
            if len(selected) == required:
                break
        if len(selected) != required:
            report['failures'].append('insufficient_nonoverlap_candidates:' + row['chart_id'])
        save()
    n = len(labels)
    inputs = dict(audio=np.stack(features), song_audio=np.stack(song_features), song_times_us=np.stack(song_times),
        audio_times_us=np.broadcast_to(np.arange(3200, dtype=np.int64) * 10_000 + 20_000, (n, 3200)).copy(),
        mode=np.array([int(x['mode'] == 'phira') for x in labels], dtype=np.int64),
        core_bounds_us=np.tile(np.array([8_000_000,16_000_000], dtype=np.int64), (n,1)),
        duration_us=np.full(n,32_000_000,dtype=np.int64),
        condition_values=np.zeros((n,8),dtype=np.float32), condition_known=np.zeros((n,8),dtype=bool))
    inputs['audio_valid'] = inputs['audio_times_us'] < inputs['duration_us'][:,None]
    targets = dict(time_us=np.full((n,256),-1,dtype=np.int64), action_valid=np.zeros((n,256),dtype=bool),
        mania_code=np.zeros((n,256),dtype=np.int64), phira_starts=np.zeros((n,256,4),dtype=np.int64),
        prefix_lane_starts_us=np.full((n,256,4),-1,dtype=np.int64),
        prefix_hold_groups=np.full((n,256,8,2),-1,dtype=np.int64),
        phira_release_counts=np.zeros((n,256,8),dtype=np.int64))
    for i, row in enumerate(labels):
        if len(row['target_actions']) > 256:
            report['failures'].append('action_capacity:' + row['id']); continue
        for j,(action,prefix) in enumerate(zip(row['target_actions'],row['teacher_forcing_prefix_states'])):
            targets['time_us'][i,j] = action['time_us']; targets['action_valid'][i,j] = True
            targets['prefix_lane_starts_us'][i,j] = [-1 if x is None else x for x in prefix['lane_starts']]
            if row['mode'] == 'mania_4k':
                targets['mania_code'][i,j] = action['code']
            else:
                groups = prefix['hold_groups']
                if len(groups)>8 or max(action['starts'])>8:
                    report['failures'].append('phira_capacity:' + row['id']); continue
                targets['phira_starts'][i,j] = action['starts']
                release = dict(action['releases'])
                for k,(start,count) in enumerate(groups):
                    targets['prefix_hold_groups'][i,j,k] = start,count
                    targets['phira_release_counts'][i,j,k] = release.get(start,0)
    mania=[r for r in report['rows'] if r['mode']=='mania_4k']; phira=[r for r in report['rows'] if r['mode']=='phira']
    by_cell=Counter((r['mode'],r['tier']) for r in report['rows'])
    types={mode:dict(sum((Counter(r['support']['types']) for r in report['rows'] if r['mode']==mode),Counter())) for mode in ['mania_4k','phira']}
    gates=dict(twelve_cores=n==12, two_per_cell=len(by_cell)==6 and all(v==2 for v in by_cell.values()),
        all_train_no_heldout=all(r['split']=='train' and r['audio_sha256'] not in held_out for r in labels),
        all_four_mania_keys=set(e['lane'] for r in labels if r['mode']=='mania_4k' for e in r['target_events'])==set(range(4)),
        mania_type_support=all(types['mania_4k'].get(k,0)>=4 for k in ['tap','hold']),
        mania_short_repress=sum(r['support']['same_key_repress_within_20ms'] for r in mania)>=4,
        mania_joint_release_attack=sum(r['support']['release_with_head_boundaries'] for r in mania)>=4,
        phira_types=all(types['phira'].get(k,0)>=4 for k in TYPES),
        phira_multiple_groups=sum(r['support']['maximum_active_hold_start_groups']>=2 for r in phira)>=2,
        finite_inputs=all(np.isfinite(v).all() for v in inputs.values()),
        empty_initial_states=all(r['empty_initial_state'] for r in labels), capacities_not_truncated=not report['failures'])
    # Label arrays stay in a separate file: a blank-generation loader must not
    # receive sequence lengths, padding, targets or teacher-forcing states.
    np.savez_compressed(output/'inputs.npz',**inputs)
    np.savez_compressed(output/'supervision.npz',**targets)
    manifest=dict(id='d50-state-covered-v1',rows=labels,condition_fields=['density','peak_density','hold_head_ratio','hold_occupancy','chord_ratio','mania_jack','phira_flick','phira_drag'],
        inputs='inputs.npz',supervision='supervision.npz',frame_center_offset_us=20_000,frame_hop_us=10_000,
        training_ready=False, scope='fixed-training-core mechanism probe, not quota or held-out quality')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report.update(complete=True,passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},
        wall_s=time.time()-started,unique_training_charts=len({r['chart_id'] for r in labels}),
        unique_training_audio=len({r['audio_sha256'] for r in labels}),excluded_heldout_audio=sorted(held_out),
        type_support=types,input_tensor_statistics={k:stats(v) for k,v in inputs.items()},
        supervision_tensor_statistics={k:stats(v) for k,v in targets.items()},
        artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(output.iterdir())})
    save()
    print(json.dumps({k:report[k] for k in ['complete','passed','gates','wall_s','type_support']}))


if __name__=='__main__':
    main()

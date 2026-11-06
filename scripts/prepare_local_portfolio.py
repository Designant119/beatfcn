"""Freeze an expanded real-data cohort; no model updates or cloud calls."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from beatfcn.train.local_request_panel import LocalRequestPanel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'bench/rhythm-v2/local-portfolio-cohort-v1.json'
SEED = 20261028


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(values, namespace):
    return sorted(values, key=lambda v: hashlib.sha256(f'{SEED}:{namespace}:{v}'.encode()).hexdigest())


def choose_windows(panel, chart_index, split):
    rows = [i for i, r in enumerate(panel.records) if r['chart_index'] == chart_index]
    full = [i for i in rows if panel.records[i]['request']['end_us'] -
            panel.records[i]['request']['start_us'] == 8_000_000]
    if len(full) < 2:
        raise ValueError(f'Chart lacks two full windows: {chart_index}')
    if split != 'train':
        # Time positions only: confirmation selection does not optimize target density.
        chronological = sorted(full, key=lambda i: panel.records[i]['request']['start_us'])
        return list(dict.fromkeys([chronological[len(chronological)//3],
                                   chronological[2*len(chronological)//3]]))
    picks = []
    for kind in ('hold', 'flick', 'drag'):
        best = max(full, key=lambda i: (panel.records[i]['types'].get(kind, 0), -i))
        if panel.records[best]['types'].get(kind, 0):
            picks.append(best)
    density = sorted(full, key=lambda i: (panel.records[i]['event_count'], i))
    picks += [density[len(density)//3], density[2*len(density)//3], density[-1]]
    empty = [i for i in full if panel.records[i]['event_count'] == 0]
    if empty:
        picks.append(empty[0])
    return list(dict.fromkeys(picks))


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve frozen cohort')
    started = time.monotonic()
    parent_path = ROOT / 'bench/rhythm-v2/local-frontend-comparison-v2.json'
    parent = json.loads(parent_path.read_text())
    panel = LocalRequestPanel(ROOT, parent['tape'], parent['features'])
    cache = json.loads((ROOT/parent['mert_cache_report']).read_text())
    cache_records = {r['audio_sha256']: r for r in cache['records']}
    by = defaultdict(list)
    for i, c in enumerate(panel.charts):
        if c['mode'] == 'phira' and c['audio_sha256'] in cache_records:
            by[(c['split'], c['component'])].append(i)
    old_train = set(panel.records[i]['component'] for i in parent['indices']['train'])
    old_val = set(panel.records[i]['component'] for i in parent['indices']['val']+parent['new_val_indices'])
    buckets = defaultdict(list)
    for (split, comp), charts in by.items():
        if split == 'train' and comp not in old_train:
            buckets[min(panel.charts[i]['stratum'] for i in charts)].append(comp)
    for k, values in buckets.items():
        buckets[k] = ordered(values, f'train:{k}')
    added = []
    while len(added) < 64-len(old_train) and any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k] and len(added) < 64-len(old_train):
                added.append(buckets[k].pop(0))
    if len(old_train | set(added)) != 64:
        raise ValueError('Insufficient independent training components')
    confirm = ordered([comp for split, comp in by if split=='val' and comp not in old_val], 'confirmation')
    if len(confirm) != 8:
        raise ValueError(f'Expected exactly 8 remaining validation components, got {len(confirm)}')
    indices = dict(train24=list(parent['indices']['train']), train64=list(parent['indices']['train']),
        dev=list(parent['indices']['val']+parent['new_val_indices']), confirm=[],
        empty_dev=list(parent['empty_val_indices']))
    for comp in added:
        charts = sorted(by[('train', comp)], key=lambda i:(panel.charts[i]['stratum'], panel.charts[i]['chart_id']))
        selected = [charts[0]]
        other = next((i for i in charts[1:] if panel.charts[i]['stratum'] != panel.charts[charts[0]]['stratum']), None)
        if other is None and len(charts)>1:
            other = charts[1]
        if other is not None:
            selected.append(other)
        for ci in selected:
            indices['train64'].extend(choose_windows(panel, ci, 'train'))
    for comp in confirm:
        ci = min(by[('val', comp)], key=lambda i:panel.charts[i]['chart_id'])
        indices['confirm'].extend(choose_windows(panel, ci, 'confirm'))
    indices = {k:list(dict.fromkeys(v)) for k,v in indices.items()}
    summary, sources, identities = {}, {}, {}
    for split, selected in indices.items():
        events, strata, support = Counter(), Counter(), defaultdict(set)
        max_cell = 0
        for i in selected:
            rec = panel.records[i]
            target = panel.targets(i)
            events.update(e.type for e in target)
            strata[rec['stratum']] += 1
            for e in target:
                support[e.type].add(rec['component'])
            per_cell = Counter(((e.onset_us-rec['request']['start_us'])//40000,e.type) for e in target)
            max_cell = max(max_cell, max(per_cell.values(), default=0))
            chart = panel.charts[rec['chart_index']]
            name = chart['target_path']
            if name not in sources:
                sources[name] = sha(ROOT/name)
                if sources[name] != chart['target_sha256']:
                    raise ValueError('Target identity drift')
            audio = rec['audio_sha256']
            identities[audio] = dict(path=cache_records[audio]['path'],
                files=cache_records[audio]['files'], samples=cache_records[audio]['samples'])
        summary[split] = dict(components=len({panel.records[i]['component'] for i in selected}),
            audio_identities=len({panel.records[i]['audio_sha256'] for i in selected}),
            charts=len({panel.records[i]['chart_id'] for i in selected}),windows=len(selected),
            events=dict(events),type_component_support={k:len(v) for k,v in support.items()},
            strata_windows=dict(strata),empty_windows=sum(panel.records[i]['event_count']==0 for i in selected),
            max_events_per_40ms_per_type=max_cell,current_16_slot_canvas_fits=max_cell<=16,
            maximum_serialized_target_tokens=max(panel.records[i]['tokens'] for i in selected))
    group_components = {k:{panel.records[i]['component'] for i in v} for k,v in indices.items()}
    group_audio = {k:{panel.records[i]['audio_sha256'] for i in v} for k,v in indices.items()}
    for left,right in (('train64','dev'),('train64','confirm'),('dev','confirm')):
        if group_components[left]&group_components[right] or group_audio[left]&group_audio[right]:
            raise ValueError('Training/validation component or audio leakage')
    plan = dict(id='local-portfolio-cohort-v1',seed=SEED,parent_sha256=sha(parent_path),
        tape=parent['tape'],features=parent['features'],mert_cache_report=parent['mert_cache_report'],
        indices=indices,summary=summary,health_indices=parent['health_indices'],
        empty_health_index=parent['empty_health_index'],
        additional_train_components=added,confirmation_components=confirm,
        target_sha256=sources,mert_cache_identities=identities,
        source_sha256={name:sha(ROOT/name) for name in [parent['tape'],parent['features'],
            parent['mert_cache_report'],'scripts/prepare_local_portfolio.py']},
        selection='Preserve old174 training and old32 dev windows; add40 training components stratified by original labels, up to2charts each. New training selects rare types, density thirds, densest and empty full windows; NO token-count exclusion. Confirmation chart identity and chronological positions only.',
        causal_scope='Data expansion separate from architecture/capacity; compare models on the SAME chosen track.',
        confirmation_policy='New to this local16-component panel, not claimed project-wide blind. No confirmation model scores used for tuning.',
        optimizer_updates=0,new_cloud_compute_calls=0,preparation_wall_s=time.monotonic()-started)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()

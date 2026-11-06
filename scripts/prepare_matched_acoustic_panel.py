"""Validate D31's real-data contract without running any model."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, fields
import hashlib
import json
import time
from itertools import combinations

import numpy as np
import torch

from beatfcn.events.canonical import Event
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.group_canvas import decode_group_targets
from scripts.matched_acoustic_data import (
    ROOT, file_sha, read_chart, target_events, make_example, metadata_sha,
    common_physical_targets, swap_audio,
)
from scripts.r09_local_data import verify_files


def select_covered_pairs(candidates, modes=('mania_4k', 'phira')):
    """First fixed-order combination with distinct audio and complete type support."""
    expected = {'mania_4k': {'tap', 'hold'}, 'phira': {'tap', 'hold', 'flick', 'drag'}}
    chosen = []
    for mode in modes:
        available = [pair for pair in candidates if pair[0]['row']['mode'] == mode]
        found = None
        for a, b in combinations(available, 2):
            if a[0]['row']['audio_sha256'] == b[0]['row']['audio_sha256']:
                continue
            support = set().union(*(set(item['types']) for item in [*a, *b]))
            if expected[mode] <= support:
                found = [*a, *b]
                break
        if found is None:
            raise ValueError(f'no distinct-song type-complete panel for {mode}')
        chosen.extend(dict(item) for item in found)
    return chosen


def stats(value):
    x = value.detach().float()
    return dict(shape=list(value.shape), dtype=str(value.dtype), nonfinite=int((~x.isfinite()).sum()),
        minimum=float(x.min()) if x.numel() else None, maximum=float(x.max()) if x.numel() else None,
        mean=float(x.mean()) if x.numel() else None, std=float(x.std(unbiased=False)) if x.numel() else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text())
    verify_files(plan)
    torch.set_num_threads(plan['threads'])
    started = time.perf_counter()
    catalog = torch.load(ROOT / plan['catalog'], map_location='cpu', weights_only=False)
    prior = json.loads((ROOT / plan['source_preparation']).read_text())
    receipt = json.loads((ROOT / 'reports/r09_feature_transfer.json').read_text())
    assert receipt['complete'] and not receipt['failures']
    known_features = {r['local']: r['sha256'] for r in receipt['files']}
    rows, entries = {}, defaultdict(list)
    for pair in catalog['train_pairs']:
        for entry in pair['entries']:
            rows[entry['row']['chart_id']] = entry['row']
            entries[entry['row']['chart_id']].append(entry)
    excluded, candidates, hashes = [], [], dict(plan['input_sha256'])
    def rank(text):
        return hashlib.sha256(f"{plan['seed']}:{text}".encode()).hexdigest()
    for cid, row in sorted(rows.items(), key=lambda x: rank(x[0])):
        if row['split'] != 'train':
            raise ValueError('non-train material in panel source')
        canonical = f"{plan['canonical_root']}/{row['source']}/{row['native_id']}.json"
        if file_sha(ROOT / canonical) != prior['train_canonical_sha256'][cid]:
            raise ValueError(f'frozen canonical changed: {cid}')
        item = dict(row=row, canonical=canonical)
        chart = read_chart(item)
        by_id = {e.event_id: e for e in chart.events}
        for entry in entries[cid]:
            expected = [RhythmEvent.from_event(Event.from_json(e), row['mode']) for e in entry['truth_events']]
            original = [by_id[e['event_id']] for e in entry['truth_events']]
            # Compare using the same canonical model quantisation as R09.
            from beatfcn.events.canonical import build_model_view
            view = build_model_view(row['mode'], [], original)
            actual = [RhythmEvent.from_event(e, row['mode']) for g in view.target_groups for e in g.events]
            if not view.ok or Counter(actual) != Counter(expected):
                raise ValueError(f'original R09 source clock differs: {cid}')
        feature = ROOT / plan['feature_root'] / row['audio_sha256']
        header = np.load(feature / 'L9.npy', mmap_mode='r')
        maximum = min(round(row['audio_sec'] * 1e6), len(header) * 40_000)
        possible = []
        for c0 in range(plan['core_origin_in_clip_us'], maximum, plan['source_core_grid_us']):
            if c0 - plan['core_origin_in_clip_us'] + plan['clip_us'] + plan['untrained_translation_us'] > maximum:
                continue
            try:
                a, ae = target_events(chart, c0, plan)
                b, be = target_events(chart, c0 + plan['untrained_translation_us'], plan)
                possible.append(dict(item, source_core_us=c0,
                    types=dict(Counter(e.type for e in ae)), target_count=len(ae), shifted_target_count=len(be)))
            except ValueError as exc:
                excluded.append(dict(chart_id=cid, source_core_us=c0, reason=str(exc)))
        possible.sort(key=lambda x: rank(f"{cid}:{x['source_core_us']}"))
        pairs = ([a, b] for a, b in combinations(possible, 2)
                 if abs(a['source_core_us'] - b['source_core_us']) >= plan['minimum_same_song_core_separation_us'])
        if plan.get('selection_strategy') == 'all_pairs_type_coverage':
            candidates.extend(pairs)
        else:
            pair = next(pairs, None)
            if pair:
                candidates.append(pair)
    if plan.get('selection_strategy') == 'all_pairs_type_coverage':
        if plan['songs_per_mode'] != 2 or plan['cores_per_song'] != 2:
            raise ValueError('coverage selection requires the registered two by two panel')
        chosen = select_covered_pairs(candidates)
        audio_seen = {x['row']['audio_sha256'] for x in chosen}
    else:
        chosen, audio_seen = [], set()
        for mode in ('mania_4k', 'phira'):
            selected = 0
            for pair in candidates:
                row = pair[0]['row']
                if row['mode'] != mode or row['audio_sha256'] in audio_seen:
                    continue
                audio_seen.add(row['audio_sha256'])
                chosen.extend(pair)
                selected += 1
                if selected == plan['songs_per_mode']:
                    break
            if selected != plan['songs_per_mode']:
                raise ValueError(f'insufficient matched real sources for {mode}')
    for i, item in enumerate(chosen):
        item['panel_id'] = f"{plan['id'].lower()}-{i:02d}"
        names = [item['canonical'], *[str((ROOT / plan['feature_root'] / item['row']['audio_sha256'] /
                    name).relative_to(ROOT)) for name in ('L9.npy', 'L16.npy', 'L24.npy', 'S20.npy')]]
        for name in names:
            actual = file_sha(ROOT / name)
            if name in known_features and actual != known_features[name]:
                raise ValueError(f'feature changed since original transfer: {name}')
            hashes[name] = actual
    health, reference_meta, coverage = [], {}, defaultdict(Counter)
    for item in chosen:
        a = make_example(item, plan)
        b = make_example(item, plan, translated=True)
        same_targets, intersection = common_physical_targets(a[1], b[1], a[2], b[2], plan)
        if not same_targets:
            raise ValueError('same physical intersection has different truth')
        for inputs, labels, identity in (a, b):
            key = item['row']['mode']
            actual = metadata_sha(inputs)
            if reference_meta.setdefault(key, actual) != actual:
                raise ValueError('non-acoustic metadata identifies a target')
            swap_audio(inputs, a[0])
            measured = {f.name: stats(getattr(inputs, f.name)) for f in fields(inputs)
                        if torch.is_tensor(getattr(inputs, f.name))}
            for name in ('counts', 'heads', 'multiplicity', 'notes'):
                measured['target_' + name] = stats(torch.from_numpy(getattr(labels, name)))
            if any(v['nonfinite'] for v in measured.values()):
                raise ValueError('nonfinite real tensor')
            events = decode_group_targets(labels)
            if not identity['translated']:
                coverage[key].update(e.type for e in events)
            health.append(dict(identity=identity, tensors=measured, metadata_sha256=actual,
                target_roundtrip_exact=True, shared_physical_targets_exact=True,
                shared_physical_interval_us=intersection, targets=[asdict(e) for e in events],
                crossing_core_end_holds=sum(e.type == 'hold' and e.end_us > labels.layout.end_us for e in events)))
    expected = {'mania_4k': {'tap', 'hold'}, 'phira': {'tap', 'hold', 'flick', 'drag'}}
    type_gate = all(expected[mode] <= set(coverage[mode]) for mode in expected)
    manifest = dict(schema='beatfcn.matched_acoustic_panel_manifest.v1', id=plan['id'], plan=args.plan,
        plan_sha256=file_sha(ROOT / args.plan), items=chosen, input_sha256=hashes,
        non_acoustic_metadata_sha256=reference_meta, original_type_counts=dict(coverage),
        source_exclusions=excluded, all_applicable_types_present=type_gate,
        training_ready=False, model_updates=0, no_chart_or_audio_download=True)
    destination = ROOT / plan['manifest']
    if destination.exists():
        raise FileExistsError('panel already exists; do not silently reselect')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    report = dict(schema='beatfcn.matched_acoustic_panel_health.v1', id=plan['id'], complete=True, passed=type_gate,
        manifest=plan['manifest'], manifest_sha256=file_sha(destination), plan_sha256=manifest['plan_sha256'],
        songs=len(audio_seen), original_cores=len(chosen), shifted_cores=len(chosen),
        applicable_type_coverage=dict(coverage), same_mode_metadata_identical=True,
        original_R09_source_clock_reconstructed=True, exact_S20_slices=True,
        full_target_roundtrips=True, tensor_health=health, exclusions=excluded,
        wall_s=time.perf_counter() - started, optimizer_updates=0, new_paid_compute_usd=0,
        model_promotion=False, formal_training_ready=False)
    (ROOT / plan['result']).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tensor_health', 'exclusions')}, ensure_ascii=False))
    if not type_gate:
        raise SystemExit('Selected panel lacks an applicable type; no model run allowed')


if __name__ == '__main__':
    main()

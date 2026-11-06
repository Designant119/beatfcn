"""Freeze Phira/MERT-only CPU data; never uploads, trains or touches budgets."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.event_conditions import encode_controls, training_labels
from beatfcn.train.event_density_drag_filter import EventDensityFilteredPanel, RULE, filtered_request_row
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.modal_event_dataset import KINDS, SCHEMA, TARGET_SLOTS, MertOnlyFeatureStore, event_array
from beatfcn.train.native_diagnostic_priors import DURATION_UPPER_US, duration_bucket
from beatfcn.train.reference_completion import partition
from beatfcn.train.typed_inventory import carry_reference_row


ROOT = Path(__file__).resolve().parents[1]
TAPE = 'reports/d94_expanded_target_tape_v1.json'
FEATURES = 'reports/d94_merged_feature_admission_v1.json'
MERT = 'reports/d96_local_mert_cache_workers_v2.json'
COHORT = 'bench/rhythm-v2/local-portfolio-cohort-v1.json'
TRAIN_VIEWS = ('draft', 'single_condition', 'all_conditions', 'random_groups', 'middle', 'continuation')
SOURCES = (
    'beatfcn/train/modal_event_dataset.py', 'scripts/prepare_modal_event_data.py',
    'tests/test_modal_event_dataset.py', 'beatfcn/train/event_density_drag_filter.py',
    'beatfcn/train/drag_density_filter.py', 'beatfcn/train/reference_completion.py',
    'beatfcn/train/typed_inventory.py', 'beatfcn/train/joint_reference.py',
    'beatfcn/train/event_reference.py', 'beatfcn/train/event_conditions.py',
    'beatfcn/train/native_diagnostic_priors.py', 'beatfcn/train/local_request_panel.py',
    'beatfcn/train/local_bundle.py', 'beatfcn/train/local_features.py',
    'beatfcn/train/mert_canvas_features.py', 'beatfcn/train/local_panel.py',
    'beatfcn/train/local_shapes.py', 'beatfcn/events/controls.py',
    'beatfcn/model/full_event_codec.py', 'beatfcn/model/event_canvas.py',
    'beatfcn/model/local_song_rollout.py', 'beatfcn/events/canonical.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class SourcePanel:
    """Metadata and selected raw targets only; no audio, token or confirm reads."""
    def __init__(self, root, tape):
        self.root = Path(root)
        self.charts = tape['charts']
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index,
                    chart_id=chart['chart_id'], audio_sha256=chart['audio_sha256'],
                    component=chart['component'], split=chart['split'], mode=chart['mode'],
                    stratum=chart['stratum']))
        self._arrays = {}
        self.opened_target_paths = set()

    def arrays(self, chart_index):
        if chart_index not in self._arrays:
            path = self.charts[chart_index]['target_path']
            with np.load(self.root / path, allow_pickle=False) as archive:
                self._arrays[chart_index] = {k: archive[k].copy() for k in archive.files}
            self.opened_target_paths.add(path)
        return self._arrays[chart_index]

    def targets(self, index):
        record = self.records[index]
        arrays = self.arrays(record['chart_index'])
        sl = slice(record['target_start'], record['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(*(arrays[key][sl] for key in ('onset_us', 'end_us', 'kind', 'lane')))]


def prepare_view(panel, filtered, index, view, *, field_index=0, manual_bands=None, mask_reference=False):
    """Partition ORIGINAL objects first, then apply the one-shot chart mask."""
    record = panel.records[index]
    original = panel.targets(index)
    original_request = event_request(record['request'])
    request, visible, hidden, regions = original_request, [], original, ()
    if view in ('middle', 'continuation', 'random_groups'):
        request, visible, hidden, regions = partition(original, request, view,
            f"{record['chart_id']}:{request.start_us}", 20261027)
    bands = {} if manual_bands is None else manual_bands
    if view in ('single_condition', 'all_conditions'):
        if record['split'] != 'train':
            raise ValueError('Development truth cannot construct caller controls')
        bands, _ = training_labels('phira', canonical(original + list(original_request.known_carry), 'phira'),
                                   (request.start_us, request.end_us))
        if view == 'single_condition' and bands:
            field = sorted(bands)[field_index % len(bands)]
            bands = {field: bands[field]}
    controls = encode_controls('phira', bands, (request.start_us, request.end_us), origin_us=request.start_us)
    raw = dict(index=index, view=view, request=request, targets=hidden, visible=visible,
        source_role='human_input', inputs=dict(controls=controls), origin_s=request.start_us / 1e6,
        known_regions=regions, public_partition_regions=regions)
    row = filtered_request_row(filtered, raw)
    if mask_reference:
        row = carry_reference_row(row, record['audio_sha256'], mask_reference=True)
        row['known_regions'] = ()
    if Counter(e for e in hidden if e.type != 'drag') != Counter(e for e in row['targets'] if e.type != 'drag'):
        raise ValueError('Non-drag multiset changed')
    # This uniform public resource contract is independent of hidden N. The
    # source request's old limits are separately retained in each record.
    row['request'] = replace(row['request'], max_events=TARGET_SLOTS, max_tokens=8192)
    return row


def validate_native_targets(events, request):
    """CPU-only complete support validation, with no rounding or clipping."""
    if len(events) > min(TARGET_SLOTS, request.max_events):
        raise ValueError(f'Explicit native target OVERFLOW {len(events)} > {min(TARGET_SLOTS, request.max_events)}')
    copies = set(request.copy_times)
    for event in events:
        if (event.lane != -1 or event.type not in KINDS
                or not request.start_us <= event.onset_us < request.end_us
                or (event.onset_us % 1000 and event.onset_us not in copies)
                or (request.allowed_types is not None and event.type not in request.allowed_types)):
            raise ValueError('Unreachable native target head/type')
        if event.type == 'hold':
            if not event.onset_us < event.end_us <= request.duration_us:
                raise ValueError('Unreachable native hold tail')
            if event.end_us % 1000 and event.end_us not in copies:
                raise ValueError('Fractional tail lacks authorized COPY')
        elif event.end_us != event.onset_us:
            raise ValueError('Non-hold duration')


def public_head_layout(request, known_regions=()):
    """Complete native-ms + authorized COPY support, using public fields only."""
    heads = np.arange((request.start_us + 999) // 1000 * 1000, request.end_us, 1000, dtype=np.int64)
    copies = np.asarray([t for t in request.copy_times if request.start_us <= t < request.end_us], np.int64)
    if len(copies):
        heads = np.unique(np.concatenate((heads, copies)))
    for lo, hi in known_regions:
        heads = heads[~((heads >= lo) & (heads < hi))]
    allowed = np.tile([request.allowed_types is None or kind in request.allowed_types for kind in KINDS],
                      (len(heads), 1))
    endpoint = (heads // 1000 + 1) * 1000 <= request.duration_us
    for value in request.copy_times:
        endpoint |= (heads < value) & (value <= request.duration_us)
    allowed[:, 1] &= endpoint
    keep = allowed.any(-1)
    return heads[keep] - request.start_us, allowed[keep]


def describe_windows(panel, filtered, indices):
    groups = defaultdict(lambda: dict(windows=0, raw_events=0, events=0, removed_drag=0,
        empty_windows=0, raw_empty_windows=0, holds=0, crossing_holds=0, incoming_holds=0,
        max_events=0, types_before=Counter(), types_after=Counter(), components=set(), charts=set()))
    for index in indices:
        rec = panel.records[index]
        raw, clean = panel.targets(index), filtered.targets(index)
        for key in ('all', rec['stratum']):
            d = groups[key]
            d['windows'] += 1
            d['raw_events'] += len(raw)
            d['events'] += len(clean)
            d['removed_drag'] += len(raw) - len(clean)
            d['raw_empty_windows'] += not raw
            d['empty_windows'] += not clean
            d['holds'] += sum(e.type == 'hold' for e in clean)
            d['crossing_holds'] += sum(e.type == 'hold' and e.end_us > rec['request']['end_us'] for e in clean)
            d['incoming_holds'] += len(rec['request']['known_carry'])
            d['max_events'] = max(d['max_events'], len(clean))
            d['types_before'].update(e.type for e in raw)
            d['types_after'].update(e.type for e in clean)
            d['components'].add(rec['component'])
            d['charts'].add(rec['chart_id'])
    return {key: {k: len(v) if isinstance(v, set) else dict(v) if isinstance(v, Counter) else v
                  for k, v in values.items()} for key, values in groups.items()}


def prepare(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Preserve frozen prepared artifacts; choose a new output identity')
    if shutil.disk_usage(root).free < 15 * 2**30:
        raise OSError('15GiB free reserve required')
    output.mkdir(parents=True)
    started = time.monotonic()
    tape = json.loads((root / TAPE).read_text())
    cohort = json.loads((root / COHORT).read_text())
    if not tape['complete'] or tape['failures']:
        raise ValueError('Complete source target tape required')
    panel = SourcePanel(root, tape)
    filtered = EventDensityFilteredPanel(panel)
    features = MertOnlyFeatureStore.from_reports(root, FEATURES, MERT)
    train = [i for i, r in enumerate(panel.records) if r['split'] == 'train' and r['mode'] == 'phira']
    dev = cohort['indices']['dev'] + cohort['indices']['empty_dev']
    if any(panel.records[i]['mode'] != 'phira' for i in dev):
        raise ValueError('Phira-only development required')
    train_components = {panel.records[i]['component'] for i in train}
    dev_components = {panel.records[i]['component'] for i in dev}
    # Only component IDs are read for sealed confirmation. No target path for
    # those rows is ever opened, copied, included in an upload list or scored.
    sealed = {panel.records[i]['component'] for i in cohort['indices']['confirm']}
    if train_components & dev_components or (train_components | dev_components) & sealed:
        raise ValueError('Train/development/sealed component overlap')
    selected = train + dev
    audios = sorted({panel.records[i]['audio_sha256'] for i in selected})
    if set(audios) - features.records.keys():
        raise ValueError('Selected source lacks admitted MERT/mel')
    if len(train) != 6699 or len({panel.records[i]['chart_id'] for i in train}) != 324 or len(dev) != 34:
        raise ValueError('Frozen full Phira train/dev population changed')
    audit = dict(train=describe_windows(panel, filtered, train),
                 dev=describe_windows(panel, filtered, dev))
    priors = dict(global_counts=[.5] * 4, raw_global_counts=[.5] * 4, first_member_counts=[.5] * 4,
        duration_counts=[.5] * len(DURATION_UPPER_US), duration_upper_us=list(DURATION_UPPER_US),
        history_counts={}, source='Training draft windows only;native duplicate objects counted separately',
        interpretation='Diagnostic/initialization priors, never output quotas or hidden caller controls')
    for index in train:
        for event in panel.targets(index):
            priors['raw_global_counts'][KINDS.index(event.type)] += 1
        events = sorted(filtered.targets(index), key=lambda e: (e.onset_us, KINDS.index(e.type), e.end_us))
        previous = -1
        if events:
            priors['first_member_counts'][KINDS.index(events[0].type)] += 1
        for rank, event in enumerate(events):
            kind = KINDS.index(event.type)
            priors['global_counts'][kind] += 1
            counts = priors['history_counts'].setdefault(f'{rank}:{previous}', [0] * 4)
            counts[kind] += 1
            previous = kind
            if event.type == 'hold':
                priors['duration_counts'][duration_bucket(event.end_us - event.onset_us)] += 1
    counts = np.asarray(priors['global_counts'])
    priors['global_probability'] = (counts / counts.sum()).tolist()
    raw_counts = np.asarray(priors['raw_global_counts'])
    priors['raw_global_probability'] = (raw_counts / raw_counts.sum()).tolist()
    arrays = dict(targets=[], reference=[], reference_times=[], controls=[], head_offsets=[], head_type_allowed=[])
    records, offsets = [], dict(targets=0, reference=0)
    layouts, layout_offset = {}, 0
    specs = [(i, view, i % 8, None, False, 'train') for i in train for view in TRAIN_VIEWS]
    specs += [(i, 'draft', 0, None, False, 'dev') for i in dev]
    specs += [(i, 'middle', 0, None, masked, 'dev') for masked in (False, True)
              for i in cohort['indices']['dev'][::2]]
    specs += [(i, 'draft', 0, {'density_notes_per_second': list(bands)}, False, 'dev')
              for bands in ((2., 3.), (6., 8.)) for i in cohort['indices']['dev'][::8]]
    maximum = 0
    exposure = defaultdict(lambda: dict(rows=0, targets=0, empty=0, types=Counter()))
    for n, (index, view, field, bands, masked, split) in enumerate(specs):
        row = prepare_view(panel, filtered, index, view, field_index=field,
                           manual_bands=bands, mask_reference=masked)
        request = row['request']
        validate_native_targets(row['targets'], request)
        target = event_array(row['targets'])
        maximum = max(maximum, len(target))
        reference = row['inputs']['reference']
        head_offsets, type_allowed = public_head_layout(request, row.get('known_regions', ()))
        layout_digest = hashlib.sha256(head_offsets.tobytes() + type_allowed.tobytes()).hexdigest()
        if layout_digest not in layouts:
            layouts[layout_digest] = [layout_offset, layout_offset + len(head_offsets)]
            arrays['head_offsets'].append(head_offsets)
            arrays['head_type_allowed'].append(type_allowed)
            layout_offset += len(head_offsets)
        mode = 'middle_masked' if masked else 'density_low' if bands and bands['density_notes_per_second'][0] == 2 else 'density_high' if bands else view
        source = panel.records[index]
        record = dict(array_index=n, index=index, view=mode, original_view=view, field_index=field,
            split=split, chart_id=source['chart_id'], audio_sha256=source['audio_sha256'],
            component=source['component'], stratum=source['stratum'], request=asdict(request),
            original_resource_limits={k:source['request'][k] for k in ('max_events','max_tokens')},
            resource_limit_change_scope='Uniform public512events/8192tokens;no target-dependent selection',
            target_slice=[offsets['targets'], offsets['targets'] + len(target)],
            reference_slice=[offsets['reference'], offsets['reference'] + len(reference)],
            head_layout_slice=layouts[layout_digest],
            known_carry_us=event_array(request.known_carry).tolist(),
            known_regions_us=list(row.get('known_regions', ())),
            public_partition_regions_us=list(row.get('public_partition_regions', ())),
            allowed_types=[request.allowed_types is None or k in request.allowed_types for k in KINDS],
            raw_target_slice=[source['target_start'], source['target_stop']],
            raw_target_path=panel.charts[source['chart_index']]['target_path'],
            public_partition='original objects before v2 filtering',
            source_role=row['source_role'], provenance=row['provenance'])
        record['reference_masked'] = masked
        record['source_ids'] = sorted({p['source_id'] for p in row['provenance'] if p.get('source_id')})
        records.append(record)
        arrays['targets'].append(target)
        arrays['reference'].append(reference)
        arrays['reference_times'].append(row['inputs']['reference_times'])
        arrays['controls'].append(row['inputs']['controls'])
        offsets['targets'] += len(target)
        offsets['reference'] += len(reference)
        e = exposure[split + ':' + mode]
        e['rows'] += 1
        e['targets'] += len(target)
        e['empty'] += not len(target)
        e['types'].update(event.type for event in row['targets'])
        if n % 2000 == 0:
            print(json.dumps(dict(event='data_prepare',rows=n,total=len(specs),wall_s=time.monotonic()-started)),flush=True)
    array_paths = {}
    for key, values in arrays.items():
        path = output / (key + '.npy')
        values = np.stack(values) if key == 'controls' else np.concatenate(values)
        np.save(path, values, allow_pickle=False)
        array_paths[key] = str(path.relative_to(root))
    source_sha = {name: sha(root / name) for name in (*SOURCES, TAPE, FEATURES, MERT, COHORT)}
    raw_sha = {path: sha(root / path) for path in sorted(panel.opened_target_paths)}
    for chart in panel.charts:
        if chart['target_path'] in raw_sha and raw_sha[chart['target_path']] != chart['target_sha256']:
            raise ValueError('Original native target identity changed')
    audio_records = [features.records[a] for a in audios]
    upload = {}
    for audio in audio_records:
        for name, info in audio['mert_files'].items():
            path = str(Path(audio['mert_path']) / name)
            upload[path] = dict(bytes=(root / path).stat().st_size, sha256=info['sha256'], role='frozen_mert')
        path = audio['complete_mel']
        upload[path] = dict(bytes=(root / path).stat().st_size, sha256=audio['mel_sha256'], role='complete_fine_mel')
    # Raw source targets are retained separately for cross-regime diagnostics.
    # Prepared filtered arrays are the runtime supervision; no re-filter on GPU.
    for path, digest in raw_sha.items():
        upload[path] = dict(bytes=(root / path).stat().st_size, sha256=digest, role='raw_target_provenance')
    payload_checked = 0
    for path, info in upload.items():
        if sha(root / path) != info['sha256']:
            raise ValueError('Selected frozen payload changed: ' + path)
        payload_checked += 1
    audit.update(schema=SCHEMA, rule=RULE, target_slots=TARGET_SLOTS,
        maximum_filtered_targets_across_all_views=maximum, overflow_rows=0,
        no_target_dependent_model_slots=True,
        source_train_charts=324, source_train_windows=6699,
        source_train_components=len(train_components), source_dev_windows=34,
        source_dev_components=len(dev_components), sealed_components_excluded=len(sealed),
        sealed_target_arrays_opened=0, raw_target_arrays_opened=len(raw_sha),
        exposure={k:{**v,'types':dict(v['types'])} for k,v in exposure.items()},
        payload_files_hash_verified=payload_checked, unique_public_head_layouts=len(layouts),
        cloud_calls=0, optimizer_updates=0)
    write(output / 'audit.json', audit)
    write(output / 'priors.json', priors)
    manifest = dict(schema=SCHEMA, complete=True, target_slots=TARGET_SLOTS,
        public_request_resource_limits=dict(max_events=TARGET_SLOTS,max_tokens=8192),
        resource_authority='Parent candidate implementation resource contract;these numbers were not directly specified by Owner',
        rule=RULE, arrays=array_paths, records=records, audio_records=audio_records,
        source_sha256=source_sha, raw_target_sha256=raw_sha,
        source_train_indices=train, source_dev_indices=dev,
        train_views=list(TRAIN_VIEWS), dev_hidden_label_controls=False,
        component_splits=dict(train=sorted(train_components),dev=sorted(dev_components)),
        priors=str((output/'priors.json').relative_to(root)), audit=str((output/'audit.json').relative_to(root)),
        preparation_wall_s=time.monotonic()-started, cloud_calls=0, optimizer_updates=0,
        runtime_contract='CPU workers slice numeric MERT/mel,targets and references;no chart filtering or partitioning in GPU loop')
    write(output / 'manifest.json', manifest)
    for path in sorted(output.glob('*.json')) + sorted(output.glob('*.npy')):
        rel = str(path.relative_to(root))
        upload[rel] = dict(bytes=path.stat().st_size, sha256=sha(path), role='prepared_cpu_data')
    upload_manifest = dict(schema=SCHEMA, upload_performed=False, files=upload,
        total_bytes=sum(r['bytes'] for r in upload.values()),
        source_asset_bytes=sum(r['bytes'] for r in upload.values() if r['role']!='prepared_cpu_data'),
        prepared_bytes=sum(r['bytes'] for r in upload.values() if r['role']=='prepared_cpu_data'),
        source_sha256=source_sha, mount_contract='Preserve each listed repo-relative path below the selected data root',
        excluded='Pupu arrays/scales/summaries,Mania labels,sealed-confirm target arrays,weights,credentials')
    write(output / 'upload-manifest.json', upload_manifest)
    print(json.dumps(dict(status='prepared',rows=len(records),max_events=maximum,
        upload_bytes=upload_manifest['total_bytes'],wall_s=time.monotonic()-started)),flush=True)
    return manifest, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/modal120m-data-v1/prepared-v3')
    args = parser.parse_args()
    prepare(ROOT, ROOT / args.output)


if __name__ == '__main__':
    main()

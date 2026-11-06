"""Read and count frozen training metadata; no app, inference or optimizer."""
import asyncio
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import modal
import torch

from beatfcn.model import features as F
from beatfcn.model.batch import unpack_batch
from beatfcn.train.window_controls import window_control_view

ROOT = Path(__file__).resolve().parents[1]


async def main():
    started = time.perf_counter(); torch.set_num_threads(4)
    plan_path = ROOT / 'bench/rhythm-v2/d10-actual-request-support-v1.json'
    plan = json.loads(plan_path.read_text())
    manifest = ROOT / 'reports/r03_window_raw/manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != plan['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    output = ROOT / 'inputs/d10_actual_request_support'
    output.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    files = []; transferred = 0
    async def fetch(part):
        nonlocal transferred
        path = output / part['file']
        if path.exists():
            raw = path.read_bytes()
        else:
            chunks = bytearray()
            async for chunk in volume.read_file.aio(f"{plan['remote_prefix']}/{part['file']}"):
                transferred += len(chunk); chunks.extend(chunk)
                if len(chunks) > part['bytes'] or transferred > plan['expected_transfer_bytes']:
                    raise ValueError('download exceeded frozen size')
            raw = bytes(chunks)
        if len(raw) != part['bytes'] or hashlib.sha256(raw).hexdigest() != part['sha256']:
            raise ValueError('compressed tape identity differs')
        path.write_bytes(raw)
        files.append(dict(file=part['file'], sha256=part['sha256'], bytes=len(raw)))
    await asyncio.wait_for(asyncio.gather(*(fetch(p) for p in plan['parts'])), timeout=55)
    with (ROOT / 'reports/d10_transfer_receipts.jsonl').open('a') as receipt:
        receipt.write(json.dumps(dict(at_unix=time.time(),
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            transferred_bytes=transferred, verified_files=files, remote_compute_started=False)) + '\n')
    rows = []
    for shard, part in zip(plan['part_indices'], plan['parts'], strict=True):
        with gzip.GzipFile(fileobj=io.BytesIO((output / part['file']).read_bytes())) as packed:
            raw = packed.read(part['raw_bytes'] + 1)
        if len(raw) != part['raw_bytes'] or hashlib.sha256(raw).hexdigest() != part['raw_sha256']:
            raise ValueError('decoded tape identity differs')
        batches = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        del raw
        if len(batches) != part['batches']:
            raise ValueError('batch count differs')
        for position, (packed, skipped) in enumerate(batches):
            bt = unpack_batch(packed)
            if len(bt['mode']) != plan['micro_batch']:
                raise ValueError('microbatch size differs')
            views = {v: window_control_view(bt, v)['controls'] for v in plan['condition_views']}
            for i, provenance in enumerate(bt['_provenance']):
                known = bt['known'][i][bt['known_valid'][i]]
                refs = int(bt['refs_valid'][i].sum())
                anchors = int(bt['copy_valid'][i].sum())
                grid_has_events = bool((bt['blk_grid'][i] != 0).any())
                controls = {view: [field for j, field in enumerate(F.CONTROL_FIELDS) if int(values[i, j]) >= 0]
                            for view, values in views.items()}
                target_groups = int(bt['grp_valid'][i].sum())
                target_notes = int(bt['grp_n'][i][bt['grp_valid'][i]].sum())
                if target_notes != provenance['notes']:
                    raise ValueError('target note count differs from actual provenance')
                rows.append(dict(shard=shard, batch_within_shard=position, core=i,
                    mode=provenance['mode'], chart_id=provenance['chart_id'], song=provenance['song'],
                    c0_us=provenance['c0_us'], kind=provenance['kind'],
                    known_events=len(known), known_before_core=int((known[:, 0] < 0).sum()),
                    known_teacher_prefix_events=int((known[:, 7] == 1).sum()),
                    reference_events=refs, copy_anchors=anchors, song_grid_has_events=grid_has_events,
                    blank_context=not(len(known) or refs or anchors or grid_has_events),
                    target_notes=target_notes, target_groups=target_groups,
                    first_target_ms=float(bt['grp_t'][i][bt['grp_valid'][i]][0]) * 1000 if target_groups else None,
                    full_editable_core=bool(bt['bin_ok'][i].all()), controls=controls))
        del batches, bt
    if len(rows) != plan['expected_batches'] * plan['micro_batch']:
        raise ValueError('record count differs')
    summary = {}
    for mode in ('mania_4k', 'phira'):
        for kind in ('all', 'density_window', 'from_scratch', 'continuation', 'middle_holes', 'sparse_anchors', 'reference'):
            subset = [r for r in rows if r['mode'] == mode and (kind == 'all' or r['kind'] == kind)]
            if not subset:
                continue
            views = {}
            for view in plan['condition_views']:
                singleton = [r for r in subset if len(r['controls'][view]) == 1]
                views[view] = dict(numeric_field_count_histogram=dict(Counter(len(r['controls'][view]) for r in subset)),
                    single_field_cores=len(singleton),
                    single_field_support=dict(Counter(r['controls'][view][0] for r in singleton)),
                    blank_context_single_field_cores=sum(r['blank_context'] for r in singleton))
            summary[f'{mode}/{kind}'] = dict(cores=len(subset), distinct_audio=len({r['song'] for r in subset}),
                blank_context_cores=sum(r['blank_context'] for r in subset),
                blank_context_nonempty_target_cores=sum(r['blank_context'] and r['target_notes'] > 0 for r in subset),
                blank_context_empty_target_cores=sum(r['blank_context'] and r['target_notes'] == 0 for r in subset),
                with_known_teacher_prefix=sum(r['known_teacher_prefix_events'] > 0 for r in subset),
                with_song_grid_events=sum(r['song_grid_has_events'] for r in subset), condition_views=views)
    result = dict(schema='beatfcn.actual_request_support.v1', complete=True,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), files=files,
        transferred_bytes=transferred, remote_compute_started=False, training_updates=0,
        rows=rows, summary=summary, limitation=plan['limitation'], function_wall_s=time.perf_counter()-started)
    (ROOT / 'reports/d10_actual_request_support.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cores=len(rows), transferred_bytes=transferred, summary={k: v for k, v in summary.items() if k.endswith('/all')}), ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())

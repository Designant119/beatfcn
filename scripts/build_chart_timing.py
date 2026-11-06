"""Charter-timing table: chart_id -> [(start_us, beat_us, meter), ...] on the MODEL clock + validity.

For every chart of a request tape, read the raw human chart's timing points
(osu: uninherited timing points time/beatLength/meter; Phira: the package
chart's BPM list, meter 4 -- PyYAML-free path of run_workflow_benchmark.
load_raw_chart), move the grid by the target's model-clock shift (0 -> target
receipt ``model_shift_us`` via --receipts-root -> modal ms estimate, each
accepted only at >= 95% of target onsets within 1 ms of a raw onset; same rule
as run_workflow_benchmark.reference_grid) and validate it: >= 90% of the chart's
distinct target onsets within 5 ms of a 1/1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12 or
1/16 point (else ``timing_invalid``). Only ``valid`` charts are served to the
model (beatfcn.train.charter_timing); every other chart gets the absent state.

Statuses: valid | timing_invalid | clock_mismatch | no_timing (e.g. Phira
official per-line BPMs) | raw_missing | parse_error | no_onsets.

Parallel (threads over chunks) and resumable: each chunk is written atomically
to ``<out>.parts/part-NNNNN.json`` and skipped on a rerun with the same
settings; the final table is merged into ``--out`` (+ coverage summary).
Read-only on the inputs; never trains, never schedules cloud work.

  # local 580 tape
  .venv/bin/python -m scripts.build_chart_timing --out reports/chart-timing/native-v2-d94-580.json
  # full database (cloud CPU next to the volume)
  python -m scripts.build_chart_timing --root / --tape /data/full-database-v1/native-v2-tape/tape.json \
      --raw-root /data/raw --receipts-root /data/full-database-v1/targets \
      --out /data/full-database-v1/chart-timing/chart_timing.json --processes 16
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatfcn.train.charter_timing import SCHEMA  # noqa: E402
from beatfcn.train.workflow_metrics import CHARTER_DIVISORS, ON_GRID_US, subdivision_classes  # noqa: E402
from scripts.run_workflow_benchmark import (RAW_ROOT, TAPE, ClockShifts, default_receipts_root,  # noqa: E402
    load_raw_chart, raw_chart_path, resolve_clock_shift, sha_file)

VERSION = 'build-chart-timing-v1'
MIN_ON_GRID = .9
CLOCK_ALIGNMENT = .95
STATUSES = ('valid', 'timing_invalid', 'clock_mismatch', 'no_timing', 'raw_missing', 'parse_error', 'no_onsets')


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_atomic(path, value, *, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=indent, separators=None if indent else (',', ':'), allow_nan=False)+'\n')
    tmp.replace(path)


# Changing only charts that previously crashed the build: parts written before this guard stay valid
# (same VERSION / settings), because a chart that hit it could never have been written.
MIN_BEAT_US, MAX_BEAT_US = 1_000.0, 60_000_000.0  # 60000 BPM .. 1 BPM


class GimmickTiming(ValueError):
    """Timing map with a stop/teleport BPM: no usable beat grid -> timing_invalid (absent state)."""


def timing_segments(raw):
    """Raw chart timing -> [[start_us (audio clock), beat_us, meter], ...] (red-line order; at equal
    starts the later point wins, osu rule). Meter: osu uninherited point's meter, else 4."""
    timing = raw.timing
    rows = (raw.sidecar or {}).get('timing_rows') if raw.source_format.startswith('osu') else None
    meters, meter_source = [4]*len(timing.segments), 'default_4'
    if rows and len(rows) == len(timing.segments):
        meters, meter_source = [max(1, int(r['meter'])) for r in rows], 'osu_timing_point'
    segments = []
    for seg, meter in zip(timing.segments, meters):
        start = float((Fraction(seg.start_sec)+Fraction(timing.offset_sec))*1_000_000)
        try:
            beat = float(Fraction(60_000_000)/Fraction(seg.bpm))
        except (OverflowError, ZeroDivisionError) as exc:  # osu "stop" gimmicks: BPM ~0 / absurd
            raise GimmickTiming(f'unusable BPM {seg.bpm!r} at {seg.start_sec}s ({type(exc).__name__})') from exc
        if not (MIN_BEAT_US <= beat <= MAX_BEAT_US):
            raise GimmickTiming(f'beat {beat:.6g} us outside [{MIN_BEAT_US}, {MAX_BEAT_US}] at {seg.start_sec}s')
        if segments and abs(segments[-1][0]-start) < 1e-6:
            segments[-1] = [start, beat, meter]
        else:
            segments.append([start, beat, meter])
    return segments, meter_source


def target_onsets(root, chart):
    path = root/chart['target_path']
    if chart.get('target_sha256') and sha_file(path) != chart['target_sha256']:
        raise ValueError(f'Human target changed: {path}')
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z['onset_us'], np.int64)


def chart_entry(root, raw_root, chart, clock_shifts, *, min_on_grid=MIN_ON_GRID):
    chart_id = chart['chart_id']
    entry = dict(valid=False, mode=chart['mode'], source=chart_id.split(':', 1)[0])
    path = raw_chart_path(raw_root, chart_id)
    if not path.is_file():
        return dict(entry, status='raw_missing', reason=f'raw chart not found: {path}')
    try:
        raw = load_raw_chart(path, chart_id)
    except Exception as exc:  # noqa: BLE001 - recorded per chart
        return dict(entry, status='parse_error', reason=f'{type(exc).__name__}: {exc}'[:300])
    entry['source_format'] = raw.source_format
    if raw.timing is None:
        return dict(entry, status='no_timing', reason='chart has no single timing map (e.g. per-line BPMs)')
    onsets = target_onsets(root, chart)
    if not len(onsets):
        return dict(entry, status='no_onsets', reason='target has no notes')
    shift, source, alignment, tried = resolve_clock_shift(raw, chart_id, onsets.tolist(), clock_shifts=clock_shifts)
    entry.update(clock_alignment_1ms=round(float(alignment), 6))
    if alignment < CLOCK_ALIGNMENT:
        return dict(entry, status='clock_mismatch', shifts_tried_us=sorted(int(k) for k in tried),
                    reason=f'raw chart clock disagrees with target ({alignment:.3f} within 1 ms at every tried shift)')
    try:
        segments, meter_source = timing_segments(raw)
    except GimmickTiming as exc:
        return dict(entry, status='timing_invalid', reason=f'gimmick timing: {exc}'[:300])
    segments = [[round(s+shift, 3), round(b, 6), int(m)] for s, b, m in segments]
    heads = np.unique(onsets)
    classes = subdivision_classes(heads, [(s, b) for s, b, _ in segments], divisors=CHARTER_DIVISORS,
                                  tolerance_us=ON_GRID_US)
    on_grid = float(np.mean(classes < len(CHARTER_DIVISORS)))
    valid = on_grid >= min_on_grid
    entry.update(valid=bool(valid), status='valid' if valid else 'timing_invalid', segments=segments,
                 meter_source=meter_source, clock_shift_us=int(shift), clock_shift_source=source,
                 on_grid_fraction=round(on_grid, 6), onsets=int(len(heads)),
                 first_bpm=round(60e6/segments[0][1], 4))
    if not valid:
        entry['reason'] = f'{on_grid:.3f} of onsets within 5 ms of a 1/1..1/16 or triplet point (< {min_on_grid})'
    return entry


def _quantiles(values, qs=(0, .1, .5, .9, 1)):
    return None if not values else {str(q): round(float(np.quantile(values, q)), 4) for q in qs}


def coverage(charts):
    def block(items):
        items = list(items)
        status = {k: 0 for k in STATUSES}
        for _, v in items:
            status[v['status']] = status.get(v['status'], 0)+1
        valid = [v for _, v in items if v['valid']]
        return dict(charts=len(items), valid=len(valid),
                    valid_fraction=round(len(valid)/len(items), 6) if items else None,
                    status={k: n for k, n in status.items() if n})
    items = sorted(charts.items())
    out = dict(overall=block(items), by_mode={}, by_source={})
    for key, field in (('by_mode', 'mode'), ('by_source', 'source')):
        for value in sorted({v[field] for _, v in items}):
            out[key][value] = block((c, v) for c, v in items if v[field] == value)
    timed = [v for _, v in items if 'clock_shift_source' in v]
    shifts = {}
    for v in timed:
        k = f"{v['clock_shift_source']}:{v['clock_shift_us']}"
        shifts[k] = shifts.get(k, 0)+1
    valid = [v for _, v in items if v['valid']]
    meters = {}
    for v in valid:
        for m in {s[2] for s in v['segments']}:
            meters[str(m)] = meters.get(str(m), 0)+1
    out.update(clock_shift=dict(sorted(shifts.items(), key=lambda kv: -kv[1])),
               on_grid_fraction_valid=_quantiles([v['on_grid_fraction'] for v in valid]),
               on_grid_fraction_invalid=_quantiles([v['on_grid_fraction'] for _, v in items
                                                    if v['status'] == 'timing_invalid' and 'on_grid_fraction' in v]),
               gimmick_timing=sum(1 for _, v in items if str(v.get('reason', '')).startswith('gimmick timing')),
               segments_per_valid_chart=_quantiles([len(v['segments']) for v in valid]),
               multi_segment_valid_fraction=(round(float(np.mean([len(v['segments']) > 1 for v in valid])), 6)
                                             if valid else None),
               meters_in_valid_charts=dict(sorted(meters.items())))
    return out


def run_chunk(root, raw_root, chunk, receipts_root, canonical, settings_sha, path, min_on_grid):
    """One resumable part (top level so a process pool can pickle it)."""
    clock_shifts = ClockShifts(receipts_root, canonical)
    out = {c['chart_id']: chart_entry(root, raw_root, c, clock_shifts, min_on_grid=min_on_grid) for c in chunk}
    part = dict(settings_sha256=settings_sha, chart_ids=[c['chart_id'] for c in chunk], charts=out)
    write_atomic(path, part)
    return part


def build(args):
    root = Path(args.root)
    tape_path = root/args.tape
    raw_root = root/args.raw_root
    manifest = json.loads(tape_path.read_text())
    charts = sorted(manifest['charts'], key=lambda c: c['chart_id'])
    if len({c['chart_id'] for c in charts}) != len(charts):
        raise ValueError('Duplicate chart ids in tape')
    if args.limit:
        charts = charts[:args.limit]
    receipts_root = args.receipts_root or default_receipts_root(root, args.tape)
    canonical = {c['chart_id']: c.get('canonical_sha256') for c in charts if c.get('canonical_sha256')}
    tape_sha = sha_file(tape_path)
    settings = dict(version=VERSION, tape=str(args.tape), tape_sha256=tape_sha, raw_root=str(args.raw_root),
                    receipts_root=None if receipts_root is None else str(receipts_root), min_on_grid=args.min_on_grid,
                    on_grid_us=ON_GRID_US, divisors=list(CHARTER_DIVISORS), clock_alignment_1ms=CLOCK_ALIGNMENT,
                    limit=args.limit)
    settings_sha = sha_json(settings)
    parts_dir = Path(str(args.out)+'.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunks = [charts[i:i+args.chunk] for i in range(0, len(charts), args.chunk)]

    def part_path(k):
        return parts_dir/f'part-{k:05d}.json'

    def done(k, chunk):
        path = part_path(k)
        if not path.is_file():
            return None
        try:
            part = json.loads(path.read_text())
        except ValueError:
            return None
        ids = [c['chart_id'] for c in chunk]
        return part if part.get('settings_sha256') == settings_sha and part.get('chart_ids') == ids else None

    started = time.monotonic()
    results, todo = {}, []
    for k, chunk in enumerate(chunks):
        part = done(k, chunk)
        if part is None:
            todo.append(k)
        else:
            results[k] = part
    print(json.dumps(dict(phase='start', charts=len(charts), chunks=len(chunks), resumed_chunks=len(results),
                          todo_chunks=len(todo), workers=args.workers, processes=args.processes,
                          settings_sha256=settings_sha)), flush=True)
    finished = 0
    executor = (ProcessPoolExecutor(max_workers=args.processes) if args.processes
                else ThreadPoolExecutor(max_workers=args.workers))
    with executor as pool:
        futures = {pool.submit(run_chunk, root, raw_root, chunks[k], receipts_root,
                               {c['chart_id']: canonical[c['chart_id']] for c in chunks[k] if c['chart_id'] in canonical},
                               settings_sha, part_path(k), args.min_on_grid): k for k in todo}
        for future in as_completed(futures):
            k = futures[future]
            results[k] = future.result()
            finished += 1
            if finished % max(1, args.progress_every) == 0 or finished == len(todo):
                elapsed = time.monotonic()-started
                print(json.dumps(dict(phase='progress', chunks_done=finished, chunks_todo=len(todo),
                                      elapsed_s=round(elapsed, 1),
                                      eta_s=round(elapsed/finished*(len(todo)-finished), 1))), flush=True)
    merged = {}
    for k in range(len(chunks)):
        merged.update(results[k]['charts'])
    table = dict(schema=SCHEMA, settings=settings, settings_sha256=settings_sha, created_unix=int(time.time()),
                 feature_rule='beatfcn.train.charter_timing (segments on the model clock; only valid charts served)',
                 coverage=coverage(merged), charts=dict(sorted(merged.items())))
    write_atomic(args.out, table)
    summary = dict(out=str(args.out), sha256=sha_file(args.out), wall_s=round(time.monotonic()-started, 1),
                   coverage=table['coverage'])
    write_atomic(Path(str(args.out)+'.summary.json'), summary, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--tape', default=TAPE)
    p.add_argument('--raw-root', default=RAW_ROOT)
    p.add_argument('--receipts-root', type=Path, help='full-database target receipts root '
                   '(<root>/targets/<key>/receipt.json); default <tape dir>/../targets when present')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16, help='threads (used when --processes is 0)')
    p.add_argument('--processes', type=int, default=0, help='process pool size (parsing is GIL-bound; '
                   'use ~#cores on a cloud CPU box); 0 = threads only')
    p.add_argument('--chunk', type=int, default=256, help='charts per resumable part file')
    p.add_argument('--min-on-grid', type=float, default=MIN_ON_GRID)
    p.add_argument('--limit', type=int, help='first N charts (by chart_id) only, for smoke tests')
    p.add_argument('--progress-every', type=int, default=10)
    args = p.parse_args(argv)
    if args.workers < 1 or args.processes < 0 or args.chunk < 1 or not 0 < args.min_on_grid <= 1:
        p.error('invalid --workers / --chunk / --min-on-grid')
    build(args)


if __name__ == '__main__':
    main()

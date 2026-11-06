"""Build a native-v1 / strict-drag-v2 tape directly from full-database receipts.

The full database already stores filtered.npy and receipts per chart; this avoids
re-reading canonical archives.  Output target/token files are written beside the
tape and referenced by absolute paths on the Modal volume. Charts are built in
parallel and resume from verified per-chart sidecars; a --limit smoke writes
tape-limitN.json and never overwrites the full tape.json.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.audio.mert2 import SR
from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES, build_model_view
from beatfcn.train.event_density_drag_filter import RULE
from scripts.prepare_local_request_tape import requests_for_chart

NAMES = {index: name for index, name in enumerate(TYPES)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chart_events(path, mode):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError('filtered target must be [N,4]')
    return [Event(f'e{index}', int(onset), int(end), NAMES[int(kind)],
                  LineLoc('rhythm', 'above', 0.) if mode != 'mania_4k' else LaneLoc(int(lane)))
            for index, (onset, end, kind, lane) in enumerate(values)]


def duration_us(features_root, audio):
    receipt = json.loads((Path(features_root)/audio/'receipt.json').read_text())
    return int(receipt['metadata']['real_samples']) * 1_000_000 // SR


def build_chart(job):
    """Heavy per-chart work; resumable through a sidecar written last."""
    folder, receipt, features_root, output = job
    chart_id, audio, mode = receipt['chart_id'], receipt['audio_sha256'], receipt['mode']
    stem = chart_id.replace(':', '-')
    tp, kp = Path(output)/f'{stem}-targets.npz', Path(output)/f'{stem}-tokens.npy'
    sidecar = Path(output)/f'{stem}-chart.json'
    if sidecar.is_file() and tp.is_file() and kp.is_file():
        chart = json.loads(sidecar.read_text())
        chart['component'] = chart.get('component') or f'a:{audio}'
        if chart.get('target_sha256') == sha(tp) and chart.get('token_sha256') == sha(kp):
            return chart
    try:
        raw = chart_events(Path(folder)/'filtered.npy', mode)
        # Same millisecond model view and lossless rule as the 580-chart tape
        # (scripts/prepare_native_training_tape.py): quantize, reject collisions.
        view = build_model_view(mode, (), tuple(raw))
        events = tuple(e for g in view.target_groups for e in g.events)
        if view.quarantine or len(events) != len(raw):
            raise ValueError('Native model view is not lossless under the event codec: '
                             + ','.join(sorted({i.code for i in view.quarantine})))
        duration = duration_us(features_root, audio)
        rows, tokens, targets = requests_for_chart(events, mode, duration, receipt['split'])
        np.savez(tp, **targets)
        np.save(kp, tokens, allow_pickle=False)
        chart = dict(chart_id=chart_id, component=receipt.get('component') or f'a:{audio}',
            split=receipt['split'], mode=mode, stratum=receipt.get('stratum'),
            audio_sha256=audio, canonical_sha256=receipt['canonical_sha256'],
            model_rhythm_sha256=receipt.get('model_rhythm_sha256'),
            duration_us=duration, target_path=str(tp), token_path=str(kp),
            target_sha256=sha(tp), token_sha256=sha(kp), requests=rows)
        temporary = sidecar.with_suffix('.tmp')
        temporary.write_text(json.dumps(chart, ensure_ascii=False))
        temporary.replace(sidecar)
        return chart
    except Exception as exc:
        return dict(failure=True, folder=str(folder), chart_id=chart_id,
                    reason=f'{type(exc).__name__}: {exc}')


def main():
    from concurrent.futures import ProcessPoolExecutor
    import os
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets-root', required=True)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--io-threads', type=int, default=32)
    args = parser.parse_args()
    root = Path(args.targets_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs, seen, splits, skipped = [], set(), {}, []
    folders = sorted(root.iterdir())

    def read_receipt(folder):
        # Network-volume metadata reads dominate; done concurrently, decided serially.
        try:
            if not (folder/'filtered.npy').is_file():
                return folder, None, None
            return folder, json.loads((folder/'receipt.json').read_text()), None
        except FileNotFoundError:
            return folder, None, None
        except Exception as exc:
            return folder, None, f'{type(exc).__name__}: {exc}'

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.io_threads) as io:
        receipts = list(io.map(read_receipt, folders))
    print(json.dumps(dict(phase='receipts', folders=len(folders),
                          readable=sum(r is not None for _, r, _ in receipts))), flush=True)
    # Cheap serial decisions: identity, de-duplication and audio-level split checks.
    for folder, receipt, error in receipts:
        if error:
            skipped.append(dict(folder=str(folder), reason=error))
            continue
        if receipt is None:
            continue
        chart_id, audio = receipt['chart_id'], receipt['audio_sha256']
        if chart_id in seen:
            continue
        seen.add(chart_id)
        if audio in splits and splits[audio] != receipt['split']:
            skipped.append(dict(chart_id=chart_id, reason='audio_cross_split'))
            continue
        splits[audio] = receipt['split']
        jobs.append((str(folder), receipt, args.features_root, str(output)))
        if args.limit and len(jobs) >= args.limit:
            break
    started, charts = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(build_chart, jobs, chunksize=8)):
            if result.get('failure'):
                skipped.append(result)
            else:
                charts.append(result)
            if index % 500 == 0:
                print(json.dumps(dict(done=index+1, of=len(jobs), charts=len(charts),
                    skipped=len(skipped), elapsed_s=round(time.monotonic()-started, 1))), flush=True)
    charts.sort(key=lambda c: c['chart_id'])
    tape = dict(complete=True, failures=[], skipped=skipped,
        clock_contract='native-v1', drag_policy_id=RULE['id'],
        source_tape_sha256=None, canonical_receipt_sha256=None,
        charts=charts, removals=[], limit=args.limit or None,
        summary={f'{m}/{s}': n for (m, s), n in sorted(Counter((c['mode'], c['split']) for c in charts).items())})
    name = 'tape.json' if not args.limit else f'tape-limit{args.limit}.json'
    temporary = output/(name+'.tmp')
    temporary.write_text(json.dumps(tape, ensure_ascii=False, indent=1)+'\n')
    temporary.replace(output/name)
    print(json.dumps(dict(charts=len(charts), requests=sum(len(c['requests']) for c in charts),
        skipped=len(skipped), complete=tape['complete'], output=str(output/name)), ensure_ascii=False))


if __name__ == '__main__':
    main()

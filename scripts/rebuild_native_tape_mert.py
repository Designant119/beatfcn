"""Rebuild a native tape with the MERT receipt clock (D94 tape can differ by >1us)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.audio.mert2 import SR
from beatfcn.events.canonical import Event, LaneLoc, LineLoc, TYPES
from beatfcn.train.event_density_drag_filter import RULE
from scripts.prepare_local_request_tape import requests_for_chart

NAMES = {i: name for i, name in enumerate(TYPES)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration(features_root, audio):
    receipt = json.loads((Path(features_root)/audio/'receipt.json').read_text())
    return int(receipt['metadata']['real_samples']) * 1_000_000 // SR


def events_from(path, mode):
    with np.load(path, allow_pickle=False) as archive:
        return [Event(f'e{i}', int(on), int(end), NAMES[int(kind)],
                      LaneLoc(int(lane)) if mode == 'mania_4k' else LineLoc('rhythm', 'above', 0.))
                for i, (on, end, kind, lane) in enumerate(zip(
                    archive['onset_us'], archive['end_us'], archive['kind'], archive['lane']))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', required=True)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.tape).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    charts, skipped = [], []
    for chart in source['charts']:
        try:
            dur = duration(args.features_root, chart['audio_sha256'])
            events = events_from(chart['target_path'], chart['mode'])
            rows, tokens, targets = requests_for_chart(events, chart['mode'], dur, chart['split'])
            stem = chart['chart_id'].replace(':', '-')
            tp, kp = output/f'{stem}-targets.npz', output/f'{stem}-tokens.npy'
            np.savez(tp, **targets)
            np.save(kp, tokens, allow_pickle=False)
            charts.append(dict(chart, duration_us=dur, target_path=str(tp), token_path=str(kp),
                target_sha256=sha(tp), token_sha256=sha(kp), requests=rows))
        except Exception as exc:
            skipped.append(dict(chart_id=chart.get('chart_id'), reason=str(exc)))
    tape = dict(complete=True, failures=[], skipped=skipped,
        clock_contract='native-v1', drag_policy_id=RULE['id'],
        source_tape_sha256=sha(args.tape), canonical_receipt_sha256=None,
        charts=charts, removals=[])
    (output/'tape.json').write_text(json.dumps(tape, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(dict(charts=len(charts), requests=sum(len(c['requests']) for c in charts),
        skipped=len(skipped), output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()

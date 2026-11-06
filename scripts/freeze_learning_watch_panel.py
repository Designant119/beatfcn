"""Freeze held-out sample identities without loading weights or starting compute."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_panel(tape, targets, expected_tape_sha, checkpoint_sha, *, per_mode=3):
    if digest(tape) != expected_tape_sha:
        raise ValueError('Tape differs from the frozen training plan')
    manifest = json.loads(Path(tape).read_text())
    if not manifest.get('complete') or manifest.get('failures'):
        raise ValueError('Incomplete or failed target tape')
    train_audio = {c['audio_sha256'] for c in manifest['charts'] if c['split'] == 'train'}
    train_components = {c['component'] for c in manifest['charts'] if c['split'] == 'train'}
    candidates, empty, offset = {}, {}, 0
    verified = set()
    for chart in manifest['charts']:
        if chart['split'] != 'val':
            offset += len(chart['requests'])
            continue
        audio, component = chart['audio_sha256'], chart['component']
        if audio in train_audio or component in train_components:
            raise ValueError('Held-out audio or component overlaps training')
        target = Path(targets) / Path(chart['target_path']).name
        if digest(target) != chart['target_sha256']:
            raise ValueError('Target identity differs from the frozen tape')
        verified.add(chart['target_sha256'])
        with np.load(target, allow_pickle=False) as arrays:
            for index, request in enumerate(chart['requests']):
                sl = slice(request['target_start'], request['target_stop'])
                kinds = Counter(TYPES[int(k)] for k in arrays['kind'][sl])
                row = dict(index=offset+index, chart_id=chart['chart_id'], audio_sha256=audio,
                    component=component, split='val', mode=chart['mode'],
                    request=request['request'], target_sha256=chart['target_sha256'],
                    event_count=sum(kinds.values()), type_counts=dict(kinds))
                if kinds:
                    key = chart['mode'], audio
                    previous = candidates.get(key)
                    rank = (len(kinds), bool(kinds['hold']), -row['index'])
                    if previous is None or rank > previous[0]:
                        candidates[key] = rank, row
                elif (chart['mode'], audio) not in empty:
                    empty[chart['mode'], audio] = row
        offset += len(chart['requests'])
    selected = []
    used_components, used_audio = set(), set()
    for mode in ('phira', 'mania_4k'):
        options = sorted((value for (m, _), value in candidates.items() if m == mode),
                         key=lambda value: (-value[0][0], -value[0][1], value[1]['index']))
        chosen = []
        for _, row in options:
            if row['component'] not in used_components and row['audio_sha256'] not in used_audio:
                chosen.append(dict(row, case_role='positive'))
                used_components.add(row['component'])
                used_audio.add(row['audio_sha256'])
            if len(chosen) == per_mode:
                break
        if len(chosen) != per_mode:
            raise ValueError(f'Insufficient independent held-out songs for {mode}')
        selected.extend(chosen)
        silence = sorted((r for (m, _), r in empty.items() if m == mode), key=lambda r: r['index'])
        if silence:
            selected.append(dict(silence[0], case_role='natural_silence'))
    return dict(schema='learning-watch-fixed-panel-v1', tape_sha256=expected_tape_sha,
        checkpoint_sha256=checkpoint_sha, per_mode=per_mode, samples=selected,
        verified_target_files=len(verified), distinct_positive_songs=len(used_components),
        selection='deterministic type-diverse/hold-positive screening; not random corpus estimate',
        required_output=['full events and tokens', 'complete/deadline/token-limit flags',
            'source row and checkpoint identities', 'matched per-type counts',
            'hold-head and hold-joint matches and conditional tail errors', 'silence false positives'],
        protocol_ready=False,
        blockers=['Legacy evaluator cannot accept this panel or save complete raw outputs.',
            'Relative-condition, audio counterfactual and whole-song protocols are not implemented.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--checkpoint-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint_sha) != 64 or any(c not in '0123456789abcdef' for c in args.checkpoint_sha):
        parser.error('checkpoint-sha must be a SHA256 hex digest')
    plan = json.loads(args.plan.read_text())
    panel = freeze_panel(args.tape, args.targets, plan['inputs_sha256'][plan['tape']],
                         args.checkpoint_sha)
    panel['source_sha256'] = digest(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(panel, indent=2)+'\n')
    temporary.replace(args.out)
    print(json.dumps(dict(samples=len(panel['samples']),
        distinct_positive_songs=panel['distinct_positive_songs'], protocol_ready=False)))


if __name__ == '__main__':
    main()

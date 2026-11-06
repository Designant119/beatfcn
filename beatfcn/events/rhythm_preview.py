"""Timing-preserving Phira inspection packages; layout is explicitly synthetic."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml

from beatfcn.events.phira import load_package, parse_phira
from scripts.build_phira_preview import plain_background, rpe


def display_positions(events):
    groups = defaultdict(list)
    for index, event in enumerate(events):
        groups[event['onset_us']].append((index, event))
    active, capacity = [], 11
    for onset, group in sorted(groups.items()):
        active = [end for end in active if end > onset]
        capacity = max(capacity, len(active)+len(group))
        active.extend(e['end_us'] for _, e in group if e['type'] == 'hold')
    positions = np.linspace(-450, 450, capacity).tolist()
    busy = [-1]*capacity
    result = []
    for onset, group in sorted(groups.items()):
        free = [i for i, end in enumerate(busy) if end <= onset]
        for (index, event), slot in zip(group, free):
            result.append(dict(id=f'event:{index}', t_us=onset, end_us=event['end_us'],
                               type=event['type'], x=positions[slot]))
            busy[slot] = event['end_us'] if event['type'] == 'hold' else onset
    if len(result) != len(events):
        raise ValueError('Display allocation lost events')
    return result


def write_phira_preview(path, events, audio, *, title, artist, audio_name):
    path, audio = Path(path), Path(audio)
    display = display_positions(events)
    document = rpe(display, 'DRAFT · unrated · display positions only')
    document['META'].update(name=title, composer=artist, song=audio_name,
                            charter='BeatFCN rhythm; synthetic inspection positions')
    raw = json.dumps(document, ensure_ascii=False).encode()
    expected = sorted((e['onset_us'], e['end_us'], e['type']) for e in events)
    parsed = parse_phira(raw)
    if parsed.quarantined or expected != sorted((e.onset_us, e.end_us, e.type) for e in parsed.events):
        raise ValueError('RPE export changed rhythm or failed parsing')
    info = dict(name=title, composer=artist, level='DRAFT · unrated', difficulty=0.,
        charter='BeatFCN rhythm / synthetic display positions', illustrator='Plain background',
        chart='chart.json', music=audio_name, illustration='background.png', offset=0.,
        aspectRatio=16/9, backgroundDim=.5, lineLength=6.,
        intro='Private rhythm inspection. Positions are synthetic, not model output. No playability certification.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('chart.json', raw)
        archive.writestr('info.yml', yaml.safe_dump(info, allow_unicode=True))
        archive.writestr('background.png', plain_background())
        archive.write(audio, audio_name)
    loaded = load_package(path)
    if loaded.quarantined or expected != sorted((e.onset_us, e.end_us, e.type) for e in loaded.events):
        raise ValueError('Packaged Phira rhythm round-trip failed')
    return dict(notes=len(events), roundtrip_exact=True, model_spatial_output=False,
                native_playtest=False, timing_shift_us=0,
                layout='Static synthetic positions; every onset, tail, type and multiplicity preserved.')

"""Blind A/B playtest packages from two saved consensus readouts of the same panel.

Phira: model rhythm + rule-based two-thumb positions on one static line (no model spatial output).
Mania: .osz with a placeholder 120 BPM point. A/B order is hash-chosen per song; the key is in ab-key.json."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

from beatfcn.events.phira import parse_phira
from beatfcn.events.phira_layout import layout_report, two_thumb_layout
from scripts import run_workflow_benchmark as W
from scripts.build_phira_preview import plain_background, rpe


def write_phira(path, events, audio, *, title, artist, audio_name, seed=0):
    placed = two_thumb_layout(events, seed=seed)
    document = rpe(placed, 'BeatFCN draft')
    document['META'].update(name=title, composer=artist, song=audio_name, charter='BeatFCN r5 rhythm + two-thumb rule layout')
    raw = json.dumps(document, ensure_ascii=False).encode()
    expected = sorted((e['onset_us'], e['end_us'], e['type']) for e in events)
    parsed = parse_phira(raw)
    if parsed.quarantined or expected != sorted((e.onset_us, e.end_us, e.type) for e in parsed.events):
        raise ValueError('RPE export changed rhythm or failed parsing')
    info = dict(name=title, composer=artist, level='BeatFCN draft', difficulty=0., charter='BeatFCN r5 + rule layout',
                illustrator='Plain background', chart='chart.json', music=audio_name, illustration='background.png',
                offset=0., aspectRatio=16/9, backgroundDim=.5, lineLength=6.,
                intro='AI draft: rhythm and types by the model, positions by a rule layout, one static line.')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('chart.json', raw)
        archive.writestr('info.yml', yaml.safe_dump(info, allow_unicode=True))
        archive.writestr('background.png', plain_background())
        archive.write(audio, audio_name)
    return dict(notes=len(events), layout=layout_report(placed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new', type=Path, required=True, help='consensus jackfree drafts json of the new rule')
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--meta', type=Path, required=True, help='run-meta.json holding the panel')
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--timing', type=Path, required=True, help='chart_timing json with clock_shift_us per chart')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phira', type=int, default=3)
    p.add_argument('--mania', type=int, default=2)
    p.add_argument('--only', help='comma separated audio sha prefixes to package instead of the hash pick')
    a = p.parse_args()
    new, old = json.loads(a.new.read_text()), json.loads(a.old.read_text())
    songs = json.loads(a.meta.read_text())['panel']['songs']
    a.out.mkdir(parents=True, exist_ok=True)
    rank = lambda s: hashlib.sha256(('ab-20261105'+s['song_id']).encode()).hexdigest()  # noqa: E731
    fit = [s for s in sorted(songs, key=rank) if 90e6 <= s['duration_us'] <= 240e6 and s['song_id'] in new and s['song_id'] in old]
    chosen = [s for s in fit if s['mode'] == 'phira'][:a.phira]+[s for s in fit if s['mode'] == 'mania_4k'][:a.mania]
    if a.only:
        chosen = [s for s in songs if any(s['audio_sha256'].startswith(x) for x in a.only.split(','))]
    key = {}
    timing = json.loads(a.timing.read_text())['charts']
    for s in chosen:
        title, artist = W.song_metadata(a.raw_root, s)
        audio = W.resolve_audio(a.raw_root, s, None)
        primary = W.primary_reference(s)['chart_id']
        shift = int((timing.get(primary) or {}).get('clock_shift_us') or 0)
        source = 'chart_timing clock_shift_us of the primary chart'
        first_is_new = int(rank(s)[0], 16) % 2 == 0
        pair = (('A', new, 'new'), ('B', old, 'old')) if first_is_new else (('A', old, 'old'), ('B', new, 'new'))
        safe = ''.join(c for c in title if c.isalnum() or c in ' -_')[:40].strip() or s['song_id'].replace(':', '-')
        for tag, drafts, which in pair:
            events = drafts[s['song_id']]['events']
            stem = f'{safe}-{tag}'
            folder = a.out/'work'/stem
            if s['mode'] == 'mania_4k':
                r = W.export_song(folder, stem, s, events, audio, title=f'{title} [{tag}]', artist=artist, version=f'BeatFCN {tag}',
                                  clock_shift_us=shift, clock_source=source)
                if r.get('failed'):
                    raise SystemExit(r['failed'])
                (folder.parent/f'{stem}.osz').rename(a.out/f'{stem}.osz')
                info = dict(notes=r['chart'].get('notes') if isinstance(r.get('chart'), dict) else len(events))
            else:
                moved = [dict(e, onset_us=int(e['onset_us'])-shift, end_us=int(e['end_us'])-shift) for e in events]
                moved = [e for e in moved if e['onset_us'] >= 0]
                folder.mkdir(parents=True, exist_ok=True)
                name = f"audio{audio['ext']}"
                W.write_audio(audio, folder/name)
                info = write_phira(a.out/f'{stem}.zip', moved, folder/name, title=f'{title} [{tag}]', artist=artist, audio_name=name)
            key.setdefault(s['song_id'], dict(title=title, mode=s['mode'], shift_us=shift, shift_source=source))[tag] = dict(rule=which, **{k: v for k, v in info.items() if k == 'notes'})
    (a.out/'ab-key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(json.dumps({k: v['title'] for k, v in key.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Write model drafts as editor-openable charts.

osu!mania 4K: a fresh osu file format v14 (not a merge into a human chart; for
that see ``osu_mania.export_osu``). Hit objects reuse ``osu_mania.hit_object_line``
(lane x = column centre, holds type 128 with ``endTime:``). Times are written
as integer milliseconds (round half up from microseconds; how many moved is
reported). Timing: the caller's charter timing points if given, otherwise ONE
placeholder uninherited point (default 120 BPM at 0 ms) so editors open the
file -- the draft carries no BPM claim and the charter is expected to retime.
The written bytes are re-parsed with the repo's own ``parse_osu`` and must
reproduce every head, tail, type and column exactly (fail closed otherwise).

Phira: use ``beatfcn.events.rhythm_preview.write_phira_preview`` (synthetic
display positions, rhythm round trip checked there).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .canonical import Event, LaneLoc, lane_conflicts, round_half_up
from .osu_mania import ExportError, hit_object_line, parse_osu


def _clean(text):
    return ' '.join(str(text).split())


def _ms(us):
    return round_half_up(Fraction(int(us), 1000))


def _format_number(x):
    x = float(x)
    return repr(x) if x != int(x) else str(int(x))


def osu_mania_events(events):
    """Draft dicts/RhythmEvents -> canonical mania Events on integer ms; returns (events, rounded_count)."""
    out, rounded = [], 0
    for i, e in enumerate(events):
        get = e.get if isinstance(e, dict) else lambda k, e=e: getattr(e, k)
        on_us, end_us, typ, lane = int(get('onset_us')), int(get('end_us')), get('type'), int(get('lane'))
        if typ not in ('tap', 'hold'):
            raise ExportError(f'event {i}: osu!mania has no {typ!r}')
        if not 0 <= lane < 4:
            raise ExportError(f'event {i}: lane {lane} outside 4K')
        on, end = _ms(on_us), _ms(end_us)
        rounded += (on*1000 != on_us) or (end*1000 != end_us)
        if typ == 'hold' and end <= on:
            raise ExportError(f'event {i}: hold has no positive length after ms rounding ({on_us}..{end_us} us)')
        if on < 0:
            raise ExportError(f'event {i}: negative time')
        end = end if typ == 'hold' else on
        out.append(Event(f'draft:{i}', on*1000, end*1000, typ, LaneLoc(lane)))
    out.sort(key=lambda e: (e.onset_us, e.location.lane))
    return out, rounded


def osu_mania_bytes(events, *, title='BeatFCN draft', artist='unknown', version='BeatFCN draft',
                    creator='BeatFCN (model draft, unreviewed)', audio_filename='audio.mp3',
                    timing_points=None, placeholder_bpm=120., placeholder_offset_ms=0,
                    overall_difficulty=8, hp_drain=8, preview_time_ms=None, bookmarks_ms=None):
    """osu v14 4K bytes. ``timing_points``: [(time_ms, bpm), ...] charter timing, or None for a placeholder.

    ``preview_time_ms`` / ``bookmarks_ms`` (optional; defaults keep the historical
    bytes) set ``PreviewTime`` and editor ``Bookmarks`` -- e.g. to mark a pilot
    segment's start/end on the full-song audio clock. They are editor metadata only.
    """
    notes, rounded = osu_mania_events(events)
    conflicts = lane_conflicts(notes)
    if conflicts:
        raise ExportError(f'{len(conflicts)} same-lane conflicts, e.g. {conflicts[0]}')
    points = list(timing_points) if timing_points else [(placeholder_offset_ms, placeholder_bpm)]
    if any(bpm <= 0 for _, bpm in points):
        raise ExportError('BPM must be positive')
    preview = -1 if preview_time_ms is None else int(round(preview_time_ms))
    editor = ['[Editor]']
    if bookmarks_ms:
        editor.append('Bookmarks: ' + ','.join(str(int(round(b))) for b in sorted(bookmarks_ms)))
    lines = ['osu file format v14', '', '[General]', f'AudioFilename: {_clean(audio_filename)}', 'AudioLeadIn: 0',
             f'PreviewTime: {preview}', 'Countdown: 0', 'SampleSet: Normal', 'StackLeniency: 0.7', 'Mode: 3',
             'LetterboxInBreaks: 0', 'SpecialStyle: 0', 'WidescreenStoryboard: 0', '', *editor,
             'DistanceSpacing: 1', 'BeatDivisor: 4', 'GridSize: 4', 'TimelineZoom: 1', '', '[Metadata]',
             f'Title:{_clean(title)}', f'TitleUnicode:{_clean(title)}', f'Artist:{_clean(artist)}',
             f'ArtistUnicode:{_clean(artist)}', f'Creator:{_clean(creator)}', f'Version:{_clean(version)}',
             'Source:', 'Tags:beatfcn draft', 'BeatmapID:0', 'BeatmapSetID:-1', '', '[Difficulty]',
             f'HPDrainRate:{hp_drain}', 'CircleSize:4', f'OverallDifficulty:{overall_difficulty}',
             'ApproachRate:5', 'SliderMultiplier:1.4', 'SliderTickRate:1', '', '[Events]',
             '//Background and Video events', '', '[TimingPoints]']
    for time_ms, bpm in sorted(points):
        lines.append(f'{int(round(time_ms))},{_format_number(60000/float(bpm))},4,1,0,100,1,0')
    lines += ['', '[HitObjects]']
    lines += [hit_object_line(e).decode() for e in notes]
    data = ('\r\n'.join(lines)+'\r\n').encode('utf-8')
    parsed = parse_osu(data, 'draft.osu')
    got = sorted((e.onset_us, e.end_us, e.type, e.location.lane) for e in parsed.events)
    want = sorted((e.onset_us, e.end_us, e.type, e.location.lane) for e in notes)
    if parsed.quarantined or got != want:
        raise ExportError('osu round trip changed the draft or was quarantined: '
                          + '; '.join(i.code for i in parsed.quarantine_reasons()))
    return data, dict(notes=len(notes), ms_rounded_events=rounded, roundtrip_exact=True,
                      timing='charter-supplied' if timing_points else
                      f'placeholder single point {placeholder_bpm:g} BPM @ {placeholder_offset_ms} ms')


def write_osu_mania(path, events, **meta):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    data, receipt = osu_mania_bytes(events, **meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(receipt, path=str(path))


def timing_points_from_map(timing):
    """Canonical TimingMap -> [(time_ms, bpm)] on the audio clock (for charter-supplied timing)."""
    return [(float((s.start_sec+timing.offset_sec)*1000), float(s.bpm)) for s in timing.segments]

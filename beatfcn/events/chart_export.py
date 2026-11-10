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



def timing_points_from_map(timing):
    """Canonical TimingMap -> [(time_ms, bpm)] on the audio clock (for charter-supplied timing)."""
    return [(float((s.start_sec+timing.offset_sec)*1000), float(s.bpm)) for s in timing.segments]

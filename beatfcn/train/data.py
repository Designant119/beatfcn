"""Corpus access for training/probes: manifest rows, canon charts, song caches (DATA_RULES §6).

Model-view time = native time + offset_decision.shift_ms + the per-(source,codec) constant c
(DATA_RULES v2: osu c = +14 ms for both mp3 and vorbis, Phira -2..-6 ms; export subtracts c).
Native charts on the Volume are never modified. Charts whose shifted events leave the audio
are skipped by the request builder rather than clipped.
"""
from __future__ import annotations
from dataclasses import replace
from fractions import Fraction
import json
import os
import numpy as np
from ..events.canonical import Chart, Event
from ..model.batch import SongCache

DATA = '/data'


DUP_REASONS = {'duplicate_object', 'duplicate_object_in_source'}
DUP_MAX_STACK = 4
DUP_MAX_FRACTION = 0.01


def dup_only(r) -> bool:
    """Quarantined solely for exact duplicate notes (candidate for model-view collapse, DATA_RULES v3)."""
    reasons = set(r.get('quarantine_reasons') or [])
    return r.get('status') == 'parsed' and bool(reasons) and reasons <= DUP_REASONS


def load_manifest(root=DATA, split=None, kept_only=True, select=True, dup_collapse=True):
    rows = []
    with open(os.path.join(root, 'manifest', 'charts.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if kept_only and not r.get('kept') and not (dup_collapse and dup_only(r)):
                continue
            if split and r.get('split') != split:
                continue
            if select and not selected_for_training(r):
                continue
            rows.append(r)
    return rows


PHIRA_MIN_RATING = 0.92
PHIRA_MIN_VOTES = 100
OSU_APPROVED = (1, 2)  # ranked, approved; loved (4) excluded (DATA_RULES §1)


def selected_for_training(r) -> bool:
    """Quality selection on top of the ingested superset (DATA_RULES §1, 宁缺毋滥)."""
    if r.get('source') == 'phira':
        return float(r.get('rating') or 0) >= PHIRA_MIN_RATING and int(float(r.get('rating_count') or 0)) >= PHIRA_MIN_VOTES
    if r.get('source') == 'osu':
        return int(r.get('approved') or 0) in OSU_APPROVED
    return False


def model_shift_us(row, apply_codec_constant=False) -> int:
    d = row.get('offset_decision') or {}
    ms = int(d.get('shift_ms') or 0) + (int(d.get('c_ms') or 0) if apply_codec_constant else 0)
    return ms * 1000


def collapse_duplicates(events, max_stack=DUP_MAX_STACK, max_fraction=DUP_MAX_FRACTION):
    """Keep one of each exact duplicate (onset, end, type, location). Returns (events, n_removed) or
    raises ValueError when stacks are too high or too frequent (performance/visual use, not a chart)."""
    from collections import Counter
    sig = lambda e: (e.onset_us, e.end_us, e.type, e.location)
    counts = Counter(sig(e) for e in events)
    if not counts:
        return events, 0
    stacked = [c for c in counts.values() if c > 1]
    heads = len({e.onset_us for e in events})
    dup_times = len({k[0] for k, c in counts.items() if c > 1})
    if stacked and (max(stacked) > max_stack or dup_times / max(heads, 1) > max_fraction):
        raise ValueError(f'duplicate stacks ineligible: max {max(stacked)}, {dup_times}/{heads} head times')
    seen, out = set(), []
    for e in events:
        k = sig(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return tuple(out), len(events) - len(out)


def chart_from_canon(doc: dict, shift_us: int = 0) -> Chart:
    evs = []
    for e in doc['events']:
        e = dict(e)
        e.pop('fake', None)
        ev = Event.from_json(e)
        if shift_us:
            ev = replace(ev, onset_us=ev.onset_us + shift_us, end_us=ev.end_us + shift_us)
        evs.append(ev)
    return Chart(doc['mode'], doc['source_format'], str(doc.get('format_version')), tuple(evs), None,
                 Fraction(0), {}, {'source_name': doc.get('native_id', 'chart')}, (), (), doc.get('source_sha256', ''))


class Corpus:
    def __init__(self, root=DATA, taps_key=None, layers=(8, 16, 24), apply_codec_constant=True):
        self.root, self.taps_key, self.layers = root, taps_key, tuple(layers)
        self.apply_c = apply_codec_constant
        self._rows = {}
        self._charts = {}

    def train_rows(self, rows):
        out = []
        for r in rows:
            self._rows[r['chart_id']] = r
            out.append({'chart_id': r['chart_id'], 'song': r['audio_sha256'], 'mode': r['mode'],
                        'duration_us': int(round(r['audio_sec'] * 1e6))})
        return out

    def chart(self, chart_id) -> Chart:
        hit = self._charts.get(chart_id)
        if hit is not None:
            return hit
        if len(self._charts) > 512:
            self._charts.clear()
        self._charts[chart_id] = c = self._load_chart(chart_id)
        return c

    def _load_chart(self, chart_id) -> Chart:
        r = self._rows[chart_id]
        with open(os.path.join(self.root, 'canon', r['source'], f"{r['native_id']}.json")) as f:
            doc = json.load(f)
        chart = chart_from_canon(doc, model_shift_us(r, self.apply_c))
        if dup_only(r):
            from dataclasses import replace as _replace
            events, _ = collapse_duplicates(chart.events)
            chart = _replace(chart, events=events)
        return chart

    def usable(self, row) -> bool:
        """False for duplicate-only charts whose stacks fail the collapse guard (checked once, cached)."""
        if not dup_only(row):
            return True
        self._rows[row['chart_id']] = row
        try:
            self.chart(row['chart_id'])
            return True
        except ValueError:
            return False

    def mel(self, sha):
        return np.load(os.path.join(self.root, 'mel10', f'{sha}.npy'), mmap_mode='r')

    def song(self, sha, in_memory=True) -> SongCache:
        """in_memory: one sequential read per file (network volumes punish many small reads)."""
        base = os.path.join(self.root, 'taps', self.taps_key, sha)
        mm = None if in_memory else 'r'
        taps = {l: np.load(os.path.join(base, f'L{l}.npy'), mmap_mode=mm) for l in self.layers}
        summary = np.load(os.path.join(base, 'S20.npy'), mmap_mode=mm)
        mel = np.load(os.path.join(self.root, 'mel10', f'{sha}.npy'), mmap_mode=mm)
        return SongCache(taps, summary, mel, int(mel.shape[0] * 10_000))

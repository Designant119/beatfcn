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


def load_manifest(root=DATA, split=None, kept_only=True, select=True):
    rows = []
    with open(os.path.join(root, 'manifest', 'charts.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if kept_only and not r.get('kept'):
                continue
            if split and r.get('split') != split:
                continue
            if select and not selected_for_training(r):
                continue
            rows.append(r)
    return rows


PHIRA_MIN_RATING = 0.93
PHIRA_MIN_VOTES = 200
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
        return chart_from_canon(doc, model_shift_us(r, self.apply_c))

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

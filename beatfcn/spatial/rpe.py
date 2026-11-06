"""Auditable RPE subset, derived from the checked-in 2026-10-24 geometry audit.

Rule source: TeamFlos/phira 4ffcb25832f633c3847afd9749c1da92f30cc49b.
This evaluates judgement strips, not sprite placement or a finger simulator.
Unsupported transformations fail explicitly at query time. Sampled packets are
approximations; the evaluator retains native curves and discontinuities.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import math

VERSION = 'rpe-operation-context-v1'
KEYS = ('moveXEvents', 'moveYEvents', 'rotateEvents')
TWEEN = [2, 2, 4, 3, 7, 6, 5, 8, 10, 9, 13, 12, 11, 14,
         16, 15, 19, 18, 22, 21, 25, 24, 23, 26, 28, 27, 31, 30, 32, 29]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EngineProfile:
    engine_revision: str = '4ffcb25832f633c3847afd9749c1da92f30cc49b'
    aspect: float = 16 / 9
    speed: float = 1.0
    mirrored: bool = False
    touch: bool = True

    def validate(self):
        if self != EngineProfile():
            raise ValueError('unsupported geometry profile; only audited 16:9 normal touch 1x is enabled')


def beat(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or value[2] == 0:
        raise ValueError('invalid RPE beat')
    return value[0] + value[1] / value[2]


def tween(tid, x):
    if tid == 2:
        return x
    family, variant = divmod(tid - 3, 3)

    def base(v):
        if family == 0: return 1 - math.cos(v * math.pi / 2)
        if family in (1, 2, 3, 4): return v ** (family + 1)
        if family == 5: return 2 ** (10 * (v - 1))
        if family == 6: return 1 - math.sqrt(max(0, 1 - v * v))
        if family == 7: return (2.70158 * v - 1.70158) * v * v
        if family == 8: return -(2 ** (10 * v - 10) * math.sin((v * 10 - 10.75) * 2 * math.pi / 3))
        if family == 9:
            z = 1 - v
            if z < 1 / 2.75: q = 7.5625 * z * z
            elif z < 2 / 2.75: q = 7.5625 * (z - 1.5 / 2.75) ** 2 + .75
            elif z < 2.5 / 2.75: q = 7.5625 * (z - 2.25 / 2.75) ** 2 + .9375
            else: q = 7.5625 * (z - 2.625 / 2.75) ** 2 + .984375
            return 1 - q
        raise ValueError('unsupported easing')
    if variant == 0: return base(x)
    if variant == 1: return 1 - base(1 - x)
    return base(2 * x) / 2 if x < .5 else 1 - base(2 - 2 * x) / 2


class SpatialChart:
    def __init__(self, raw, info_offset_s=0.0, profile=EngineProfile()):
        profile.validate()
        self.doc = json.loads(raw)
        if 'BPMList' not in self.doc or 'judgeLineList' not in self.doc:
            raise ValueError('operation geometry currently supports RPE only')
        self.profile = profile
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        self.offset = float(self.doc.get('META', {}).get('offset', 0)) / 1000 + info_offset_s
        if not math.isfinite(self.offset):
            raise ValueError('nonfinite audio offset')
        self.cache_identity = {'source_sha256': self.source_sha256, 'compiler': VERSION,
                               'profile': asdict(profile), 'info_offset_s': info_offset_s}
        self.cache_key = digest(self.cache_identity)
        self.bpms = sorted((beat(p['startTime']), float(p['bpm'])) for p in self.doc['BPMList'])
        if (not self.bpms or self.bpms[0][0] != 0 or any(not math.isfinite(t) or not math.isfinite(b) or b <= 0 for t, b in self.bpms)
                or len({t for t, _ in self.bpms}) != len(self.bpms)):
            raise ValueError('unsupported BPM map')
        self.lines = self.doc['judgeLineList']
        self.anim, self.errors, self.notes = {}, {}, {}
        for li, line in enumerate(self.lines):
            try:
                self._compile_line(li, line)
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                self.errors[li] = str(exc)
            for ni, note in enumerate(line.get('notes') or []):
                if type(note.get('isFake', 0)) not in (bool, int) or note.get('isFake', 0) not in (0, 1):
                    raise ValueError('ambiguous isFake in spatial source')
                if note.get('isFake', 0):
                    continue
                self.notes[f'rpe:{li}:{ni}'] = (li, note)

    def seconds(self, b):
        sec = 0.0
        for i, (start, bpm) in enumerate(self.bpms):
            stop = self.bpms[i + 1][0] if i + 1 < len(self.bpms) else b
            sec += (min(b, stop) - start) * 60 / bpm
            if b <= stop:
                break
        return sec

    def _compile_line(self, li, line):
        if line.get('bpmfactor', 1) != 1:
            raise ValueError('unsupported bpmfactor')
        if line.get('father', -1) >= 0:
            raise ValueError('parented geometry requires independent parity validation')
        if line.get('attachUI') not in (None, '', -1):
            raise ValueError('UI line unsupported')
        if any(p.get('pos', 1) != 1 for p in line.get('posControl', [])):
            raise ValueError('nonidentity position control unsupported')
        for key in KEYS:
            layers = []
            for layer in line.get('eventLayers', []):
                kfs = []
                for event in (layer or {}).get(key) or []:
                    start, end = self.seconds(beat(event['startTime'])), self.seconds(beat(event['endTime']))
                    if not all(math.isfinite(v) for v in (start, end, float(event['start']), float(event['end']))):
                        raise ValueError('nonfinite geometry event')
                    if end < start:
                        raise ValueError('inverted geometry event')
                    if event.get('bezier', 0) and event['start'] != event['end']:
                        raise ValueError('nonconstant Bezier geometry unsupported')
                    typ = event.get('easingType', 1)
                    if not isinstance(typ, int) or not 0 <= typ < len(TWEEN):
                        raise ValueError('unrecognised easing')
                    kfs.extend([(start, float(event['start']), event), (end, float(event['end']), None)])
                if any(a[0] > b[0] for a, b in zip(kfs, kfs[1:])):
                    raise ValueError('unordered or overlapping geometry events')
                if kfs:
                    layers.append((kfs, [x[0] for x in kfs]))
            self.anim[li, key] = layers

    def value(self, li, key, t):
        if li in self.errors:
            raise ValueError(self.errors[li])
        value = 0.0
        for kfs, ts in self.anim[li, key]:
            i = max(0, bisect_right(ts, t) - 1)
            start, v, event = kfs[i]
            if i + 1 == len(kfs) or event is None:
                value += v
                continue
            end, v1, _ = kfs[i + 1]
            if v == v1:
                value += v
                continue
            if t < start:
                raise ValueError('pre-first-event extrapolation unsupported')
            x = (t - start) / (end - start)
            tid = TWEEN[max(1, event.get('easingType', 1))]
            l, r = max(0, min(1, event.get('easingLeft', 0))), max(0, min(1, event.get('easingRight', 1)))
            if tid > 2 and l < r and (abs(l) > 1e-7 or abs(r - 1) > 1e-7):
                den = tween(tid, r) - tween(tid, l)
                if abs(den) < 1e-12:
                    raise ValueError('degenerate easing crop')
                weight = (tween(tid, l + (r - l) * x) - tween(tid, l)) / den
            else:
                weight = tween(tid, x)
            value += v + (v1 - v) * weight
        return value

    def region(self, line_id, native_x, audio_us, *, judge_area=1.0):
        li = int(line_id)
        if not 0 <= li < len(self.lines):
            raise ValueError('unknown line')
        t = audio_us / 1e6 - self.offset
        x = self.value(li, 'moveXEvents', t) / 675
        y = self.value(li, 'moveYEvents', t) / 450 / self.profile.aspect
        rotation = -math.radians(self.value(li, 'rotateEvents', t))
        nx, ny = math.cos(rotation), math.sin(rotation)
        width = .21 / self.profile.aspect * 2 * judge_area
        values = [nx, ny, nx * x + ny * y + native_x / 675, width]
        if not all(math.isfinite(v) for v in values) or width <= 0:
            raise ValueError('nonfinite or nonpositive region')
        return values

    def breakpoints(self, line_id, a_us, b_us):
        li = int(line_id)
        if li in self.errors:
            raise ValueError(self.errors[li])
        return sorted({int(round((t + self.offset) * 1e6)) for key in KEYS
                       for kfs, _ in self.anim[li, key] for t, _, _ in kfs
                       if a_us <= (t + self.offset) * 1e6 <= b_us})


def compile_rpe(raw, *, info_offset_s=0.0, profile=EngineProfile()):
    """Compile original bytes; caller persists cache identity, never a projected truth pool."""
    return SpatialChart(raw, info_offset_s, profile)

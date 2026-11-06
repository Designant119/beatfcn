from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from beatfcn.events.canonical import TimingMap
from beatfcn.events.chart_export import (osu_mania_bytes, timing_points_from_map, write_osu_mania)
from beatfcn.events.osu_mania import ExportError, parse_osu


def ev(on_ms, lane, end_ms=None):
    end = on_ms if end_ms is None else end_ms
    return dict(onset_us=int(on_ms*1000), end_us=int(end*1000), type='hold' if end_ms else 'tap', lane=lane)


DRAFT = [ev(1000, 0), ev(1000, 3), ev(1250, 1, 1750), ev(1500, 2), ev(2000, 1), ev(2125.4, 0)]


class OsuManiaExport(unittest.TestCase):
    def test_round_trip_through_repo_parser(self):
        data, receipt = osu_mania_bytes(DRAFT, title='Song', artist='Artist')
        chart = parse_osu(data)
        self.assertFalse(chart.quarantined, chart.issues)
        got = sorted((e.onset_us, e.end_us, e.type, e.location.lane) for e in chart.events)
        want = sorted((round(d['onset_us'], -3), round(d['end_us'], -3), d['type'], d['lane']) for d in DRAFT)
        self.assertEqual(got, want)
        self.assertEqual(receipt['ms_rounded_events'], 1)
        self.assertTrue(receipt['roundtrip_exact'])
        self.assertEqual(len(chart.timing.segments), 1)
        self.assertEqual(chart.timing.segments[0].bpm, 120)
        self.assertIn(b'Mode: 3', data)
        self.assertIn(b'CircleSize:4', data)

    def test_charter_timing_points(self):
        tm = TimingMap.from_bpm_points([(0, 150), (8, 200)], first_sec=Fraction(1, 2))
        points = timing_points_from_map(tm)
        data, receipt = osu_mania_bytes(DRAFT, timing_points=points)
        chart = parse_osu(data)
        self.assertEqual([s.bpm for s in chart.timing.segments], [150, 200])
        self.assertEqual(receipt['timing'], 'charter-supplied')

    def test_fail_closed(self):
        for bad in ([ev(1000, 0), ev(1000, 0)],                  # same-lane duplicate
                    [ev(1000, 0, 1500), ev(1200, 0)],            # head inside hold
                    [dict(onset_us=0, end_us=0, type='flick', lane=0)],
                    [dict(onset_us=1000, end_us=1200, type='hold', lane=0)],  # rounds to zero length
                    [ev(0, 4)]):
            with self.assertRaises(ExportError):
                osu_mania_bytes(bad)

    def test_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'draft.osu'
            receipt = write_osu_mania(path, DRAFT)
            self.assertEqual(receipt['notes'], len(DRAFT))
            self.assertFalse(parse_osu(path.read_bytes()).quarantined)
            with self.assertRaises(FileExistsError):
                write_osu_mania(path, DRAFT)

    def test_empty_draft_is_openable(self):
        data, receipt = osu_mania_bytes([])
        self.assertEqual(receipt['notes'], 0)
        self.assertEqual(parse_osu(data).events, ())


if __name__ == '__main__':
    unittest.main()

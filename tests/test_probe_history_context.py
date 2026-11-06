import unittest

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.event_reference import REFERENCE_COLUMNS
from beatfcn.train.exposure_loader import make_vocabulary, request_sequence
from beatfcn.train.history_context import chart_arrays, draw_history, history_sources, normalize_history_context, reference_inputs
from beatfcn.train.local_request_panel import event_request
from scripts.probe_recipe_v2_overfit import (HISTORY, foreign_sources, reference_grad_norms, reference_stats,
                                              repeated_events, verify_reference_leak)

WINDOW = 8_000_000
DURATION = 48_000_000
AUDIO = 'ab' * 32


class SimplePanel:
    """One chart. Taps on a 500 ms grid; every window has an interior hold; the hold at 31.0-33.0 s crosses
    the 32 s boundary so that window has an incoming (carry) hold."""

    def __init__(self, mode='mania_4k'):
        self.mode = mode
        rows = [(t, t, 0, (t // 500_000) % 2) for t in range(0, DURATION, 500_000)]
        rows += [(w + 1_100_000, w + 3_100_000, 1, 2) for w in range(0, DURATION, WINDOW) if w != 24_000_000]
        rows.append((31_000_000, 33_000_000, 1, 3))
        rows.sort()
        self.arrays = {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}
        self.records = []
        a = self.arrays
        for start in range(0, DURATION, WINDOW):
            end = start+WINDOW
            inside = np.flatnonzero((a['onset_us'] >= start) & (a['onset_us'] < end))
            carry = [dict(onset_us=int(a['onset_us'][i]), end_us=int(a['end_us'][i]), type='hold', lane=int(a['lane'][i]))
                     for i in np.flatnonzero((a['kind'] == 1) & (a['onset_us'] < start) & (a['end_us'] > start))]
            self.records.append(dict(split='train', chart_index=0, chart_id='c:1', audio_sha256=AUDIO, mode=mode,
                target_start=int(inside[0]), target_stop=int(inside[-1])+1,
                request=dict(mode=mode, start_us=start, end_us=end, duration_us=DURATION, known_carry=carry)))

    def _arrays(self, chart_index):
        return None, self.arrays

    def targets(self, index):
        r, a = self.records[index], self.arrays
        sl = slice(r['target_start'], r['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l))
                for h, t, k, l in zip(a['onset_us'][sl], a['end_us'][sl], a['kind'][sl], a['lane'][sl])]


class ProbeHistoryContextTests(unittest.TestCase):
    def reference(self, panel, index):
        record = panel.records[index]
        request = event_request(record['request'])
        settings = normalize_history_context(HISTORY)
        sources, info = history_sources(chart_arrays(panel, 0), request, draw_history(settings, 0, index, 0), settings,
                                        chart_id='c:1', audio_sha256=AUDIO)
        self.assertEqual(info['prefix_s'], 16.)
        reference, provenance = reference_inputs(request, AUDIO, sources)
        return request, reference, provenance

    def test_reference_leak_check_passes_and_counts_authorized_carry(self):
        panel = SimplePanel()
        request, reference, provenance = self.reference(panel, 4)          # window 32 s: one carry hold
        leak = verify_reference_leak(reference['reference'], provenance, request, chart_arrays(panel, 0))
        self.assertEqual(leak['crossing_holds'], 1)
        self.assertEqual(leak['authorized_carry_holds'], 1)
        self.assertEqual(leak['events_inside_window'], 0)
        stats = reference_stats(reference['reference'])
        self.assertEqual(stats['unknown_region_rows'], 1)
        self.assertEqual(stats['human_input_rows'], stats['rows']-1)
        self.assertEqual(stats['nan'], 0)

    def test_reference_leak_check_catches_a_target_window_event(self):
        panel = SimplePanel()
        request, reference, provenance = self.reference(panel, 3)
        col = REFERENCE_COLUMNS.index('event')
        bad = np.array(reference['reference'])
        row = int(np.flatnonzero(bad[:, col] == 1)[0])
        bad[row, 0] = bad[row, 1] = 1_000_000/1.2e9                       # an event 1 s into the window
        with self.assertRaises(AssertionError):
            verify_reference_leak(bad, provenance, request, chart_arrays(panel, 0))
        with self.assertRaises(AssertionError):                           # prefix must hold every chart event
            verify_reference_leak(reference['reference'], provenance, request, chart_arrays(panel, 0), prefix_us=20_000_000)

    def test_foreign_prefix_is_retimed_leak_free_and_differs(self):
        panel = SimplePanel()
        record = panel.records[3]
        request = event_request(record['request'])
        sources, info = foreign_sources(panel, record, request, gap_us=16_000_000)
        reference, provenance = reference_inputs(request, AUDIO, sources)
        times = [e.onset_us for e in sources[0].events]
        self.assertTrue(all(request.start_us-16_000_000 <= t < request.start_us for t in times))
        self.assertTrue(all(e.end_us <= request.start_us for e in sources[0].events))
        self.assertEqual(info['shift_us'], request.start_us-info['donor_end_us'])
        self.assertNotEqual(info['donor_end_us'], request.start_us)

    def test_repeated_events_are_previous_window_shifted_and_legal(self):
        panel = SimplePanel()
        events = repeated_events(panel, 2)
        previous = panel.targets(1)
        self.assertEqual([(e.onset_us-WINDOW, e.end_us-WINDOW, e.type, e.lane) for e in events],
                         [(e.onset_us, e.end_us, e.type, e.lane) for e in previous])
        request = event_request(panel.records[2]['request'])
        self.assertTrue(all(request.start_us <= e.onset_us <= e.end_us <= request.end_us for e in events))
        request_sequence(panel, 2, request, make_vocabulary({}), events=events)   # encodes without error
        with self.assertRaises(ValueError):                                       # window with an incoming hold
            repeated_events(panel, 4)

    def test_reference_grad_norms_groups(self):
        import torch
        model = torch.nn.ModuleDict(dict(reference_projection=torch.nn.Linear(2, 2)))
        model.reference_projection(torch.ones(1, 2)).sum().backward()
        norms = reference_grad_norms(model)
        self.assertGreater(norms['reference_projection'], 0)
        self.assertEqual(norms['reference_encoder'], 0)


if __name__ == '__main__':
    unittest.main()

import unittest

from beatfcn.events.canonical import Event, LaneLoc
from scripts.prepare_local_request_tape import requests_for_chart


class LocalRequestTapeTests(unittest.TestCase):
    def test_complete_song_keeps_silence_tail_and_incoming_hold_without_hidden_val_controls(self):
        events = [Event('hold', 7_000_000, 9_000_000, 'hold', LaneLoc(0)),
                  Event('tap', 8_100_000, 8_100_000, 'tap', LaneLoc(1))]
        rows, tokens, targets = requests_for_chart(events, 'mania_4k', 18_000_000, 'val')
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['event_count'] for r in rows], [1, 1, 0])
        self.assertEqual(rows[0]['crossing_outgoing_holds'], 1)
        self.assertEqual(rows[1]['incoming_holds'], 1)
        self.assertEqual(rows[-1]['request']['end_us'], 18_000_000)
        self.assertEqual(rows[-1]['request']['start_us'], 16_000_000)
        self.assertEqual(len(targets['onset_us']), 2)
        self.assertEqual(sum(r['tokens'] for r in rows), len(tokens))
        self.assertTrue(all(not any(c[2] for c in r['controls']) and r['bands'] is None for r in rows))

    def test_training_occupancy_includes_known_incoming_holds(self):
        events = [Event('hold', 7_000_000, 12_000_000, 'hold', LaneLoc(0))]
        rows, _, _ = requests_for_chart(events, 'mania_4k', 16_000_000, 'train')
        self.assertGreater(rows[1]['bands']['hold_occupancy_ratio'][1], 0)
        with self.assertRaisesRegex(ValueError, 'never clip'):
            requests_for_chart(events, 'mania_4k', 10_000_000, 'train')

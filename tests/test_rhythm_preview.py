from collections import Counter
from pathlib import Path
import tempfile
import unittest
import wave

from beatfcn.events.rhythm_preview import display_positions, write_phira_preview


class RhythmPreviewTests(unittest.TestCase):
    def test_dense_artistic_drags_are_not_removed_to_fit_a_display(self):
        events = [dict(onset_us=1000, end_us=1000, type='drag') for _ in range(40)]
        events += [dict(onset_us=0, end_us=10000, type='hold')]
        display = display_positions(events)
        self.assertEqual(Counter((e['t_us'], e['end_us'], e['type']) for e in display),
                         Counter((e['onset_us'], e['end_us'], e['type']) for e in events))

    def test_export_preserves_types_times_and_hold_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); audio = root/'audio.wav'
            with wave.open(str(audio), 'wb') as file:
                file.setnchannels(1); file.setsampwidth(2); file.setframerate(24000); file.writeframes(b'\0\0'*24000)
            events = [dict(onset_us=123000, end_us=987000, type='hold'),
                      dict(onset_us=200000, end_us=200000, type='flick')]
            result = write_phira_preview(root/'chart.zip', events, audio,
                                         title='Test', artist='Test artist', audio_name='audio.wav')
            self.assertTrue(result['roundtrip_exact'])
            self.assertEqual(result['timing_shift_us'], 0)

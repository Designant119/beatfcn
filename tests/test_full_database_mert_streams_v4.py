from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertStreamsV4Tests(unittest.TestCase):
    def test_streams_use_single_song_forward_not_closed_batch_route(self):
        text = (ROOT / 'beatfcn/cloud/full_database_streams_app.py').read_text()
        self.assertIn('GPU_STREAMS = 8', text)
        self.assertIn('range(GPU_STREAMS - 1)', text)
        self.assertIn('max_workers=1', text)
        self.assertNotIn('encode_windows_batch(model', text)
        self.assertIn("feature_payloads(prefetched['wave'], model, mel=mel)", text)
        self.assertIn("report['max_abs'] == 0", text)
        self.assertIn("report['float16_exact_fraction'] == 1", text)
        self.assertLess(text.index('parity = _parity_preflight'), text.index('monitor.start()'))


if __name__ == '__main__':
    unittest.main()

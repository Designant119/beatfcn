from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MertGraphProbeTests(unittest.TestCase):
    def test_probe_requires_bitwise_graph_single_equivalence(self):
        text = (ROOT / 'beatfcn/cloud/mert_graph_probe_app.py').read_text()
        self.assertIn("parity['max_abs'] == 0", text)
        self.assertIn("parity['float16_exact_fraction'] == 1", text)
        self.assertIn('torch.cuda.CUDAGraph()', text)
        self.assertIn('model.model._encode(value, True)', text)
        self.assertIn('if not bool(torch.isfinite(static_input).all())', text)
        self.assertIn('(np.arange(expected_frames, dtype=np.int64) * 960 + 360) / 24000', text)
        self.assertIn('graph.replay()', text)
        self.assertIn('decoded_audio_from_receipt', text)
        self.assertNotIn('@app.local_entrypoint', text)

    def test_probe_is_bounded_diagnostic_not_formal_acceptance(self):
        text = (ROOT / 'scripts/run_mert_graph_probe.py').read_text()
        self.assertIn("diagnostic_only=True, probe='mert_cuda_graph_v1'", text)
        self.assertIn('formal_85_percent_acceptance_claim=False', text)
        self.assertIn('full_corpus_encode=False', text)


if __name__ == '__main__':
    unittest.main()

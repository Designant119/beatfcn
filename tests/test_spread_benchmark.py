import unittest

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train import spread_benchmark as sb

AUDIO = ['%064x' % (i + 1) for i in range(2)]
DURATION = 48_000_000
WINDOW = 8_000_000


class Features:
    def __init__(self):
        self.records = {a: dict(duration_us=DURATION) for a in AUDIO}

    def window(self, audio, start_us, end_us):
        n = 40
        return dict(local=np.ones((2, n, 1, 4), np.float32), local_times=np.linspace(-1, 9, n).astype(np.float32),
                    local_valid=np.ones(n, bool), mel=np.ones((n, 2), np.float32),
                    mel_times=np.linspace(-1, 9, n).astype(np.float32), mel_valid=np.ones(n, bool),
                    song=np.ones((3, 4), np.float32), song_times=np.zeros(3, np.float32), song_valid=np.ones(3, bool))


def _chart(step):
    rows = [(t, t, 0, ((t // step) ** 2 + t // 1_000_000) % 4) for t in range(0, DURATION, step)]  # not shift-invariant
    return {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


class Panel:
    """Val mania song with dense / medium / sparse charts + a lone val Phira chart."""

    def __init__(self):
        self.features = Features()
        specs = [('osu:dense', 'mania_4k', 0, 125_000), ('osu:medium', 'mania_4k', 0, 250_000),
                 ('osu:sparse', 'mania_4k', 0, 500_000), ('phira:lone', 'phira', 1, 250_000)]
        self.charts, self.arrays, self.records = [], [], []
        for i, (cid, mode, audio, step) in enumerate(specs):
            self.charts.append(dict(chart_id=cid, audio_sha256=AUDIO[audio], mode=mode, split='val',
                                    canonical_sha256=f'c{i}', duration_us=DURATION))
            a = _chart(step)
            if mode == 'phira':
                a['lane'][:] = -1
            self.arrays.append(a)
            for start in range(0, DURATION, WINDOW):
                end = start + WINDOW
                inside = np.flatnonzero((a['onset_us'] >= start) & (a['onset_us'] < end))
                self.records.append(dict(split='val', chart_index=i, chart_id=cid, audio_sha256=AUDIO[audio], mode=mode,
                    target_start=int(inside[0]), target_stop=int(inside[-1]) + 1,
                    request=dict(mode=mode, start_us=start, end_us=end, duration_us=DURATION, copy_times=[],
                                 known_carry=[], allowed_types=None, max_events=4096, max_tokens=32768)))

    def _arrays(self, chart_index):
        return self.arrays[chart_index]

    def targets(self, index):
        row = self.records[index]
        a = self.arrays[row['chart_index']]
        sl = slice(row['target_start'], row['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(a['onset_us'][sl], a['end_us'][sl], a['kind'][sl], a['lane'][sl])]


class SpreadProbeTest(unittest.TestCase):
    def test_panel_pairs_next_denser_sibling_and_teacher_runs(self):
        from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
        panel = Panel()
        protocol = sb.freeze_spread_panel(panel, windows=8, seed=1)
        pairs = {(w['chart_id'], w['sibling_chart_id']) for w in protocol['windows']}
        self.assertEqual(pairs, {('osu:medium', 'osu:dense'), ('osu:sparse', 'osu:medium')})   # next denser
        self.assertEqual(protocol['skipped'], {'no_denser_sibling': 2})                      # dense + lone phira
        window = protocol['windows'][0]
        _, _, _, variants, info = sb._prepare(panel, window, None, sb.CONDITIONS, protocol)
        self.assertEqual(info['none']['reference_events'], 0)
        self.assertGreater(info['sibling']['reference_events'], 0)
        ref = variants['sibling']['reference']
        rel = ref[:, 0] * 1_200_000_000 / 1e6
        self.assertTrue(((ref[:, 10] == 1) & (ref[:, 13] == 1) & (rel >= 0) & (rel < 8)).any())
        a, b = variants['sibling']['reference'], variants['sibling_shift1s']['reference']
        self.assertFalse(a.shape == b.shape and np.array_equal(a, b))
        torch.manual_seed(0)
        model = FullEventModel(FullEventConfig(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=1,
            tap_dim=4, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=2, intensity_dim=3)).eval()
        result = sb.run_spread_teacher(model, panel, protocol, device='cpu', batch_size=4)
        summary = result['summary']
        self.assertTrue(summary['complete'])
        gaps = summary['by_mode']['mania_4k']['gaps']
        self.assertEqual(set(gaps), {'sibling_vs_none', 'alignment_specific', 'shifted_vs_none'})
        self.assertIsNotNone(gaps['sibling_vs_none']['joint']['paired_mean_nats'])


if __name__ == '__main__':
    unittest.main()

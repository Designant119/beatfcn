import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from beatfcn.model import sample
from beatfcn.model.batch import collate_meta, SongInfo
from beatfcn.model.generation_batch import GenerationBatch, INFERENCE_PREFIX_KEYS
from beatfcn.model.p1 import P1Model
from test_p1_model import CFG, batch_parts


def fixture():
    examples, songs = batch_parts(2, seed=3)
    for ex, song in zip(examples, songs):
        groups = []
        for i in range(97):
            g = copy.deepcopy(ex.groups[i % len(ex.groups)])
            g['t_rel_us'] = i * 60000
            groups.append(g)
        yield ex, song, groups


def metadata(ex, song):
    bt = collate_meta([ex], [SongInfo.of(song, 0, CFG.taps[0])], CFG.taps, CFG.top_tap,
                      CFG.tap_dim, song.summary.shape[-1], False)
    bt.pop('_audio')
    return bt


class GenerationBatchTests(unittest.TestCase):
    def test_every_prefix_matches_collator_including_holds_chords_and_bucket_growth(self):
        for ex, song, groups in fixture():
            work = copy.copy(ex)
            work.groups = []
            right = np.random.default_rng(0).random((98, 4), dtype=np.float32)
            work.right = right[:1]
            initial = metadata(work, song)
            cache = GenerationBatch(initial, work, 'cpu')
            for count in range(len(groups) + 1):
                work.groups, work.right = groups[:count], right[:count + 1]
                expected = metadata(work, song)
                for key in INFERENCE_PREFIX_KEYS:
                    torch.testing.assert_close(cache.bt[key], expected[key], atol=0, rtol=0,
                                               msg=f'{ex.mode=} {count=} {key=}')
                self.assertIs(cache.bt['known'], initial['known'])
                self.assertIs(cache.bt['controls'], initial['controls'])
                if count < len(groups):
                    cache.append(groups[count], right[count + 1])

    def test_generation_outputs_and_retries_identical_without_rebuilding_metadata(self):
        from beatfcn.events.requests import build_request
        from beatfcn.model import features as F
        from events_helpers import synthetic_mania, synthetic_phira, DURATION
        from test_p1_model import song
        for chart in (synthetic_mania(3), synthetic_phira(4)):
            request = build_request(chart, kind='from_scratch', seed=9, audio_id='a', audio_duration_us=DURATION)
            ex = F.build_core(request, chart.events, 8 * F.SEC, DURATION)
            holds, runs, heads = sample.core_context(request, ex, F.line_index_of(chart.events))
            torch.manual_seed(5)
            model = P1Model(replace(CFG, lane_history_groups=8, selection=True)).eval()
            with torch.no_grad():
                model.kern.weight.normal_(std=.2)
            original = sample._inner
            for greedy in (True, False):
                outputs = []
                for reuse in (False, True):
                    attempts = []
                    def inner(*args, **kwargs):
                        attempts.append(args[4])
                        if len(attempts) == 2:
                            return []
                        return original(*args, **kwargs)
                    with patch.object(sample, '_inner', side_effect=inner), patch.object(
                            sample, 'collate_meta', wraps=sample.collate_meta) as collator:
                        groups = sample.generate_core(model, ex, song(1), holds, runs, heads,
                                                      greedy=greedy, seed=4, reuse_metadata=reuse)
                    outputs.append([(g['t_rel_us'], g['notes']) for g in groups])
                    if reuse:
                        self.assertEqual(collator.call_count, 0)
                self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()

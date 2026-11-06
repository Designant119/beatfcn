import copy
import unittest

from scripts.prepare_local_data_scale_probe import choose_case, merge_admission, padding_shapes
from scripts.freeze_local_data_scale import validate_health


class DataScaleProbeTests(unittest.TestCase):
    def test_case_selection_excludes_validation_and_seen_songs_before_ranking(self):
        pool = [dict(split=split, component=component, tokens=tokens) for split, component, tokens in
                [('val', 'v', 1), ('train', 'old', 2), ('train', 'a', 5),
                 ('train', 'b', 100), ('train', 'c', 1000)]]
        used = {'old'}
        index, row, audit = choose_case(pool, lambda r: True, used, 'test')
        self.assertEqual((index, row['tokens'], audit['eligible_requests']), (3, 100, 3))
        self.assertIn('b', used)
        with self.assertRaisesRegex(ValueError, 'No real'):
            choose_case(pool, lambda r: r['split'] == 'val', set(), 'test')

    def test_padding_keeps_old_bucket_and_preserves_long_context(self):
        self.assertEqual(padding_shapes(896)['song'], 896)
        self.assertEqual(padding_shapes(1011)['song'], 1024)
        for length in (0, 1025, 1011.5):
            with self.assertRaises(ValueError):
                padding_shapes(length)

    def report(self, audio):
        return dict(complete=True, failures=[], scales='scale.npz', records=[dict(
            audio_sha256=audio, passed=True, legacy_prefix_parity=dict(interior_exact_equal=True))])

    def test_merge_requires_full_frozen_identity_and_passed_audits(self):
        a, b = self.report('a'), self.report('b')
        merged = merge_admission([a, b], {'a', 'b'})
        self.assertEqual(merged['completed_songs'], 2)
        self.assertFalse(merged['training_ready'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            merge_admission([a], {'a', 'b'})
        for field, bad in [('complete', False), ('failures', ['failed']), ('scales', 'other.npz')]:
            broken = copy.deepcopy(b)
            broken[field] = bad
            with self.assertRaisesRegex(ValueError, 'incomplete, failed, or incompatible'):
                merge_admission([a, broken], {'a', 'b'})
        b['records'][0]['passed'] = False
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            merge_admission([a, b], {'a', 'b'})

    def test_merge_rejects_conflicting_duplicate_audio(self):
        a, b = self.report('a'), self.report('a')
        b['records'][0]['different_content'] = True
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            merge_admission([a, b], {'a'})

    def test_quality_requires_actual_full_free_generation_health(self):
        plan = dict(cases=[dict(record_index=i) for i in range(8)], max_request_mean_ce=.05)
        health = dict(id='D94-new-data-probe-v1', complete=True, passed=True,
            optimizer_updates=100, plan_sha256='frozen', parent_unchanged=True,
            generation=[dict(record_index=i, complete=True, exact_tokens=True, exact_events=True) for i in range(8)],
            final=[dict(request_mean_ce=.01, bos_correct=True) for _ in range(8)],
            padding_proofs=[dict(passed=True)])
        validate_health(health, plan, 'frozen')
        broken = copy.deepcopy(health)
        broken['generation'][2]['exact_events'] = False
        with self.assertRaisesRegex(ValueError, 'Per-request'):
            validate_health(broken, plan, 'frozen')
        broken = copy.deepcopy(health)
        broken['final'][0]['request_mean_ce'] = .5
        with self.assertRaisesRegex(ValueError, 'Per-request'):
            validate_health(broken, plan, 'frozen')
        broken = copy.deepcopy(health)
        broken['padding_proofs'] = []
        with self.assertRaisesRegex(ValueError, 'Per-request'):
            validate_health(broken, plan, 'frozen')
        with self.assertRaisesRegex(ValueError, 'actual D94'):
            validate_health(health, plan, 'another plan')


if __name__ == '__main__':
    unittest.main()

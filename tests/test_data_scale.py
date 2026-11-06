import json
from pathlib import Path

import unittest

from beatfcn.ingest.coverage import native_tier, phira_native_difficulty, quota_report, policy_identity

POLICY = json.loads((Path(__file__).resolve().parents[1] / 'bench/rhythm-v2/data-scale-policy-v1.json').read_text())


def example(**changes):
    return dict(dict(mode='mania_4k', difficulty_tier='low', split='train',
        audio_sha256='audio1', pure_rhythm_hash='target1', song_group='song1',
        technical_audit_pass=True, difficulty_review_status='verified',
        difficulty_policy_sha256=policy_identity(POLICY),
        feature_cache_status='verified', training_ready=True), **changes)


class DataScale(unittest.TestCase):
    def test_exact_boundaries_and_invalid_labels(self):
        assert [native_tier('mania_4k', x, POLICY) for x in [1.99, 2, 3.99, 4]] == ['low','medium','medium','high']
        assert [native_tier('phira', x, POLICY) for x in [7.99, 8, 12.99, 13]] == ['low','medium','medium','high']
        assert native_tier('phira', 0, POLICY) is None
        assert native_tier('mania_4k', float('nan'), POLICY) is None


    def test_contradictory_phira_label_cannot_fill_easy_quota(self):
        assert phira_native_difficulty(dict(level='IN Lv.15', difficulty=1))[0] is None
        assert phira_native_difficulty(dict(level='EZ Lv.4', difficulty=4))[0] == 4
        assert phira_native_difficulty(dict(level='HD Lv.12', difficulty=12.5))[0] == 12.5


    def test_ineligible_rows_do_not_count(self):
        for change in [dict(split='val'), dict(split='test'), dict(pure_rhythm_hash=None),
                       dict(difficulty_review_status='unverified'), dict(training_ready=False),
                       dict(feature_cache_status='not_verified'), dict(quarantined=True)]:
            with self.subTest(change=change):
                cell = quota_report([example(**change)], POLICY)['cells']['mania_4k/low']
                self.assertEqual(cell['accepted_unique_train'], 0)


    def test_duplicates_and_true_same_song_contrasts(self):
        report = quota_report([example(), example(), example(pure_rhythm_hash='target2')], POLICY)
        cell = report['cells']['mania_4k/low']
        assert cell['accepted_unique_train'] == 2
        assert cell['unique_audio'] == cell['unique_song_groups'] == 1
        assert report['exclusions']['duplicate_uploads'] == 1
        assert cell['shortfall'] == 1498
        assert not report['scale_gate_passed']


    def test_duplicates_cannot_fill_different_tiers(self):
        report = quota_report([example(), example(difficulty_tier='medium')], POLICY)
        assert report['exclusions']['duplicate_targets_with_conflicting_tiers'] == 2
        assert sum(c['accepted_unique_train'] for c in report['cells'].values()) == 0


    def test_one_full_bucket_cannot_cover_other_buckets(self):
        policy = dict(POLICY, minimum_unique_train_charts_per_tier={'mania_4k': 1, 'phira': 1})
        report = quota_report([example(difficulty_policy_sha256=policy_identity(policy))], policy)
        assert report['cells']['mania_4k/low']['shortfall'] == 0
        assert not report['scale_gate_passed']

    def test_stale_difficulty_policy_cannot_count(self):
        report = quota_report([example(difficulty_policy_sha256='old')], POLICY)
        assert report['cells']['mania_4k/low']['accepted_unique_train'] == 0

    def test_validation_song_cannot_also_fill_training_quota(self):
        report = quota_report([example(), example(split='val', pure_rhythm_hash='other')], POLICY)
        assert report['exclusions']['held_out_song_overlap'] == 1
        assert report['cells']['mania_4k/low']['accepted_unique_train'] == 0

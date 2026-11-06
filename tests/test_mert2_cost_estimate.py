import unittest

from scripts.estimate_mert2_campaign import encoded_seconds, estimate


class CampaignCostTests(unittest.TestCase):
    def setUp(self):
        self.scenario = dict(encode_audio_seconds_per_wall_second=100,
            s1_seconds_per_effective_update=2, s2_seconds_per_effective_update=1,
            routine_review_minutes=10, expanded_review_minutes=20, final_audit_minutes=20,
            s2_review_minutes=10, overhead_hours=1, probe_hours=.25)
        self.platform = dict(provider='runpod', compute_usd_h=1, storage_kind='runpod_network')

    def calculate(self, **kw):
        args = dict(storage_gb=400, retention_hours=72, egress_gib=100,
                    reserve_fraction=.2, remaining_usd=5.655470851657537)
        args.update(kw)
        return estimate(self.scenario, self.platform, 3600, **args)

    def test_long_songs_are_not_truncated_or_assumed_free(self):
        self.assertEqual(encoded_seconds(360), 360)
        self.assertEqual(encoded_seconds(361), 461)
        self.assertEqual(encoded_seconds(600), 800)

    def test_effective_updates_and_all_reviews_are_billed(self):
        r = self.calculate()
        self.assertAlmostEqual(r['phase_hours']['s1_10000_effective_updates'], 20000/3600)
        self.assertAlmostEqual(r['phase_hours']['ten_thousand_step_reviews'], 100/60)
        self.assertEqual(r['phase_hours']['three_expanded_reviews'], 1)
        self.assertEqual(r['migration_egress_usd'], 4)
        self.assertAlmostEqual(r['storage_usd'], 2.8)
        self.assertAlmostEqual(r['total_scenario_usd'], r['subtotal_usd']*1.2)
        self.assertFalse(r['dispatch_ready'])

    def test_stopped_community_disk_still_costs_money(self):
        self.platform['storage_kind'] = 'runpod_local_volume'
        r = self.calculate()
        self.assertGreater(r['storage_usd'], 400*.1*72/720)

    def test_no_false_dispatch_approval_even_with_plenty_of_budget(self):
        r = self.calculate(remaining_usd=1000)
        self.assertTrue(r['fits_numeric_scenario'])
        self.assertFalse(r['dispatch_ready'])

    def test_invalid_speed_and_impossible_retention_rejected(self):
        with self.assertRaises(ValueError):
            self.calculate(retention_hours=.01)
        self.scenario['encode_audio_seconds_per_wall_second'] = 0
        with self.assertRaises(ValueError):
            self.calculate()


if __name__ == '__main__':
    unittest.main()

from datetime import datetime, timezone
import unittest

from scripts.refresh_rhythm_state import billing_start_date


class BillingRangeTests(unittest.TestCase):
    def test_utc_midnight_does_not_drop_yesterday(self):
        ledger = dict(billing_through_exclusive_utc='2026-10-24T13:00:00+00:00',
            paid_stages_dispatched=[dict(reserved_at=datetime(2026,10,25,0,5,tzinfo=timezone.utc).timestamp())])
        self.assertEqual(billing_start_date(ledger), '2026-10-24')

    def test_earliest_stage_can_precede_historical_cutoff(self):
        ledger = dict(billing_through_exclusive_utc='2026-10-24T13:00:00+00:00',
            paid_stages_dispatched=[dict(reserved_at=datetime(2026,10,23,23,55,tzinfo=timezone.utc).timestamp())])
        self.assertEqual(billing_start_date(ledger), '2026-10-23')

    def test_ambiguous_or_invalid_clock_fails(self):
        with self.assertRaises(ValueError):
            billing_start_date(dict(billing_through_exclusive_utc='2026-10-24T13:00:00', paid_stages_dispatched=[]))
        for value in (float('nan'), float('inf'), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                billing_start_date(dict(billing_through_exclusive_utc='2026-10-24T13:00:00+00:00',
                    paid_stages_dispatched=[dict(reserved_at=value)]))

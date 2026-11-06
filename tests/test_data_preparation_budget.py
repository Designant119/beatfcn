from pathlib import Path
import tempfile
import unittest
from unittest import mock

from beatfcn.train.data_preparation_budget import (
    CAP, GRANT, authorize, record_usage, reserve,
)


class DataPreparationBudgetTests(unittest.TestCase):
    def test_authorize_once_and_reserve_one_bounded_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json'
            evidence = root / 'approval.md'
            evidence.write_text('USD 5 pilot\n')
            ledger.write_text('{}\n')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                authorize(ledger, owner_evidence_path='approval.md', root=root)
            contract = dict(contract_sha256='a', source_sha256={'x.py': 'b'})
            resources = dict(timeout_s=600, startup_allowance_s=120, stop_allowance_s=60,
                cpu=4, memory_gib=16, retries=0, max_containers=1, other_upper_usd=0)
            rates = dict(gpu_usd_s='0.0005', cpu_usd_s='0.00001', memory_gib_usd_s='0.000002')
            connection = dict(fresh_check_succeeded=True, checked_at_unix=10**12,
                evidence_sha256='c', full_lifecycle_bound_verified=True)
            with mock.patch('beatfcn.train.data_preparation_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                job = reserve(ledger, 'pilot-v1', contract, root, resources, rates, connection)
            self.assertEqual(job['status'], 'reserved')
            self.assertLess(float(job['reserved_upper_usd']), float(CAP))
            with mock.patch('beatfcn.train.data_preparation_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                with self.assertRaisesRegex(ValueError, 'active'):
                    reserve(ledger, 'pilot-v2', contract, root, resources, rates, connection)
            usage = record_usage(ledger, 'pilot-v1', elapsed_s_since_dispatch=10,
                app_id='ap-test', closed=True, stop_evidence=dict(state='stopped', tasks=0))
            self.assertEqual(usage['job']['status'], 'closed')
            self.assertFalse(usage['grant']['stop_required'])

    def test_gate_requires_all_preparation_receipts_and_zero_updates(self):
        from beatfcn.train.data_preparation_budget import gate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = dict(schema='beatfcn.full-database-mert-pilot.v1', grant_id=GRANT,
                optimizer_updates=0, source_sha256={'x.py': 'b'}, jobs=[dict(audio_sha256='a')])
            errors = gate(contract, root)
            self.assertTrue(any('receipt missing' in error for error in errors))
            self.assertTrue(any('GPU-utilization gate' in error for error in errors))


if __name__ == '__main__':
    unittest.main()

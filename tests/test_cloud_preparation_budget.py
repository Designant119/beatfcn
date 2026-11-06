from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from beatfcn.train.cloud_preparation_budget import (
    GRANT, authorize, authorize_increase, record_usage, reserve)
from beatfcn.train.cloud_parallel_checkpoint_budget import reserve_parallel_checkpoint


class CloudPreparationBudgetTests(unittest.TestCase):
    def test_cpu_only_phase_does_not_require_gpu_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json';ledger.write_text('{}\n')
            approval = root / 'approval.md';approval.write_text('approved')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            contract = dict(contract_sha256='x', source_sha256={'x.py': 'y'})
            resources = dict(timeout_s=10, startup_allowance_s=0, stop_allowance_s=1,
                cpu=8, memory_gib=32, retries=0, max_containers=1, other_upper_usd=0)
            rates = dict(gpu_usd_s='0', cpu_usd_s='0.00001', memory_gib_usd_s='0.000002')
            connection = dict(fresh_check_succeeded=True, checked_at_unix=10**12,
                evidence_sha256='e', full_lifecycle_bound_verified=True)
            with mock.patch('beatfcn.train.cloud_preparation_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                job = reserve(ledger, 'targets', contract, root, resources, rates, connection)
            self.assertEqual(job['status'], 'reserved')
            self.assertGreater(float(job['rate_upper_usd_s']), 0)
            self.assertLess(float(job['reserved_upper_usd']), 5)

    def test_manual_recovery_requires_closed_stopped_prior(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json';ledger.write_text('{}\n')
            (root / 'approval.md').write_text('approved')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            contract = dict(contract_sha256='x', source_sha256={'x.py': 'y'})
            resources = dict(timeout_s=10, startup_allowance_s=0, stop_allowance_s=1,
                cpu=8, memory_gib=32, retries=0, max_containers=1, other_upper_usd=0)
            rates = dict(gpu_usd_s='0', cpu_usd_s='0.00001', memory_gib_usd_s='0.000002')
            connection = dict(fresh_check_succeeded=True, checked_at_unix=10**12,
                evidence_sha256='e', full_lifecycle_bound_verified=True)
            with mock.patch('beatfcn.train.cloud_preparation_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                reserve(ledger, 'targets', contract, root, resources, rates, connection)
                record_usage(ledger, 'targets', elapsed_s_since_dispatch=1, app_id='ap-old',
                    closed=True, stop_evidence=dict(state='stopped', tasks=0))
                job = reserve(ledger, 'targets-v5', contract, root, resources, rates, connection,
                    phase='targets', recovery_of='targets')
                self.assertEqual(job['phase'], 'targets')
                self.assertEqual(job['recovery_of'], 'targets')
                with self.assertRaisesRegex(ValueError, 'manual recovery lineage'):
                    reserve(ledger, 'raw-v5', contract, root, resources, rates, connection,
                        phase='raw')

    def test_recovery_rejects_function_return_without_app_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json';ledger.write_text('{}\n')
            (root / 'approval.md').write_text('approved')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            contract = dict(contract_sha256='x', source_sha256={'x.py': 'y'})
            resources = dict(timeout_s=10, startup_allowance_s=0, stop_allowance_s=1,
                cpu=8, memory_gib=32, retries=0, max_containers=1, other_upper_usd=0)
            rates = dict(gpu_usd_s='0', cpu_usd_s='0.00001', memory_gib_usd_s='0.000002')
            connection = dict(fresh_check_succeeded=True, checked_at_unix=10**12,
                evidence_sha256='e', full_lifecycle_bound_verified=True)
            with mock.patch('beatfcn.train.cloud_preparation_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                reserve(ledger, 'targets', contract, root, resources, rates, connection)
                record_usage(ledger, 'targets', elapsed_s_since_dispatch=1, app_id='ap-old',
                    closed=True, stop_evidence=dict(function_returned=True))
                with self.assertRaisesRegex(ValueError, 'stopped/tasks=0'):
                    reserve(ledger, 'targets-v5', contract, root, resources, rates, connection,
                        phase='targets', recovery_of='targets')

    def test_checkpoint_can_overlap_only_mel_with_independent_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json';ledger.write_text('{}\n')
            (root / 'approval.md').write_text('approved')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            contract = dict(contract_sha256='x', source_sha256={'x.py': 'y'},
                execution_phases=['checkpoint'], detach=True)
            resources = dict(timeout_s=10, startup_allowance_s=0, stop_allowance_s=1,
                cpu=8, memory_gib=4, retries=0, max_containers=1, other_upper_usd=0)
            rates = dict(gpu_usd_s='0', cpu_usd_s='0.00001', memory_gib_usd_s='0.000002')
            connection = dict(fresh_check_succeeded=True, checked_at_unix=10**12,
                evidence_sha256='e', full_lifecycle_bound_verified=True)
            with mock.patch('beatfcn.train.cloud_preparation_budget.gate', return_value=[]), \
                    mock.patch('beatfcn.train.cloud_parallel_checkpoint_budget.gate', return_value=[]), \
                    mock.patch('time.time', return_value=10**12):
                reserve(ledger, 'mel', contract, root, resources, rates, connection)
                job = reserve_parallel_checkpoint(ledger, contract, root,
                    resources, rates, connection)
                self.assertEqual(job['parallel_with'], 'mel')
                self.assertEqual(job['phase'], 'checkpoint')
                with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint'):
                    reserve_parallel_checkpoint(ledger, contract, root,
                        resources, rates, connection)

    def test_owner_three_dollar_amendment_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json';ledger.write_text('{}\n')
            (root / 'approval.md').write_text('original')
            (root / 'increase.md').write_text('add 3 USD')
            authorize(ledger, owner_evidence_path='approval.md', root=root)
            value = json.loads(ledger.read_text())
            grant = value['cloud_preparation_authorizations'][GRANT]
            grant['new_usd_cap'] = '5.00'
            ledger.write_text(json.dumps(value))
            authorize_increase(ledger, owner_evidence_path='increase.md', root=root)
            value = json.loads(ledger.read_text())
            grant = value['cloud_preparation_authorizations'][GRANT]
            self.assertEqual(grant['new_usd_cap'], '8.00')
            self.assertEqual(grant['remaining_usd'], '8.00')
            self.assertEqual(grant['cap_amendments'][0]['increase_usd'], '3.00')
            self.assertEqual(grant['cap_amendments'][0]['previous_evidence']['path'], 'approval.md')


if __name__ == '__main__':
    unittest.main()

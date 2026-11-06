from copy import deepcopy
import unittest

from scripts.reconcile_hn2_closed_reservation import GRANT, proposal


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        proof = dict(app_id='ap-test', state='stopped', tasks=0)
        job = dict(id='cpu-only', status='closed', app_id='ap-test', stop_evidence=proof,
                   reserved_upper_usd='4.059323019597', live_upper_usd='.8904983893455469',
                   reported_actual_usd='0')
        self.ledger = {'conditional_candidate_authorizations': {GRANT: {'jobs': [job]}}}
        self.receipt = dict(job_id='cpu-only', status='failed', failure=dict(type='TimeoutError'),
            stop_verified=True, accounting_closed=True, cleanup=dict(passed=True),
            supervisor=dict(stop_evidence=proof), cloud_calls=[dict(stage='cpu_asset_admission')])
        self.apps = [proof]

    def test_rounds_closed_upper_up_without_mutating_or_claiming_invoice(self):
        before = deepcopy(self.ledger)
        result = proposal(self.ledger, self.receipt, self.apps)
        self.assertEqual(result['retained_closed_cost_upper_usd'], '0.90')
        self.assertEqual(self.ledger, before)
        self.assertIn('not an invoice', result['statement'])

    def test_gpu_attempt_is_not_reconciled_by_cpu_only_rule(self):
        self.receipt['cloud_calls'].append(dict(stage='cuda_admission_and_discarded_health_only'))
        with self.assertRaises(ValueError):
            proposal(self.ledger, self.receipt, self.apps)

    def test_unstopped_or_uncleaned_attempt_keeps_full_reservation(self):
        for field in ('stop_verified', 'accounting_closed'):
            r = deepcopy(self.receipt);r[field] = False
            with self.assertRaises(ValueError):
                proposal(self.ledger, r, self.apps)
        r = deepcopy(self.receipt);r['cleanup']['passed'] = False
        with self.assertRaises(ValueError):
            proposal(self.ledger, r, self.apps)


if __name__ == '__main__':
    unittest.main()

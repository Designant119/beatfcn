import json
from pathlib import Path
import unittest

from scripts.reconcile_hn2_audit_failure_reservation import proposal


ROOT = Path(__file__).resolve().parents[1]


class AuditCostTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text())
        self.receipt = json.loads((ROOT/'reports/council-modal120m-c27/runtime-launch.json').read_text())
        self.apps = json.loads((ROOT/'reports/council-modal120m-c27/postrun-apps.json').read_text())

    def test_specific_stopped_25_update_audit_failure_retains_upward_upper(self):
        self.assertEqual(proposal(self.ledger, self.receipt, self.apps)
                         ['retained_closed_cost_upper_usd'], '0.70')

    def test_unknown_progress_is_not_reconciled_as_known_case(self):
        phase = self.receipt['result']['evidence']['phase.json']
        phase['text'] = '{"health_applied_updates":24,"quality_applied_updates":0}'
        with self.assertRaises(ValueError):
            proposal(self.ledger, self.receipt, self.apps)

    def test_uncleaned_attempt_keeps_full_reservation(self):
        self.receipt['cleanup']['passed'] = False
        with self.assertRaises(ValueError):
            proposal(self.ledger, self.receipt, self.apps)


if __name__ == '__main__':
    unittest.main()

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.reconcile_hn2_premodel_reservation import proposal


ROOT = Path(__file__).resolve().parents[1]


class PreModelTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text())
        self.receipt = json.loads((ROOT/'reports/council-modal120m-c26/runtime-launch.json').read_text())
        self.apps = json.loads((ROOT/'reports/council-modal120m-c26/postrun-apps.json').read_text())

    def test_specific_closed_run05_rounds_upper_up_without_mutation(self):
        original = deepcopy(self.ledger)
        value = proposal(self.ledger, self.receipt, self.apps)
        self.assertEqual(value['retained_closed_cost_upper_usd'], '0.63')
        self.assertEqual(self.ledger, original)

    def test_any_other_gpu_failure_is_not_released_by_this_case_rule(self):
        self.receipt['failure']['reason'] = 'CUDA OOM after training started'
        with self.assertRaises(ValueError):
            proposal(self.ledger, self.receipt, self.apps)

    def test_unverified_cleanup_cannot_release_gpu_reservation(self):
        self.receipt['cleanup']['passed'] = False
        with self.assertRaises(ValueError):
            proposal(self.ledger, self.receipt, self.apps)


if __name__ == '__main__':
    unittest.main()

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/run_full_database_mert_pilot.py'


class FullDatabaseMertPilotLauncherTests(unittest.TestCase):
    def test_modal_import_and_spawn_stay_behind_launch_gate(self):
        tree = ast.parse(SOURCE.read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(getattr(node, 'module', '') == 'modal' or
            any(alias.name == 'modal' for alias in getattr(node, 'names', []))
            for node in imports))
        text = SOURCE.read_text()
        self.assertIn('reserve(LEDGER, contract[\'run_id\']', text)
        self.assertIn('import modal', text.split('# The first cloud-capable import', 1)[1])
        self.assertNotIn('--yes', text)

    def test_contract_defaults_are_zero_update_non_retry_l40s(self):
        text = SOURCE.read_text()
        self.assertIn("optimizer_updates=0", text)
        self.assertIn("full_corpus_encode=False", text)
        self.assertIn("automatic_retry=False", text)
        self.assertIn("gpu='L40S'", text)
        self.assertIn("other_upper_usd='0.30'", text)
        self.assertIn("method='legacy_GpuUtil_take_interval_means_with_last8_resource_guard'", text)
        self.assertIn('legacy_stable_period_gpu_mean_at_least_85_percent=True', text)
        self.assertIn('last_eight_interval_mean_at_least_85_percent_after_480s=True', text)


if __name__ == '__main__':
    unittest.main()

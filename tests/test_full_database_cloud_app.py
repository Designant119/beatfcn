import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'beatfcn/cloud/full_database_app.py'


class FullDatabaseCloudAppTests(unittest.TestCase):
    def test_encode_is_bounded_single_container_without_local_entrypoint(self):
        tree = ast.parse(SOURCE.read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('encode', functions)
        self.assertNotIn('main', functions)
        decorators = functions['encode'].decorator_list
        calls = [node for node in decorators if isinstance(node, ast.Call)]
        self.assertEqual(len(calls), 1)
        keywords = {}
        for node in calls[0].keywords:
            if node.arg == 'timeout' and isinstance(node.value, ast.BinOp):
                keywords[node.arg] = ast.literal_eval(node.value.left) * ast.literal_eval(node.value.right)
            elif node.arg in {'gpu', 'retries', 'max_containers'}:
                keywords[node.arg] = ast.literal_eval(node.value)
        self.assertEqual(keywords['gpu'], 'H100')
        self.assertEqual(keywords['retries'], 0)
        self.assertEqual(keywords['max_containers'], 1)
        self.assertGreaterEqual(keywords['timeout'], 3600)
        self.assertNotIn('@app.local_entrypoint', SOURCE.read_text())

    def test_static_contract_pins_checkpoint_and_output_root(self):
        text = SOURCE.read_text()
        self.assertIn("CHECKPOINT = '/data/full-database-v1/mert2-checkpoint'", text)
        self.assertIn("REMOTE_ROOT = 'full-database-v1'", text)
        self.assertIn('device=\'cuda\', local_source=CHECKPOINT', text)
        self.assertIn('output_root.startswith(REMOTE_ROOT + \'/\')', text)


    def test_gpu_monitor_uses_legacy_stable_period_last_eight_average(self):
        import time
        from beatfcn.cloud.full_database_app import StableGpuMonitor
        monitor = StableGpuMonitor()
        with monitor.lock:
            monitor.interval_means = [90.] * 8
            monitor.sample_counts = [60] * 8
            monitor.started_monotonic = time.monotonic() - 500
        summary = monitor.summary()
        self.assertTrue(summary['passed'])
        self.assertGreaterEqual(summary['stable_period_mean_utilization_percent'], 85)
        self.assertGreaterEqual(summary['last_eight_interval_mean_utilization_percent'], 85)
        self.assertEqual(summary['rolling_intervals'], 8)
        self.assertEqual(summary['check_after_s'], 480)

    def test_gpu_monitor_fails_short_stable_period_or_low_legacy_mean(self):
        import time
        from beatfcn.cloud.full_database_app import StableGpuMonitor
        short = StableGpuMonitor()
        with short.lock:
            short.interval_means = [90.] * 8
            short.sample_counts = [60] * 8
            short.started_monotonic = time.monotonic() - 100
        self.assertFalse(short.summary()['passed'])
        low = StableGpuMonitor()
        with low.lock:
            low.interval_means = [90.] * 5 + [20.] * 3
            low.sample_counts = [60] * 8
            low.started_monotonic = time.monotonic() - 500
        summary = low.summary()
        self.assertFalse(summary['passed'])
        self.assertLess(summary['last_eight_interval_mean_utilization_percent'], 85)

    def test_gpu_monitor_reuses_existing_gpu_util_sampler(self):
        text = SOURCE.read_text()
        self.assertIn('from beatfcn.cloud.gpu_util import GpuUtil', text)
        self.assertIn('GpuUtil(period_ms=500)', text)
        self.assertNotIn('phase_callback=monitor.phase', text)
        self.assertNotIn('overall_mean_utilization_percent', text)


if __name__ == '__main__':
    unittest.main()

import copy
import unittest
from pathlib import Path

from scripts.freeze_local_capacity_quality import execution_ast, validate_health


class CapacityQualityTests(unittest.TestCase):
    def test_execution_comparison_catches_unregistered_loss_change(self):
        source = Path('scripts/run_local_capacity.py').read_text()
        old = Path('inputs/code_baselines/d94-data-scale-v1/scripts/run_local_data_scale.py').read_text()
        self.assertEqual(execution_ast(source), execution_ast(old))
        changed = source.replace('loss = event_loss(logits, sequence)*len(ids)/4',
                                 'loss = event_loss(logits, sequence)*len(ids)/8')
        self.assertNotEqual(execution_ast(changed), execution_ast(old))

    def test_incomplete_exact_generation_cannot_pass_health(self):
        plan = dict(id='health', cases=[dict(record_index=i) for i in range(8)], max_request_mean_ce=.05)
        report = dict(id='health', complete=True, passed=True, optimizer_updates=100,
            plan_sha256='identity', parent_unchanged=True,
            final=[dict(request_mean_ce=.01, bos_correct=True)] * 8,
            generation=[dict(record_index=i, complete=True, exact_tokens=True, exact_events=True) for i in range(8)],
            padding_proofs=[dict(passed=True)])
        validate_health(report, plan, 'identity')
        for field in ('complete', 'exact_events', 'exact_tokens'):
            damaged = copy.deepcopy(report)
            damaged['generation'][3][field] = False
            with self.assertRaises(ValueError):
                validate_health(damaged, plan, 'identity')


if __name__ == '__main__':
    unittest.main()

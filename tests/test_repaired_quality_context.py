import unittest
import inspect

from beatfcn.train.field_integrity import replay_fields
from beatfcn.train.native_trial_evaluation import evaluate as evaluate_trial
from scripts import run_native_cell_recovery as original
from scripts import run_repaired_native_quality as repaired


class BindingTests(unittest.TestCase):
    def test_isolated_binding_preserves_original_namespace_and_training(self):
        original_art=original.ART;original_reader=original.replay_fields
        runner=repaired.load_runner()
        self.assertEqual(original.ART,original_art);self.assertIs(original.replay_fields,original_reader)
        self.assertEqual(runner.ART,repaired.ART);self.assertNotEqual(runner.ART,original.ART)
        self.assertIs(runner.replay_fields,replay_fields)
        self.assertIs(runner.cell_loss,original.cell_loss)
        self.assertEqual(runner.train.__code__.co_code,original.train.__code__.co_code)
        self.assertIsNot(runner.train.__globals__,original.train.__globals__)
        self.assertEqual(runner.train.__globals__['ART'],repaired.ART)
        self.assertEqual(inspect.unwrap(runner.field_panel).__globals__['replay_fields'],replay_fields)
        self.assertIs(runner.evaluate.func,evaluate_trial)
        self.assertIs(runner.evaluate.args[0],runner)


if __name__=='__main__':unittest.main()

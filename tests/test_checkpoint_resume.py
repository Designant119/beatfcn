from copy import deepcopy
from dataclasses import asdict
import unittest

from beatfcn.model.full_event_model import FullEventConfig
from beatfcn.train.checkpoint_resume import validate_resume


class ResumeTests(unittest.TestCase):
    def fixture(self):
        parent = dict(config={}, steps=300000, accum_steps=16, device='cuda',
                      seed=1, inputs_sha256={'data': 'frozen'}, loss={'type_weight': .1})
        plan = dict(parent, compile_loss=True, output='new.json', max_train_wall_s=100,
                    resume_checkpoint=dict(sha256='checkpoint', parent_plan_sha256='parent'))
        saved = dict(config=asdict(FullEventConfig()), updates=1504,
                     progress=dict(updates=1504, optimizer_updates=94), optimizer={'state': {}},
                     cpu_rng='cpu', cuda_rng=['gpu'], plan_sha256='parent')
        return plan, parent, saved

    def check(self, plan, parent, saved):
        return validate_resume(plan, parent, saved, parent_sha='parent', checkpoint_sha='checkpoint')

    def test_execution_changes_preserve_exact_next_exposure(self):
        self.assertEqual(self.check(*self.fixture()), 1504)

    def test_data_loss_and_optimizer_changes_are_rejected(self):
        for field in ['seed', 'loss', 'inputs_sha256', 'optimizer']:
            plan, parent, saved = self.fixture()
            plan[field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'scientific field'):
                self.check(plan, parent, saved)

    def test_recipe_v2_fields_are_scientific(self):
        changes = dict(vocabulary=dict(relative_blocks=True), ema=dict(decay=.99, evaluate=True),
                       counterfactual=dict(treatment_mix=[1], condition_dropout=dict(all_probability=.5)),
                       counterfactual_curve_cell_us=500_000, onset_supervision=dict(scope='core'),
                       optimizer_updates=99)
        for field, value in changes.items():
            plan, parent, saved = self.fixture()
            plan[field] = value
            with self.assertRaisesRegex(ValueError, 'scientific field: '+field):
                self.check(plan, parent, saved)

    def test_default_vocabulary_spellings_are_equal_and_checkpoint_vocab_is_checked(self):
        from beatfcn.model.full_event_codec import EventVocabulary
        plan, parent, saved = self.fixture()
        plan['vocabulary'], parent['vocabulary'] = {}, None
        saved['vocabulary'] = asdict(EventVocabulary())
        self.assertEqual(self.check(plan, parent, saved), 1504)
        plan, parent, saved = self.fixture()
        plan['vocabulary'] = parent['vocabulary'] = dict(relative_blocks=True, tail_query=True)
        saved['vocabulary'] = asdict(EventVocabulary())
        with self.assertRaisesRegex(ValueError, 'vocabulary mismatch'):
            self.check(plan, parent, saved)

    def test_ema_plan_requires_ema_state(self):
        plan, parent, saved = self.fixture()
        plan['ema'] = parent['ema'] = dict(decay=.999, evaluate=True)
        with self.assertRaisesRegex(ValueError, 'EMA'):
            self.check(plan, parent, saved)
        saved['ema'] = dict(decay=.999)
        self.assertEqual(self.check(plan, parent, saved), 1504)

    def test_partial_accumulation_or_weights_only_are_rejected(self):
        plan, parent, saved = self.fixture()
        saved['updates'] = 1505
        with self.assertRaisesRegex(ValueError, 'optimizer boundary'):
            self.check(plan, parent, saved)
        plan, parent, saved = self.fixture()
        saved['optimizer'] = None
        with self.assertRaisesRegex(ValueError, 'optimizer and RNG'):
            self.check(plan, parent, saved)

    def test_wrong_parent_is_rejected(self):
        plan, parent, saved = self.fixture()
        saved['plan_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'parent plan'):
            self.check(plan, parent, saved)

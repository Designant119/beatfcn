import copy
import json
from pathlib import Path
import unittest

from scripts.validate_mechanism_reviews import validate_review, validate_successor


ROOT=Path(__file__).resolve().parents[1]


class MechanismReviewGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review=json.loads((ROOT/'reports/local_whole_song_arrangement_review_v2.json').read_text())

    def test_actual_deep_review_has_identifiable_evidence(self):
        self.assertEqual(validate_review(self.review),[])

    def test_result_without_structure_layer_is_not_complete(self):
        value=copy.deepcopy(self.review);del value['layers']['structure_conditions']
        self.assertIn('Missing result layer: structure_conditions',validate_review(value))

    def test_causal_story_requires_a_falsifiable_prediction(self):
        value=copy.deepcopy(self.review);del value['mechanism_hypotheses'][0]['prediction']
        self.assertTrue(any('prediction' in x for x in validate_review(value)))

    def test_changed_raw_identity_blocks_closure(self):
        value=copy.deepcopy(self.review);name=next(iter(value['identity']));value['identity'][name]='0'*64
        self.assertTrue(any('Evidence identity changed' in x for x in validate_review(value)))

    def test_shared_failure_without_a_contract_change_blocks_successor(self):
        contract=dict(predecessors=['B'],shared_failure_modes=['latent_collapse'],mechanism_changes={},
                      pretraining_probe_receipts=[dict(status='completed',evidence='reports/local_whole_song_mechanism_v2.json')],execution_ready=True)
        self.assertIn('Unaddressed shared failure: latent_collapse',validate_successor(contract,{'B'}))

    def test_planned_probe_is_not_an_executed_receipt(self):
        contract=dict(predecessors=['B'],shared_failure_modes=[],pretraining_probe_receipts=[dict(status='planned',evidence='reports/local_whole_song_mechanism_v2.json')],execution_ready=True)
        self.assertIn('Unperformed pretraining diagnostic',validate_successor(contract,{'B'}))

    def test_old_weight_diagnosis_does_not_certify_a_new_implementation(self):
        contract=dict(predecessors=['B'],shared_failure_modes=[],
                      pretraining_probe_receipts=[dict(status='completed',
                          scope='frozen_predecessor_diagnostic_only',
                          evidence='reports/local_whole_song_mechanism_v2.json')],execution_ready=True)
        self.assertIn('Predecessor diagnosis cannot certify a new implementation',validate_successor(contract,{'B'}))


if __name__=='__main__':unittest.main()

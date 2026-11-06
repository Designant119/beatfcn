import unittest
import torch

from beatfcn.model.logit_audit import compare_valid_logits


class LogitAuditTests(unittest.TestCase):
    def test_padding_cannot_create_a_false_equivalence_failure(self):
        a = torch.zeros(1, 3, 4); b = a.clone()
        b[0, 2] = float('nan')
        valid = torch.tensor([[True, True, False]])
        report = compare_valid_logits(a, b, valid)
        self.assertTrue(report['passed'])
        self.assertEqual(report['maximum_logit_difference'], 0.)
        self.assertEqual(report['excluded_padding_queries'], 1)
        b[0, 1, 0] = .1
        self.assertFalse(compare_valid_logits(a, b, valid)['passed'])

    def test_nonfinite_valid_logits_fail_without_nonfinite_json_metrics(self):
        a = torch.zeros(1, 2, 4); b = a.clone(); b[0, 0, 0] = float('nan')
        report = compare_valid_logits(a, b, torch.ones(1, 2, dtype=torch.bool))
        self.assertFalse(report['passed']); self.assertEqual(report['nonfinite'], 1)
        self.assertIsNone(report['maximum_logit_difference'])

    def test_legal_decisions_and_required_valid_support(self):
        a = torch.tensor([[[100., 1., 2.]]]); b = a.clone(); b[..., 0] += 1e-5
        valid = torch.ones(1, 1, dtype=torch.bool); allowed = torch.tensor([[[False, True, True]]])
        self.assertTrue(compare_valid_logits(a, b, valid, allowed=allowed)['passed'])
        for mask in (torch.zeros_like(valid), torch.ones(1, 2, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                compare_valid_logits(a, b, mask)
        with self.assertRaises(ValueError):
            compare_valid_logits(a, b, valid, allowed=torch.zeros_like(allowed))

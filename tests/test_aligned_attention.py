import unittest

import torch
from torch.nn import functional as F

from beatfcn.model.full_event_model import aligned_attention


class AlignedAttentionTests(unittest.TestCase):
    def test_non_aligned_heads_preserve_mask_scale_outputs_and_gradients(self):
        torch.manual_seed(42)
        q = torch.randn(2, 2, 5, 68, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 2, 7, 68, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, 2, 7, 68, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(2, 1, 5, 7, dtype=torch.bool)
        mask[:, :, :, -2:] = False
        mask[:, :, -1] = False
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        actual = aligned_attention(q, k, v, mask, 0.)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        weight = torch.randn_like(reference)
        expected_grads = torch.autograd.grad((reference*weight).sum(), (q, k, v))
        actual_grads = torch.autograd.grad((actual*weight).sum(), (q, k, v))
        for actual_grad, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected, atol=1e-12, rtol=1e-12)

    def test_aligned_heads_keep_shape_and_gradients(self):
        q = torch.randn(1, 2, 4, 16, requires_grad=True)
        result = aligned_attention(q, q, q, None, 0.)
        self.assertEqual(result.shape, q.shape)
        result.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

import unittest
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from beatfcn.audio.mert_attention import QueryChunkedMertAttention


def identity_attention():
    old = nn.Module()
    old.config = SimpleNamespace(_attn_implementation='sdpa')
    old.num_heads, old.head_dim = 1, 4
    for name in ('query_proj', 'key_proj', 'value_proj', 'out_proj'):
        setattr(old, name, nn.Identity())
    return old.eval()


class MertQueryChunksTests(unittest.TestCase):
    def test_partial_query_chunk_keeps_full_key_softmax(self):
        model = QueryChunkedMertAttention(identity_attention(), 2)
        signal = torch.zeros(1, 5, 4)
        signal[:, -1] = 1.
        position = (torch.ones(1, 5, 1, 4), torch.zeros(1, 5, 1, 4))
        with torch.inference_mode():
            actual = model(signal, position)
            expected = F.scaled_dot_product_attention(signal[:, None], signal[:, None], signal[:, None],
                dropout_p=0., is_causal=False)[:, 0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, :4], torch.full((1, 4, 4), .2))

    def test_batched_nondivisible_rotary_queries_match_full_attention(self):
        torch.manual_seed(20261026)
        signal = torch.randn(2, 13, 4)
        angle = torch.randn(1, 13, 1, 4)
        cos, sin = angle.cos(), angle.sin()
        with torch.inference_mode():
            actual = QueryChunkedMertAttention(identity_attention(), 5)(signal, (cos, sin))
            shaped = signal[:, :, None]
            a, b = shaped.chunk(2, -1)
            rotated = (shaped*cos+torch.cat((-b, a), -1)*sin).transpose(1, 2)
            expected = F.scaled_dot_product_attention(rotated, rotated, shaped.transpose(1, 2),
                dropout_p=0., is_causal=False).transpose(1, 2).reshape_as(signal)
        torch.testing.assert_close(actual, expected)

    def test_training_cannot_silently_use_frozen_path(self):
        model = QueryChunkedMertAttention(identity_attention(), 2)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 5, 4), (torch.ones(1, 5, 1, 4), torch.zeros(1, 5, 1, 4)))


if __name__ == '__main__':
    unittest.main()

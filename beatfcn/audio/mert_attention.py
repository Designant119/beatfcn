"""Exact full-key attention with bounded query rows for frozen local MERT."""
import torch
from torch import nn
from torch.nn import functional as F


class QueryChunkedMertAttention(nn.Module):
    """Change only the SDPA query batch; every row retains every key/value."""

    def __init__(self, original, chunk_size):
        super().__init__()
        if original.config._attn_implementation != 'sdpa' or chunk_size < 1:
            raise ValueError('Expected SDPA and a positive query chunk')
        self.config = original.config
        self.num_heads, self.head_dim = original.num_heads, original.head_dim
        self.chunk_size = chunk_size
        for name in ('query_proj', 'key_proj', 'value_proj', 'out_proj'):
            setattr(self, name, getattr(original, name))
        self.train(original.training)

    def forward(self, hidden_states, position_embeddings):
        if self.training or torch.is_grad_enabled():
            raise ValueError('This frozen encoder execution path requires inference mode')
        batch, length, width = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.query_proj(hidden_states).reshape(shape)
        key = self.key_proj(hidden_states).reshape(shape)
        value = self.value_proj(hidden_states).reshape(shape)
        cos, sin = position_embeddings

        def rotate_half(x):
            left, right = x.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)

        query = (query*cos+rotate_half(query)*sin).transpose(1, 2)
        key = (key*cos+rotate_half(key)*sin).transpose(1, 2)
        value = value.transpose(1, 2)
        rows = [F.scaled_dot_product_attention(query[:, :, start:start+self.chunk_size],
                key, value, dropout_p=0., is_causal=False)
                for start in range(0, length, self.chunk_size)]
        attended = torch.cat(rows, dim=2).transpose(1, 2)
        return self.out_proj(attended.reshape(batch, length, width))


def install_query_chunks(model, chunk_size=512):
    """Keep parameter objects/names intact; replace only attention execution."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze and eval the encoder before installing query chunks')
    original = dict(model.named_parameters())
    if any(isinstance(layer.attn, QueryChunkedMertAttention) for layer in model.layers):
        raise ValueError('Query chunks already installed')
    for layer in model.layers:
        layer.attn = QueryChunkedMertAttention(layer.attn, chunk_size)
    installed = dict(model.named_parameters())
    if original.keys() != installed.keys() or any(installed[k] is not v for k, v in original.items()):
        raise AssertionError('Attention installation changed pretrained parameters')

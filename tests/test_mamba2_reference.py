"""Differential tests against frozen official source and direct SSD equations."""
import ast
import math
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from beatfcn.model.mamba2_reference import Mamba2Reference, ssd_scan

ROOT = Path(__file__).resolve().parents[1]


def rearrange(x, pattern, **kw):
    """Only shape operations used by the pinned upstream verification functions."""
    if pattern == 'b (c l) ... -> b c l ...':
        return x.reshape(x.shape[0], x.shape[1] // kw['l'], kw['l'], *x.shape[2:])
    if pattern == 'b c l h -> b h c l':
        return x.permute(0, 3, 1, 2)
    if pattern == 'b c l h p -> b (c l) h p':
        return x.reshape(x.shape[0], -1, *x.shape[-2:])
    if pattern == 'b l (h p) -> b l h p':
        return x.reshape(*x.shape[:2], x.shape[-1] // kw['p'], kw['p'])
    if pattern == 'b l (g n) -> b l g n':
        return x.reshape(*x.shape[:2], kw['g'], x.shape[-1] // kw['g'])
    if pattern == 'b l h p -> b l (h p)':
        return x.flatten(-2)
    raise ValueError('Unexpected upstream shape contract: ' + pattern)


def repeat(x, pattern, **kw):
    if pattern == '... d -> ... d e':
        return x[..., None].expand(*x.shape, kw['e'])
    raise ValueError('Unexpected upstream repeat contract: ' + pattern)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(body) != len(names):
        raise ValueError('Pinned upstream definitions changed')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


def upstream_ssd():
    scope = dict(torch=torch, F=F, rearrange=rearrange, repeat=repeat)
    extract(ROOT / 'inputs/fullstack_mamba_v1/upstream/ssd_minimal.py',
            {'segsum', 'ssd_minimal_discrete'}, scope)
    return scope['ssd_minimal_discrete']


def direct_scan(x, a, b, c, initial=None):
    state = torch.zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1], dtype=x.dtype, device=x.device) if initial is None else initial
    output = []
    for i in range(x.shape[1]):
        state = state * a[:, i].exp()[..., None, None] + x[:, i, :, :, None] * b[:, i, :, None, :]
        output.append((state * c[:, i, :, None, :]).sum(-1))
    return torch.stack(output, 1), state


def official_simple(dim, state, head, chunk):
    scope = dict(torch=torch, F=F, nn=nn, math=math, rearrange=rearrange, repeat=repeat,
                 causal_conv1d_fn=None)
    extract(ROOT / 'inputs/fullstack_mamba_v1/upstream/layernorm_gated.py', {'rms_norm_ref'}, scope)
    norm_ref = scope['rms_norm_ref']
    class Norm(nn.Module):
        def __init__(self, width, eps, norm_before_gate, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width, **kwargs))
            self.eps = eps
            self.norm_before_gate = norm_before_gate
        def forward(self, x, z):
            return norm_ref(x, self.weight, None, z=z, eps=self.eps,
                            norm_before_gate=self.norm_before_gate)
    def kernel(x, dt, a, b, c, chunk_size, D, **kwargs):
        b = b.expand(-1, -1, x.shape[2], -1)
        c = c.expand(-1, -1, x.shape[2], -1)
        y, _ = direct_scan(x * dt[..., None], dt * a, b, c)
        return y + x * D[None, None, :, None]
    scope.update(RMSNormGated=Norm, mamba_chunk_scan_combined=kernel)
    extract(ROOT / 'inputs/fullstack_mamba_v1/upstream/mamba2_simple.py', {'Mamba2Simple'}, scope)
    return scope['Mamba2Simple'](dim, d_state=state, headdim=head, chunk_size=chunk, use_mem_eff_path=False)


class Mamba2ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261026)

    def test_ssd_outputs_states_and_gradients_match_official_and_direct_scan(self):
        tensors = [torch.randn(2, 24, 2, 4), -torch.rand(2, 24, 2),
                   torch.randn(2, 24, 2, 3), torch.randn(2, 24, 2, 3)]
        functions = [lambda *x: ssd_scan(*x, 8), lambda *x: upstream_ssd()(*x, 8), direct_scan]
        versions = []
        for fn in functions:
            values = [t.clone().requires_grad_() for t in tensors]
            out, state = fn(*values)
            (out.square().mean() + state.square().mean()).backward()
            versions.append((out.detach(), state.detach(), [v.grad for v in values]))
        for other in versions[1:]:
            torch.testing.assert_close(versions[0][0], other[0], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(versions[0][1], other[1], atol=2e-5, rtol=2e-5)
            for a, b in zip(versions[0][2], other[2]):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)

    def test_full_layer_matches_official_simple_equations_including_gradients(self):
        official = official_simple(16, 8, 8, 8)
        actual = Mamba2Reference(16, d_state=8, headdim=8, chunk_size=8)
        actual.load_state_dict(official.state_dict())
        x = torch.randn(2, 19, 16)
        outputs = []
        for model in (official, actual):
            output = model(x)
            output.square().mean().backward()
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
        for name, p in actual.named_parameters():
            torch.testing.assert_close(p.grad, dict(official.named_parameters())[name].grad, atol=2e-5, rtol=2e-5)

    def test_non_chunk_prefix_padding_and_cached_steps_match_full_forward(self):
        model = Mamba2Reference(16, d_state=8, headdim=8, chunk_size=8).eval()
        x = torch.randn(2, 29, 16)
        with torch.no_grad():
            full = model(x)
            prefix, cache = model(x[:, :13], return_cache=True)
            pieces = [prefix]
            for i in range(13, len(x[0])):
                y, cache = model.step(x[:, i:i + 1], cache)
                pieces.append(y)
            torch.testing.assert_close(full, torch.cat(pieces, 1), atol=1e-5, rtol=1e-5)
            _, whole_cache = model(x, return_cache=True)
            torch.testing.assert_close(cache['state'], whole_cache['state'], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(cache['conv'], whole_cache['conv'])


if __name__ == '__main__':
    unittest.main()

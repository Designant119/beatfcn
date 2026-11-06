"""Bias-corrected exponential moving average of model parameters.

The shadow after n optimizer updates is the normalized exponentially weighted
average of the post-update parameters theta_1..theta_n with weights
decay**(n-k) / sum_j decay**(n-j). It is computed in place with the per-update
coefficient beta_n = decay*(1-decay**(n-1))/(1-decay**n) (beta_1 = 0), so the
random initialization never leaks into the average: a plain EMA started from
theta_0 would still hold decay**n of it (13.5% after 2000 updates at 0.999).
beta_n -> decay as n grows. Shadows are float32 on the parameters' device; the
EMA never touches gradients or optimizer state.
"""
from contextlib import contextmanager

import torch


class ParameterEMA:
    def __init__(self, model, decay=.999):
        decay = float(decay)
        if not 0 < decay < 1:
            raise ValueError('EMA decay must lie in (0, 1)')
        self.decay, self.updates = decay, 0
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.names = [name for name, _ in named]
        self.params = [p for _, p in named]
        self.shadow = [p.detach().float().clone() for p in self.params]

    def coefficient(self, n):
        """Weight kept on the previous shadow at the n-th (1-based) update."""
        d = self.decay
        return d * (1 - d ** (n - 1)) / (1 - d ** n)

    @torch.no_grad()
    def update(self):
        self.updates += 1
        beta = self.coefficient(self.updates)
        current = [p.detach().float() for p in self.params]
        if hasattr(torch, '_foreach_lerp_'):
            torch._foreach_lerp_(self.shadow, current, 1 - beta)
        else:  # pragma: no cover - very old torch
            for s, p in zip(self.shadow, current):
                s.lerp_(p, 1 - beta)

    def model_state_dict(self, model):
        """``model.state_dict()`` keys with EMA parameter values (buffers as-is), on CPU."""
        by_id = {id(p): s for p, s in zip(self.params, self.shadow)}
        state = {}
        for key, value in model.state_dict(keep_vars=True).items():
            source = by_id.get(id(value))
            tensor = (source.to(value.dtype) if source is not None else value).detach()
            state[key] = tensor.cpu().clone()
        return state

    def state_dict(self, model):
        return dict(decay=self.decay, updates=self.updates, bias_corrected=True,
                    model=self.model_state_dict(model))

    @torch.no_grad()
    def load_state_dict(self, state, model):
        if float(state['decay']) != self.decay or not state.get('bias_corrected'):
            raise ValueError('Checkpoint EMA has a different decay or definition')
        values = state['model']
        names = dict(model.named_parameters())
        missing = [name for name in self.names if name not in values]
        if missing:
            raise ValueError(f'Checkpoint EMA lacks {len(missing)} parameters')
        for name, shadow in zip(self.names, self.shadow):
            if values[name].shape != names[name].shape:
                raise ValueError(f'EMA shape mismatch for {name}')
            shadow.copy_(values[name].to(shadow.device, torch.float32))
        self.updates = int(state['updates'])

    @contextmanager
    def swapped(self):
        """Temporarily load EMA values into the live parameters (e.g. for evaluation)."""
        backup = [p.detach().clone() for p in self.params]
        try:
            with torch.no_grad():
                for p, s in zip(self.params, self.shadow):
                    p.copy_(s.to(p.dtype))
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)
            del backup

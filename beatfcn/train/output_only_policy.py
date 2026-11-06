"""A frozen decoder with a trainable output matrix and an immutable prior."""
from pathlib import Path
import hashlib

import torch
from torch.nn import functional as F


class OutputOnlyPolicy:
    def __init__(self, model, parent_sha256):
        self.model = model.eval()
        self.parent_sha256 = parent_sha256
        for p in model.parameters():
            p.requires_grad_(False)
        self.weight = model.output.weight
        self.weight.requires_grad_(True)
        self.base = self.weight.detach().clone()
        self.hidden = None
        self.hook = model.output.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self.hidden = args[0].detach()

    def features(self, inputs, sequence):
        with torch.no_grad():
            logits = self.model(inputs, sequence)
            hidden = self.hidden
            self.hidden = None
        return hidden, logits

    def prior(self, hidden):
        return F.linear(hidden, self.base)

    def logits(self, hidden):
        return F.linear(hidden, self.weight)

    def restore(self):
        with torch.no_grad():
            self.weight.copy_(self.base)

    def save(self, path, *, updates, metadata):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix('.partial')
        if temp.exists():
            raise FileExistsError(temp)
        torch.save(dict(schema='beatfcn.output-only.v1', parent_sha256=self.parent_sha256,
                        output_weight=self.weight.detach().cpu(), updates=updates, metadata=metadata), temp)
        temp.replace(path)

    def load(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('schema') != 'beatfcn.output-only.v1' or saved['parent_sha256'] != self.parent_sha256:
            raise ValueError('Output adapter parent identity mismatch')
        weight = saved['output_weight']
        if weight.shape != self.weight.shape or not torch.isfinite(weight).all():
            raise ValueError('Malformed output adapter')
        with torch.no_grad():
            self.weight.copy_(weight)
        return saved

    def close(self):
        self.hook.remove()
        self.hidden = None


def save_interrupted_output(policy, path, *, updates, plan_sha256):
    """Keep one inspectable delta; a partial run can never become eligible here."""
    if type(updates) is not int or updates <= 0:
        raise ValueError('Interrupted delta requires actual updates')
    path=Path(path)
    policy.save(path,updates=updates,metadata={'plan_sha256':plan_sha256,
        'scope':'interrupted_diagnostic_only','quality_eligible':False})
    return dict(interrupted_adapter=str(path),interrupted_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quality_eligible=False)

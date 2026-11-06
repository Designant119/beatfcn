"""Explicit CPU log-mel placement for the local MERT numerical contract."""
import torch
from .mert2 import FrozenMert2, Mert2Spec


class CpuMelTransfer(torch.nn.Module):
    """Keep the official FP32 frontend on CPU, then transfer its result once.

    Install after model loading and placement. This is an explicit execution
    policy, not an unsupported-operator fallback. Waveforms remain on CPU.
    """

    def __init__(self, frontend, destination='mps'):
        super().__init__()
        self.frontend = frontend.cpu().eval().requires_grad_(False)
        self.destination = torch.device(destination)

    @torch.inference_mode()
    def forward(self, waveform):
        if waveform.device.type != 'cpu' or waveform.dtype != torch.float32:
            raise ValueError('Explicit CPU frontend requires CPU float32 waveform')
        for value in self.frontend.buffers():
            if value.device.type != 'cpu' or (value.is_floating_point() and value.dtype != torch.float32):
                raise ValueError('Official frontend buffers must remain CPU float32')
        values = self.frontend(waveform)
        if values.device.type != 'cpu' or values.dtype != torch.float32:
            raise ValueError('Official frontend output must be CPU float32')
        return values.to(self.destination)


class LocalFrozenMert2(FrozenMert2):
    """Reuse the versioned extraction clock with explicit CPU waveform input."""

    def __init__(self, model, spec=None):
        if not isinstance(model.feature_extractor, CpuMelTransfer):
            raise ValueError('Local MERT requires explicitly placed CPU frontend')
        if any(p.device.type != 'mps' or p.requires_grad for p in model.parameters()):
            raise ValueError('Local MERT encoder parameters must be frozen on MPS')
        self.model = model.eval()
        self.device = 'mps'
        self.spec = spec or Mert2Spec()

    def _waveform_tensor(self, padded):
        return torch.from_numpy(padded)[None]

"""Optional official Mamba-2 history mixer; no CPU surrogate or silent fallback."""
import torch
from torch import nn


class EventMamba2(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except ImportError as exc:
            raise ImportError('Mamba-2 candidates require the pinned CUDA experiment image') from exc
        self.mixer = Mamba2(d_model=config.d, d_state=config.ssm_state,
            d_conv=4, expand=2, headdim=config.ssm_headdim,
            chunk_size=64, layer_idx=layer_idx, use_mem_eff_path=True)

    def forward(self, x, valid, cache=None):
        # Each row is a separate request. Never flatten/persist state across rows.
        # A hole/left padding would change the recurrence, so reject it explicitly.
        # FullEventModel validates this once per forward, outside the layer loop.
        if cache is None:
            return self.mixer(x * valid[..., None]) * valid[..., None]
        if self.training or x.shape[1] != 1:
            raise ValueError('SSM cache requires one eval token per fixed batch row')
        if 'ssm' not in cache:
            # Allocate independent state per request/layer; step works from zeros
            # even for the first token and avoids short-prefix prefill ambiguities.
            cache['ssm'] = self.mixer.allocate_inference_cache(x.shape[0], 1, dtype=x.dtype)
        out, conv, state = self.mixer.step(x, *cache['ssm'])
        cache['ssm'] = (conv, state)
        return out

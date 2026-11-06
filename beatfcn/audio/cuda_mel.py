"""CUDA implementation of the audited complete-song mel recipe.

Constants and equations mirror audio.preprocess.waveform_mel. Read-only filter and
window tensors are cached per device; CPU mode is used by local equivalence tests.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .preprocess import (
    CELL_SAMPLES, HOP, MEAN, NFFT, NMEL, PAD, SR, STD, slaney_mel_filter)


_CACHE = {}


def _constants(device):
    key = str(device)
    if key not in _CACHE:
        _CACHE[key] = dict(
            window=torch.hann_window(NFFT, device=device),
            mel_filter=slaney_mel_filter().to(device))
    return _CACHE[key]


def complete_song_mel_cuda(wave, device='cuda'):
    """Compute complete-tail mel on CUDA without changing the physical clock."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or len(wave) < 393 or not np.isfinite(wave).all():
        raise ValueError('Expected a finite mono waveform with at least 393 samples')
    samples = len(wave)
    padded_samples = (samples + CELL_SAMPLES - 1) // CELL_SAMPLES * CELL_SAMPLES
    padded = np.pad(wave, (0, padded_samples - samples)) if padded_samples != samples else wave
    constants = _constants(device)
    x = torch.from_numpy(padded).to(device)
    x = F.pad(x[None, None], (PAD, PAD), mode='reflect').squeeze(1)
    spec = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
        window=constants['window'], center=False, return_complex=True)
    mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
    lm = (torch.log(torch.clamp(constants['mel_filter'] @ mag, min=1e-5)) - MEAN) / (STD + 1e-8)
    mel = lm.transpose(1, 2)[:, None]
    expected_frames = padded_samples // HOP
    if mel.shape != (1, 1, expected_frames, NMEL) or not torch.isfinite(mel).all():
        raise ValueError('Malformed CUDA complete-song mel')
    values = mel[0, 0].float().cpu().numpy().astype(np.float16, order='C')
    starts = np.arange(expected_frames, dtype=np.int64) * HOP
    ends = np.minimum(starts + HOP, samples)
    valid = starts < samples
    starts = np.minimum(starts, samples)
    return dict(mel=values, frame_start_samples=starts, frame_end_samples=ends,
        valid=valid, decoded_samples=samples, padded_samples=padded_samples,
        padding_samples=padded_samples - samples)


def mel_parity(reference, candidate):
    """Compare CPU/CUDA numerical output without hiding float16 quantization differences."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError('CPU/CUDA mel shapes differ')
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    exact = reference == candidate
    return dict(shape=list(reference.shape), max_abs=float(difference.max()) if difference.size else 0.,
        mean_abs=float(difference.mean()) if difference.size else 0.,
        float16_exact_fraction=float(exact.mean()) if exact.size else 1.,
        finite=bool(np.isfinite(candidate).all()))

"""Local complete-song mel preparation, isolated from existing training caches."""
import numpy as np

from beatfcn.audio.preprocess import CELL_SAMPLES, HOP, waveform_mel


def complete_song_mel(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or len(wave) < 393 or not np.isfinite(wave).all():
        raise ValueError('Expected a finite mono waveform with at least 393 samples')
    samples = len(wave)
    padded_samples = (samples+CELL_SAMPLES-1)//CELL_SAMPLES*CELL_SAMPLES
    padded = np.pad(wave, (0, padded_samples-samples)) if padded_samples != samples else wave
    mel = np.ascontiguousarray(waveform_mel(padded)[0, 0].numpy().astype(np.float16))
    starts = np.arange(len(mel), dtype=np.int64)*HOP
    ends = np.minimum(starts+HOP, samples)
    valid = starts < samples
    # Invalid padding carries an empty interval, never a backwards interval.
    starts = np.minimum(starts, samples)
    return dict(mel=mel, frame_start_samples=starts, frame_end_samples=ends,
                valid=valid, decoded_samples=samples, padded_samples=padded_samples,
                padding_samples=padded_samples-samples)


def legacy_prefix_comparison(legacy, complete):
    """The old reflection boundary affects its last frames; compare only the interior."""
    legacy = np.asarray(legacy)
    current = complete['mel']
    expected = complete['decoded_samples']//CELL_SAMPLES*CELL_SAMPLES//HOP
    if legacy.shape != (expected, 128):
        raise ValueError('Legacy mel shape differs from decoded audio')
    # Four hops exceed the STFT right context (631 samples). These four frames
    # remain in both arrays; only the parity comparison excludes that boundary.
    interior = len(legacy) if complete['padding_samples'] == 0 else max(0, len(legacy)-4)
    difference = np.abs(legacy[:interior].astype(np.float32)-current[:interior].astype(np.float32))
    return dict(compared_frames=interior, legacy_boundary_frames=len(legacy)-interior,
                interior_exact_equal=bool(np.array_equal(legacy[:interior], current[:interior])),
                max_abs_difference=float(difference.max()) if difference.size else 0.,
                missing_tail_frames=len(current)-len(legacy),
                parity_does_not_certify_pupu_tail=True)

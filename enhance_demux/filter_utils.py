import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_lowpass_sos(
    cutoff_hz: float,
    sr: float,
    order: int = 8,
) -> np.ndarray:
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz:.1f}) must be < Nyquist ({nyquist:.1f})"
        )
    return butter(order, cutoff_hz / nyquist, btype="low", output="sos")


def split_bands(
    audio: np.ndarray,
    cutoff_hz: float,
    sr: float,
    order: int = 8,
    pad_samples: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    sos = design_lowpass_sos(cutoff_hz, sr, order)

    n = audio.shape[-1]
    pad = min(pad_samples, n // 4) if n >= pad_samples * 2 else 0

    if pad > 0:
        padded = np.concatenate(
            [audio[..., :pad][..., ::-1], audio, audio[..., -pad:][..., ::-1]],
            axis=-1,
        )
        low_padded = sosfiltfilt(sos, padded, axis=-1)
        low_band = low_padded[..., pad : pad + n]
    else:
        low_band = sosfiltfilt(sos, audio, axis=-1)

    high_band = audio - low_band
    return low_band.astype(np.float32), high_band.astype(np.float32)
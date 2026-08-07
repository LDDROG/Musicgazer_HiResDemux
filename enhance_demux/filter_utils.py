"""
Linear-phase band-splitting filters.

Provides zero-phase (forward-backward) Butterworth filters to split
audio into low and high frequency bands with perfect reconstruction
(low + high == original).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_lowpass_sos(
    cutoff_hz: float,
    sr: float,
    order: int = 8,
) -> np.ndarray:
    """
    Design a Butterworth lowpass filter, returned as second-order sections.

    Args:
        cutoff_hz: -3 dB cutoff frequency in Hz.
        sr: Sample rate in Hz.
        order: Butterworth filter order (applied twice via filtfilt,
               so effective order is 2 * order).

    Returns:
        SOS array suitable for scipy.signal.sosfiltfilt.
    """
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
    """
    Split audio into low and high frequency bands with zero-phase filtering.

    The high band is computed as ``audio - low_band``, guaranteeing
    perfect reconstruction for any filter.

    A reflection pad / trim is applied to suppress filtfilt edge
    transients.

    Args:
        audio: Shape ``(..., time)`` array.
        cutoff_hz: Crossover frequency in Hz.
        sr: Sample rate in Hz.
        order: Butterworth filter order.
        pad_samples: Number of samples to pad at each end for transient
                     suppression. Default 8192 (~186ms @ 44.1kHz).

    Returns:
        (low_band, high_band): Both same shape as *audio*.
    """
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

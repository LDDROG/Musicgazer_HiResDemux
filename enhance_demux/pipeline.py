"""
Unified high-sample-rate separation pipeline with auto-strategy selection.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from enhance_demux.hires_separator import separate_varispeed, separate_band_split

VARISPEED_MAX_RATIO = 1.15


def separate_hires(
    mix: np.ndarray,
    model,
    config,
    device,
    model_type: str,
    orig_sr: int,
    target_sr: int = 44100,
    method: Literal["auto", "varispeed", "band_split"] = "auto",
    cutoff_ratio: float = 0.9,
    filter_order: int = 8,
    callback=None,
) -> dict[str, np.ndarray]:
    """
    Separate high-sample-rate audio through a model trained at *target_sr*.

    Parameters
    ----------
    mix : np.ndarray, shape (channels, time)
        Input audio at *orig_sr*.
    orig_sr : int
        Native sample rate (e.g. 48000, 96000, 192000).
    target_sr : int
        Sample rate the model expects (default 44100).
    method : str
        ``"auto"`` — picks varispeed when ratio <= 1.15, else band-split.
    cutoff_ratio : float
        (band-split only) Fraction of *target_sr* Nyquist for crossover.
    filter_order : int
        (band-split only) Butterworth order for the crossover filter.
    callback : callable or None
        Progress callback, passed through to ``demix``.

    Returns
    -------
    dict[str, np.ndarray]
        Separated stems at *orig_sr*.
    """
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]

    if orig_sr == target_sr:
        return separate_varispeed(
            mix, model, config, device, model_type,
            orig_sr=orig_sr, target_sr=target_sr,
            callback=callback,
        )

    ratio = orig_sr / target_sr

    if method == "auto":
        strategy = "varispeed" if ratio <= VARISPEED_MAX_RATIO else "band_split"
    else:
        strategy = method

    if strategy == "varispeed":
        return separate_varispeed(
            mix, model, config, device, model_type,
            orig_sr=orig_sr, target_sr=target_sr,
            callback=callback,
        )
    else:
        return separate_band_split(
            mix, model, config, device, model_type,
            orig_sr=orig_sr, target_sr=target_sr,
            cutoff_ratio=cutoff_ratio,
            filter_order=filter_order,
            callback=callback,
        )

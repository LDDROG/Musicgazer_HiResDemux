from __future__ import annotations
import numpy as np
from typing import Literal
from enhance_demux.hires_separator import separate_varispeed, separate_band_split

VARISPEED_MAX_RATIO = 1.15    # 采样率比值，超过该值则使用频带分割

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
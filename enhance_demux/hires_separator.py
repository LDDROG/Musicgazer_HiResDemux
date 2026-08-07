"""
High sample rate audio source separation.

Two strategies for processing audio at sample rates above 44.1kHz
through models trained exclusively on 44.1kHz audio.

Varispeed
---------
For moderate rate differences (e.g. 48kHz). The raw audio samples are
fed to the model unchanged. The model's internal STFT treats them as
44.1kHz audio, which linearly compresses all frequencies into the
model's 0--22.05kHz bandwidth.  After separation the output is
interpreted at the original rate, restoring correct speed and pitch.
Zero quality loss.

Band-split + per-bin mask extrapolation
---------------------------------------
For large rate differences (e.g. 96/192kHz). The audio is split into
a low band (0--~20kHz) and a high band (~20kHz--Nyquist).  The low
band is downsampled, processed normally, then upsampled back.  Per-stem
soft masks are computed from the stems' STFTs for every frequency bin
and extrapolated above the model's effective Nyquist.  The original
high-band STFT is masked per-bin and iSTFT'd back, preserving the
original recording's spectral shape while distributing HF energy
according to each stem's separation profile.
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import Dict

from enhance_demux.filter_utils import split_bands


def separate_varispeed(
    mix: np.ndarray,
    model,
    config,
    device,
    model_type: str,
    orig_sr: int,
    target_sr: int = 44100,
    callback=None,
) -> Dict[str, np.ndarray]:
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]

    from utils.utils import demix
    return demix(config, model, mix, device, model_type=model_type, callback=callback)


def separate_band_split(
    mix: np.ndarray,
    model,
    config,
    device,
    model_type: str,
    orig_sr: int,
    target_sr: int = 44100,
    cutoff_ratio: float = 0.9,
    filter_order: int = 8,
    callback=None,
) -> Dict[str, np.ndarray]:
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]

    n_channels, n_samples = mix.shape
    cutoff_hz = (target_sr / 2.0) * cutoff_ratio

    # ---- 1. split bands ------------------------------------------------
    mix_low, mix_high = split_bands(mix, cutoff_hz, orig_sr, order=filter_order)

    # ---- 2. downsample low band, run model, upsample -------------------
    mix_low_ds = librosa.resample(
        mix_low, orig_sr=orig_sr, target_sr=target_sr, axis=-1, res_type="kaiser_fast"
    )

    from utils.utils import demix
    waveforms = demix(config, model, mix_low_ds, device, model_type=model_type, callback=callback)

    instruments = sorted(waveforms.keys())

    for instr in instruments:
        wav = waveforms[instr]
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]
        if wav.shape[0] != n_channels:
            wav = np.broadcast_to(wav[:1], (n_channels, wav.shape[-1]))
        waveforms[instr] = librosa.resample(
            wav, orig_sr=target_sr, target_sr=orig_sr, axis=-1, res_type="kaiser_fast"
        )

    # ---- 3. align lengths ----------------------------------------------
    out_len = min(w.shape[-1] for w in waveforms.values())
    out_len = min(out_len, n_samples)

    for instr in instruments:
        if waveforms[instr].shape[-1] > out_len:
            waveforms[instr] = waveforms[instr][..., :out_len]

    mix_low = mix_low[..., :out_len]
    mix_high = mix_high[..., :out_len]

    # ---- 4. per-bin STFT masks + apply to high band --------------------
    n_fft = 4096
    hop_length = 1024

    mask_stfts, freqs = _compute_per_bin_masks(
        waveforms, orig_sr, cutoff_hz, instruments, n_channels,
        n_fft=n_fft, hop_length=hop_length,
    )

    # ---- 4a. energy-guided gate: suppress mask in quiet regions -------
    mix_low_stft = librosa.stft(mix_low, n_fft=n_fft, hop_length=hop_length)
    if n_channels == 1:
        mix_low_stft = mix_low_stft[np.newaxis, ...]
    gate_bins = np.where((freqs >= 10000.0) & (freqs <= cutoff_hz))[0]
    if len(gate_bins) > 0:
        frame_energy = np.mean(
            np.abs(mix_low_stft[..., gate_bins, :]) ** 2, axis=(-3, -2)
        )
        frame_energy = np.maximum(frame_energy, 0.0)
        if frame_energy.max() > 1e-12:
            frame_energy = frame_energy / frame_energy.max()
        gate_threshold = 0.001
        gate = np.clip(frame_energy / gate_threshold, 0.0, 1.0)

        n_frames_common = min(
            *(m.shape[-1] for m in mask_stfts.values()),
            mix_low_stft.shape[-1],
        )
        for instr in instruments:
            mask_stfts[instr][..., :n_frames_common] *= gate[
                np.newaxis, np.newaxis, :n_frames_common
            ]

    # ---- 4b. apply masks to high band (no crossfade on mask itself) ---
    mix_high_stft = librosa.stft(mix_high, n_fft=n_fft, hop_length=hop_length)
    if n_channels == 1:
        mix_high_stft = mix_high_stft[np.newaxis, ...]
    n_frames_h = mix_high_stft.shape[-1]

    for instr in instruments:
        n_frames_common = min(mask_stfts[instr].shape[-1], n_frames_h)
        mask = mask_stfts[instr][..., :n_frames_common]

        high_contrib = librosa.istft(
            mix_high_stft[..., :n_frames_common] * mask,
            hop_length=hop_length, length=out_len,
        )
        if high_contrib.ndim == 1:
            high_contrib = high_contrib[np.newaxis, :]
        high_contrib = high_contrib[:n_channels]

        # ---- 4c. transition-band blend: seamless crossover ------------
        # 3-band split: model output kept untouched below fade_start,
        # blended in the transition [fade_start, fade_end], HF extension
        # added above fade_end.  Both signals are derived from the
        # original recording — no generative processing, no synthesis.
        waveforms[instr] = _blend_transition(
            model_out=waveforms[instr],
            hf_ext=high_contrib,
            sr=orig_sr,
            fade_start=19000.0,
            fade_end=22500.0,
        )

    return waveforms


# -----------------------------------------------------------------------
# Internal: per-bin soft masks
# -----------------------------------------------------------------------

def _compute_per_bin_masks(
    separated: dict[str, np.ndarray],
    sr: float,
    cutoff_hz: float,
    instruments: list[str],
    n_channels: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_freq = len(freqs)
    n_stems = len(instruments)
    eps = 1e-8

    # STFT each stem
    stfts: dict[str, np.ndarray] = {}
    n_frames = None
    for instr in instruments:
        s = librosa.stft(separated[instr], n_fft=n_fft, hop_length=hop_length)
        if n_channels == 1:
            s = s[np.newaxis, ...]
        stfts[instr] = s
        if n_frames is None:
            n_frames = s.shape[-1]

    # Power
    power = {instr: np.abs(stfts[instr]) ** 2 for instr in instruments}
    total_power = sum(power.values())

    # Per-bin masks with power transform:
    #   mask_i = power_i^α / Σ(power_j^α)
    # α > 1 amplifies dominant stems and suppresses minor ones,
    # so a piano with 2% energy near 20 kHz gets near-zero HF contribution.
    alpha = 3.0

    masks: dict[str, np.ndarray] = {}
    for instr in instruments:
        m = np.divide(power[instr], total_power + eps, dtype=np.float32)
        m[..., total_power < eps] = 1.0 / n_stems
        m = m ** alpha
        masks[instr] = m

    # Re-normalize after power transform
    total = sum(masks.values())
    for instr in instruments:
        masks[instr] = np.divide(masks[instr], total + eps, dtype=np.float32)

    # Highest bin with real stem energy: model Nyquist at orig_sr
    # (content was upsampled from 44.1 kHz → no info above ~22 kHz)
    model_nyquist_bin = int(22050 / (sr / n_fft))
    cutoff_bin = int(cutoff_hz / (sr / n_fft))
    reliable_bin = min(model_nyquist_bin, cutoff_bin)
    reliable_bin = max(1, min(reliable_bin, n_freq - 2))

    # ---- gating: which stems' peaks continue past reliable_bin? -------
    trend_bins = 8
    trend_start = max(1, reliable_bin - trend_bins)
    continues: dict[str, np.ndarray] = {}
    for instr in instruments:
        m_early = masks[instr][..., trend_start:trend_start + 4, :].mean(axis=-2)
        m_late = masks[instr][..., reliable_bin - 4:reliable_bin + 1, :].mean(axis=-2)
        cont = (m_late >= m_early * 0.5) & (masks[instr][..., reliable_bin, :] > 0.05)
        continues[instr] = cont

    # ---- extrapolate above reliable_bin via harmonic sub-mapping -------
    if reliable_bin + 1 < n_freq:
        for b in range(reliable_bin + 1, n_freq):
            f_high = freqs[b]
            source_bin = reliable_bin
            for div in [2, 3, 4, 5]:
                f_sub = f_high / div
                if f_sub < freqs[1]:
                    break
                cand = int(f_sub / (sr / n_fft))
                if 1 <= cand <= reliable_bin:
                    source_bin = cand
                    break
            for instr in instruments:
                masks[instr][..., b, :] = (
                    masks[instr][..., source_bin, :] * continues[instr]
                )

        # Re-normalize per bin
        for b in range(reliable_bin + 1, n_freq):
            total_b = sum(masks[i][..., b, :] for i in instruments)
            for i in instruments:
                masks[i][..., b, :] = np.divide(masks[i][..., b, :], total_b + eps, dtype=np.float32)

    return masks, freqs


def _blend_transition(
    model_out: np.ndarray,
    hf_ext: np.ndarray,
    sr: float,
    fade_start: float,
    fade_end: float,
    order: int = 4,
) -> np.ndarray:
    """
    Blend the transition band [*fade_start*, *fade_end*] between the
    model output and the HF extension.  Below *fade_start* the model
    is kept untouched; above *fade_end* only the HF extension
    contributes.  The transition band is a linear crossfade — both
    signals derive from the original recording, so no content is
    synthesised.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    sos_bp = butter(order, [fade_start / nyq, fade_end / nyq],
                    btype='band', output='sos')
    sos_hp = butter(order, fade_end / nyq, btype='high', output='sos')

    # Extract transition band from both signals (zero-phase)
    model_trans = sosfiltfilt(sos_bp, model_out, axis=-1)
    hf_trans = sosfiltfilt(sos_bp, hf_ext, axis=-1)

    # Energy-guided per-sample blend weight
    eps = 1e-10
    model_pow = model_trans ** 2
    hf_pow = hf_trans ** 2
    weight = model_pow / (model_pow + hf_pow + eps)

    # Light temporal smoothing to avoid fluttering at the crossover
    kernel = np.ones(256, dtype=np.float32) / 256.0
    if weight.ndim == 1:
        weight = np.convolve(weight, kernel, mode='same')
    else:
        for ch in range(weight.shape[0]):
            weight[ch] = np.convolve(weight[ch], kernel, mode='same')

    blended_trans = model_trans * weight + hf_trans * (1.0 - weight)

    # Highpass the HF extension at fade_end — ultrasonic-only part
    hf_high = sosfiltfilt(sos_hp, hf_ext, axis=-1)

    # Recombine: remove model's transition band, add blended transition
    # and the ultrasonic HF extension.
    return (model_out - model_trans + blended_trans + hf_high).astype(np.float32)

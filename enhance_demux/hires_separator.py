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

    mix_low, mix_high = split_bands(mix, cutoff_hz, orig_sr, order=filter_order)

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

    out_len = min(w.shape[-1] for w in waveforms.values())
    out_len = min(out_len, n_samples)

    for instr in instruments:
        if waveforms[instr].shape[-1] > out_len:
            waveforms[instr] = waveforms[instr][..., :out_len]

    mix_low = mix_low[..., :out_len]
    mix_high = mix_high[..., :out_len]

    n_fft = 4096
    hop_length = 1024

    mask_stfts, freqs = _compute_per_bin_masks(
        waveforms, orig_sr, target_sr, cutoff_hz, instruments, n_channels,
        n_fft=n_fft, hop_length=hop_length,
    )

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

        waveforms[instr] = _blend_transition(
            model_out=waveforms[instr],
            hf_ext=high_contrib,
            sr=orig_sr,
            fade_start=19000.0,
            fade_end=22500.0,
        )

    return waveforms



# LLLLL 每个bin单独计算掩码 DDDDD

def _compute_per_bin_masks(
    separated: dict[str, np.ndarray],
    sr: float,
    model_sr: float,
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


    stfts: dict[str, np.ndarray] = {}
    n_frames = None
    for instr in instruments:
        s = librosa.stft(separated[instr], n_fft=n_fft, hop_length=hop_length)
        if n_channels == 1:
            s = s[np.newaxis, ...]
        stfts[instr] = s
        if n_frames is None:
            n_frames = s.shape[-1]

    power = {instr: np.abs(stfts[instr]) ** 2 for instr in instruments}
    total_power = sum(power.values())

    alpha = 3.0

    masks: dict[str, np.ndarray] = {}
    for instr in instruments:
        m = np.divide(power[instr], total_power + eps, dtype=np.float32)
        m[..., total_power < eps] = 1.0 / n_stems
        m = m ** alpha
        masks[instr] = m

    total = sum(masks.values())
    for instr in instruments:
        masks[instr] = np.divide(masks[instr], total + eps, dtype=np.float32)

    model_nyquist_bin = int((model_sr / 2) / (sr / n_fft))    # 奈奎斯特频率
    cutoff_bin = int(cutoff_hz / (sr / n_fft))
    reliable_bin = min(model_nyquist_bin, cutoff_bin)
    reliable_bin = max(1, min(reliable_bin, n_freq - 2))

    trend_bins = 8
    trend_start = max(1, reliable_bin - trend_bins)
    continues: dict[str, np.ndarray] = {}
    for instr in instruments:
        m_early = masks[instr][..., trend_start:trend_start + 4, :].mean(axis=-2)
        m_late = masks[instr][..., reliable_bin - 4:reliable_bin + 1, :].mean(axis=-2)
        cont = (m_late >= m_early * 0.5) & (masks[instr][..., reliable_bin, :] > 0.05)
        continues[instr] = cont

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

    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    sos_bp = butter(order, [fade_start / nyq, fade_end / nyq],
                    btype='band', output='sos')
    sos_hp = butter(order, fade_end / nyq, btype='high', output='sos')

    model_trans = sosfiltfilt(sos_bp, model_out, axis=-1)
    hf_trans = sosfiltfilt(sos_bp, hf_ext, axis=-1)

    eps = 1e-10
    model_pow = model_trans ** 2
    hf_pow = hf_trans ** 2
    weight = model_pow / (model_pow + hf_pow + eps)

    # 低通滤波
    kernel = np.ones(256, dtype=np.float32) / 256.0
    if weight.ndim == 1:
        weight = np.convolve(weight, kernel, mode='same')
    else:
        for ch in range(weight.shape[0]):
            weight[ch] = np.convolve(weight[ch], kernel, mode='same')

    blended_trans = model_trans * weight + hf_trans * (1.0 - weight)

    hf_high = sosfiltfilt(sos_hp, hf_ext, axis=-1)

    return (model_out - model_trans + blended_trans + hf_high).astype(np.float32)
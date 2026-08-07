# enhance_demux

Model-agnostic hi-res audio source separation. Enables 44.1 kHz-trained MSST models to separate 48/96/192 kHz audio **without losing high-frequency content**.

## How It Works

With standard MSST inference, all audio is resampled to the model's training rate (44.1 kHz). Content above 22.05 kHz is permanently discarded. This module provides two strategies to preserve the original sample rate:

| Input SR | Strategy | Description |
|----------|----------|-------------|
| 48 kHz | **Varispeed** | Raw samples fed to the model without resampling. The model's STFT treats them as 44.1 kHz audio, linearly compressing all frequencies into 0–22.05 kHz. Output is interpreted at the original rate, restoring correct speed and pitch. Zero quality loss. |
| 96/192 kHz | **Band-split** | Audio is split at ~20 kHz. The low band goes through the model normally. The high band is redistributed across stems via per-bin soft masks extrapolated from the model's separation decisions. |

### Band-split pipeline

```
Original audio
    │
    ├─ low-pass (0~20 kHz) → downsample → model → upsample ─┐
    │                                                        ├─ blend → full-rate output
    └─ high-pass (20 kHz~) ─── per-bin mask × HF STFT ──────┘
```

All high-frequency content comes from the **original recording**. No generative synthesis — no harmonic generation, no neural upsampling, no saturation.

### Key technical details

- **Per-bin soft masks** — computed from each stem's STFT magnitude in the model output, then extrapolated above the model's effective Nyquist via harmonic sub-mapping (f/2, f/3, f/4, f/5).
- **Power transform (α=3.0)** — `mask_i = power_i^α / Σ(power_j^α)`. Amplifies dominant stems and suppresses minor ones, so a piano with 2% energy near 20 kHz gets near-zero HF contribution.
- **Continuity gate** — a stem's mask is only extrapolated if its energy in the upper octave of the reliable band is sustained (not fading out) and above a significance threshold.
- **Energy gate** — suppresses mask contribution in quiet regions (frame energy < 0.1% of peak) to prevent noise artifacts.
- **Transition blend** — 19–22.5 kHz crossover with per-sample energy-guided weight and 256-sample temporal smoothing. Below 19 kHz the model output is kept untouched; above 22.5 kHz only the HF extension contributes.
- **Perfect reconstruction** — the crossover filter uses zero-phase Butterworth with `high = audio - low`, guaranteeing lossless recombination.

## Dependencies

`numpy`, `scipy`, `librosa` — all already required by MSST.

## Usage

### Integrated (MSST WebUI / CLI)

Already wired into `inference/msst_infer.py`. `MSSeparator` defaults to `use_hires=True` — when input SR exceeds the model's training rate, enhance_demux activates automatically:

```
Hi-res separation enabled | orig_sr: 96000 Hz → model_sr: 44100 Hz | strategy: band_split
Separation complete: your_audio.flac → output_sr: 96000 Hz
```

No extra configuration needed. 44.1 kHz input takes the standard inference path with zero overhead.

### Standalone

```python
from enhance_demux import separate_hires

# Load audio at native sample rate — do NOT resample
mix, orig_sr = librosa.load("audio_96k.wav", sr=None, mono=False)

# model, config, device, model_type are set up by your pipeline
waveforms = separate_hires(
    mix, model, config, device, model_type,
    orig_sr=orig_sr,          # e.g. 96000
    target_sr=44100,
    method="auto",            # "auto" | "varispeed" | "band_split"
    cutoff_ratio=0.9,         # band-split crossover point (fraction of target Nyquist)
    filter_order=8,           # Butterworth crossover filter order
)
# waveforms["vocals"] → (channels, time) at orig_sr
```

### Porting to other projects (UVR5, etc.)

1. Copy `enhance_demux/` into the project.
2. Load audio at its native sample rate (`sr=None` in librosa).
3. Add a hires branch after the model call:

```python
if sr > 44100:
    from enhance_demux import separate_hires
    results = separate_hires(mix, model, config, device, model_type, orig_sr=sr)
else:
    results = demix(config, model, mix, device, model_type=model_type)
```

4. Write output at `sr` instead of hardcoding `44100`.

The strategies are model-agnostic — they work with any architecture exposing a `(audio) → {stem: audio}` interface (MDX-Net, VR Arch, Demucs, RoFormer, etc.).

## Configuration

Tunable parameters in `hires_separator.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 3.0 | Mask power transform exponent. Higher → more aggressive stem separation in HF. |
| Continuity threshold | `m_late >= m_early * 0.5` | Lower → more stems extend into HF. Higher → more conservative. |
| Continuity floor | `mask > 0.05` | Lower → noise may leak into HF. Higher → only strong peaks extend. |
| `fade_start` | 19000 Hz | Start of transition band crossover. |
| `fade_end` | 22500 Hz | End of transition band; above this only HF extension contributes. |
| `cutoff_ratio` | 0.9 | Band-split crossover as fraction of target Nyquist (0.9 × 22050 ≈ 19845 Hz). |
| `filter_order` | 8 | Butterworth order for crossover (applied twice via filtfilt → effective order 16). |

## Known Limitations

- The 19–22.5 kHz transition band may have subtle crossover artifacts in pathological cases (e.g., isolated pure tones sitting exactly at the crossover).
- Varispeed is only recommended for `orig_sr / 44100 ≤ 1.15`. For larger ratios the speed/pitch shift becomes noticeable.
- Currently integrated with MSST inference only. Porting to other frameworks requires the steps listed above.

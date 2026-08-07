"""
enhance_demux: High sample rate audio source separation for MSST.

Supports processing audio at 48kHz, 96kHz, 192kHz (and beyond) through
models trained exclusively on 44.1kHz audio, without losing high-frequency
information from the original recording.

Two strategies are provided:

1. Varispeed — For moderate rate differences (e.g. 48kHz -> 44.1kHz).
   The audio is processed without resampling; the model "sees" it as
   44.1kHz audio, linearly compressing all frequencies into the model's
   bandwidth. After separation, the output is interpreted at the original
   rate, restoring speed/pitch. Zero quality loss, all frequencies
   benefit from the model's full separation capability.

2. Band-split + mask extrapolation — For large rate differences
   (e.g. 96kHz/192kHz -> 44.1kHz). The audio is split into low and high
   frequency bands. The low band is downsampled and processed normally.
   The high band (above ~20kHz) retains the original audio energy, with
   per-stem energy distribution guided by soft masks extrapolated from
   the model's separation decisions in the upper octave of the low band.

The auto mode selects varispeed when orig_sr / 44100 <= 1.15 and
band-split otherwise.
"""

from enhance_demux.pipeline import separate_hires
from enhance_demux.hires_separator import separate_varispeed, separate_band_split

__all__ = ["separate_hires", "separate_varispeed", "separate_band_split"]

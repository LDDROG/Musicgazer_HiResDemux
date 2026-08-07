<h1 align="center">Musicgazer</h1>
<div align="center">

English | [简体中文](docs/README_zh.md) | [繁體中文](docs/README_zh-TW.md) | [日本語](docs/README_ja.md)<br>

</div>

## Introduction

Musicgazer is an enhanced fork of [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) that adds hi-res audio source separation — output always at the original sample rate regardless of the model's training rate (except when the input rate is lower). MSST-WebUI is a WebUI for [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training), supporting model training and inference, customizable preset processing pipelines, model installation, advanced ensemble mode, and [SOME](https://github.com/openvpi/SOME/) integration.

> [!IMPORTANT]
> [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) is ceasing maintenance. Check the original repository for the latest status and migration options (e.g., [pymss](https://github.com/pymss-project/pymss)), or port to UVR5 or other hosts. This project only maintains the hi-res audio separation feature.

## Key Features

- **Lossless Varispeed**: For the common case of separating 48 kHz audio with a 44.1 kHz model, this strategy linearly compresses all frequencies into the model bandwidth, runs inference at a slightly lower pitch, then restores the outputs to the original speed and pitch. Zero quality loss.
- **Band-split**: Splits audio at ~20 kHz. The low band goes through the model normally; the high band is separated using masks derived from each stem's behavior around 20 kHz. The HF distribution is essentially inferred from the low-band separation. This differs from a model genuinely trained on hi-res audio, but all spectral content comes from the original recording — no generative restoration.
- **Model-agnostic** — works with any MSST architecture (BS-RoFormer, Mel-Band RoFormer, MDX23C, HTDemucs, SCNet, Bandit, Apollo, Mamba2, etc.). The strategies only depend on the `(audio) → {stem: audio}` interface.
- **Auto strategy selection** — picks varispeed when `orig_sr / model_sr ≤ 1.15`, band-split otherwise. Manual override available.

See [docs/enhance_demux.md](docs/enhance_demux.md) for the full technical breakdown.

## Quick Start

1. Install the WebUI following [Run from source](#run-from-source) below.
2. Download models via the "Install Models" page, or place `.ckpt` files in `pretrain/` manually.
3. Upload a 48/96/192 kHz audio file. The console will show:
   ```
   Hi-res separation enabled | orig_sr: 96000 Hz → model_sr: 44100 Hz | strategy: band_split
   Separation complete: your_audio.flac → output_sr: 96000 Hz
   ```
4. Verify that the output files retain the original sample rate.

<div align="center">
    <img src="test_result_image/1.png" alt="test result 1" width="45%">
    <img src="test_result_image/2.png" alt="test result 2" width="45%">
</div>

## Usage

- **Windows**: Download the installer from [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) or clone this repository and run from source.
- **Linux/macOS**: Clone this repository and run from source.
- **Google Colab**: [Click here](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) to run the WebUI on Google Colab.

### Download links

For installers, models, and mirror download options, see the upstream [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) repository.

### Documents

Chinese documentation is available [here](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc). Additional docs can be found in the `docs/` folder. See also [deton24](https://github.com/deton24)'s [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c).

## Run from source

- Clone this repository.

  ```bash
  git clone https://github.com/LDDROG/Musicgazer_HiResDemux.git
  cd Musicgazer_HiResDemux
  ```

- Create a Python environment and install dependencies. Python 3.10 is recommended.

  ```bash
  conda create -n msst python=3.10 -y
  conda activate msst
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt --only-binary=samplerate
  ```

- Fix a librosa compatibility issue: in `site-packages/librosa/util/utils.py` line 2185, change `np.dtype(complex): np.dtype(np.float).type,` to `np.dtype(complex): np.dtype(float).type,`. Alternatively:

  ```bash
  pip uninstall librosa -y
  pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
  ```

- Start the WebUI:

  ```bash
  python webUI.py
  ```

- For advanced launch arguments or cloud deployment, see [docs/webui.md](docs/webui.md).

- If you encounter `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration` when using `swin_upernet`, refer to [this issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24).

## CLI & Python API

See [docs/inference.md](docs/inference.md) for details.

### Integration example

The `enhance_demux` module is designed to drop into any MSST-compatible pipeline:

```python
import librosa
from enhance_demux import separate_hires

mix, orig_sr = librosa.load("audio_96k.wav", sr=None, mono=False)

# model, config, device, model_type are set up by your pipeline
waveforms = separate_hires(
    mix, model, config, device, model_type,
    orig_sr=orig_sr,    # e.g. 96000
    target_sr=44100,
)
# waveforms["vocals"] → (channels, time) at orig_sr
```

### Porting to UVR5

UVR5 follows the same `load → demix → save` pattern. To add native sample rate support:

1. Copy `enhance_demux/` into UVR5's `gui/` directory.
2. Load audio at its native rate:
   ```python
   mix, sr = librosa.load(path, sr=None, mono=False)
   ```
3. Add the hires branch after the model call:
   ```python
   if sr > 44100:
       from enhance_demux import separate_hires
       results = separate_hires(mix, model, config, device, model_type,
                                orig_sr=sr)
   else:
       results = demix(config, model, mix, device, model_type=model_type)
   ```
4. Write output at `sr` instead of hardcoding `44100`.

The strategies are model-agnostic — compatible with MDX-Net, VR Arch, Demucs, and any architecture exposing a `(audio) → {stem: audio}` interface.

## Training & Validation

See [docs/training.md](docs/training.md).

## Reference

### Upstream projects

- [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) — the WebUI this project is forked from
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — MSST training framework by ZFTurbo
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) — VR model inference

### Tools & libraries

- [librosa](https://github.com/librosa/librosa) — audio analysis (STFT, resampling, filters)
- [scipy](https://github.com/scipy/scipy) — signal processing (zero-phase Butterworth filters)
- [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) — UI components
- [PySide6](https://doc.qt.io/qtforpython-6) — Qt for Python
- [Singing-Oriented MIDI Extractor (SOME)](https://github.com/openvpi/SOME/) — MIDI extraction tool

### Community

- [deton24's Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS) — ComfyUI nodes for music source separation
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — Python audio separation library

### Thanks to all contributors

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>

### Thanks to all model providers

<table>
  <tr>
    <td style="text-align: center;"><img src="https://github.com/anvuew.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="anvuew"></td>
    <td style="text-align: center;"><img src="https://github.com/aufr33.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="aufr33"></td>
    <td style="text-align: center;"><img src="https://github.com/deton24.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="deton24"></td>
    <td style="text-align: center;"><img src="https://github.com/jarredou.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="jarredou"></td>
    <td style="text-align: center;"><img src="https://github.com/pcunwa.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="pcunwa"></td>
    <td style="text-align: center;"><img src="https://github.com/SUC-DriverOld.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="Sucial"></td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://github.com/anvuew">anvuew</a></td>
    <td style="text-align: center;"><a href="https://github.com/aufr33">aufr33</a></td>
    <td style="text-align: center;"><a href="https://github.com/deton24">deton24</a></td>
    <td style="text-align: center;"><a href="https://github.com/jarredou">jarredou</a></td>
    <td style="text-align: center;"><a href="https://github.com/pcunwa">pcunwa</a></td>
    <td style="text-align: center;"><a href="https://github.com/SUC-DriverOld">Sucial</a></td>
  </tr>
  <tr>
    <td style="text-align: center;"><img src="https://github.com/Super-YH.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="Super-YH"></td>
    <td style="text-align: center;"><img src="https://github.com/playdasegunda.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="viperx"></td>
    <td style="text-align: center;"><img src="https://github.com/wesleyr36.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="wesleyr36"></td>
    <td style="text-align: center;"><img src="https://github.com/yxlllc.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="yxlllc"></td>
    <td style="text-align: center;"><img src="https://github.com/ZFTurbo.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="ZFturbo"></td>
    <td></td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://github.com/Super-YH">Super-YH</a></td>
    <td style="text-align: center;"><a href="https://github.com/playdasegunda">viperx</a></td>
    <td style="text-align: center;"><a href="https://github.com/wesleyr36">wesleyr36</a></td>
    <td style="text-align: center;"><a href="https://github.com/yxlllc">yxlllc</a></td>
    <td style="text-align: center;"><a href="https://github.com/ZFTurbo">ZFturbo</a></td>
    <td></td>
  </tr>
</table>

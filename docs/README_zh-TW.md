<h1 align="center">Musicgazer</h1>
<p align="center"><sub>望樂者</sub></p>
<div align="center">

[English](../README.md) | [简体中文](README_zh.md) | 繁體中文 | [日本語](README_ja.md)<br>

</div>

## 簡介

Musicgazer 是 [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) 的改進版本，新增了高解析度音訊分離功能，即無論使用什麼採樣率的分離模型都能按原音訊採樣率輸出（低於模型採樣率的除外）。MSST-WebUI 是 [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training) 的 WebUI 應用，支援各種音訊模型的訓練和推理、自訂預設處理流程、模型安裝、進階合奏模式，以及 [SOME](https://github.com/openvpi/SOME/) 整合。

> [!IMPORTANT]
> [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) 即將停止維護，建議查閱原倉庫了解最新狀態及遷移方案（如 [pymss](https://github.com/pymss-project/pymss)），或移植到 UVR5 等其他宿主。本專案僅維護高解析度音訊分離功能。

## 重要特性

- **無損變頻策略**：以最常見的 44.1 kHz 模型分離 48 kHz 音訊為例，該策略會將頻率線性壓縮到模型頻帶，以略微低於原來音高的頻率輸入模型並輸出分軌，再將分軌恢復原速。
- **頻帶分離**：在約 20 kHz 處分割音訊，低頻段送入模型正常分離，高頻段使用 20 kHz 左右的分軌產生的遮罩直接分離，本質上是由低頻段的分離情況猜測高頻段情況，與真正使用 Hi-Res 訓練的模型還是有區別，但可以確保頻譜來自原始音訊，不存在生成式恢復手段。
- **模型無關**：適用於所有 MSST 架構（BS-RoFormer、Mel-Band RoFormer、MDX23C、HTDemucs、SCNet、Bandit、Apollo、Mamba2 等），策略僅依賴 `(audio) → {stem: audio}` 介面。
- **策略自動選擇** — `orig_sr / model_sr ≤ 1.15` 時走 varispeed，否則走 band-split，也可手動指定。

完整技術細節見 [docs/enhance_demux.md](enhance_demux.md)。

## 快速開始

1. 按照下方[從原始碼運行](#從原始碼運行)安裝 WebUI。
2. 透過「安裝模型」頁面下載模型，或將 `.ckpt` 檔案手動放入 `pretrain/` 目錄。
3. 上傳 48/96/192 kHz 音訊檔案，控制台將顯示：
   ```
   Hi-res separation enabled | orig_sr: 96000 Hz → model_sr: 44100 Hz | strategy: band_split
   Separation complete: your_audio.flac → output_sr: 96000 Hz
   ```
4. 觀察輸出檔案已保持原始採樣率。

<div align="center">
    <img src="../test_result_image/1.png" alt="test result 1" width="45%">
    <img src="../test_result_image/2.png" alt="test result 2" width="45%">
</div>

## 使用方式

- **Windows**：從 [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) 下載安裝程式，或複製倉庫從原始碼運行。
- **Linux/macOS**：複製倉庫從原始碼運行。
- **Google Colab**：[點擊此處](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) 在 Colab 上運行。

### 下載連結

安裝程式、模型及鏡像下載渠道，請參見上游 [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) 倉庫。

### 文件

詳細的飛書中文教學見 [此處](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)。更多文件見 `docs/` 資料夾。也可參考 [deton24](https://github.com/deton24) 的 [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)。

## 從原始碼運行

- 複製倉庫。

  ```bash
  git clone https://github.com/LDDROG/Musicgazer_HiResDemux.git
  cd Musicgazer_HiResDemux
  ```

- 建立 Python 環境並安裝依賴，推薦 Python 3.10。

  ```bash
  conda create -n msst python=3.10 -y
  conda activate msst
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt --only-binary=samplerate
  ```

- 修復 librosa 相容性問題：開啟 `site-packages/librosa/util/utils.py` 第 2185 行，將 `np.dtype(complex): np.dtype(np.float).type,` 改為 `np.dtype(complex): np.dtype(float).type,`。或使用以下命令：

  ```bash
  pip uninstall librosa -y
  pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
  ```

- 啟動 WebUI：

  ```bash
  python webUI.py
  ```

- 進階啟動參數或雲端部署見 [docs/webui.md](webui.md)。

- 使用 `swin_upernet` 時若遇到 `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration`，參考 [此 issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24)。

## CLI 與 Python API

詳見 [docs/inference.md](inference.md)。

### 整合範例

`enhance_demux` 模組設計為可嵌入任何相容 MSST 的推理管道：

```python
import librosa
from enhance_demux import separate_hires

mix, orig_sr = librosa.load("audio_96k.wav", sr=None, mono=False)

# model, config, device, model_type 由你的管線設定
waveforms = separate_hires(
    mix, model, config, device, model_type,
    orig_sr=orig_sr,    # 例如 96000
    target_sr=44100,
)
# waveforms["vocals"] → (channels, time)，採樣率為 orig_sr
```

### 移植到 UVR5

UVR5 遵循相同的 `載入 → 分離 → 儲存` 模式。加入原生採樣率支援：

1. 將 `enhance_demux/` 複製到 UVR5 的 `gui/` 目錄。
2. 以原生採樣率載入音訊：
   ```python
   mix, sr = librosa.load(path, sr=None, mono=False)
   ```
3. 在模型呼叫後新增高解析度分支：
   ```python
   if sr > 44100:
       from enhance_demux import separate_hires
       results = separate_hires(mix, model, config, device, model_type,
                                orig_sr=sr)
   else:
       results = demix(config, model, mix, device, model_type=model_type)
   ```
4. 以 `sr` 而非寫死的 `44100` 寫入輸出檔案。

策略與模型架構無關——相容 MDX-Net、VR Arch、Demucs 等任何遵循 `(audio) → {stem: audio}` 介面的架構。

## 訓練與驗證

詳見 [docs/training.md](training.md)。

## 參考

### 上游專案

- [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) — 本專案 fork 來源
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — ZFTurbo 的 MSST 訓練框架
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) — VR 模型推理

### 工具與庫

- [librosa](https://github.com/librosa/librosa) — 音訊分析（STFT、重採樣、濾波）
- [scipy](https://github.com/scipy/scipy) — 訊號處理（零相位 Butterworth 濾波器）
- [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) — UI 元件
- [PySide6](https://doc.qt.io/qtforpython-6) — Qt for Python
- [Singing-Oriented MIDI Extractor (SOME)](https://github.com/openvpi/SOME/) — MIDI 提取工具

### 社群

- [deton24 的音訊分離與母帶處理指南](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS) — ComfyUI 音樂源分離節點
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — Python 音訊分離庫

### 感謝所有貢獻者

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>

### 感謝所有模型提供者

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

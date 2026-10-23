<h1 align="center">Musicgazer</h1>
<p align="center"><sub>望楽者</sub></p>
<div align="center">

## from LDD_ROG REMASTER

[English](../README.md) | [简体中文](README_zh.md) | [繁體中文](README_zh-TW.md) | 日本語<br>

</div>

## 概要

Musicgazer は [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) の改良版で、高解像度オーディオ音源分離機能を追加しています。分離モデルの学習サンプリングレートに関係なく、元のサンプリングレートで出力できます（入力がモデルのサンプリングレートより低い場合を除く）。MSST-WebUI は [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training) の WebUI アプリケーションで、各種音声モデルの学習・推論、カスタムプリセット処理、モデルインストール、高度なアンサンブルモード、[SOME](https://github.com/openvpi/SOME/) 統合をサポートしています。

> [!IMPORTANT]
> [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) はメンテナンスを終了します。最新の状況や移行オプション（[pymss](https://github.com/pymss-project/pymss) など）については元のリポジトリを確認するか、UVR5 などの他のホストに移植してください。本プロジェクトは高解像度オーディオ分離機能のみをメンテナンスします。

## 主な機能

- **ロスレスバリスピード**：最も一般的な 44.1 kHz モデルで 48 kHz 音声を分離する場合、この戦略は全周波数をモデルの帯域幅に線形圧縮し、わずかに低いピッチで推論を実行し、出力を元の速度とピッチに復元します。品質損失はゼロです。
- **バンド分離**：約 20 kHz で音声を分割します。低域はモデルで通常通り分離し、高域は 20 kHz 付近の各ステムの振る舞いから得たマスクを使用して分離します。高域の分配は本質的に低域の分離結果から推測されます。これは真に Hi-Res で学習されたモデルとは異なりますが、すべてのスペクトル成分は元の録音に由来し、生成的な復元は行いません。
- **モデル非依存** — すべての MSST アーキテクチャ（BS-RoFormer、Mel-Band RoFormer、MDX23C、HTDemucs、SCNet、Bandit、Apollo、Mamba2 など）で動作します。戦略は `(audio) → {stem: audio}` インターフェースのみに依存します。
- **戦略の自動選択** — `orig_sr / model_sr ≤ 1.15` の場合はバリスピード、それ以外はバンド分離。手動指定も可能。

詳細な技術情報は [docs/enhance_demux.md](enhance_demux.md) を参照してください。

## クイックスタート

1. 以下の[ソースから実行](#ソースから実行)に従って WebUI をインストールします。
2. 「モデルをインストール」ページからモデルをダウンロードするか、`.ckpt` ファイルを手動で `pretrain/` に配置します。
3. 48/96/192 kHz の音声ファイルをアップロードします。コンソールに以下が表示されます：
   ```
   Hi-res separation enabled | orig_sr: 96000 Hz → model_sr: 44100 Hz | strategy: band_split
   Separation complete: your_audio.flac → output_sr: 96000 Hz
   ```
4. 出力ファイルが元のサンプリングレートを保持していることを確認します。

<div align="center">
    <img src="../test_result_image/1.png" alt="test result 1" width="45%">
    <img src="../test_result_image/2.png" alt="test result 2" width="45%">
</div>

## 使用方法

- **Windows**：[Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) からインストーラをダウンロードするか、リポジトリをクローンしてソースから実行します。
- **Linux/macOS**：リポジトリをクローンしてソースから実行します。
- **Google Colab**：[こちらをクリック](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) して Colab で WebUI を実行します。

### ダウンロードリンク

インストーラ、モデル、ミラーダウンロードオプションについては、アップストリームの [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) リポジトリを参照してください。

### ドキュメント

中国語のドキュメントは[こちら](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)から入手できます。その他のドキュメントは `docs/` フォルダを参照してください。[deton24](https://github.com/deton24) の [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c) も参考になります。

## ソースから実行

- リポジトリをクローンします。

  ```bash
  git clone https://github.com/LDDROG/Musicgazer_HiResDemux.git
  cd Musicgazer_HiResDemux
  ```

- Python 環境を作成し依存関係をインストールします。Python 3.10 を推奨します。

  ```bash
  conda create -n msst python=3.10 -y
  conda activate msst
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt --only-binary=samplerate
  ```

- librosa の互換性の問題を修正します：`site-packages/librosa/util/utils.py` の 2185 行目を開き、`np.dtype(complex): np.dtype(np.float).type,` を `np.dtype(complex): np.dtype(float).type,` に変更します。または以下のコマンドを使用します：

  ```bash
  pip uninstall librosa -y
  pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
  ```

- WebUI を起動します：

  ```bash
  python webUI.py
  ```

- 高度な起動引数やクラウドデプロイについては [docs/webui.md](webui.md) を参照してください。

- `swin_upernet` 使用時に `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration` が発生した場合は、[この issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) を参照してください。

## CLI と Python API

詳細は [docs/inference.md](inference.md) を参照してください。

### 統合例

`enhance_demux` モジュールは、MSST 互換のパイプラインに簡単に組み込めるように設計されています：

```python
import librosa
from enhance_demux import separate_hires

mix, orig_sr = librosa.load("audio_96k.wav", sr=None, mono=False)

# model, config, device, model_type はパイプラインで設定
waveforms = separate_hires(
    mix, model, config, device, model_type,
    orig_sr=orig_sr,    # 例: 96000
    target_sr=44100,
)
# waveforms["vocals"] → (channels, time)、サンプリングレートは orig_sr
```

### UVR5 への移植

UVR5 は同じ `読み込み → 分離 → 保存` パターンに従います。ネイティブサンプリングレート対応を追加するには：

1. `enhance_demux/` を UVR5 の `gui/` ディレクトリにコピーします。
2. ネイティブサンプリングレートで音声を読み込みます：
   ```python
   mix, sr = librosa.load(path, sr=None, mono=False)
   ```
3. モデル呼び出しの後に高解像度ブランチを追加します：
   ```python
   if sr > 44100:
       from enhance_demux import separate_hires
       results = separate_hires(mix, model, config, device, model_type,
                                orig_sr=sr)
   else:
       results = demix(config, model, mix, device, model_type=model_type)
   ```
4. `44100` とハードコーディングせずに `sr` で出力を書き込みます。

この戦略はモデルアーキテクチャに依存しません — MDX-Net、VR Arch、Demucs など、`(audio) → {stem: audio}` インターフェースを公開するあらゆるアーキテクチャと互換性があります。

## 学習と検証

詳細は [docs/training.md](training.md) を参照してください。

## 参考

### アップストリームプロジェクト

- [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) — 本プロジェクトのフォーク元
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — ZFTurbo による MSST 学習フレームワーク
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) — VR モデル推論

### ツールとライブラリ

- [librosa](https://github.com/librosa/librosa) — 音声分析（STFT、リサンプリング、フィルタ）
- [scipy](https://github.com/scipy/scipy) — 信号処理（ゼロ位相 Butterworth フィルタ）
- [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) — UI コンポーネント
- [PySide6](https://doc.qt.io/qtforpython-6) — Qt for Python
- [Singing-Oriented MIDI Extractor (SOME)](https://github.com/openvpi/SOME/) — MIDI 抽出ツール

### コミュニティ

- [deton24 の音源分離とマスタリングガイド](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS) — ComfyUI 音源分離ノード
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — Python 音声分離ライブラリ


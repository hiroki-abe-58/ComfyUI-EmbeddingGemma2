# EmbeddingGemma 2 for ComfyUI（日本語概要）

Google DeepMind のマルチモーダル埋め込みモデル **EmbeddingGemma 2**（[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)）を
ComfyUI で使うノードパックです。テキスト・画像・動画・音声を同じ 768 次元の空間に写すので、互いに比較できます。たとえば
「プロンプトに一番合う生成画像を選ぶ」「説明文で動画のフレームを並べ替える」「画像に最も近いキャプションを付ける」「どのクリップが
テキストに合うかを探す」といった処理をワークフローの中で行えます。処理はモデルカードが示す
[sentence-transformers](https://github.com/huggingface/sentence-transformers) 経由で **ComfyUI のプロセス内**で行い、ノードが何かを
ダウンロードすることはありません。

> **非公式**のインテグレーションです。Google / Google DeepMind、sentence-transformers / Transformers、Comfy Org とは関係がなく、
> 承認も受けていません。詳細は英語版 [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 内容 |
| --- | --- |
| **EmbeddingGemma 2 Loader** | `models/embeddinggemma` のモデル、読み込むエンコーダー（全部 / テキスト＋画像・動画 / テキスト＋音声 / テキストのみ）、dtype、デバイス、常駐 |
| **Encode Text** | 行ごと（または全体で 1 つ）の埋め込み。学習時のタスク接頭辞付き |
| **Encode Images** | `IMAGE` バッチの画像ごとの埋め込み |
| **Encode Video** | `VIDEO` 全体で 1 つの埋め込み（公式処理どおり毎秒 1 フレーム、最大 32 フレーム） |
| **Encode Audio** | `AUDIO` のクリップごとの埋め込み（モノラル 16 kHz、最初の 30 秒） |
| **Similarity** | すべての組み合わせのコサイン類似度。先頭クエリの最良ラベル・番号・スコア |
| **Rank Images** | 画像バッチをクエリとの類似度順に並べ替え（上位 k 枚） |
| **Unload** | モデルを解放 |

埋め込みは長さ 1 のベクトルです。各エンコードノードで 512 / 256 / 128 次元に短縮できます（Matryoshka、再正規化あり）。比較できるのは
同じ長さのベクトル同士だけです。

## 確認した範囲

Windows 11、RTX 5090、ComfyUI v0.38.0、torch 2.14.1、transformers 5.20.1、sentence-transformers 6.1.0：

- **公式の使い方とビット単位で一致**：モデルカードどおりの使い方（別環境、ファイルを直接読み込み）と、ComfyUI 標準の Load Image /
  動画ローダー / Load Audio から入力したノードの出力が 18 件中 18 件で完全に一致しました（テキスト 4 タスク、256 次元、タイトル付き
  文書、画像 6 枚、動画 2 本、音声 4 本）。
- **実際の ComfyUI キュー**：同梱ワークフロー 3 本のスコア（小数 6 桁）とフレーム順が、別途計算した公式の結果と一致しました。
- **動画のフレーム**：ComfyUI は通常の動画を浮動小数点で（独自の色差処理で）デコードし、ノードはそのフレームを使います。同じファイルの
  8 ビットデコードとの差は平均約 1.3/255、埋め込みのコサインは 0.9996 です。
- **CPU テストと CI**：公式のトークナイザーと設定ファイルから作った小さなランダムモデルで実際の処理を CPU 上で実行し、全ノードを
  sentence-transformers の直接呼び出しと比較。CPU での実キューも実行します。

## インストール

1. V3 ノード API のある ComfyUI（**v0.38.0** で確認）、Python 3.10 以上。GPU は任意です（740M パラメータ、bfloat16・全エンコーダーで
   GPU ピーク 2.5 GiB。CPU では float32 で動作し、遅くなります）。
2. このリポジトリを `ComfyUI/custom_nodes/` に clone し、依存パッケージを入れます：

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-EmbeddingGemma2 custom_nodes/ComfyUI-EmbeddingGemma2
   python -m pip install -r custom_nodes/ComfyUI-EmbeddingGemma2/requirements.txt
   ```

3. モデル（約 1.5 GB）をダウンロードします。確認済みのリビジョンに固定し、全ファイルの SHA-256 を検証します：

   ```
   python custom_nodes/ComfyUI-EmbeddingGemma2/tools/download_model.py --models-dir models/embeddinggemma
   ```

4. ComfyUI を再起動し、`workflows/gui/` のワークフローを開きます（`image_captions` は `input/example.png`、`rank_frames` と
   `video_search` は `input/example.mp4` を使います）。

## 注意点

- **音声は最初の 30 秒だけ**が埋め込みに影響します。公式処理（transformers 5.20.1）が 1 クリップを 750 音声トークン（毎秒 25）で
  打ち切るためです（60 秒のクリップは最初の 30 秒と完全に同じ埋め込みになることを確認）。長い音声は分割してください。
- **テキストは切り詰められません**（12,000 語のテキストは 12,003 トークン）。学習時のコンテキスト 8,192 トークン以内に収めてください。
- **動画**は ComfyUI がすべてのフレームを先にメモリへ読み込みます（832 x 480 で 1 フレーム約 4.8 MB）。長い動画は先に切り出すか
  縮小してください。
- **dtype**：`auto` は対応 GPU なら bfloat16、それ以外は float32。float16 はモデルがオーバーフローするため選べません（モデルカード）。

## ライセンス

- このノードパック：**MIT**（[LICENSE](LICENSE)）。第三者のコードは含みません。
- モデル（同梱しません。`tools/download_model.py` を実行したときだけ取得）：`google/embeddinggemma-2`、モデルカード上は
  **Apache-2.0**。モデルカードは [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) の順守も求めています。
  [NOTICE](NOTICE) を参照してください。

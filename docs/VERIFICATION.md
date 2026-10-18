# Verification

What was run to check this node pack, with the raw numbers. Everything below used the real released weights unless it says
"tiny model".

## 1. Setup

- Windows 11 Pro (build 26200), Intel Core Ultra 9 285K, 64 GB RAM, NVIDIA RTX 5090 32 GB (driver 610.88).
- ComfyUI v0.38.0 (commit `6b747c0428c343e1417219641db93a4fb7cb69ae`).
- Model: `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` (SHA-256 of every file in
  `tools/model_files.sha256`).
- Two Python 3.12 environments with the same libraries: torch 2.14.1+cu130, transformers 5.20.1, sentence-transformers 6.1.0,
  huggingface-hub 1.33.0, numpy 2.5.2, PyAV 19.0.1.
  - "official": a plain virtual environment with sentence-transformers (the model card's usage), no ComfyUI;
  - "comfy": ComfyUI's requirements plus this pack's requirements.

## 2. Nodes vs the official usage

**Official side** (official environment): `SentenceTransformer(<model folder>, device="cuda", model_kwargs={"torch_dtype":
torch.bfloat16})` as on the model card, `encode(..., prompt_name=<task>)`, `truncate_dim=256, normalize_embeddings=True` for the
short vectors, a titled document written as `title: {title} | text: {content}`. Inputs taken directly from the files: images with
`PIL.Image.open(...).convert("RGB")`; WAV files read as 16-bit samples / 32768, channels averaged, 44.1 kHz resampled to 16 kHz with
`scipy.signal.resample_poly` (the pack's documented resampling; the model accepts 16 kHz only); video as the frame list with
`{"fps", "total_num_frames", "duration"}` metadata.

**Pack side** (comfy environment, `scripts/run_parity.py`): the nodes, fed by ComfyUI's own Load Image, video loader (`VideoFromFile`)
and Load Audio, exactly as in a workflow; dtype `auto` (bfloat16 on this GPU).

| Case | Vectors | Identical |
| --- | --- | --- |
| text, task SearchQuery / Document / SentenceSimilarity / none (6 texts: English, Japanese, code) | 4 x [6, 768] | yes |
| text, SearchQuery, 256 dimensions | [6, 256] | yes |
| text, Document with a title | [1, 768] | yes |
| images: 832 x 480 (3), 768 x 768, 517 x 301, 640 x 420 crop | 6 x [1, 768] | yes |
| videos: 81 frames and 160 frames at 16 fps (the frames ComfyUI decoded) | 2 x [1, 768] | yes |
| audio: 4 s chirp at 16 kHz and 44.1 kHz, 3 s noise at 16 kHz, 2 s stereo tones at 44.1 kHz | 4 x [1, 768] | yes |

**18 of 18 identical** (every float of every vector).

**Video decoding.** ComfyUI v0.38.0 decodes ordinary (yuv420p) video in floating point (`gbrpf32le`) with its own chroma
handling; the nodes encode those frames, rounded to 8 bit. For comparison, an 8-bit `rgb24` PyAV decode of the same files:

| Video | Mean pixel difference to ComfyUI's frames | Cosine of the two embeddings |
| --- | --- | --- |
| fox (81 frames) | 1.31 / 255 | 0.99962 |
| river (160 frames) | 1.50 / 255 | 0.99961 |

The official usage for a video *file* decodes it with torchcodec (or with torchvision's video API, which torchvision 0.26 and later
no longer has); with in-memory frames, as here, the decoder is the caller's choice.

## 3. Real ComfyUI queue (comfy environment)

A ComfyUI server with this pack in `custom_nodes` and the model in an `extra_model_paths.yaml` folder; the shipped API workflows
submitted unchanged through `POST /prompt` (`scripts/queue_client.py`), `input/example.mp4` = an 81-frame clip generated for this
repository.

| Workflow | Result | Equal to the official usage recomputed separately |
| --- | --- | --- |
| `image_captions` (ComfyUI's `example.png` vs 5 captions) | best: "a drawing of a girl with big fox ears and blonde hair", 0.702 (others 0.495 - 0.531) | all 5 scores (6 decimals) |
| `rank_frames` (81 frames vs a text, top 4) | 4 frames saved, sorted | order and the 4 scores |
| `video_search` (the clip vs 5 descriptions) | best: "an animal in a snowy forest", 0.748 (others 0.517 - 0.550) | all 5 scores |
| Unload | `unloaded: true` | - |

Wall time per workflow including loading the model: 3.8 - 5.3 s.

## 4. Behaviour of the official processing worth knowing (checked)

- **Audio is capped at 30 seconds.** The processor gives 25 audio tokens per second up to 750 tokens: 4 s -> 100, 20 s -> 500,
  60 s / 300 s / 400 s -> 750. A 60 s clip (with a different second half) gives exactly the embedding of its first 30 s
  (identical vectors); its first 29 s give a different one (cosine 0.998).
- **Text is not truncated**: `max_seq_length` is unbounded and a 12,000-word text becomes 12,003 tokens, beyond the 8,192-token
  training context.
- **Video sampling** needs `fps` and `duration` in the metadata; without them the processor keeps every frame up to its 32-frame
  budget instead of sampling 1 per second. The nodes always pass both.

## 5. Tests without the weights (tiny model)

- `tests/` (CPU): ComfyUI value conversion (exact image pixels, video metadata and the official sampler's frame choice, mono 16 kHz
  audio), the runtime against direct sentence-transformers calls for every modality, task and length, the model card's options
  (encoder subsets, dtypes, prompt names), the nodes as ComfyUI loads them (ComfyUI's own `validate_prompt` on the shipped workflows;
  Load Image / video loader / Load Audio feeding the nodes, compared with the files decoded directly), the download manifest, and a
  scan of the package for environment / network / process code.
- The tiny model uses the official tokenizer, processor and sentence-transformers files with small random towers (9.2M parameters),
  so the real processing runs; CI downloads those files (no weights) with `tools/download_model.py --no-weights`.
- `scripts/ci_queue_check.py`: a stock ComfyUI server, the tiny model on the CPU, the three shipped workflows plus an audio workflow
  through the queue; repeated inputs give identical scores, the ranking saves its frames, Unload works.

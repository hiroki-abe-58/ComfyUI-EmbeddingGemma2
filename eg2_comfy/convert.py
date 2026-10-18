"""ComfyUI values -> the inputs sentence-transformers takes for EmbeddingGemma 2.

* IMAGE (float [B, H, W, C] in 0..1) -> RGB PIL images. Pixels are rounded back to uint8, so an image loaded from a file with
  ComfyUI's Load Image gives exactly the pixels of the file.
* VIDEO -> {"array": frames, "video_metadata": {fps, total_num_frames, duration}}: with fps and duration the processor samples
  frames at its configured rate (1 frame per second, at most 32 frames), as it does for a video file.
* AUDIO ({"waveform": [B, C, N], "sample_rate"}) -> {"array": mono float32, "sampling_rate": 16000} per batch item: channels are
  averaged (as sentence-transformers does for decoded audio); other rates are resampled to the model's 16 kHz with a polyphase
  filter (scipy.signal.resample_poly), since the model's feature extractor accepts 16 kHz only.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

AUDIO_RATE = 16000


def to_uint8(images: torch.Tensor) -> np.ndarray:
    return (images.detach().float().cpu() * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()


def images_to_pil(images: torch.Tensor) -> list[Image.Image]:
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected an IMAGE batch [B, H, W, C], got shape {tuple(images.shape)}")
    arr = to_uint8(images)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return [Image.fromarray(a[..., :3]) for a in arr]


def video_input(frames: torch.Tensor, fps: float) -> dict:
    """frames: IMAGE-like [T, H, W, 3] float in 0..1; fps: the video's frame rate."""
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("the video has no frames")
    if not fps or fps <= 0:
        raise ValueError("the video has no frame rate")
    arr = to_uint8(frames)[..., :3]
    count = int(arr.shape[0])
    return {"array": list(arr), "video_metadata": {"fps": float(fps), "total_num_frames": count, "duration": count / float(fps)}}


def audio_inputs(audio: dict) -> list[dict]:
    waveform, rate = audio["waveform"], int(audio["sample_rate"])
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[-1] == 0:
        raise ValueError(f"expected an AUDIO waveform [B, C, N], got shape {tuple(waveform.shape)}")
    mono = waveform.detach().float().cpu().mean(dim=1).numpy()  # [B, N]
    if rate != AUDIO_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(rate, AUDIO_RATE)
        mono = resample_poly(mono, AUDIO_RATE // g, rate // g, axis=-1)
    return [{"array": np.ascontiguousarray(item, dtype=np.float32), "sampling_rate": AUDIO_RATE} for item in mono]

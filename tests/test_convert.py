"""ComfyUI values -> model inputs: exact pixels, the video metadata the official frame sampler needs, mono 16 kHz audio."""

import numpy as np
import pytest
import torch

from eg2_comfy import convert


def _as_comfy_image(arr: np.ndarray) -> torch.Tensor:
    """What ComfyUI's Load Image produces from uint8 pixels."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0)[None]


def test_images_round_trip_to_the_file_pixels():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    (pil,) = convert.images_to_pil(_as_comfy_image(arr))
    assert pil.mode == "RGB" and pil.size == (53, 37) and np.array_equal(np.asarray(pil), arr)


def test_alpha_and_gray_images():
    rgba = torch.rand(2, 8, 9, 4)
    gray = torch.rand(1, 8, 9, 1)
    assert [p.mode for p in convert.images_to_pil(rgba)] == ["RGB", "RGB"]
    (g,) = convert.images_to_pil(gray)
    assert np.array_equal(np.asarray(g)[..., 0], np.asarray(g)[..., 2])
    with pytest.raises(ValueError):
        convert.images_to_pil(torch.rand(8, 9, 3))


def test_video_input_carries_fps_and_duration():
    frames = _as_comfy_image(np.full((20, 16, 24, 3), 7, np.uint8))[0].expand(20, 16, 24, 3).contiguous()
    item = convert.video_input(frames, 8.0)
    assert len(item["array"]) == 20 and item["array"][0].dtype == np.uint8 and int(item["array"][3][0, 0, 0]) == 7
    assert item["video_metadata"] == {"fps": 8.0, "total_num_frames": 20, "duration": 2.5}
    with pytest.raises(ValueError):
        convert.video_input(frames, 0.0)
    with pytest.raises(ValueError):
        convert.video_input(frames[:0], 8.0)


def test_the_official_sampler_takes_one_frame_per_second(reference_files):
    """The metadata makes the processor sample 1 frame per second, at most 32 (here checked on the processor itself)."""
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    processor = AutoProcessor.from_pretrained(str(reference_files))
    for count, fps, expected in ((81, 16.0, [0, 16, 32, 48, 64]), (40, 10.0, [0, 10, 20, 30]), (3, 16.0, [0])):
        meta = convert.video_input(torch.zeros(count, 4, 4, 3), fps)["video_metadata"]
        indices = processor.video_processor.sample_frames(VideoMetadata(**meta), fps=1, max_frames=32, overflow_strategy="uniform")
        assert list(indices) == expected
    meta = convert.video_input(torch.zeros(1200, 4, 4, 3), 10.0)["video_metadata"]  # 120 s: 120 candidates spread to 32
    indices = processor.video_processor.sample_frames(VideoMetadata(**meta), fps=1, max_frames=32, overflow_strategy="uniform")
    assert len(indices) == 32 and indices[0] == 0 and indices[-1] == 1190


def test_audio_is_mono_16k():
    t = torch.arange(32000) / 16000
    left, right = torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 660 * t)
    (item,) = convert.audio_inputs({"waveform": torch.stack([left, right])[None], "sample_rate": 16000})
    assert item["sampling_rate"] == 16000 and item["array"].dtype == np.float32
    assert np.allclose(item["array"], ((left + right) / 2).numpy())
    resampled = convert.audio_inputs({"waveform": torch.zeros(2, 1, 44100), "sample_rate": 44100})
    assert len(resampled) == 2 and resampled[0]["array"].shape == (16000,)
    with pytest.raises(ValueError):
        convert.audio_inputs({"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000})

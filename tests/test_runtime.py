"""The runtime against direct sentence-transformers calls, with the tiny random model on the CPU (the real model is compared with the
official usage in docs/VERIFICATION.md)."""

import numpy as np
import pytest
import torch
from PIL import Image

from eg2_comfy import runtime


@pytest.fixture
def loaded(tiny_model_dir):
    yield runtime.acquire(runtime.make_handle(str(tiny_model_dir), "all", "auto", "cpu", True))
    runtime.release()


def _image(seed=0, size=(40, 30)):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def test_options_follow_the_model_card():
    assert runtime.MODALITIES["text"] == {"vision_config": None, "audio_config": None}
    assert runtime.MODALITIES["text + image/video"] == {"audio_config": None}
    assert runtime.MODALITIES["text + audio"] == {"vision_config": None} and runtime.MODALITIES["all"] == {}
    assert runtime.DIMENSIONS == (768, 512, 256, 128) and "float16" not in runtime.DTYPES
    assert runtime.resolve_dtype("auto", "cpu") == torch.float32 and runtime.resolve_dtype("bfloat16", "cpu") == torch.bfloat16
    with pytest.raises(ValueError):
        runtime.resolve_dtype("float16", "cuda")


def test_every_task_is_a_prompt_of_the_model(loaded, reference_files):
    import json

    prompts = json.loads((reference_files / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]
    assert set(runtime.TASKS) - {"none"} <= set(prompts)
    assert loaded.model.prompts["SearchQuery"] == "task: search result | query: "


@pytest.mark.parametrize("task", ["SearchQuery", "Document", "Clustering", "none"])
def test_text_equals_sentence_transformers(loaded, task):
    texts = ["a fox in the snow", "un renard dans la neige", "def f(x): return x"]
    kwargs = {} if task == "none" else {"prompt_name": task}
    expected = loaded.model.encode(texts, convert_to_tensor=True, **kwargs).float().cpu()
    got = runtime.encode(loaded, texts, task=task)
    assert got.dtype == torch.float32 and torch.equal(got, expected)
    assert torch.allclose(got.norm(dim=1), torch.ones(3), atol=1e-4)


@pytest.mark.parametrize("dims", [512, 256, 128])
def test_matryoshka_truncation_is_renormalised(loaded, dims):
    texts = ["a fox", "a boat"]
    expected = loaded.model.encode(texts, prompt_name="SearchQuery", convert_to_tensor=True, truncate_dim=dims, normalize_embeddings=True)
    got = runtime.encode(loaded, texts, task="SearchQuery", dimensions=dims)
    assert got.shape == (2, dims) and torch.equal(got, expected.float().cpu())
    full = runtime.encode(loaded, texts, task="SearchQuery")
    assert torch.allclose(got, torch.nn.functional.normalize(full[:, :dims], dim=1), atol=1e-5)


def test_documents_with_a_title():
    assert runtime.document_text("body", " My title ") == "title: My title | text: body"


def test_media_equal_sentence_transformers(loaded):
    images = [_image(0), _image(1, (61, 17))]
    assert torch.equal(runtime.encode(loaded, images), loaded.model.encode(images, convert_to_tensor=True).float().cpu())
    frames = [np.full((24, 32, 3), i * 10, np.uint8) for i in range(12)]
    video = {"array": frames, "video_metadata": {"fps": 4.0, "total_num_frames": 12, "duration": 3.0}}
    assert torch.equal(runtime.encode(loaded, [video]), loaded.model.encode([video], convert_to_tensor=True).float().cpu())
    audio = {"array": np.sin(np.arange(16000, dtype=np.float32) / 7), "sampling_rate": 16000}
    assert torch.equal(runtime.encode(loaded, [audio]), loaded.model.encode([audio], convert_to_tensor=True).float().cpu())


def test_cosine_equals_model_similarity(loaded):
    a = runtime.encode(loaded, ["a fox", "a boat"], task="SearchQuery")
    b = runtime.encode(loaded, [_image(2), _image(3)])
    assert torch.allclose(runtime.cosine(a, b), loaded.model.similarity(a, b).float(), atol=1e-6)


def test_bad_arguments(loaded):
    with pytest.raises(ValueError):
        runtime.encode(loaded, ["x"], task="search")
    with pytest.raises(ValueError):
        runtime.encode(loaded, ["x"], dimensions=300)
    with pytest.raises(ValueError):
        runtime.encode(loaded, [])
    with pytest.raises(ValueError):
        runtime.make_handle("m", "vision", "auto", "cpu", True)


def test_text_only_loading_skips_the_media_encoders(tiny_model_dir):
    text_only = runtime.acquire(runtime.make_handle(str(tiny_model_dir), "text", "auto", "cpu", True))
    assert not runtime.supports(text_only, "image") and not runtime.supports(text_only, "audio") and runtime.supports(text_only, "text")
    names = {n for n, _ in text_only.model.named_parameters()}
    assert not any("vision" in n or "audio" in n for n in names)
    full = runtime.acquire(runtime.make_handle(str(tiny_model_dir), "all", "auto", "cpu", True))
    assert full is not text_only and runtime._SLOT is full
    assert torch.equal(runtime.encode(full, ["a fox"], "SearchQuery"), runtime.encode(full, ["a fox"], "SearchQuery"))
    runtime.release()
    assert runtime._SLOT is None


def test_loading_under_inference_mode_gives_ordinary_tensors(tiny_model_dir):
    with torch.inference_mode():
        loaded = runtime.acquire(runtime.make_handle(str(tiny_model_dir), "all", "auto", "cpu", True))
    assert not any(p.is_inference() for p in loaded.model.parameters())
    runtime.release()

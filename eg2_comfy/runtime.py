"""EmbeddingGemma 2 through sentence-transformers (the usage the model card documents), one resident model.

Every embedding is computed by `SentenceTransformer.encode` of the loaded model folder: the processor, chat template, mean pooling
and normalisation are the official ones. This module only chooses the documented options (task prompt, encoder subset, dtype,
Matryoshka truncation) and returns float32 vectors on the CPU. Nothing here downloads anything.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

# model card, "Selective Encoder Loading": config_kwargs per set of active modalities
MODALITIES = {
    "all": {},
    "text + image/video": {"audio_config": None},
    "text + audio": {"vision_config": None},
    "text": {"vision_config": None, "audio_config": None},
}
# prompt names of the model's config_sentence_transformers.json, as listed on the model card; "none" = no prefix
TASKS = ("SearchQuery", "Document", "QuestionAnswering", "FactChecking", "CodeRetrieval", "Classification", "Clustering",
         "SentenceSimilarity", "none")
DIMENSIONS = (768, 512, 256, 128)
DTYPES = ("auto", "bfloat16", "float32")  # float16 overflows (model card); not offered

_LOCK = threading.Lock()
_SLOT = None


@dataclass
class Loaded:
    key: tuple
    model: object
    info: dict = field(default_factory=dict)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """auto: bfloat16 on a CUDA GPU that supports it, otherwise float32 (the model card's rule)."""
    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if name == "auto":
        cuda_bf16 = device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if cuda_bf16 else torch.float32
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def make_handle(folder: str, modalities: str, dtype: str, device: str, keep_loaded: bool) -> dict:
    if modalities not in MODALITIES:
        raise ValueError(f"modalities must be one of {list(MODALITIES)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device cuda was chosen but CUDA is not available")
    return {"folder": folder, "modalities": modalities, "dtype": dtype, "device": device, "keep_loaded": bool(keep_loaded)}


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["folder"], handle["modalities"], handle["dtype"], handle["device"])
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        _SLOT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ComfyUI runs nodes under torch.inference_mode(); build the weights as ordinary tensors
        with torch.inference_mode(False):
            _SLOT = _load(key)
        return _SLOT


def _load(key) -> Loaded:
    from sentence_transformers import SentenceTransformer

    folder, modalities, dtype_name, device = key
    dtype = resolve_dtype(dtype_name, device)
    started = time.perf_counter()
    model = SentenceTransformer(folder, device=device, local_files_only=True, model_kwargs={"torch_dtype": dtype},
                                config_kwargs=dict(MODALITIES[modalities]))
    info = {"model": Path(folder).name, "modalities": modalities, "dtype": str(dtype).replace("torch.", ""), "device": device,
            "load_seconds": round(time.perf_counter() - started, 2)}
    return Loaded(key=key, model=model, info=info)


def supports(loaded: Loaded, modality: str) -> bool:
    disabled = MODALITIES[loaded.key[1]]
    return not ((modality in ("image", "video") and "vision_config" in disabled) or (modality == "audio" and "audio_config" in disabled))


def encode(loaded: Loaded, inputs: list, task: str = "none", dimensions: int = 768) -> torch.Tensor:
    """[N, dimensions] float32 on the CPU, unit length. task: a prompt name of TASKS (text only; media takes no prefix).
    Below 768 the vector is truncated to its leading dimensions and re-normalised, as the model card asks."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if dimensions not in DIMENSIONS:
        raise ValueError(f"dimensions must be one of {DIMENSIONS}")
    if not inputs:
        raise ValueError("nothing to encode")
    kwargs = {"convert_to_tensor": True, "show_progress_bar": False}
    if task != "none":
        kwargs["prompt_name"] = task
    if dimensions != 768:
        kwargs.update(truncate_dim=dimensions, normalize_embeddings=True)
    vectors = loaded.model.encode(inputs, **kwargs)
    return vectors.float().cpu()


def document_text(text: str, title: str) -> str:
    """Model card: documents with a real title are written as "title: {title} | text: {content}"."""
    return f"title: {title.strip()} | text: {text}"


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[len(a), len(b)] cosine similarities (what SentenceTransformer.similarity computes for this model)."""
    a = torch.nn.functional.normalize(a.float(), p=2, dim=1)
    b = torch.nn.functional.normalize(b.float(), p=2, dim=1)
    return a @ b.T

"""EmbeddingGemma 2 for ComfyUI (unofficial integration of https://huggingface.co/google/embeddinggemma-2)."""

import logging

try:
    from .eg2_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[EmbeddingGemma 2] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

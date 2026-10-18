"""A tiny random EmbeddingGemma 2 in the official folder layout, so sentence-transformers runs the real processor, chat template,
model classes, pooling and normalisation on the CPU in about a second.

The tokenizer, processor and sentence-transformers files are the official ones (EG2_REFERENCE_FILES points to a folder with them;
CI downloads them without the weights). config.json is the official one with small text, vision and audio towers; the weights are
random (seeded) and written with `save_pretrained`.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

SMALL_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
               "preprocessor_config.json", "modules.json", "sentence_bert_config.json", "config_sentence_transformers.json",
               "1_Pooling/config.json", "2_Normalize/config.json", "config.json")


def tiny_config(real: dict) -> dict:
    c = copy.deepcopy(real)
    c["text_config"].update(hidden_size=32, hidden_size_per_layer_input=4, intermediate_size=64, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                            per_layer_config={"01": {"head_dim": 32, "num_key_value_heads": 1}}, sliding_window=16)
    c["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                              head_dim=16, global_head_dim=16)
    # the audio front end's first conv width is tied to the 128 mel bins; keep the official [128, 32] channels
    c["audio_config"].update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=64)
    return c


def make_tiny_model(reference: Path, out: Path, seed: int = 0) -> Path:
    """Writes the tiny model folder `out` (named like a model folder in models/embeddinggemma) and returns it."""
    import torch
    from transformers import AutoConfig, AutoModel

    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, out / name)
    (out / "config.json").write_text(json.dumps(tiny_config(json.loads((reference / "config.json").read_text(encoding="utf-8"))), indent=2),
                                     encoding="utf-8")
    torch.manual_seed(seed)
    model = AutoModel.from_config(AutoConfig.from_pretrained(str(out)), dtype=torch.float32)
    model.save_pretrained(str(out))
    return out

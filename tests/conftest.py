"""Test setup. References come from environment variables, never from this repository:

EG2_COMFYUI_DIR          ComfyUI checkout (v0.38.0 or later)
EG2_REFERENCE_FILES      folder with the official non-weight files of google/embeddinggemma-2 (tokenizer, processor and
                         sentence-transformers configs, config.json); a full model folder works too
EG2_REQUIRE_REFERENCES=1 fail instead of skip when one of them is missing (CI sets this)
No trained weights are used: the model tests build a tiny random EmbeddingGemma 2 (tests/tiny_model.py) and run it on the CPU through
sentence-transformers, the same code path the nodes use with the real model.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("EG2_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("EG2_COMFYUI_DIR")
REFERENCE = _path("EG2_REFERENCE_FILES")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (EG2_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (EG2_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def reference_files():
    return Path(_need(REFERENCE, "EmbeddingGemma 2 tokenizer / processor files (EG2_REFERENCE_FILES)"))


@pytest.fixture(scope="session")
def tiny_model_dir(reference_files, tmp_path_factory):
    """<tmp>/embeddinggemma/tiny-embeddinggemma-2: a model folder as the Loader lists it."""
    from tiny_model import make_tiny_model

    return make_tiny_model(reference_files, tmp_path_factory.mktemp("models") / "embeddinggemma" / "tiny-embeddinggemma-2")

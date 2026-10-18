"""Download the official EmbeddingGemma 2 model into ComfyUI's models/embeddinggemma and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/embeddinggemma [--no-weights] [--verify-only]

Source: https://huggingface.co/google/embeddinggemma-2, pinned to the revision this node pack was verified with (about 1.5 GB;
Apache-2.0 per its model card, which also asks deployments to follow the Gemma Prohibited Use Policy). The files land in
models/embeddinggemma/embeddinggemma-2/ and are checked against the SHA-256 in tools/model_files.sha256. --no-weights fetches only
the tokenizer and configuration files (what the test suite needs). Needs `huggingface_hub` (installed with transformers). Nothing is
downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FOLDER = "embeddinggemma-2"
WEIGHTS = "model.safetensors"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path inside the model folder: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/embeddinggemma folder")
    parser.add_argument("--no-weights", action="store_true", help="only the tokenizer and configuration files")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    files = {k: v for k, v in expected_files().items() if not (args.no_weights and k == WEIGHTS)}
    target = args.models_dir / FOLDER
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(target), allow_patterns=sorted(files))
    ok = True
    for name, digest in files.items():
        path = target / name
        good = path.is_file() and sha256(path) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {FOLDER}/{name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

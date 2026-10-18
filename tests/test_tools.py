"""The download tool's manifest covers every file the nodes require, and its verification reports what it finds."""

import hashlib
import importlib.util
from pathlib import Path

from eg2_comfy import paths

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_model", ROOT / "tools" / "download_model.py")
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_manifest_covers_the_required_files():
    files = download_model.expected_files()
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in files.values())
    assert set(paths.REQUIRED) <= set(files) and download_model.WEIGHTS in files
    assert {"modules.json", "1_Pooling/config.json", "2_Normalize/config.json", "config_sentence_transformers.json"} <= set(files)


def test_the_manifest_matches_the_reference_files(reference_files):
    for name, digest in download_model.expected_files().items():
        path = reference_files / name
        if name != download_model.WEIGHTS or path.is_file():  # CI fetches the files without the weights
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_verify_only_reports_missing_and_mismatching_files(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "embeddinggemma" / download_model.FOLDER
    folder.mkdir(parents=True)
    (folder / "modules.json").write_bytes(b"[]")
    (folder / "config.json").write_bytes(b"{}")
    manifest = tmp_path / "model_files.sha256"
    manifest.write_text(f"# test\n{hashlib.sha256(b'[]').hexdigest()}  modules.json\n{'0' * 64}  config.json\n{'1' * 64}  model.safetensors\n",
                        encoding="ascii")
    monkeypatch.setattr(download_model, "MANIFEST", manifest)
    assert download_model.main(["--models-dir", str(tmp_path / "embeddinggemma"), "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "ok       embeddinggemma-2/modules.json" in out and "MISMATCH embeddinggemma-2/config.json" in out
    assert "MISSING  embeddinggemma-2/model.safetensors" in out
    assert download_model.main(["--models-dir", str(tmp_path / "embeddinggemma"), "--verify-only", "--no-weights"]) == 1
    assert "model.safetensors" not in capsys.readouterr().out

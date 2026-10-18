"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed and run the
shipped workflows through POST /prompt with a tiny random EmbeddingGemma 2 in the official folder layout.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --reference <folder with the official tokenizer / config files>
                                     [--workdir DIR] [--keep-pack]

The trained model is not needed: the tiny model (tests/tiny_model.py) uses the official tokenizer, processor and sentence-transformers
files with small random towers, so the real processor, chat template, pooling and the nodes run unchanged on the CPU. The workflows
are run with `device` set to cpu. This proves that the pack installs into a stock ComfyUI, that the shipped workflows (image
captions, frame ranking, video search) and an audio workflow execute through the queue, that results repeat exactly, and that Unload
works. The trained-model runs are in docs/VERIFICATION.md. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402
from tiny_model import make_tiny_model  # noqa: E402

NODE_IDS = {"EG2Loader", "EG2EncodeText", "EG2EncodeImages", "EG2EncodeVideo", "EG2EncodeAudio", "EG2Similarity", "EG2RankImages", "EG2Unload"}
AUDIO_WORKFLOW = {
    "1": {"class_type": "EG2Loader", "inputs": {"model": "embeddinggemma-2", "modalities": "text + audio", "dtype": "auto", "device": "cpu",
                                                "keep_loaded": False}},
    "2": {"class_type": "LoadAudio", "inputs": {"audio": "eg2_ci.wav"}},
    "3": {"class_type": "EG2EncodeAudio", "inputs": {"model": ["1", 0], "audio": ["2", 0], "dimensions": "256"}},
    "4": {"class_type": "EG2EncodeText", "inputs": {"model": ["1", 0], "text": "a rising tone\nrain on a window", "one_per_line": True,
                                                    "task": "SearchQuery", "title": "", "dimensions": "256"}},
    "5": {"class_type": "EG2Similarity", "inputs": {"queries": ["3", 0], "candidates": ["4", 0]}},
    "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def write_media(folder: Path) -> None:
    """example.mp4 (24 frames, 8 fps, with a moving bar) and eg2_ci.wav (1 s rising tone, 16 kHz)."""
    import wave

    import av
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    with av.open(str(folder / "example.mp4"), "w") as c:
        s = c.add_stream("libx264", rate=8)
        s.width, s.height, s.pix_fmt = 96, 64, "yuv420p"
        for i in range(24):
            frame = np.full((64, 96, 3), 30, np.uint8)
            frame[:, 4 * i:4 * i + 8] = (240, 200, 40)
            for p in s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    t = np.arange(16000) / 16000
    with wave.open(str(folder / "eg2_ci.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((0.3 * np.sin(2 * np.pi * (300 + 400 * t) * t) * 32767).astype("<i2").tobytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8196)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="eg2-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-EmbeddingGemma2"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    make_tiny_model(args.reference, work / "models" / "embeddinggemma" / "embeddinggemma-2")  # the name the shipped workflows use
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"eg2_ci:\n  base_path: {(work / 'models').as_posix()}\n  embeddinggemma: embeddinggemma\n", encoding="utf-8")
    input_dir = work / "input"
    write_media(input_dir)
    shutil.copyfile(comfy / "input" / "example.png", input_dir / "example.png")
    output = work / "output"

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--input-directory", str(input_dir), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(name: str, workflow: dict | None = None, **overrides) -> dict:
            wf = workflow or json.loads((ROOT / "workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
            wf = json.loads(json.dumps(wf))
            wf["1"]["inputs"]["device"] = "cpu"
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            result = queue_client.run(server, wf, None, 900)
            error = result.get("error") or {}
            print(json.dumps({"workflow": name, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:300] if isinstance(error, dict) else error}))
            if result["status"] != "success":
                failures.append(f"{name}: {error}")
            return result

        def text(result: dict, node: str):
            values = (result.get("text") or {}).get(node) or [None]
            return values[0]

        a = run("image_captions")
        table = json.loads(text(a, "7") or "{}")
        if len(table.get("scores", [[]])[0]) != 5 or text(a, "6") not in table.get("candidates", []):
            failures.append(f"image_captions: unexpected output {table}")
        b = run("image_captions", **{"4.text": "first caption\nsecond caption"})
        a2 = run("image_captions")  # node 4 changed in between, so this executes again
        if text(a2, "7") != text(a, "7"):
            failures.append("image_captions: the same inputs gave different scores")
        if len(json.loads(text(b, "7") or "{}").get("scores", [[]])[0]) != 2:
            failures.append("image_captions: two captions did not give two scores")
        r = run("rank_frames")
        ranking = json.loads(text(r, "8") or "{}")
        saved = sorted((output / "embeddinggemma").glob("best_frames*.png"))
        if len(ranking.get("order", [])) != 4 or len(saved) != 4 or ranking["scores"] != sorted(ranking["scores"], reverse=True):
            failures.append(f"rank_frames: ranking {ranking}, saved {saved}")
        v = run("video_search")
        if not text(v, "6"):
            failures.append("video_search: no best label")
        au = run("audio_search", AUDIO_WORKFLOW)
        if len(json.loads(text(au, "6") or "{}").get("scores", [[]])[0]) != 2:
            failures.append("audio_search: unexpected output")
        unload = queue_client.run(server, {"1": {"class_type": "EG2Unload", "inputs": {}}}, None, 120)
        print(json.dumps({"workflow": "unload", "status": unload["status"], "text": unload.get("text")}))
        if unload["status"] != "success":
            failures.append(f"unload: {unload.get('error')}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

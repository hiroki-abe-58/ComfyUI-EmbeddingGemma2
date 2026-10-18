"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and the nodes
fed by ComfyUI's Load Image / Load Video / Load Audio with the tiny random model on the CPU, compared with the inputs the official
usage takes (files decoded directly)."""

import asyncio
import importlib
import json
import shutil
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"EG2Loader", "EG2EncodeText", "EG2EncodeImages", "EG2EncodeVideo", "EG2EncodeAudio", "EG2Similarity", "EG2RankImages", "EG2Unload"}


def make_media(folder: Path) -> None:
    """eg2_test.png, eg2_test_16k.wav (mono), eg2_test_44k.wav (stereo), eg2_test.mp4 and example.mp4 (20 frames at 8 fps)."""
    import av

    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(folder / "eg2_test.png")
    for name, rate, channels in (("eg2_test_16k.wav", 16000, 1), ("eg2_test_44k.wav", 44100, 2)):
        t = np.arange(rate) / rate
        x = np.stack([0.3 * np.sin(2 * np.pi * (300 + 200 * c) * t) for c in range(channels)], 1)
        with wave.open(str(folder / name), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((x * 32767).astype("<i2").tobytes())
    for name in ("eg2_test.mp4", "example.mp4"):
        with av.open(str(folder / name), "w") as c:
            s = c.add_stream("libx264", rate=8)
            s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
            for i in range(20):
                frame = np.zeros((48, 64, 3), np.uint8)
                frame[:, : 3 * i + 4] = (12 * i, 255 - 12 * i, 90)
                for p in s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_model_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    make_media(base / "input")
    shutil.copyfile(Path(comfy_dir) / "input" / "example.png", base / "input" / "example.png")
    folder_paths.set_input_directory(str(base / "input"))
    folder_paths.set_output_directory(str(base / "output"))
    models = tiny_model_dir.parent
    if not (models / "embeddinggemma-2").exists():  # the name the shipped workflows use
        shutil.copytree(tiny_model_dir, models / "embeddinggemma-2")
    (models / "incomplete").mkdir(exist_ok=True)
    (models / "incomplete" / "config.json").write_text("{}", encoding="utf-8")
    folder_paths.add_model_folder_path("embeddinggemma", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    yield nodes
    pack_module(nodes, "runtime").release()


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `eg2_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["EG2EncodeText"].__module__.rsplit(".", 1)[0] + "." + name)


def run(comfy_nodes, name, **kwargs):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS[name]
    if hasattr(cls, "define_schema"):
        out = cls.execute(**kwargs)
        out = asyncio.run(out) if asyncio.iscoroutine(out) else out
        return out.args
    return getattr(cls(), cls.FUNCTION)(**kwargs)


def model(comfy_nodes, modalities="all", keep_loaded=True):
    return run(comfy_nodes, "EG2Loader", model="tiny-embeddinggemma-2", modalities=modalities, dtype="auto", device="cpu", keep_loaded=keep_loaded)[0]


def reference(comfy_nodes, inputs, task="none"):
    runtime = pack_module(comfy_nodes, "runtime")
    return runtime.encode(runtime.acquire(model(comfy_nodes)), inputs, task=task)


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_only_complete_model_folders_are_listed(comfy_nodes):
    paths = pack_module(comfy_nodes, "paths")
    assert paths.models() == ["embeddinggemma-2", "tiny-embeddinggemma-2"]
    for bad in ("incomplete", "../embeddinggemma-2", "missing", "..\\x"):
        with pytest.raises(FileNotFoundError):
            paths.model_path(bad)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_text_node(comfy_nodes):
    embeddings, report = run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text="a fox\n\n  a boat  \n", one_per_line=True,
                             task="SearchQuery", title="", dimensions="256")
    parsed = json.loads(report)
    assert embeddings["labels"] == ["a fox", "a boat"] and embeddings["vectors"].shape == (2, 256) and parsed["task"] == "SearchQuery"
    runtime = pack_module(comfy_nodes, "runtime")
    assert torch.equal(embeddings["vectors"], runtime.encode(runtime.acquire(model(comfy_nodes)), ["a fox", "a boat"], "SearchQuery", 256))
    whole, _ = run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text="a fox\na boat", one_per_line=False, task="none", title="",
                   dimensions="768")
    assert whole["vectors"].shape == (1, 768)
    titled, report = run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text="body text", one_per_line=True, task="Document",
                         title="Notes", dimensions="768")
    assert torch.equal(titled["vectors"], reference(comfy_nodes, ["title: Notes | text: body text"])) and json.loads(report)["task"] == "none"
    with pytest.raises(ValueError):
        run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text=" \n ", one_per_line=True, task="none", title="", dimensions="768")


def test_image_node_equals_the_file_pixels(comfy_nodes):
    image, _ = run(comfy_nodes, "LoadImage", image="eg2_test.png")
    embeddings, report = run(comfy_nodes, "EG2EncodeImages", model=model(comfy_nodes), images=image, dimensions="768")
    import folder_paths

    pil = Image.open(Path(folder_paths.get_input_directory()) / "eg2_test.png").convert("RGB")
    assert torch.equal(embeddings["vectors"], reference(comfy_nodes, [pil])) and json.loads(report)["count"] == 1


def _decode(path):
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return [f.to_ndarray(format="rgb24") for f in c.decode(s)], float(s.average_rate)


def test_video_node_passes_comfyui_frames_with_the_file_rate(comfy_nodes):
    """The node encodes the frames ComfyUI decoded (rounded to 8 bit) with the file's frame rate. ComfyUI decodes ordinary videos in
    float precision with its own chroma upsampling, so its frames differ from an 8-bit (rgb24) decode of the same file, mostly at
    sharp colour edges."""
    import folder_paths
    from comfy_api.latest import InputImpl

    path = Path(folder_paths.get_input_directory()) / "eg2_test.mp4"
    video = InputImpl.VideoFromFile(str(path))
    embeddings, _ = run(comfy_nodes, "EG2EncodeVideo", model=model(comfy_nodes), video=video, dimensions="768")
    comfy_frames = (video.get_components().images * 255).round().to(torch.uint8).numpy()
    frames, fps = _decode(path)
    expected = reference(comfy_nodes, [{"array": list(comfy_frames), "video_metadata": {"fps": fps, "total_num_frames": 20, "duration": 20 / fps}}])
    assert len(frames) == 20 and fps == 8.0 and torch.equal(embeddings["vectors"], expected)
    assert np.abs(comfy_frames.astype(int) - np.stack(frames).astype(int)).mean() < 3


def test_video_from_frames_and_rate(comfy_nodes):
    from comfy_api.latest import InputImpl, Types

    frames = torch.rand(12, 32, 48, 3)
    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=None, frame_rate=Fraction(6)))
    embeddings, _ = run(comfy_nodes, "EG2EncodeVideo", model=model(comfy_nodes), video=video, dimensions="512")
    assert embeddings["vectors"].shape == (1, 512)


def test_audio_node_equals_the_file_samples(comfy_nodes):
    import folder_paths

    path = Path(folder_paths.get_input_directory()) / "eg2_test_16k.wav"
    audio = run(comfy_nodes, "LoadAudio", audio="eg2_test_16k.wav")[0]
    embeddings, _ = run(comfy_nodes, "EG2EncodeAudio", model=model(comfy_nodes), audio=audio, dimensions="768")
    with wave.open(str(path)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    assert torch.equal(embeddings["vectors"], reference(comfy_nodes, [{"array": samples, "sampling_rate": 16000}]))
    stereo = run(comfy_nodes, "LoadAudio", audio="eg2_test_44k.wav")[0]
    embeddings, report = run(comfy_nodes, "EG2EncodeAudio", model=model(comfy_nodes), audio=stereo, dimensions="768")
    assert embeddings["vectors"].shape == (1, 768) and json.loads(report)["modality"] == "audio"


def test_similarity_and_ranking(comfy_nodes):
    m = model(comfy_nodes)
    images = torch.rand(5, 24, 24, 3, generator=torch.Generator().manual_seed(3))
    image_emb, _ = run(comfy_nodes, "EG2EncodeImages", model=m, images=images, dimensions="768")
    query, _ = run(comfy_nodes, "EG2EncodeText", model=m, text="a bright picture", one_per_line=True, task="SearchQuery", title="",
                   dimensions="768")
    scores, best_label, best_index, best_score = run(comfy_nodes, "EG2Similarity", queries=query, candidates=image_emb)
    table = json.loads(scores)
    assert len(table["scores"]) == 1 and len(table["scores"][0]) == 5 and best_label == f"image {best_index}"
    assert best_score == pytest.approx(max(table["scores"][0]), abs=1e-6)
    ranked, ranking = run(comfy_nodes, "EG2RankImages", images=images, image_embeddings=image_emb, query=query, top_k=3)
    order = json.loads(ranking)["order"]
    assert ranked.shape[0] == 3 and order[0] == best_index and torch.equal(ranked, images[order])
    assert json.loads(ranking)["scores"] == sorted(json.loads(ranking)["scores"], reverse=True)
    short, _ = run(comfy_nodes, "EG2EncodeText", model=m, text="x", one_per_line=True, task="none", title="", dimensions="128")
    with pytest.raises(ValueError, match="different lengths"):
        run(comfy_nodes, "EG2Similarity", queries=short, candidates=image_emb)
    with pytest.raises(ValueError, match="images but"):
        run(comfy_nodes, "EG2RankImages", images=images[:4], image_embeddings=image_emb, query=query, top_k=0)


def test_modalities_not_loaded_are_reported(comfy_nodes):
    with pytest.raises(ValueError, match="does not include image"):
        run(comfy_nodes, "EG2EncodeImages", model=model(comfy_nodes, "text"), images=torch.rand(1, 8, 8, 3), dimensions="768")


def test_keep_loaded_false_releases_and_unload_is_safe(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes, keep_loaded=False), text="x", one_per_line=True, task="none", title="",
        dimensions="768")
    assert runtime._SLOT is None
    assert json.loads(run(comfy_nodes, "EG2Unload")[0])["unloaded"] is False
    run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text="x", one_per_line=True, task="none", title="", dimensions="768")
    assert json.loads(run(comfy_nodes, "EG2Unload")[0])["unloaded"] is True and runtime._SLOT is None


def test_interrupt_before_encoding(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    def interrupt():
        raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        run(comfy_nodes, "EG2EncodeText", model=model(comfy_nodes), text="x", one_per_line=True, task="none", title="", dimensions="768")


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= NODE_IDS | {"LoadImage", "LoadVideo", "GetVideoComponents", "SaveImage", "PreviewAny", "LoadAudio"}, path.name

"""Encode media files through the nodes, in-process in a ComfyUI environment, and save every vector for a comparison with the official
sentence-transformers usage (maintainer tool; the official side is run separately on the same files).

    python scripts/run_parity.py --comfyui <ComfyUI checkout> --model <model folder> --media <folder> --out <file.npz> [--device cuda]

Images go through ComfyUI's Load Image, videos through ComfyUI's video loader, audio through Load Audio, exactly as in a workflow.
The npz holds one array per case plus `frames__<video>` (the 8-bit frames the video node encoded) so the official side can be fed
the same frames.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["a red fox walking through snow in a forest", "a sailboat on a calm bay at sunset", "aerial view of a river through autumn trees",
         "a child's drawing of a girl with big fox ears", "白い雪の中を歩く赤いキツネ", "def add(a, b): return a + b"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.comfyui))
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import folder_paths
    import nodes

    folder_paths.set_input_directory(str(args.media))
    folder_paths.add_model_folder_path("embeddinggemma", str(args.model.parent), is_default=True)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    m = nodes.NODE_CLASS_MAPPINGS

    def run(name, **kw):
        cls = m[name]
        if hasattr(cls, "define_schema"):
            return cls.execute(**kw).args
        return getattr(cls(), cls.FUNCTION)(**kw)

    from comfy_api.latest import InputImpl

    model = run("EG2Loader", model=args.model.name, modalities="all", dtype="auto", device=args.device, keep_loaded=True)[0]
    out, report = {}, {}
    for task in ("SearchQuery", "Document", "SentenceSimilarity", "none"):
        emb, rep = run("EG2EncodeText", model=model, text="\n".join(TEXTS), one_per_line=True, task=task, title="", dimensions="768")
        out[f"text__{task}"], report[f"text__{task}"] = emb["vectors"].numpy(), json.loads(rep)
    emb, _ = run("EG2EncodeText", model=model, text="\n".join(TEXTS), one_per_line=True, task="SearchQuery", title="", dimensions="256")
    out["text__SearchQuery__256"] = emb["vectors"].numpy()
    emb, _ = run("EG2EncodeText", model=model, text=TEXTS[0], one_per_line=True, task="Document", title="Wildlife", dimensions="768")
    out["text__Document_titled"] = emb["vectors"].numpy()
    for path in sorted(args.media.glob("*.png")):
        image = run("LoadImage", image=path.name)[0]
        emb, rep = run("EG2EncodeImages", model=model, images=image, dimensions="768")
        out[f"image__{path.stem}"], report[f"image__{path.stem}"] = emb["vectors"].numpy(), json.loads(rep)
    for path in sorted(args.media.glob("*.mp4")):
        video = InputImpl.VideoFromFile(str(path))
        emb, rep = run("EG2EncodeVideo", model=model, video=video, dimensions="768")
        out[f"video__{path.stem}"], report[f"video__{path.stem}"] = emb["vectors"].numpy(), json.loads(rep)
        components = video.get_components()
        out[f"frames__{path.stem}"] = (components.images * 255).round().clamp(0, 255).byte().numpy()
        out[f"fps__{path.stem}"] = np.array(float(components.frame_rate))
    for path in sorted(args.media.glob("*.wav")):
        audio = run("LoadAudio", audio=path.name)[0]
        emb, rep = run("EG2EncodeAudio", model=model, audio=audio, dimensions="768")
        out[f"audio__{path.stem}"], report[f"audio__{path.stem}"] = emb["vectors"].numpy(), json.loads(rep)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: list(v.shape) for k, v in out.items() if not k.startswith(("frames__", "fps__"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

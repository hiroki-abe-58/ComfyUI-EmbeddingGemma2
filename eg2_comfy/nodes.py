"""ComfyUI nodes for EmbeddingGemma 2 multimodal embeddings (run in the ComfyUI process; nothing is downloaded at run time).

Encode nodes turn text, images, a video or audio into unit-length vectors in one shared space; Similarity and Rank Images compare
them (cosine), e.g. to pick the generated image that matches a prompt best or to label an image with the closest caption."""

from __future__ import annotations

import json
import time

import torch
from comfy_api.latest import ComfyExtension, io

from . import convert, paths, runtime

EG2Model = io.Custom("EG2_MODEL")
EG2Embeddings = io.Custom("EG2_EMBEDDINGS")
DIMENSION_OPTIONS = [str(d) for d in runtime.DIMENSIONS]
DIMENSIONS_TOOLTIP = ("Matryoshka length: the leading dimensions, re-normalised. 768 = full; 512 / 256 are close to lossless; 128 loses "
                      "noticeably on images, video and audio (model card). Only vectors of the same length can be compared.")


def _interrupt_check():
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def _embed(handle: dict, modality: str, inputs: list, labels: list[str], task: str, dimensions: str) -> tuple[dict, str]:
    _interrupt_check()
    loaded = runtime.acquire(handle)
    info = loaded.info
    try:
        if not runtime.supports(loaded, modality):
            raise ValueError(f"the model was loaded with modalities {info['modalities']!r}, which does not include {modality}")
        started = time.perf_counter()
        vectors = runtime.encode(loaded, inputs, task=task, dimensions=int(dimensions))
        seconds = round(time.perf_counter() - started, 3)
    finally:
        del loaded
        if not handle.get("keep_loaded", True):
            runtime.release()
    embeddings = {"vectors": vectors, "labels": list(labels), "modality": modality}
    report = {"model": {k: info[k] for k in ("model", "modalities", "dtype", "device")}, "modality": modality, "count": int(vectors.shape[0]),
              "dimensions": int(vectors.shape[1]), "task": task if modality == "text" else None, "seconds": seconds}
    return embeddings, json.dumps(report, indent=1, ensure_ascii=False)


class EG2Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2Loader",
            display_name="EmbeddingGemma 2 Loader",
            category="loaders/embeddinggemma",
            description=("Loads EmbeddingGemma 2 (Google DeepMind's multimodal embedding model: text, images, video and audio in one "
                         "768-dimensional space) from models/embeddinggemma through sentence-transformers. One model stays resident."),
            inputs=[
                io.Combo.Input("model", options=paths.models(), tooltip="A model folder in models/embeddinggemma (e.g. embeddinggemma-2)"),
                io.Combo.Input("modalities", options=list(runtime.MODALITIES), default="all",
                               tooltip="Encoders to load. text = 270M parameters, + image/video 440M, + audio 570M, all 740M."),
                io.Combo.Input("dtype", options=list(runtime.DTYPES), default="auto",
                               tooltip="auto = bfloat16 on a GPU that supports it, otherwise float32. float16 is not offered: the model "
                                       "overflows in float16 (model card)."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after each encode node."),
            ],
            outputs=[EG2Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, modalities, dtype, device, keep_loaded) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.model_path(model), modalities, dtype, device, keep_loaded))


class EG2EncodeText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2EncodeText",
            display_name="EmbeddingGemma 2 Encode Text",
            category="utils/embeddinggemma",
            description=("Embeds text with the task prefix the model was trained with. Queries: SearchQuery (also for text-to-image / "
                         "video / audio search), QuestionAnswering, FactChecking, CodeRetrieval. Corpus texts: Document. Comparing like "
                         "with like: Classification, Clustering, SentenceSimilarity (use the same task on both sides)."),
            inputs=[
                EG2Model.Input("model"),
                io.String.Input("text", multiline=True, default="a red fox in the snow\na sailboat at sunset\nan aerial view of a river"),
                io.Boolean.Input("one_per_line", default=True, tooltip="Each non-empty line is one item; off = the whole text is one item."),
                io.Combo.Input("task", options=list(runtime.TASKS), default="SearchQuery"),
                io.String.Input("title", default="", tooltip="Document task only: the documents' title (empty = 'title: none')."),
                io.Combo.Input("dimensions", options=DIMENSION_OPTIONS, default="768", tooltip=DIMENSIONS_TOOLTIP),
            ],
            outputs=[EG2Embeddings.Output(display_name="embeddings"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, text, one_per_line, task, title, dimensions) -> io.NodeOutput:
        items = [line.strip() for line in text.splitlines() if line.strip()] if one_per_line else ([text] if text.strip() else [])
        if not items:
            raise ValueError("no text to encode")
        inputs, used_task = items, task
        if task == "Document" and title.strip():
            inputs, used_task = [runtime.document_text(t, title) for t in items], "none"
        embeddings, report = _embed(model, "text", inputs, items, used_task, dimensions)
        return io.NodeOutput(embeddings, report)


class EG2EncodeImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2EncodeImages",
            display_name="EmbeddingGemma 2 Encode Images",
            category="utils/embeddinggemma",
            description="One embedding per image of the batch (the official image processing: 280 tokens per image, no task prefix).",
            inputs=[EG2Model.Input("model"), io.Image.Input("images"),
                    io.Combo.Input("dimensions", options=DIMENSION_OPTIONS, default="768", tooltip=DIMENSIONS_TOOLTIP)],
            outputs=[EG2Embeddings.Output(display_name="embeddings"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, images, dimensions) -> io.NodeOutput:
        pil = convert.images_to_pil(images)
        embeddings, report = _embed(model, "image", pil, [f"image {i}" for i in range(len(pil))], "none", dimensions)
        return io.NodeOutput(embeddings, report)


class EG2EncodeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2EncodeVideo",
            display_name="EmbeddingGemma 2 Encode Video",
            category="utils/embeddinggemma",
            description=("One embedding for the whole video. As in the official processing, frames are sampled at 1 per second (at most "
                         "32 frames, evenly spread over longer videos); the soundtrack is not used (encode it with Encode Audio)."),
            inputs=[EG2Model.Input("model"), io.Video.Input("video"),
                    io.Combo.Input("dimensions", options=DIMENSION_OPTIONS, default="768", tooltip=DIMENSIONS_TOOLTIP)],
            outputs=[EG2Embeddings.Output(display_name="embeddings"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, video, dimensions) -> io.NodeOutput:
        components = video.get_components()
        item = convert.video_input(components.images, float(components.frame_rate))
        embeddings, report = _embed(model, "video", [item], ["video"], "none", dimensions)
        return io.NodeOutput(embeddings, report)


class EG2EncodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2EncodeAudio",
            display_name="EmbeddingGemma 2 Encode Audio",
            category="utils/embeddinggemma",
            description=("One embedding per audio clip of the batch. Channels are averaged to mono and other sample rates are resampled "
                         "to 16 kHz (the model's rate). Only the first 30 seconds of a clip count: the official processing truncates "
                         "audio at 480,000 samples (750 audio tokens); cut longer audio into parts to cover all of it."),
            inputs=[EG2Model.Input("model"), io.Audio.Input("audio"),
                    io.Combo.Input("dimensions", options=DIMENSION_OPTIONS, default="768", tooltip=DIMENSIONS_TOOLTIP)],
            outputs=[EG2Embeddings.Output(display_name="embeddings"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, audio, dimensions) -> io.NodeOutput:
        items = convert.audio_inputs(audio)
        embeddings, report = _embed(model, "audio", items, [f"audio {i}" for i in range(len(items))], "none", dimensions)
        return io.NodeOutput(embeddings, report)


def _check_pair(a: dict, b: dict) -> None:
    if a["vectors"].shape[1] != b["vectors"].shape[1]:
        raise ValueError(f"the embeddings have different lengths ({a['vectors'].shape[1]} and {b['vectors'].shape[1]}); "
                         "use the same dimensions on both encode nodes")


class EG2Similarity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2Similarity",
            display_name="EmbeddingGemma 2 Similarity",
            category="utils/embeddinggemma",
            description=("Cosine similarity of every query against every candidate. The single outputs refer to the first query: its "
                         "best candidate (e.g. the caption that fits an image best)."),
            inputs=[EG2Embeddings.Input("queries"), EG2Embeddings.Input("candidates")],
            outputs=[io.String.Output(display_name="scores"), io.String.Output(display_name="best_label"),
                     io.Int.Output(display_name="best_index"), io.Float.Output(display_name="best_score")],
        )

    @classmethod
    def execute(cls, queries, candidates) -> io.NodeOutput:
        _check_pair(queries, candidates)
        scores = runtime.cosine(queries["vectors"], candidates["vectors"])
        best = int(torch.argmax(scores[0]))
        table = {"queries": queries["labels"], "candidates": candidates["labels"], "scores": [[round(float(s), 6) for s in row] for row in scores],
                 "best": [candidates["labels"][int(i)] for i in scores.argmax(dim=1)]}
        return io.NodeOutput(json.dumps(table, indent=1, ensure_ascii=False), candidates["labels"][best], best, float(scores[0, best]))


class EG2RankImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2RankImages",
            display_name="EmbeddingGemma 2 Rank Images",
            category="image/embeddinggemma",
            description=("Sorts an image batch by similarity to a query (the first vector of `query`, e.g. a SearchQuery text), best "
                         "first, and keeps the top_k. image_embeddings must come from Encode Images on the same batch."),
            inputs=[io.Image.Input("images"), EG2Embeddings.Input("image_embeddings"), EG2Embeddings.Input("query"),
                    io.Int.Input("top_k", default=0, min=0, max=4096, tooltip="0 = keep all, sorted")],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="ranking")],
        )

    @classmethod
    def execute(cls, images, image_embeddings, query, top_k) -> io.NodeOutput:
        _check_pair(query, image_embeddings)
        if image_embeddings["vectors"].shape[0] != images.shape[0]:
            raise ValueError(f"{images.shape[0]} images but {image_embeddings['vectors'].shape[0]} image embeddings; encode this batch")
        scores = runtime.cosine(query["vectors"][:1], image_embeddings["vectors"])[0]
        order = torch.argsort(scores, descending=True, stable=True)
        if top_k:
            order = order[:top_k]
        ranking = {"query": query["labels"][0], "order": [int(i) for i in order], "scores": [round(float(scores[i]), 6) for i in order]}
        return io.NodeOutput(images[order.to(images.device)], json.dumps(ranking, indent=1, ensure_ascii=False))


class EG2Unload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EG2Unload",
            display_name="EmbeddingGemma 2 Unload",
            category="loaders/embeddinggemma",
            description="Frees the resident EmbeddingGemma 2 model (GPU and RAM).",
            inputs=[EG2Embeddings.Input("after", optional=True, tooltip="Optional: connect embeddings so this runs after them")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class EG2Extension(ComfyExtension):
    async def get_node_list(self):
        return [EG2Loader, EG2EncodeText, EG2EncodeImages, EG2EncodeVideo, EG2EncodeAudio, EG2Similarity, EG2RankImages, EG2Unload]


async def comfy_entrypoint() -> EG2Extension:
    paths.register()
    return EG2Extension()

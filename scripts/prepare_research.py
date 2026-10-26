"""Cache and validate the full research inputs before reserving a GPU."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "research_workflow"


def record(manifest, name, operation):
    try:
        manifest[name] = {"status": "ready", **operation()}
    except Exception as error:
        manifest[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        traceback.print_exc()
    destination = OUTPUT / "prerequisites.json"
    with (OUTPUT / "prerequisites.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = json.loads(destination.read_text()) if destination.exists() else {}
        merged[name] = manifest[name]
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(merged, indent=2) + "\n")
        temporary.replace(destination)
    print(name, manifest[name], flush=True)


def offline_check(code):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def cache_model(model_name):
    info = HfApi().model_info(model_name)
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "*.tiktoken"]
    if not any(item.rfilename.endswith(".safetensors") for item in info.siblings):
        patterns.append("*.bin")
    path = Path(snapshot_download(model_name, revision="main", allow_patterns=patterns))
    if path.name != info.sha:
        raise ValueError(f"Model revision changed during preparation: {model_name}")
    for index_path in path.rglob("*.safetensors.index.json"):
        for shard in set(json.loads(index_path.read_text())["weight_map"].values()):
            if not (index_path.parent / shard).is_file():
                raise FileNotFoundError(f"Missing model shard {shard}")
    if info.library_name == "diffusers":
        offline_check(f"import torch; from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained({str(path)!r}, local_files_only=True, torch_dtype=torch.float32)")
    else:
        offline_check(f"from transformers import AutoConfig,AutoTokenizer; AutoConfig.from_pretrained({model_name!r},local_files_only=True,trust_remote_code={not model_name.startswith('tiiuae/')!r}); AutoTokenizer.from_pretrained({model_name!r},local_files_only=True,trust_remote_code={not model_name.startswith('tiiuae/')!r})")
    return {"revision": info.sha, "path": str(path), "weight_files": sum(1 for _ in path.rglob("*.safetensors")), "offline_ready": True}


def cache_dataset(name, configuration=None):
    from datasets import load_dataset
    info = HfApi().dataset_info(name)
    patterns = ["README.md", "*.parquet"] if configuration is None else ["README.md", f"{configuration}/*.parquet"]
    path = snapshot_download(name, repo_type="dataset", revision="main", allow_patterns=patterns, max_workers=2)
    data = load_dataset(name, *([configuration] if configuration else []))
    counts = {split: len(values) for split, values in data.items()}
    offline_check(f"from datasets import load_dataset; ds=load_dataset({name!r}, *{[configuration] if configuration else []!r}); assert {{k:len(v) for k,v in ds.items()}} == {counts!r}")
    return {"revision": info.sha, "path": path, "splits": counts, "offline_ready": True}


def text_configs(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "QLORA_CONFIGS" for target in node.targets
                ):
                    return ast.literal_eval(node.value)
    raise ValueError("Notebook QLORA_CONFIGS was not found")


def cache_peer_reviews(notebook):
    import pandas as pd
    from datasets import load_dataset, load_dataset_builder

    name = "IntelLabs/AI-Peer-Review-Detection-Benchmark"
    info = HfApi().dataset_info(name)
    snapshot = Path(snapshot_download(name, repo_type="dataset", revision="main", allow_patterns=["README.md", "test/*.csv", "test/**/*.csv"], max_workers=2))
    builder = load_dataset_builder(name, "test", revision=info.sha)
    files = {}
    for split, urls in builder.config.data_files.items():
        files[split] = []
        for url in urls:
            relative = "test/" + str(url).split("/test/", 1)[1]
            local = snapshot / relative
            if not local.is_file():
                raise FileNotFoundError(local)
            files[split].append(str(local))

    def local_loader(path, configuration, *, split, streaming):
        if path != name or configuration != "test" or not streaming:
            raise ValueError("Unexpected peer-review loader protocol")
        return load_dataset("csv", data_files={split: files[split]}, split=split, streaming=True)

    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "def peer_review_text" in "".join(cell["source"]))
    context = {"load_dataset": local_loader, "pd": pd, "SEED": 42}
    exec(compile("".join(cell["source"]), "peer_review_sampling", "exec"), context)
    frame = context["df_external_ood"]
    destination = ROOT / "data" / "peer_review_evaluation.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    fingerprint = hashlib.sha256(destination.read_bytes()).hexdigest()
    sampling = {"seed": 42, "buffer_size": 2000, "sample_count": 2000,
                "generators": context["OOD_GENERATORS"], "content_tokens": list(context["OOD_CONTENT_TOKENS"]),
                "metadata_fields": sorted(context["OOD_METADATA_FIELDS"])}
    metadata = {"revision": info.sha, "path": str(destination), "sha256": fingerprint,
                "sampling_cell_sha256": hashlib.sha256("".join(cell["source"]).encode()).hexdigest(),
                "csv_files_in_order": files, "sampling": sampling,
                "labels": {str(label): int(count) for label, count in frame.label.value_counts().items()},
                "offline_ready": True}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_text(manifest):
    notebook = json.loads((ROOT / "ai_content_detector/notebooks/bench-aitextdetect.ipynb").read_text())
    for config in text_configs(notebook).values():
        name = config["hf_model"]
        def model(name=name):
            result = cache_model(name)
            offline_check(f"from transformers import AutoConfig,AutoTokenizer; AutoConfig.from_pretrained({name!r},local_files_only=True,trust_remote_code=True); AutoTokenizer.from_pretrained({name!r},local_files_only=True,trust_remote_code=True)")
            return {**result, "offline_ready": True}
        record(manifest, name, model)
    for name in ["yaful/MAGE", "artem9k/ai-text-detection-pile"]:
        record(manifest, name, lambda name=name: cache_dataset(name))
    record(manifest, "IntelLabs/AI-Peer-Review-Detection-Benchmark", lambda: cache_peer_reviews(notebook))


def prepare_images(manifest):
    from ml_pipeline.utils.kaggle_utils import ensure_kaggle_dataset
    for slug, directory in [
        ("awsaf49/artifact-dataset", "artifact-real-fake"),
        ("birdy654/cifake-real-and-ai-generated-synthetic-images", "cifake"),
        ("sunnykakar/shoes-dataset-real-and-ai-generated-images", "shoes-ai-vs-real"),
    ]:
        def dataset(slug=slug, directory=directory):
            target = ROOT / "data" / directory
            if slug == "awsaf49/artifact-dataset":
                marker = target / ".artifact-complete.json"
                if not marker.is_file():
                    raise ValueError("Full ArtiFact CRC/count validation has not completed")
                result = json.loads(marker.read_text())
                if not result.get("complete") or not result.get("crc_verified"):
                    raise ValueError("ArtiFact completion marker does not validate the full archive")
                return {**result, "path": str(target.resolve()), "offline_ready": True}
            previous = manifest.get(slug, {}).get("path")
            if not target.exists() and previous and Path(previous).exists():
                target.symlink_to(Path(previous), target_is_directory=True)
            path = ensure_kaggle_dataset(slug, target, slug)
            count = sum(1 for entry in path.rglob("*") if entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if count == 0:
                raise FileNotFoundError(f"No images for {slug}")
            if not target.exists():
                target.symlink_to(path.resolve(), target_is_directory=True)
            return {"path": str(path.resolve()), "image_count": count, "offline_ready": True}
        record(manifest, slug, dataset)
    from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY
    names = ["binary_efficientnet_b4_ns", "clip_classifier", "timm_convnextv2_tiny", "timm_dinov2_vit_base", "resnet50"]
    for name in names:
        def vision(name=name):
            parameters = {"num_classes": 2}
            if name.startswith("timm_"):
                parameters["img_size"] = 196 if "dinov2" in name else 200
            model = MODEL_REGISTRY[name](**parameters)
            del model
            gc.collect()
            offline_check(f"from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY; model=MODEL_REGISTRY[{name!r}](**{parameters!r})")
            return {"offline_ready": True, "parameters": parameters}
        record(manifest, f"vision/{name}", vision)


def prepare_rl(manifest):
    prepare_language_resources(manifest)
    models = ["Qwen/Qwen3.5-9B-Base", "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct",
              "openai-community/roberta-base-openai-detector", "intfloat/e5-base-v2",
              "rrivera1849/LUAR-MUD", "tuner007/pegasus_paraphrase",
              "stable-diffusion-v1-5/stable-diffusion-v1-5", "openai/clip-vit-base-patch32"]
    for name in models:
        record(manifest, name, lambda name=name: cache_model(name))
    record(manifest, "abisee/cnn_dailymail", lambda: cache_dataset("abisee/cnn_dailymail", "3.0.0"))
    record(manifest, "Gustavosta/Stable-Diffusion-Prompts", lambda: cache_dataset("Gustavosta/Stable-Diffusion-Prompts"))
    def hc3():
        name = "Hello-SimpleAI/HC3"
        info = HfApi().dataset_info(name)
        snapshot = Path(snapshot_download(name, repo_type="dataset", revision="main", allow_patterns=["all.jsonl", "README.md"]))
        source = snapshot / "all.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ROOT / "data/hc3-all.jsonl"
        if not destination.exists():
            destination.symlink_to(source)
        with source.open() as stream:
            count = sum(1 for _ in stream)
        return {"revision": info.sha, "path": str(source), "rows": count,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "offline_ready": True}
    record(manifest, "Hello-SimpleAI/HC3", hc3)


def prepare_language_resources(manifest):
    import urllib.request
    import nltk
    import spacy
    from importlib.metadata import version

    corpora = ROOT / ".venv/nltk_data/corpora"
    corpora.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(corpora.parent))
    hashes = {"cmudict": "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6",
              "wordnet": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"}
    for name, expected in hashes.items():
        def corpus(name=name, expected=expected):
            destination = corpora / f"{name}.zip"
            if not destination.exists():
                url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{name}.zip"
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Unexpected NLTK {name} archive hash")
                temporary = destination.with_suffix(".zip.tmp")
                temporary.write_bytes(content)
                temporary.replace(destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Cached NLTK {name} archive differs from the pinned version")
            if name == "cmudict":
                assert len(nltk.corpus.cmudict.entries()) == 133737
            else:
                assert len(nltk.corpus.wordnet.synsets("example")) == 6
            return {"path": str(destination), "sha256": expected, "offline_ready": True}
        record(manifest, f"nltk/{name}", corpus)

    def english():
        model = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        if not model("The researchers compare generated articles with human writing.").has_annotation("POS"):
            raise ValueError("The English model did not supply POS tags")
        return {"version": version("en-core-web-sm"), "pos_tagged": True, "offline_ready": True}
    record(manifest, "spacy/en_core_web_sm", english)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["text", "image", "rl", "all"], default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "prerequisites.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.group in ("text", "all"):
        prepare_text(manifest)
    if args.group in ("image", "all"):
        prepare_images(manifest)
    if args.group in ("rl", "all"):
        prepare_rl(manifest)
    selected = list(text_configs(json.loads((ROOT / "ai_content_detector/notebooks/bench-aitextdetect.ipynb").read_text())).values())
    if args.group in ("text", "all"):
        required = [item["hf_model"] for item in selected] + ["yaful/MAGE", "artem9k/ai-text-detection-pile", "IntelLabs/AI-Peer-Review-Detection-Benchmark"]
        if any(manifest.get(name, {}).get("status") != "ready" for name in required):
            raise SystemExit(1)
    if args.group in ("image", "all") and any(value.get("status") == "failed" for name, value in manifest.items() if name.startswith("vision/") or name == "awsaf49/artifact-dataset"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

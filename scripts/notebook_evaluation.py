"""Resume the full text notebook's inference without repeating completed batches."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest


def _digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _input_digest(values):
    digest = hashlib.sha256()
    if isinstance(values, np.ndarray) and values.dtype.kind not in "OU":
        digest.update(str((values.shape, values.dtype.str)).encode())
        for start in range(0, len(values), 32):
            digest.update(np.ascontiguousarray(values[start:start + 32]).tobytes())
    else:
        for value in values:
            encoded = str(value).encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def cached_probabilities(loader, values, directory, signature, stop, batch_size=32):
    signature = {**signature, "inputs": _input_digest(values), "samples": len(values),
                 "batch_size": batch_size}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    destination = Path(directory) / key
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination / "progress.json"
    array_path = destination / "probabilities.npy"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "signature": signature, "completed": 0,
    }
    if progress["signature"] != signature:
        raise ValueError("Notebook evaluation cache signature differs")
    completed = progress["completed"]
    if not 0 <= completed <= len(values):
        raise ValueError("Invalid notebook evaluation cache offset")
    probabilities = np.load(array_path, mmap_mode="r+") if array_path.exists() else None
    if probabilities is not None and probabilities.shape != (len(values), 2):
        raise ValueError("Notebook evaluation cache shape differs")
    if completed and probabilities is None:
        raise ValueError("Committed notebook evaluation probabilities are missing")
    if stop.requested:
        raise CheckpointPause("Notebook inference checkpoint retained")
    model = loader() if completed < len(values) else None
    try:
        for start in range(completed, len(values), batch_size):
            batch = np.asarray(model.predict_proba(values[start:start + batch_size]))
            if batch.shape != (min(batch_size, len(values) - start), 2):
                raise ValueError("Expected two-class notebook probabilities")
            if not np.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
                raise ValueError("Invalid notebook probabilities")
            if not np.allclose(batch.sum(axis=1), 1, atol=1e-5):
                raise ValueError("Notebook probability rows do not sum to one")
            if probabilities is None:
                probabilities = np.lib.format.open_memmap(
                    array_path, mode="w+", dtype=np.float64, shape=(len(values), 2),
                )
            probabilities[start:start + len(batch)] = batch
            probabilities.flush()
            with array_path.open("rb") as stream:
                os.fsync(stream.fileno())
            progress["completed"] = start + len(batch)
            temporary = progress_path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                json.dump(progress, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(progress_path)
            if progress["completed"] % 1280 == 0 or progress["completed"] == len(values):
                print(f"  Cached {signature.get('model_name')}: {progress['completed']}/{len(values)}", flush=True)
            if stop.requested:
                raise CheckpointPause("Notebook inference probabilities saved after a batch")
    finally:
        del model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.asarray(probabilities)


def install_text_evaluation_cache(namespace):
    from ml_pipeline.utils import utils
    import ml_pipeline.pipelines_torch.models as models

    original_loader = namespace["load_model_by_name"]
    original_evaluations = {name: namespace[name] for name in (
        "evaluate_sklearn_models", "evaluate_qlora_models",
    )}
    stop = None
    hashes = {}
    implementation = {_path: _digest_file(_path) for _path in (
        str(Path(__file__).resolve()), str(Path(utils.__file__).resolve()),
        str(Path(models.__file__).resolve()),
    )}

    class CachedModel:
        def __init__(self, model_class, model_name, params, path_start):
            self.loader = lambda: original_loader(model_class, model_name, params, path_start=path_start)
            entry = utils._load_index(path_start)[utils.get_checkpoint_id(path_start, model_name)]
            artifact = Path(utils.RESULTS_DIR_OUT) / path_start / entry["artifact_path"]
            if str(artifact) not in hashes:
                paths = sorted(artifact.rglob("*")) if artifact.is_dir() else [artifact]
                hashes[str(artifact)] = {str(path.relative_to(artifact) if artifact.is_dir() else path.name):
                                        _digest_file(path) for path in paths if path.is_file()}
            self.signature = {"model_name": model_name, "checkpoint": entry,
                              "artifact_hashes": hashes[str(artifact)], "implementation": implementation}

        def predict_proba(self, values):
            return cached_probabilities(self.loader, values,
                                        "results/research_workflow/text_inference", self.signature, stop)

    def load(model_class, model_name, params, path_start):
        if stop is None:
            return original_loader(model_class, model_name, params, path_start=path_start)
        return CachedModel(model_class, model_name, params, path_start)

    def wrap(function):
        @functools.wraps(function)
        def evaluate(*args, **kwargs):
            nonlocal stop
            with StopRequest() as request:
                stop = request
                try:
                    return function(*args, **kwargs)
                finally:
                    stop = None
        return evaluate

    namespace["load_model_by_name"] = load
    for name, function in original_evaluations.items():
        namespace[name] = wrap(function)

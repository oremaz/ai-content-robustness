"""Atomic restart state for the local research training loops."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import random
import signal
import shutil
import tempfile
import time

import numpy as np
import torch


class CheckpointPause(RuntimeError):
    """Training stopped at a safe boundary after saving restart state."""


class StopRequest:
    def __init__(self):
        self.requested = False
        self._previous = {}

    def __enter__(self):
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request)
        return self

    def _request(self, signum, frame):
        self.requested = True
        previous = self._previous.get(signum)
        if isinstance(getattr(previous, "__self__", None), StopRequest) and previous.__self__ is not self:
            previous(signum, frame)

    def __exit__(self, *args):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def rng_state(generator=None):
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["generator"] = generator.get_state()
    return state


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and "generator" in state:
        generator.set_state(state["generator"])


def model_state(model, names=None):
    names = set(names) if names is not None else {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def restore_model(model, state):
    missing = set(state) - set(model.state_dict())
    if missing:
        raise ValueError(f"Checkpoint parameters absent from model: {sorted(missing)}")
    model.load_state_dict(state, strict=False)


def gradient_state(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }


def restore_gradients(model, state):
    for name, parameter in model.named_parameters():
        parameter.grad = state[name].to(parameter.device) if name in state else None


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def map_tensors(value, operation):
    if torch.is_tensor(value):
        return operation(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, operation) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tensors(item, operation) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, operation) for item in value)
    return value


def atomic_transformers_save(trainer, save_method, model, trial=None, extra_state=None):
    output_dir = trainer.args.output_dir
    destination = Path(output_dir) / f"checkpoint-{trainer.state.global_step}"
    if (destination / "restart_complete.pt").is_file():
        return
    staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=output_dir))
    previous_best = getattr(trainer.state, "best_model_checkpoint", None)
    committed = False
    try:
        trainer.args.output_dir = str(staging)
        save_method(model, trial)
        completed = staging / destination.name
        best = getattr(trainer.state, "best_model_checkpoint", None)
        if best is not None and str(best).startswith(str(staging) + os.sep):
            trainer.state.best_model_checkpoint = str(Path(output_dir) / Path(best).relative_to(staging))
        trainer_state = completed / "trainer_state.json"
        if trainer_state.exists():
            values = json.loads(trainer_state.read_text())
            best = values.get("best_model_checkpoint")
            if best is not None and str(best).startswith(str(staging) + os.sep):
                values["best_model_checkpoint"] = str(Path(output_dir) / Path(best).relative_to(staging))
                trainer_state.write_text(json.dumps(values, indent=2) + "\n")
        atomic_save(extra_state or {}, completed / "restart_complete.pt")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(completed, destination)
        committed = True
    finally:
        trainer.args.output_dir = output_dir
        if not committed:
            trainer.state.best_model_checkpoint = previous_best
        shutil.rmtree(staging, ignore_errors=True)
    if hasattr(trainer, "_rotate_checkpoints"):
        trainer._rotate_checkpoints(use_mtime=False, output_dir=output_dir)
    else:
        from transformers.trainer_utils import rotate_checkpoints
        rotate_checkpoints(
            output_dir=output_dir, save_total_limit=trainer.args.save_total_limit,
            best_model_checkpoint=trainer.state.best_model_checkpoint, use_mtime=False,
        )


def load_checkpoint(path, config=None):
    path = Path(path)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = asdict(config) if is_dataclass(config) else config
    if config is not None and state.get("config") != expected:
        raise ValueError(f"Restart configuration differs from {path}")
    return state


class CheckpointClock:
    def __init__(self, seconds=300):
        self.seconds = seconds
        self.last_saved = time.monotonic()

    def due(self):
        return time.monotonic() - self.last_saved >= self.seconds

    def saved(self):
        self.last_saved = time.monotonic()


def evaluation_state(path, config, model):
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str((parameter.shape, parameter.dtype)).encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    signature = {"evaluation": config, "trainable_weights_sha256": digest.hexdigest()}
    state = load_checkpoint(path, signature)
    if state is None:
        state = {"config": signature, "generated": [], "scores": [], "rng": rng_state()}
        atomic_save(state, path)
    restore_rng(state["rng"])
    return state

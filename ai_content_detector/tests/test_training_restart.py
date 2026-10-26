import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import (
    CheckpointPause, atomic_save, atomic_transformers_save, load_checkpoint, restore_rng, rng_state,
)
from ai_content_detector.rl_evasion.config import ImageEvasionConfig, MultiSPINConfig, seed_everything


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    @property
    def device(self):
        return self.weight.device

    def save_pretrained(self, directory):
        from pathlib import Path
        Path(directory).mkdir(parents=True, exist_ok=True)


def make_spin(path, interrupt_after=None):
    from ai_content_detector.rl_evasion.text_evasion.multispin import SPINTrainerWithFeatureMonitoring
    seed_everything(42)
    trainer = SPINTrainerWithFeatureMonitoring.__new__(SPINTrainerWithFeatureMonitoring)
    trainer.config = MultiSPINConfig(
        output_dir=str(path), num_iterations=2, steps_per_iteration=6,
        batch_size=2, gradient_accumulation_steps=4,
    )
    trainer.model = ToyPolicy()
    trainer.ref_model = copy.deepcopy(trainer.model)
    trainer.ref_model.weight.requires_grad_(False)
    trainer._generator = torch.Generator().manual_seed(42)
    trainer.human_texts = ["human reference " * 30 for _ in range(8)]
    trainer.human_stylo = trainer.human_emb = np.zeros((8, 2))
    trainer.human_style_emb = None
    trainer.feature_bank = None
    trainer.tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    trainer._generate_batch = lambda prompts: [
        str(torch.rand((), generator=trainer._generator).item()) for _ in prompts
    ]
    trainer._compute_feature_matching_loss = lambda texts: {
        "stylo_loss": 0.0, "emb_mmd": 0.0, "style_mmd": 0.0,
    }
    calls = 0

    def loss(prompts, references, generated):
        nonlocal calls
        calls += 1
        if calls == interrupt_after:
            trainer._stop_request.requested = True
        target = sum(map(float, generated)) + np.random.random() + torch.rand(())
        return (trainer.model.weight - trainer.ref_model.weight - target).square()

    trainer._compute_spin_loss = loss
    return trainer


def test_spin_restart_preserves_partial_accumulation_and_reference(tmp_path):
    uninterrupted = make_spin(tmp_path / "continuous")
    uninterrupted.train()
    interrupted = make_spin(tmp_path / "restart", interrupt_after=3)
    with pytest.raises(CheckpointPause):
        interrupted.train()
    state = load_checkpoint(tmp_path / "restart/restart.pt")
    assert state["step"] == 3
    assert state["gradients"]
    resumed = make_spin(tmp_path / "restart")
    resumed.train()
    torch.testing.assert_close(resumed.model.weight, uninterrupted.model.weight, rtol=0, atol=0)
    torch.testing.assert_close(resumed.ref_model.weight, uninterrupted.ref_model.weight, rtol=0, atol=0)
    assert resumed.training_history == uninterrupted.training_history
    assert len(resumed.training_history) == 12


def make_ddpo(path, monkeypatch, interrupt_after=None):
    from ai_content_detector.rl_evasion.image_evasion import ddpo_trainer as module
    seed_everything(42)
    trainer = module.DDPOImageEvasionTrainer(ImageEvasionConfig(
        output_dir=str(path), num_train_epochs=2, num_inference_steps=6,
        sample_batch_size=2, train_batch_size=4, kl_coeff=0.0, reward_reference_weight=0.0,
    ))
    model = ToyPolicy()
    scheduler = SimpleNamespace(
        timesteps=torch.arange(6), set_timesteps=lambda count, device: None,
    )
    trainer.pipeline = SimpleNamespace(unet=model, scheduler=scheduler)
    trainer.prompts, trainer.eval_prompts = ["one", "two", "three"], ["four"]
    trainer._encode_prompts = lambda prompts: (torch.ones((2, 1)), torch.zeros((2, 1)))
    trainer.reward_fn = lambda image, prompt: {"total": float(image)}
    calls = 0

    def sample(**kwargs):
        values = torch.rand((7, 2, 1), generator=kwargs["generator"])
        return values[-1], list(values), [torch.zeros(2) for _ in range(6)]

    def step(**kwargs):
        nonlocal calls
        calls += 1
        if calls == interrupt_after:
            trainer._stop_request.requested = True
        probabilities = model.weight * kwargs["latents"].flatten()
        return kwargs["prev_sample"], probabilities

    monkeypatch.setattr(module, "pipeline_sample_with_logprob", sample)
    monkeypatch.setattr(module, "decode_latents_to_pil", lambda pipeline, latents: latents.flatten())
    monkeypatch.setattr(module, "ddim_step_with_logprob", step)
    return trainer


def test_ddpo_restart_reuses_trajectory_and_partial_gradients(tmp_path, monkeypatch):
    uninterrupted = make_ddpo(tmp_path / "continuous", monkeypatch)
    uninterrupted.train()
    interrupted = make_ddpo(tmp_path / "restart", monkeypatch, interrupt_after=3)
    with pytest.raises(CheckpointPause):
        interrupted.train()
    state = load_checkpoint(tmp_path / "restart/restart.pt")
    assert state["progress"]["position"] == 3
    assert state["progress"]["trajectory"] is not None
    assert state["gradients"]
    resumed = make_ddpo(tmp_path / "restart", monkeypatch)
    resumed.train()
    torch.testing.assert_close(resumed.pipeline.unet.weight, uninterrupted.pipeline.unet.weight, rtol=0, atol=0)
    assert resumed.stat_tracker._stats == uninterrupted.stat_tracker._stats
    assert resumed.training_history == uninterrupted.training_history
    assert len(resumed.training_history) == 2


def test_atomic_checkpoint_failure_preserves_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "restart.pt"
    atomic_save({"value": 1}, path)
    monkeypatch.setattr(torch, "save", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        atomic_save({"value": 2}, path)
    assert load_checkpoint(path)["value"] == 1
    assert list(tmp_path.iterdir()) == [path]


def test_rng_round_trip_preserves_all_sampling_sources():
    import random
    generator = torch.Generator().manual_seed(12)
    state = rng_state(generator)
    expected = (random.random(), np.random.random(), torch.rand(()), torch.rand((), generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.random(), torch.rand(()), torch.rand((), generator=generator))
    assert actual == expected


def test_native_hf_restart_preserves_best_model_paths_and_optimizer(tmp_path):
    import json
    from pathlib import Path
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.layer = torch.nn.Linear(2, 1)

        def forward(self, input_ids, labels=None):
            logits = self.layer(self.dropout(input_ids.float())).flatten()
            return {"loss": (logits - labels).square().mean(), "logits": logits}

    dataset = [
        {"input_ids": torch.tensor([float(i), 1.0]), "labels": torch.tensor(0.0)}
        for i in range(8)
    ]

    def make(directory, stop=False):
        seed_everything(42)
        trainer = Trainer(
            model=Regression(), train_dataset=dataset, eval_dataset=dataset,
            args=TrainingArguments(
                output_dir=str(directory), max_steps=3, per_device_train_batch_size=2,
                eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
                load_best_model_at_end=True, save_total_limit=2, learning_rate=0.01,
                report_to="none", use_cpu=True, disable_tqdm=True,
            ),
        )
        native_save = trainer._save_checkpoint
        trainer._save_checkpoint = lambda model, trial=None: atomic_transformers_save(
            trainer, native_save, model, trial,
        )

        class Interrupt(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if stop and state.global_step == 2:
                    control.should_training_stop = True
                return control

        trainer.add_callback(Interrupt())
        return trainer

    continuous = make(tmp_path / "continuous")
    continuous.train()
    interrupted = make(tmp_path / "restart", stop=True)
    interrupted.train()
    checkpoint = tmp_path / "restart/checkpoint-2"
    values = json.loads((checkpoint / "trainer_state.json").read_text())
    assert Path(values["best_model_checkpoint"]).is_dir()
    assert ".saving-" not in values["best_model_checkpoint"]
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "rng_state.pth").is_file()
    resumed = make(tmp_path / "restart")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, dict(continuous.model.named_parameters())[name], rtol=0, atol=0)
    assert len(list((tmp_path / "restart").glob("checkpoint-*"))) <= 2


def test_papermill_forwards_scheduler_signal_to_checkpointing_kernel(tmp_path):
    import os
    import signal
    import threading
    import time
    import nbformat
    from papermill.exceptions import PapermillExecutionError
    from scripts.research_workflow import execute_notebook

    armed, saved = tmp_path / "armed", tmp_path / "saved.pt"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import os, time\n"
        "from pathlib import Path\n"
        "from ai_content_detector.rl_evasion.checkpointing import StopRequest, CheckpointPause, atomic_save\n"
        "with StopRequest() as stop:\n"
        f"    Path({str(armed)!r}).write_text('ready')\n"
        "    while not stop.requested:\n"
        "        time.sleep(0.05)\n"
        f"    atomic_save({{'saved': True, 'pid': os.getpid()}}, {str(saved)!r})\n"
        "    raise CheckpointPause('scheduler checkpoint saved')\n"
    )])
    source, output = tmp_path / "input.ipynb", tmp_path / "output.ipynb"
    nbformat.write(notebook, source)

    def signal_after_kernel_armed():
        deadline = time.monotonic() + 120
        while not armed.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if armed.exists():
            os.kill(os.getpid(), signal.SIGUSR1)

    thread = threading.Thread(target=signal_after_kernel_armed, daemon=True)
    thread.start()
    with pytest.raises(PapermillExecutionError) as caught:
        execute_notebook(source, output)
    thread.join(timeout=1)
    assert caught.value.ename == "CheckpointPause"
    state = load_checkpoint(saved)
    assert state["saved"] is True
    assert state["pid"] != os.getpid()
    executed = nbformat.read(output, as_version=4)
    assert any(
        item.get("ename") == "CheckpointPause"
        for cell in executed.cells if cell.cell_type == "code"
        for item in cell.get("outputs", [])
    )

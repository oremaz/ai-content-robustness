from types import SimpleNamespace
from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.text_evasion.rewards import SemanticSimilarityReward
from ai_content_detector.rl_evasion.defender_comparison.workflow import BatchReward
from ai_content_detector.rl_evasion.defender_comparison.defenders import DefenderPool


class Tokens(dict):
    def to(self, device):
        return self


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        ids = [[ord(c) % 19 + 1 for c in text][:512] for text in texts]
        width = max(map(len, ids))
        return Tokens(input_ids=torch.tensor([x + [0] * (width - len(x)) for x in ids]),
                      attention_mask=torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids]))

    def batch_decode(self, ids, **kwargs):
        return [" ".join(["word"] * 24)] * len(ids)


class Encoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=torch.stack([
            input_ids.float(), input_ids.float().square(), torch.ones_like(input_ids).float()], dim=-1))


def semantic():
    value = SemanticSimilarityReward(device="cpu")
    value._model, value._tokenizer = Encoder(), Tokenizer()
    return value


def test_batched_semantics_preserve_scalar_results_padding_truncation_and_order():
    value = semantic()
    sources = ["a short source", "a substantially longer source", "x" * 700, "a short source"]
    generated = ["some completion", "tiny", "x" * 512, "some completion"]
    expected = [value(a, b) for a, b in zip(sources, generated)]
    value._model.calls = 0
    assert value.batch(sources, generated) == pytest.approx(expected, abs=1e-7)
    assert value._model.calls == 1
    assert value.batch(sources, generated) == pytest.approx(expected, abs=1e-7)
    assert value._model.calls == 1
    with pytest.raises(ValueError, match="aligned"):
        value.batch(sources, [])
    assert value.batch([], []) == []


def test_embedding_cache_is_bounded_without_losing_current_batch_entries():
    value = semantic()
    value._embedding_cache.update((str(i), np.ones(3)) for i in range(4096))
    value.batch(["new source"], ["new completion"])
    assert len(value._embedding_cache) == 4096
    assert "new source" in value._embedding_cache


def test_batched_reward_keeps_formula_and_short_output_rejection():
    calls = []
    pool = SimpleNamespace(specifications={"one": {}, "two": {}}, predict_batch=lambda texts:
                           {"one": [0.2] * len(texts), "two": [0.4] * len(texts)})
    reward = BatchReward(pool)
    reward.semantic = SimpleNamespace(batch=lambda sources, generated: calls.append((sources, generated)) or [0.9] * len(sources))
    source = " ".join(["source"] * 30)
    output = " ".join(["output"] * 20)
    rows = reward.batch([source, source], [output, "short"])
    quality = max(0, 1 - (1 - 20 / 30) ** 2)
    assert rows[0] == {"total": 0.6 * 0.7 + 0.3 * 0.9 + 0.1 * quality,
                       "evasion": 0.7, "semantic": 0.9, "quality": quality,
                       "rejected": False, "per_detector": {"one": 0.8, "two": 0.6}}
    assert rows[1]["total"] == 0 and rows[1]["rejected"]
    assert calls == [([source], [output])]


def test_resident_detector_keeps_scores_and_offloads_after_errors(tmp_path):
    moves = []
    class Detector:
        def move(self, device):
            moves.append(device)
        def predict_batch(self, texts):
            return [0.25] * len(texts)
    pool = DefenderPool({"one": {}}, tmp_path, device="cpu")
    detector = Detector()
    pool.defenders["one"] = detector
    expected = pool.predict_batch(["sample"])
    assert moves == ["cpu", "cpu"]
    moves.clear()
    pool.keep_on_device = True
    assert pool.predict_batch(["sample"]) == expected
    assert moves == ["cpu"]
    detector.predict_batch = lambda texts: [float("nan")]
    with pytest.raises(ValueError, match="Invalid scores"):
        pool.predict_batch(["sample"])
    assert moves[-2:] == ["cpu", "cpu"]


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("sample_count", [32, 100])
def test_logging_restores_rng_training_mode_and_tokenizer_and_excludes_fit_prompts(tmp_path, monkeypatch, fail, sample_count):
    from ai_content_detector.rl_evasion.defender_comparison import diagnostics, workflow
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    pairs = [{"sample_id": str(i), "prompt": "held out " + str(i), "reference": "human reference",
              "human": "human text", "eligible": True} for i in range(160)]
    (tmp_path / "base_pairs.json").write_text(json.dumps({"evaluation": pairs}))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0
        def generate(self, input_ids, **kwargs):
            self.calls += 1
            random.random()
            np.random.random()
            if fail:
                raise RuntimeError("fixture failure")
            return torch.cat([input_ids, torch.randint(0, 5, (len(input_ids), 4))], dim=1)
    attacker = SimpleNamespace(model=Model(), tokenizer=Tokenizer(), prompts=["fitting prompt"],
                               diagnostic_samples=sample_count,
                               config=SimpleNamespace(output_dir=str(tmp_path / "training"), max_seq_length=512,
                                                      max_new_tokens=256, temperature=0.8, top_p=0.95),
                               pool=SimpleNamespace(checkpoints={}),
                               reward_fn=SimpleNamespace(semantic=SimpleNamespace(batch=lambda a, b: [0.9] * len(a))))
    monkeypatch.setattr(diagnostics, "evaluate_pool", lambda pool, rows, directory:
                        [{"detector": "fixture", "attack_success_rate": 0.25}])
    from ai_content_detector.rl_evasion.checkpointing import rng_state
    before = rng_state()
    if fail:
        with pytest.raises(RuntimeError, match="fixture failure"):
            diagnostics.log_training_diagnostic(attacker, 50)
    else:
        value = diagnostics.log_training_diagnostic(attacker, 50)
        assert value["num_generated"] == sample_count and value["logging_only"]
        assert len(value["sample_ids"]) == sample_count
        assert diagnostics.log_training_diagnostic(attacker, 50) == value
        assert attacker.model.calls == (sample_count + 7) // 8
        confirmation = diagnostics.log_training_diagnostic(attacker, 50, subset_size=128)
        assert confirmation["num_generated"] == 128
        assert attacker.model.calls == (sample_count + 7) // 8 + (128 - sample_count // 8 * 8) // 8
        attacker.diagnostic_samples = 100 if sample_count == 32 else 32
        with pytest.raises(ValueError, match="different sample count"):
            diagnostics.log_training_diagnostic(attacker, 50)
        attacker.diagnostic_samples = sample_count
    after = rng_state()
    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert attacker.model.training and attacker.tokenizer.padding_side == "right"
    attacker.prompts = [pairs[0]["prompt"]]
    with pytest.raises(ValueError, match="excluded"):
        diagnostics.log_training_diagnostic(attacker, 100)


@pytest.mark.parametrize("resume_step", [None, 0, 50])
def test_training_callback_logs_every_50_updates_without_ending_training(tmp_path, monkeypatch, resume_step):
    import trl
    from dataclasses import asdict
    from ai_content_detector.rl_evasion import checkpointing
    from ai_content_detector.rl_evasion.config import TextEvasionConfig
    from ai_content_detector.rl_evasion.text_evasion.grpo_trainer import GRPOTextEvasionTrainer
    events = []
    class Trainer:
        def __init__(self, model, **kwargs):
            self.model = model
            self.state = SimpleNamespace(global_step=0, log_history=[])
            self.callbacks = []
        def _save_checkpoint(self, model, trial=None):
            events.append(("save", self.state.global_step))
            (tmp_path / f"checkpoint-{self.state.global_step}").mkdir(exist_ok=True)
        def _load_from_checkpoint(self, path, model):
            self.state.global_step = int(Path(path).name.split("-")[-1])
            events.append(("load", self.state.global_step))
        def add_callback(self, callback):
            self.callbacks.append(callback)
        def train(self, resume_from_checkpoint=None):
            assert bool(resume_from_checkpoint) == (resume_step is not None)
            control = SimpleNamespace()
            for c in self.callbacks:
                c.on_train_begin(None, self.state, control)
            for step in [49, 50, 51, 299, 300]:
                if resume_step is not None and step <= resume_step:
                    continue
                self.state.global_step = step
                self.state.log_history.append({"step": step})
                control = SimpleNamespace(should_save=False, should_training_stop=False)
                for c in self.callbacks:
                    c.on_step_end(None, self.state, control)
                if control.should_save:
                    self._save_checkpoint(self.model)
                    for c in self.callbacks:
                        c.on_save(None, self.state, control)
                assert not control.should_training_stop
            for c in self.callbacks:
                c.on_train_end(None, self.state, control)
        def save_model(self, path):
            pass
    monkeypatch.setattr(trl, "GRPOTrainer", Trainer)
    configuration = trl.GRPOConfig
    monkeypatch.setattr(trl, "GRPOConfig", lambda **kwargs: configuration(use_cpu=True, bf16=False, **kwargs))
    monkeypatch.setattr(checkpointing, "atomic_transformers_save",
                        lambda trainer, native, model, trial, extra: native(model, trial))
    attacker = GRPOTextEvasionTrainer(TextEvasionConfig(output_dir=str(tmp_path), learning_rate=5e-5,
        lr_scheduler_type="constant_with_warmup", warmup_steps=50))
    attacker.model = torch.nn.Linear(1, 1)
    attacker.tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    attacker.prompts, attacker.references = ["fit"], ["reference"]
    attacker.eval_prompts, attacker.eval_references = ["held out"], ["other reference"]
    attacker._stop_request = SimpleNamespace(requested=False)
    if resume_step is not None:
        checkpointing.atomic_save({"config": asdict(attacker.config), "prompts": attacker.prompts,
            "references": attacker.references, "eval_prompts": attacker.eval_prompts,
            "eval_references": attacker.eval_references, "max_steps": 300, "completed": False},
            tmp_path / "restart_context.pt")
        native = tmp_path / f"checkpoint-{resume_step}"
        native.mkdir()
        checkpointing.atomic_save({"step": resume_step, "buffered_inputs": None}, native / "restart_complete.pt")
    def diagnostic(step):
        assert events[-1] in [("save", step), ("load", step)]
        events.append(("log", step))
    attacker.log_training_diagnostic = diagnostic
    attacker._train_with_trl(max_steps=300)
    expected = [("log", 50), ("log", 300), ("log", 300)]
    if resume_step in (None, 0):
        expected.insert(0, ("log", 0))
    assert [event for event in events if event[0] == "log"] == expected
    arguments = json.loads((tmp_path / "trl_training_arguments.json").read_text())
    assert arguments["lr_scheduler_type"] == "constant_with_warmup"
    assert arguments["warmup_steps"] == 50 and arguments["learning_rate"] == 5e-5
    assert arguments["max_steps"] == 300
    assert attacker.training_history[-1]["step"] == 300
    assert checkpointing.load_checkpoint(tmp_path / "restart_context.pt")["completed"]

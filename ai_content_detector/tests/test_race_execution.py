import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, rng_state
from ai_content_detector.rl_evasion.config import TextEvasionConfig
from ai_content_detector.rl_evasion.defender_comparison import generation, evaluation


class Tokens(dict):
    def to(self, device):
        return self


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        assert self.padding_side == "left"
        values = [[int(text) + 1] * (int(text) % 3 + 1) for text in texts]
        width = max(map(len, values))
        return Tokens(input_ids=torch.tensor([[0] * (width - len(row)) + row for row in values]),
                      attention_mask=torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in values]))

    def batch_decode(self, values, **kwargs):
        return [str(row.tolist()) for row in values]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        assert torch.equal(input_ids != 0, attention_mask.bool())
        assert not self.training
        self.calls.append(len(input_ids))
        return torch.cat([input_ids, torch.randint(1, 100, (len(input_ids), 3))], dim=1)


def attacker():
    return SimpleNamespace(model=Model(), tokenizer=Tokenizer(), config=TextEvasionConfig())


def test_batched_generation_restarts_without_replaying_batches_and_restores_caller_state(tmp_path, monkeypatch):
    original = attacker()
    prompts = [str(i) for i in range(7)]
    before = rng_state()
    expected = generation.generate_batched(original, prompts, tmp_path / "full", "fixture", batch_size=3)
    assert original.model.calls == [3, 3, 1]
    assert original.model.training and original.tokenizer.padding_side == "right"
    assert torch.equal(before["torch"], rng_state()["torch"])
    save = generation.atomic_save

    def interrupt(state, path):
        save(state, path)
        if len(state.get("generated", [])) == 3:
            raise CheckpointPause("fixture interruption")

    monkeypatch.setattr(generation, "atomic_save", interrupt)
    interrupted = attacker()
    with pytest.raises(CheckpointPause):
        generation.generate_batched(interrupted, prompts, tmp_path / "resume", "fixture", batch_size=3)
    assert interrupted.model.training and interrupted.tokenizer.padding_side == "right"
    assert torch.equal(before["torch"], rng_state()["torch"])
    monkeypatch.setattr(generation, "atomic_save", save)
    resumed = attacker()
    assert generation.generate_batched(resumed, prompts, tmp_path / "resume", "fixture", batch_size=3) == expected
    assert resumed.model.calls == [3, 1]
    resumed.model.calls.clear()
    assert generation.generate_batched(resumed, prompts, tmp_path / "resume", "fixture", batch_size=3) == expected
    assert resumed.model.calls == []
    with torch.no_grad():
        resumed.model.weight.add_(1)
    with pytest.raises(ValueError, match="configuration differs"):
        generation.generate_batched(resumed, prompts, tmp_path / "resume", "fixture", batch_size=3)


def test_semantic_batches_resume_and_reuse_identical_post_attack_texts(tmp_path, monkeypatch):
    calls = []
    encoder = SimpleNamespace(_model_name="fixture", batch=lambda sources, texts:
                              calls.append(texts) or [len(text) / 10 for text in texts])
    pairs = [{"reference": "source", "ai": text} for text in ["a", "bb", "", "cccc", "eeeee"]]
    expected = [0.1, 0.2, 0.4, 0.5]
    save = evaluation.atomic_save

    def interrupt(state, path):
        save(state, path)
        if len(state["values"]) == 2:
            raise CheckpointPause("fixture interruption")

    monkeypatch.setattr(evaluation, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        evaluation.semantic_values(encoder, pairs, tmp_path, batch_size=2)
    monkeypatch.setattr(evaluation, "atomic_save", save)
    calls.clear()
    assert evaluation.semantic_values(encoder, pairs, tmp_path, batch_size=2) == expected
    assert calls == [["cccc"], ["eeeee"]]
    calls.clear()
    assert evaluation.semantic_values(encoder, pairs, tmp_path, batch_size=2) == expected
    assert calls == []


@pytest.mark.parametrize("attacker_only", [False, True])
@pytest.mark.parametrize("cosine", [False, True])
def test_race_pilot_continues_without_retraining_or_repeating_before_evaluation(tmp_path, monkeypatch, attacker_only, cosine):
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    base = {"evaluation": [{"prompt": "held out"}],
            "warmup": [{"prompt": str(i), "reference": "human " * 24} for i in range(6)]}
    (tmp_path / "base_pairs.json").write_text(json.dumps(base))
    warm = tmp_path / "warm"
    warm.mkdir()
    torch.save({"round": 0}, warm / "completed.pt")
    trained, evaluated, defended = [], [], []
    budget = 750 if cosine else 300
    schedule = {"learning_rate": 5e-5, "lr_scheduler_type": "warmup_stable_decay" if cosine else "constant_with_warmup",
                "warmup_steps": 50}
    if cosine:
        schedule["lr_scheduler_kwargs"] = {"num_decay_steps": 100, "min_lr_ratio": 0.2}

    def setup(plan, group, directory, names):
        actor = attacker()
        actor.training_history = []

        def train(max_steps):
            assert max_steps == budget
            assert actor.config.learning_rate == 5e-5
            assert actor.config.lr_scheduler_type == schedule["lr_scheduler_type"]
            assert actor.config.lr_scheduler_kwargs == schedule.get("lr_scheduler_kwargs", {})
            assert actor.config.warmup_steps == 50
            assert actor.diagnostic_samples == 100
            index = int(Path(actor.config.output_dir).parent.name.removeprefix("round_"))
            trained.append(index)
            steps = max_steps
            with torch.no_grad():
                actor.model.weight.add_(steps / 1000)
            actor.training_history = [{"step": steps, "reward": 0.5}]

        actor.train = train
        defender = SimpleNamespace(_round=0)

        def retrain(ai, human):
            defender._round += 1
            defended.append(defender._round)
            directory = defender.output_dir / f"round_{defender._round}"
            directory.mkdir(parents=True)
            torch.save({"round": defender._round}, directory / "completed.pt")

        defender.retrain = retrain
        pool = SimpleNamespace(checkpoints={"roberta": str(warm)}, specifications={"roberta": {}},
                               get=lambda name: defender, release=lambda name: None, clear=lambda: None)
        return actor, pool

    def measure(actor, pool, directory, context, **kwargs):
        evaluated.append(context)
        return {"records": [{"detector": "roberta", "attack_success_rate": float(actor.model.weight / 2)}]}

    monkeypatch.setattr(workflow, "attacker_setup", setup)
    monkeypatch.setattr(workflow, "evaluate_attacker", measure)
    monkeypatch.setattr(workflow, "generation", lambda actor, prompts, *args, **kwargs: ["ai " * 24] * len(prompts))
    monkeypatch.setattr(workflow, "matched_pairs", lambda prompts, refs, outputs, tokenizer:
                        [{"ai": text, "human": ref, "eligible": True} for text, ref in zip(outputs, refs)])
    plan = {"execution": {"generation_batch_size": 8, "reuse_before_evaluation": True,
                           "batch_semantic_evaluation": True, "diagnostic_samples": 100,
                           "attacker_training": schedule}}
    options = dict(directory=tmp_path / "race", defender_names=["roberta"], rounds=10,
                   attacker_steps=budget, reuse_standalone=False)
    if attacker_only:
        first = workflow.race(plan, "roberta", pause_after_attacker_round=1, **options)
        assert first["attacker_only_complete"] and first["history"] == []
        assert trained == [1] and len(evaluated) == 2
        assert not list((tmp_path / "race").glob("round_*/defenders"))
        repeated = workflow.race(plan, "roberta", pause_after_attacker_round=1, **options)
        assert repeated["attacker_only_complete"] and trained == [1] and len(evaluated) == 2
        first = workflow.race(plan, "roberta", pause_after_round=1, **options)
        assert trained == [1]
    else:
        first = workflow.race(plan, "roberta", pause_after_round=1, **options)
    assert first["pilot_complete"] and len(first["history"]) == 1
    resumed = workflow.race(plan, "roberta", pause_after_round=3, **options)
    assert trained == [1, 2, 3]
    assert defended == [1, 2, 3]
    assert len(evaluated) == 7
    assert resumed["pilot_complete"] and len(resumed["history"]) == 3
    assert [row["after_attack"]["attacker_updates"] for row in resumed["history"]] == [budget, 2 * budget, 3 * budget]
    assert resumed["history"][1]["before"] == resumed["history"][0]["after_defense"]
    assert (tmp_path / "race/round_2/before/reuse.json").is_file()

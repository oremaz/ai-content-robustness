from types import MethodType, SimpleNamespace

import pytest
import torch

from ai_content_detector.rl_evasion.arms_race import equilibrium
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
from ai_content_detector.rl_evasion.config import ArmsRaceConfig, TextEvasionConfig, seed_everything


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.tensor(0.1))
        self.generations = 0

    def generate(self, input_ids, **kwargs):
        self.generations += 1
        return torch.cat([input_ids, torch.randint(1, 100, (1, 3))], dim=1)


def make_race(path):
    race = equilibrium.ArmsRaceExperiment(ArmsRaceConfig(
        output_dir=str(path), num_rounds=3, attacker_steps_per_round=2,
        defender_retrain_samples=4, eval_prompts=2,
    ))
    actor = TinyModel()
    defender = TinyModel()
    calls = []

    def train(max_steps):
        calls.append(race.attacker.config.output_dir)
        with torch.no_grad():
            actor.policy.add_(torch.rand(()) * 0.05)

    def retrain(ai_texts, human_texts, warmup=False):
        if not warmup:
            race.defender._round += 1
        with torch.no_grad():
            defender.policy.add_(0.02)
        return {"round": race.defender._round, "accuracy": 0.5, "auroc": 0.5}

    def score(text):
        return 0.2 if text.startswith("human") else float(defender.policy + float(text.split()[0]))

    race.attacker = SimpleNamespace(
        model=actor, config=TextEvasionConfig(), train=train,
        reward_fn=SimpleNamespace(detector_rewards=[]), calls=calls,
    )
    race.defender = SimpleNamespace(
        model=defender, _round=0, retrain=retrain, predict=score,
        predict_batch=lambda texts: [score(text) for text in texts],
    )
    race.train_prompts, race.train_references = ["train0", "train1"], ["human0", "human1"]
    race.eval_prompts, race.eval_references = ["eval0", "eval1"], ["human2", "human3"]

    def generate(self, prompts, seed):
        generator = torch.Generator().manual_seed(seed)
        return [f"{float(actor.policy) + float(torch.rand((), generator=generator)) * 0.1} {p}"
                for p in prompts]

    race._generate = MethodType(generate, race)
    race._restore_restart()
    if race._restart["phase"] == "warmup":
        race._warmup_defender()
        race._restart["phase"] = "before_attack"
        race._save_restart()
    return race


def test_race_resume_skips_completed_attacker_and_preserves_paired_history(tmp_path, monkeypatch):
    seed_everything(42)
    baseline = make_race(tmp_path / "baseline")
    baseline.run()
    seed_everything(42)
    interrupted = make_race(tmp_path / "resumed")
    save = equilibrium.atomic_save

    def interrupt(state, path):
        save(state, path)
        if state.get("round") == 1 and state.get("phase") == "defender":
            raise CheckpointPause("simulated phase interruption")

    monkeypatch.setattr(equilibrium, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.run()
    assert len(interrupted.attacker.calls) == 2
    monkeypatch.setattr(equilibrium, "atomic_save", save)
    resumed = make_race(tmp_path / "resumed")
    resumed.run()
    assert len(resumed.attacker.calls) == 1
    assert resumed.attacker.calls[0].endswith("round_3")
    assert resumed.history == baseline.history
    assert resumed.defender._round == 3
    assert torch.equal(resumed.attacker.model.policy, baseline.attacker.model.policy)
    assert torch.equal(resumed.defender.model.policy, baseline.defender.model.policy)


def test_generation_restart_preserves_sampling_stream(tmp_path, monkeypatch):
    class Tokens(dict):
        def to(self, device):
            return self

    tokenizer = SimpleNamespace(
        __call__=lambda *args, **kwargs: Tokens(input_ids=torch.tensor([[1, 2]])),
        decode=lambda tokens, **kwargs: str(tokens.tolist()),
    )

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return tokenizer.__call__(*args, **kwargs)

        decode = staticmethod(tokenizer.decode)

    def factory(path):
        race = equilibrium.ArmsRaceExperiment(ArmsRaceConfig(output_dir=str(path)))
        race.attacker = SimpleNamespace(model=TinyModel(), tokenizer=Tokenizer(),
                                       config=TextEvasionConfig())
        return race

    prompts = [f"p{i}" for i in range(5)]
    baseline = factory(tmp_path / "baseline")
    before = torch.get_rng_state().clone()
    expected = baseline._generate(prompts, seed=42)
    assert torch.equal(before, torch.get_rng_state())
    interrupted = factory(tmp_path / "resumed")
    save = equilibrium.atomic_save

    def interrupt(state, path):
        save(state, path)
        if len(state.get("texts", [])) == 2:
            raise CheckpointPause("simulated generation interruption")

    monkeypatch.setattr(equilibrium, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted._generate(prompts, seed=42)
    monkeypatch.setattr(equilibrium, "atomic_save", save)
    resumed = factory(tmp_path / "resumed")
    assert resumed._generate(prompts, seed=42) == expected
    assert resumed.attacker.model.generations == 3

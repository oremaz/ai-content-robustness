import pytest
import torch

from ai_content_detector.rl_evasion.arms_race import radar_defender
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
from ai_content_detector.rl_evasion.config import seed_everything


def make_defender(path):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            inputs = [[1, 3 if text.startswith("ai") else 4, int(text[-1]) + 5, 2] for text in texts]
            return {"input_ids": inputs, "attention_mask": [[1] * 4 for _ in inputs]}

    defender = radar_defender.AdaptiveClassifierDefender(
        output_dir=str(path), retrain_samples=16, epochs=3, seed=42,
    )
    defender.model = RobertaForSequenceClassification(RobertaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
        num_labels=2, hidden_dropout_prob=0.2,
    ))
    defender.tokenizer = Tokenizer()
    return defender


def test_defender_native_resume_preserves_best_model_and_validation(tmp_path, monkeypatch):
    texts = [f"ai{i}" for i in range(8)]
    references = [f"human{i}" for i in range(8)]
    seed_everything(42)
    baseline = make_defender(tmp_path / "baseline")
    expected = baseline.retrain(texts, references)
    seed_everything(42)
    interrupted = make_defender(tmp_path / "resumed")
    save = radar_defender.atomic_transformers_save

    def interrupt(trainer, method, model, trial=None, extra_state=None):
        save(trainer, method, model, trial, extra_state)
        if trainer.state.global_step == 2:
            raise CheckpointPause("simulated epoch checkpoint interruption")

    monkeypatch.setattr(radar_defender, "atomic_transformers_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.retrain(texts, references)
    monkeypatch.setattr(radar_defender, "atomic_transformers_save", save)
    resumed = make_defender(tmp_path / "resumed")
    actual = resumed.retrain(texts, references)
    for name, weight in baseline.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], weight), name
    for key in ("accuracy", "auroc", "eval_loss", "train_samples", "val_samples"):
        assert actual[key] == expected[key]
    replay = make_defender(tmp_path / "resumed")
    assert replay.retrain(texts, references) == actual
    for name, weight in resumed.model.state_dict().items():
        assert torch.equal(replay.model.state_dict()[name], weight), name

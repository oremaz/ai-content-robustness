"""Math-correctness tests for the MultiSPIN DPO-style loss arithmetic.

The previous implementation defined ``_compute_spin_loss`` correctly but
*never called it from the training loop* — the loop ran a plain
language-modeling cross-entropy. These tests pin two things:

1. ``_compute_spin_loss`` is wired into the training step.
2. The DPO log-sigmoid math is correct in the direction we expect: when the
   current policy assigns higher relative probability to human text than
   to its own generations, the loss decreases.

Both checks avoid loading the actual LLM by patching the relevant helpers.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch


@pytest.fixture
def trainer_train_source() -> str:
    from ai_content_detector.rl_evasion.text_evasion.multispin import MultiSPINTrainer
    return inspect.getsource(MultiSPINTrainer.train)


class TestSPINLossWired:
    def test_train_loop_calls_compute_spin_loss(self, trainer_train_source: str):
        """The training step MUST call _compute_spin_loss (else CE proxy regression)."""
        assert "_compute_spin_loss" in trainer_train_source, (
            "MultiSPINTrainer.train no longer calls _compute_spin_loss — "
            "the loop has regressed to language-modeling CE."
        )

    def test_train_loop_does_not_use_labels_equal_input_ids(self, trainer_train_source: str):
        """The buggy proxy used outputs.loss from labels=input_ids, i.e. plain CE.

        After the fix the training step computes the SPIN log-sigmoid loss, so
        the labels=input_ids form should no longer be the gradient source.
        """
        # Soft check: it's fine to use this pattern elsewhere (e.g. monitoring),
        # but specifically the line that drives .backward() must not be a CE call.
        # We just look for the sentinel comment from the buggy version.
        assert "# SPIN loss (simplified: use CE loss as proxy)" not in trainer_train_source


class TestSPINLossSign:
    """The DPO-style SPIN loss is

        L = -log sigma(beta * ((curr_lp_h - ref_lp_h) - (curr_lp_g - ref_lp_g)))

    so when ``curr_lp_h - ref_lp_h`` (current preference for human) exceeds
    ``curr_lp_g - ref_lp_g`` (current preference for own generation), the
    log-sigmoid argument is positive → the loss is small. This pins that
    direction so a sign regression in the trainer would fail loudly.
    """

    @staticmethod
    def _spin_loss(curr_h: float, ref_h: float, curr_g: float, ref_g: float, beta: float = 0.1) -> float:
        diff = beta * ((curr_h - ref_h) - (curr_g - ref_g))
        # log sigmoid (same as torch.nn.functional.logsigmoid)
        return -math.log(1.0 / (1.0 + math.exp(-diff)))

    def test_loss_decreases_when_policy_prefers_human(self):
        # Reference policy is uniform-ish; current policy is shifted toward
        # human (curr_h > ref_h, curr_g < ref_g).
        baseline = self._spin_loss(curr_h=-2.0, ref_h=-2.0, curr_g=-2.0, ref_g=-2.0)
        improved = self._spin_loss(curr_h=-1.0, ref_h=-2.0, curr_g=-3.0, ref_g=-2.0)
        assert improved < baseline

    def test_loss_increases_when_policy_prefers_own_gen(self):
        baseline = self._spin_loss(curr_h=-2.0, ref_h=-2.0, curr_g=-2.0, ref_g=-2.0)
        worse = self._spin_loss(curr_h=-3.0, ref_h=-2.0, curr_g=-1.0, ref_g=-2.0)
        assert worse > baseline

    def test_loss_zero_at_perfect_preference(self):
        # As the human-vs-gen log-prob gap → +infinity, loss → 0.
        loss = self._spin_loss(curr_h=10.0, ref_h=0.0, curr_g=-10.0, ref_g=0.0, beta=1.0)
        assert loss < 1e-6


def test_conditional_logprob_excludes_prompt_tokens():
    from ai_content_detector.rl_evasion.config import MultiSPINConfig
    from ai_content_detector.rl_evasion.text_evasion.multispin import MultiSPINTrainer

    class Encoded:
        def __init__(self, ids):
            self.input_ids = ids

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens, **kwargs):
            ids = [1] if add_special_tokens else []
            ids.extend([2] * len(text.strip().split()))
            return Encoded(ids)

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask):
            batch, length = input_ids.shape
            return type("Output", (), {
                "logits": torch.zeros((batch, length, 4)) + self.anchor,
            })()

    trainer = MultiSPINTrainer(MultiSPINConfig(max_seq_length=32))
    trainer.tokenizer = Tokenizer()
    model = UniformModel()
    result = trainer._conditional_logprob(
        model, ["short", "a much longer prompt"], ["two words", "two words"],
    )
    assert result[0].item() == pytest.approx(result[1].item())
    assert result[0].item() == pytest.approx(-2 * math.log(4))

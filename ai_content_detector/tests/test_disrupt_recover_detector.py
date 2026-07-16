"""CPU-only guards for the D&R-style text detector hook."""

from __future__ import annotations

import pytest

from ai_content_detector.detectors import DisruptRecoverDetector


LONG_TEXT = (
    "Machine-generated passages often maintain unusually regular structure "
    "across sentences, even when the topic changes. This makes a local recovery "
    "test useful: after disrupting several words, a black-box model can sometimes "
    "reconstruct the original with suspiciously high fidelity. Human prose is "
    "usually less concentrated because there are many plausible ways to say the "
    "same thing."
)


def test_shuffle_disruption_is_deterministic():
    det = DisruptRecoverDetector(recover_fn=lambda corrupted: corrupted)

    first = det._disrupt(LONG_TEXT)
    second = det._disrupt(LONG_TEXT)

    assert first == second
    assert "[MASK]" not in first
    assert first != LONG_TEXT


def test_default_shuffle_disruption_preserves_tokens_without_masks():
    det = DisruptRecoverDetector(recover_fn=lambda corrupted: corrupted)

    corrupted = det._disrupt(LONG_TEXT)

    assert corrupted != LONG_TEXT
    assert "[MASK]" not in corrupted
    assert sorted(det._word_tokens(corrupted.lower())) == sorted(det._word_tokens(LONG_TEXT.lower()))


def test_exact_recovery_scores_high():
    det = DisruptRecoverDetector(
        recover_fn=lambda corrupted: LONG_TEXT,
        semantic_scorer=lambda source, recovered: 1.0,
        min_words=5,
    )

    result = det.detect(LONG_TEXT)

    assert result.score > 0.9
    assert result.label == "ai"
    assert result.details["recovery_similarity"] == 1.0
    assert result.details["structural_rank_similarity"] == pytest.approx(1.0)


def test_poor_recovery_scores_low():
    det = DisruptRecoverDetector(
        recover_fn=lambda corrupted: "short unrelated answer with little overlap",
        semantic_scorer=lambda source, recovered: 0.0,
        min_words=5,
    )

    result = det.detect(LONG_TEXT)

    assert result.score < 0.2
    assert result.label == "human"
    assert result.details["recovery_similarity"] < 0.4


def test_short_text_is_uncertain_without_recovery_call():
    called = False

    def recover(_: str) -> str:
        nonlocal called
        called = True
        return "should not be used"

    det = DisruptRecoverDetector(
        recover_fn=recover, semantic_scorer=lambda source, recovered: 1.0, min_words=10,
    )

    result = det.detect("Too short.")

    assert result.score == 0.5
    assert result.label == "uncertain"
    assert called is False

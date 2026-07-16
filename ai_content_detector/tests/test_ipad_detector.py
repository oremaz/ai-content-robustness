from __future__ import annotations

import math

import pytest

from ai_content_detector.detectors.text_detectors import IPADDetector


def test_binary_probability_renormalizes_yes_and_no(monkeypatch):
    detector = IPADDetector(regenerator=lambda prompt: "regenerated")

    def log_probability(adapter, prompt, answer):
        return math.log(0.2 if answer == "yes" else 0.3)

    monkeypatch.setattr(detector, "_answer_log_probability", log_probability)
    assert detector._yes_probability("ptcv", "prompt") == pytest.approx(0.4)


def test_adapter_roles_fusion_and_paper_threshold(monkeypatch):
    detector = IPADDetector(
        regenerator=lambda prompt: "regenerated text",
        fusion_weight=0.45,
        threshold=0.54,
    )
    detector._available = True
    calls = []

    monkeypatch.setattr(detector, "_load", lambda: None)
    monkeypatch.setattr(detector, "_generate", lambda adapter, prompt, max_new_tokens: "predicted prompt")

    def probability(adapter, prompt):
        calls.append(adapter)
        return {"ptcv": 0.8, "rc": 0.6}[adapter]

    monkeypatch.setattr(detector, "_yes_probability", probability)
    result = detector.detect("A sufficiently long text to exercise the complete IPAD workflow.")

    assert calls == ["ptcv", "rc"]
    assert result.score == pytest.approx(0.45 * 0.8 + 0.55 * 0.6)
    assert result.label == "ai"
    assert result.details["regeneration_backend"] == "callable"


def test_ipad_uses_binary_paper_decision_without_uncertain_band(monkeypatch):
    detector = IPADDetector(regenerator=lambda prompt: "regenerated", threshold=0.54)
    detector._available = True
    monkeypatch.setattr(detector, "_load", lambda: None)
    monkeypatch.setattr(detector, "_generate", lambda *args, **kwargs: "predicted")
    monkeypatch.setattr(detector, "_yes_probability", lambda *args, **kwargs: 0.54)

    result = detector.detect("A sufficiently long text to exercise the complete IPAD workflow.")
    assert result.score == pytest.approx(0.54)
    assert result.label == "human"

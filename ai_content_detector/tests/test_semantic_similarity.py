"""Pair alignment and embedding metric checks."""

import numpy as np
import pytest
import torch

from ai_content_detector.watermarking.semantic_similarity import (
    aligned_rows, cosine_pairs, last_token_pool, paired_scores, summarize, text_key,
)


def test_pair_alignment_uses_prompt_ids_and_rejects_missing_or_duplicate_rows():
    original = [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]
    candidate = [{"id": "b", "text": "other second"}, {"id": "a", "text": "other first"}]
    assert [x["id"] for x in aligned_rows(original, candidate)] == ["a", "b"]
    with pytest.raises(ValueError, match="differ"):
        aligned_rows(original, candidate[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        aligned_rows(original, candidate+candidate[:1])


def test_last_token_pool_handles_left_and_right_padding():
    hidden = torch.arange(24).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    torch.testing.assert_close(last_token_pool(hidden, mask), torch.stack([hidden[0, 3], hidden[1, 1]]))
    with pytest.raises(ValueError, match="mask"):
        last_token_pool(hidden, torch.zeros_like(mask))


def test_cosine_is_scale_invariant_and_not_probability():
    first = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
    second = np.asarray([[2, 0], [0, -2], [1, -1]], dtype=float)
    np.testing.assert_allclose(cosine_pairs(first, second), [1, -1, 0])
    with pytest.raises(ValueError, match="nonzero"):
        cosine_pairs(first, np.zeros_like(first))


def test_pair_scores_and_bootstrap_preserve_observation_units():
    original = [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]
    candidate = [{"id": "b", "text": "third"}, {"id": "a", "text": "first"}]
    vectors = {text_key("first"): [1, 0], text_key("second"): [0, 1], text_key("third"): [1, 0]}
    rows = paired_scores(original, candidate, vectors)
    assert [r["cosine"] for r in rows] == [1, 0]
    result = summarize([r["cosine"] for r in rows])
    assert result == summarize([1, 0])
    assert result["mean"] == .5
    assert result["mean_ci95"][0] <= .5 <= result["mean_ci95"][1]


def test_empty_output_is_reported_separately():
    rows = [{"id": "a", "text": "original"}]
    result = paired_scores(rows, [{"id": "a", "text": ""}],
                           {text_key("original"): [1, 0], text_key(""): [0, 1]})
    assert result[0]["empty"] is True

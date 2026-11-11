import torch

from ai_content_detector.watermarking.text_attacks import (
    TopKResponse, constrained_api_choice, source_ngram_completions)


def test_source_map_and_blocking_preserve_all_alternative_completions():
    table = source_ngram_completions([1, 2, 3, 4, 1, 2, 3, 5])
    assert table[(1, 2, 3)] == {4, 5}
    response = TopKResponse(torch.tensor([[4, 9]]), torch.tensor([[.9, .1]]), 20)
    choice, weight, fallback = constrained_api_choice(response, {4, 5})
    assert choice.tolist() == [9] and weight is None and not fallback.any()


def test_smoothing_blocks_both_supports_but_preserves_original_confidence_weight():
    n = 12000
    target = TopKResponse(torch.tensor([[4, 9]]).expand(n, -1), torch.tensor([[.2, .3]]).expand(n, -1), 20)
    reference = TopKResponse(torch.tensor([[4, 10]]).expand(n, -1), torch.tensor([[.3, .2]]).expand(n, -1), 20)
    choice, weight, fallback = constrained_api_choice(target, {4}, reference, 0., 1.,
                                                     generator=torch.Generator().manual_seed(412))
    expected_weight = .2**2 + .3**2 + .5**2 / 18
    assert torch.allclose(weight, torch.full_like(weight, expected_weight))
    expected_p9 = .6 * expected_weight / (.6 * expected_weight + .4 * (1-expected_weight))
    assert abs(float((choice == 9).float().mean()) - expected_p9) < .015
    assert not (choice == 4).any() and not fallback.any()


def test_fully_blocked_response_reports_fallback_instead_of_hiding_it():
    response = TopKResponse(torch.tensor([[7]]), torch.tensor([[1.]]), 20)
    choice, _, fallback = constrained_api_choice(response, {7})
    assert choice.tolist() == [7] and fallback.tolist() == [True]

import importlib.util
from pathlib import Path

import pytest
import torch

from ai_content_detector.watermarking.text_attacks import (
    TopKResponse, confidence, target_weight, resample, smooth, suppression_from_logits, degenerated, SuppressTokens,
)


def response(ids, probabilities, vocabulary=4):
    return TopKResponse(torch.tensor([ids]), torch.tensor([probabilities]), vocabulary)


def test_confidence_uses_unrenormalized_topk_and_uniform_tail():
    r = response([0, 1], [.6, .2])
    assert confidence(r).item() == pytest.approx(.6**2+.2**2+.2**2/2)
    assert confidence(response([0, 1], [.6, .4], 2)).item() == pytest.approx(.52)
    assert target_weight(r, .2, .6, 2).item() == pytest.approx(((.42-.2)/.4)**2)
    assert target_weight(r, .5, .6).item() == 0
    assert target_weight(r, .1, .2).item() == 1
    with pytest.raises(ValueError):
        target_weight(r, .3, .3)


def test_mixture_adds_overlap_mass_and_respects_endpoints():
    target = response([1, 2], [.8, .1])
    reference = response([2, 3], [.2, .8])
    assert resample(response([2], [.01])).item() == 2
    for _ in range(20):
        value, weight = smooth(target, reference, .9, 1.)
        assert value.item() in {2, 3} and weight.item() == 0
        value, weight = smooth(target, reference, 0., .1)
        assert value.item() in {1, 2} and weight.item() == 1
    with pytest.raises(ValueError):
        smooth(target, response([0], [1.], 3), 0., 1.)
    count = 20000
    a = TopKResponse(target.ids.expand(count, -1), target.probabilities.expand(count, -1), 4)
    b = TopKResponse(reference.ids.expand(count, -1), reference.probabilities.expand(count, -1), 4)
    values, weights = smooth(a, b, 0., 1., generator=torch.Generator().manual_seed(7))
    expected_overlap = weights[0].item() * (.1/.9) + (1-weights[0].item()) * .2
    assert (values == 2).float().mean().item() == pytest.approx(expected_overlap, abs=.015)


def test_suppression_shifts_next_token_labels_and_includes_quantile_ties():
    ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.tensor([[[0., 10., 0., 0.], [0., 10., -10., 0.], [0., 0., 0., 10.], [0., 0., 10., 0.]]])
    assert suppression_from_logits(logits, ids, .9).tolist() == [2]
    uniform = torch.zeros(1, 4, 4)
    assert suppression_from_logits(uniform, ids, .5).tolist() == [1, 2, 3]
    processor = SuppressTokens([torch.tensor([2])], [-4.])
    scores = processor(ids, torch.zeros(1, 4))
    assert scores.tolist() == [[0., 0., -4., 0.]]
    assert not degenerated([])
    assert not degenerated(list(range(450)))
    assert degenerated([1, 2]*225)


def script():
    path = Path(__file__).resolve().parents[2]/"scripts/textseal_api_attacks.py"
    spec = importlib.util.spec_from_file_location("textseal_api_attacks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_rejects_signature_and_prompt_changes(tmp_path):
    module = script()
    path = tmp_path/"rows.json"
    rows = [{"id": "a"}, {"id": "b"}]
    module.write_json(path, {"signature": "original", "rows": rows[:1]})
    assert module.cached_rows(path, rows, "original") == rows[:1]
    with pytest.raises(ValueError):
        module.cached_rows(path, rows, "changed")
    with pytest.raises(ValueError):
        module.cached_rows(path, rows[::-1], "original")


def test_probability_oracle_topk_preserves_tail_mass():
    module = script()
    logits = torch.log(torch.tensor([[.6, .2, .15, .05]]))
    r = module.exposed(logits, {"temperature": 1., "top_p": .9, "k": 2})
    assert r.ids.tolist() == [[0, 1]]
    assert r.probabilities.sum().item() == pytest.approx(.8/.95)
    assert r.probabilities.sum() < 1


def test_only_time_warning_permits_automatic_requeue():
    import signal
    module = script()
    stop = module.DevStop()
    stop._request(signal.SIGTERM, None)
    assert stop.requested and stop.signum == signal.SIGTERM
    with pytest.raises(module.CheckpointPause):
        module.pause(stop)


def test_rewriter_recipe_excludes_owner_and_target_information():
    module = script()
    config = {"directory": "run", "rewrite_tokens": 1500, "rewrite_temperature": .7, "percentile": .5,
              "owner": {"key": 123}, "target_probabilities": [.1, .9]}
    assert set(module.public_recipe(config)) == {"directory", "rewrite_tokens", "rewrite_temperature", "percentile"}


def test_rewrite_resume_keeps_completed_retry_without_regeneration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = script()
    original = tmp_path/"original"
    original.mkdir()
    rows = [{"id": "a", "prompt": "p", "text": "source a"}, {"id": "b", "prompt": "p", "text": "source b"}]
    module.write_json(original/"watermarked.json", rows)
    module.write_json(tmp_path/"beta.json", {"beta": -4})
    module.write_json(tmp_path/"vanilla_paraphrase.json", {"signature": "fixed", "rows": rows})
    config = {"directory": str(tmp_path), "original": str(original), "rewriter": "public",
              "protocol_sha256": "fixed", "batch": 2, "seed": 7, "beta_step": .125,
              "max_restarts": 97, "rewrite_tokens": 1500, "rewrite_temperature": .7, "percentile": .5}
    monkeypatch.setattr(module, "load_model", lambda _: (None, None))
    monkeypatch.setattr(module, "suppression_sets", lambda model, tokenizer, texts, config: [[] for _ in texts])
    stop = SimpleNamespace(requested=False)
    calls = []
    def generate(model, tokenizer, texts, sets, betas, recipe, seed):
        calls.append(list(texts))
        if len(calls) == 1:
            return [{"text": "repeated", "degenerated": True}, {"text": "good b", "degenerated": False}]
        stop.requested = True
        return [{"text": "good a", "degenerated": False}]
    monkeypatch.setattr(module, "rewrite_once", generate)
    with pytest.raises(module.CheckpointPause):
        module.rewrite(config, stop)
    stop.requested = False
    module.rewrite(config, stop)
    saved = module.read(tmp_path/"bira.json")["rows"]
    assert len(calls) == 2
    assert [r["text"] for r in saved] == ["good a", "good b"]
    assert saved[0]["attempts"] == 2 and saved[0]["beta"] == -3.875
    assert not (tmp_path/"bira_pending.json").exists()

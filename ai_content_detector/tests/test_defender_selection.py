import pytest

from scripts.defender_selection import select_notebook_defenders


def fixture():
    configurations = {"qlora_modernbert_base": {"hf_model": "answerdotai/ModernBERT-base"},
                      "qlora_modernbert_large": {"hf_model": "answerdotai/ModernBERT-large"},
                      "qlora_qwen": {"hf_model": "Qwen/Qwen3.5-4B-Base"},
                      "qlora_gemma": {"hf_model": "google/gemma-4-E4B"},
                      "qlora_other": {"hf_model": "other/decoder"}}
    names = ["tfidf_logreg", "tfidf_random_forest", "tfidf_xgboost", "tfidf_lightgbm", *configurations]
    scores = [0.65, 0.66, 0.67, 0.7, 0.79, 0.81, 0.99, 0.83, 0.8]
    payload = {"tables": {"MAGE test": [{"model": name, "f1_macro": score}
                                          for name, score in zip(names, scores)]}}
    return payload, configurations


def test_selects_one_per_architecture_and_excludes_qwen_even_when_it_wins():
    payload, configurations = fixture()
    selection = select_notebook_defenders(payload, configurations)
    assert selection["selected"] == {"tfidf": "tfidf_lightgbm", "encoder": "qlora_modernbert_large",
                                      "decoder": "qlora_gemma"}
    assert selection["selected_scores"]["decoder"] == 0.83
    assert selection["selection_is_not_independent_evaluation"] is True


def test_incomplete_ranking_cannot_select_a_best_decoder():
    payload, configurations = fixture()
    payload["tables"]["MAGE test"].pop()
    with pytest.raises(ValueError, match="Complete MAGE evaluation"):
        select_notebook_defenders(payload, configurations)


def test_invalid_score_does_not_silently_drop_a_candidate():
    payload, configurations = fixture()
    payload["tables"]["MAGE test"][-1]["f1_macro"] = float("nan")
    with pytest.raises(ValueError, match="finite MAGE scores"):
        select_notebook_defenders(payload, configurations)

"""The requested seven-defender comparison and existing training budgets."""

STAGES = ["comparison_baseline", "comparison_warmup", "comparison_rl_encoder",
          "comparison_rl_decoder", "comparison_rl_ensemble", "comparison_race_encoder",
          "comparison_race_decoder", "comparison_race_ensemble"]
GROUPS = {"encoder": ["modernbert_large"], "decoder": ["decoder_classifier"], "ensemble": None}


def specifications(selection, root):
    return {
        "binoculars": {"family": "binoculars", "adaptation": "performer_lora_frozen_observer"},
        "fast_detect_gpt": {"family": "fast_detect_gpt", "adaptation": "scorer_lora_frozen_reference"},
        "diveye": {"family": "diveye", "feature_cache": str(root / "diveye_features"),
                   "adaptation": "continue_authors_xgboost_frozen_extractors"},
        "roberta": {"family": "roberta", "model_name": "openai-community/roberta-base-openai-detector"},
        "tfidf": {"family": "tfidf", "model_name": selection["selected"]["tfidf"],
                  "checkpoint_id": selection["checkpoints"]["tfidf"]["checkpoint_id"]},
        "modernbert_large": {"family": "notebook_hf", "model_name": selection["selected"]["encoder"],
                             "checkpoint_id": selection["checkpoints"]["encoder"]["checkpoint_id"]},
        "decoder_classifier": {"family": "notebook_hf", "model_name": selection["selected"]["decoder"],
                               "checkpoint_id": selection["checkpoints"]["decoder"]["checkpoint_id"]},
    }

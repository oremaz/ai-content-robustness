"""Architecture-specific defender updates with a common AI-score interface."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

from ..arms_race.radar_defender import AdaptiveClassifierDefender
from ..checkpointing import (CheckpointPause, StopRequest, atomic_save, load_checkpoint,
                             model_state, restore_model, restore_rng, rng_state)


class DefenderTrainer(Trainer):
    """Keep native optimizer recovery while saving only adapted detector weights."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        ignored = list(ignore_keys if ignore_keys is not None else
                       getattr(getattr(model, "config", None), "keys_to_ignore_at_inference", []) or [])
        ignored += ["hidden_states", "attentions", "past_key_values"]
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignored)

    def _save(self, output_dir=None, state_dict=None):
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_save(model_state(self.model), directory / "defender_trainable.pt")
        torch.save(self.args, directory / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        restore_model(model or self.model, load_checkpoint(Path(resume_from_checkpoint) / "defender_trainable.pt"))

    def _load_best_model(self):
        self._load_from_checkpoint(self.state.best_model_checkpoint)


class ScoringClassifier(nn.Module):
    """Supervised adaptation of a paper statistic; the second LM stays fixed."""

    def __init__(self, family, scorer, reference, threshold=0.0):
        super().__init__()
        self.family = family
        self.scorer = scorer
        self.reference = reference.eval().requires_grad_(False)
        self.head = nn.Linear(1, 1, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.fill_(-20.0 if family == "binoculars" else 1.0)
            self.head.bias.fill_(20.0 * threshold if family == "binoculars" else -threshold)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        from ...detectors.text_detectors import FastDetectGPTDetector
        import torch.nn.functional as functional

        mask = (torch.ones_like(input_ids) if attention_mask is None else attention_mask)[:, 1:]
        with torch.no_grad():
            reference_logits = self.reference(input_ids=input_ids, attention_mask=attention_mask,
                                              use_cache=False).logits[:, :-1].float()
        scoring_logits = self.scorer(input_ids=input_ids, attention_mask=attention_mask,
                                     use_cache=False).logits[:, :-1].float()
        if self.family == "binoculars":
            # The observer is frozen; the performer distribution is adapted.
            observer_log_probs = reference_logits.log_softmax(-1)
            observed = -observer_log_probs.gather(2, input_ids[:, 1:, None]).squeeze(-1)
            cross = -(scoring_logits.softmax(-1) * observer_log_probs).sum(-1)
            weights = mask.to(cross.dtype)
            statistic = (observed * weights).sum(-1) / (cross * weights).sum(-1).clamp_min(1e-6)
        else:
            statistic = FastDetectGPTDetector._criterion_from_logits(
                scoring_logits, reference_logits, input_ids[:, 1:], mask,
            )
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            ai_logits = self.head(statistic.float()[:, None]).squeeze(-1)
        logits = torch.stack([torch.zeros_like(ai_logits), ai_logits], dim=-1)
        loss = functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class TorchDefender(AdaptiveClassifierDefender):
    def __init__(self, specification, output_dir, device="cuda", seed=42):
        self.specification = specification
        self.device = torch.device(device)
        super().__init__(model_name=json.dumps(specification, sort_keys=True), output_dir=str(output_dir),
                         seed=seed, trainer_class=DefenderTrainer, paired_validation=True,
                         mixed_precision=self.device.type == "cuda")

    def setup(self):
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from ...detectors.text_detectors import ModernBERTDetector

        family = self.specification["family"]
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        torch.manual_seed(self.seed)
        if family == "notebook_hf":
            from ml_pipeline.utils.utils import get_checkpoint_id
            if get_checkpoint_id("bench_aitextdetect", self.specification["model_name"]) != self.specification["checkpoint_id"]:
                raise ValueError("Selected notebook checkpoint identity changed")
            detector = ModernBERTDetector(model_name=self.specification["model_name"], device=str(self.device))
            detector._load()
            if not detector._available:
                raise RuntimeError("Notebook classifier checkpoint did not load")
            self.model, self.tokenizer = detector._model.model, detector._model.tokenizer
            self.model.set_adapter(self.model.active_adapter)
            self.ai_label = 0
        elif family == "roberta":
            name = "openai-community/roberta-base-openai-detector"
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).to(self.device)
            matches = [int(key) for key, label in self.model.config.id2label.items()
                       if label.lower() in {"fake", "ai", "machine", "generated"}]
            if len(matches) != 1:
                raise ValueError("RoBERTa AI class is not identified by its published configuration")
            self.ai_label = matches[0]
        else:
            scorer_name = "tiiuae/falcon-7b-instruct" if family == "binoculars" else "tiiuae/falcon-7b"
            reference_name = "tiiuae/falcon-7b"
            self.tokenizer = AutoTokenizer.from_pretrained(reference_name)
            reference = AutoModelForCausalLM.from_pretrained(reference_name, torch_dtype=dtype).to(self.device)
            # Separate objects are essential when the two initial checkpoints coincide.
            scorer = AutoModelForCausalLM.from_pretrained(scorer_name, torch_dtype=dtype).to(self.device)
            scorer = get_peft_model(scorer, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM",
            ))
            scorer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model = ScoringClassifier(family, scorer, reference,
                                           threshold=0.9015 if family == "binoculars" else 0.0).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.model.eval()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def move(self, device):
        self.model.to(device)

    def predict_batch(self, texts):
        self.model.eval()
        results = []
        device = next(self.model.parameters()).device
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            for start in range(0, len(texts), 8):
                tokens = self.tokenizer(texts[start:start + 8], return_tensors="pt", padding=True,
                                        truncation=True, max_length=self.max_length).to(device)
                logits = self.model(**tokens).logits
                results.extend(logits.float().softmax(-1)[:, self.ai_label].cpu().tolist())
        return results

    def predict(self, text):
        return self.predict_batch([text])[0]


def atomic_joblib(value, path):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FeatureDefender:
    def __init__(self, specification, output_dir, device="cuda", seed=42):
        self.specification = specification
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed
        self._round = 0
        self.classifier = self.extractor = None
        self.ai_label = 1 if specification["family"] == "diveye" else 0

    def setup(self):
        from ...detectors.text_detectors import DivEyeDetector, TFIDFDetector
        if self.specification["family"] == "diveye":
            self.extractor = DivEyeDetector(device=self.device)
            self.extractor._load_classifier()
            self.classifier = self.extractor._classifier
        else:
            from ml_pipeline.utils.utils import get_checkpoint_id
            if get_checkpoint_id("bench_aitextdetect", self.specification["model_name"]) != self.specification["checkpoint_id"]:
                raise ValueError("Selected TF-IDF checkpoint identity changed")
            detector = TFIDFDetector(model_name=self.specification["model_name"])
            detector._load()
            if not detector._available:
                raise RuntimeError("TF-IDF classifier/vectorizer did not load")
            self.classifier = getattr(detector._model, "model", detector._model)
            self.extractor = detector._vectorizer

    def move(self, device):
        if self.specification["family"] == "diveye":
            self.extractor._device_spec = str(device)
            self.extractor._device = torch.device(device)
            for name in ("_model", "_bi_model"):
                model = getattr(self.extractor, name)
                if model is not None:
                    model.to(device)

    def features(self, texts):
        if self.specification["family"] != "diveye":
            return self.extractor.transform(texts).toarray().astype(np.float32)
        rows = []
        directory = Path(self.specification.get("feature_cache", self.output_dir.parent / "diveye_features"))
        with StopRequest() as stop:
            for text in texts:
                if stop.requested:
                    raise CheckpointPause("DivEye feature extraction checkpoint retained")
                key = hashlib.sha256((self.extractor.SPACE_REVISION + text).encode()).hexdigest()
                path = directory / f"{key}.pt"
                row = load_checkpoint(path)
                if row is None:
                    row = self.extractor.extract_features(text)
                    atomic_save(row, path)
                rows.append(row)
        return np.asarray(rows)

    def predict_batch(self, texts):
        columns = list(self.classifier.classes_)
        return self.classifier.predict_proba(self.features(texts))[:, columns.index(self.ai_label)].tolist()

    def retrain(self, ai_texts, human_texts, *, warmup=False):
        import joblib
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        if not warmup:
            self._round += 1
        destination = self.output_dir / ("warmup" if warmup else f"round_{self._round}")
        count = min(1000, len(ai_texts), len(human_texts))
        if count < 5:
            raise ValueError("At least five aligned pairs are required for feature-detector warm-up")
        texts = ai_texts[:count] + human_texts[:count]
        labels = np.array([self.ai_label] * count + [1 - self.ai_label] * count)
        signature = {"specification": self.specification, "texts": texts, "labels": labels.tolist(),
                     "seed": self.seed + self._round, "feature_update": "continue_existing_estimator"}
        complete = load_checkpoint(destination / "completed.pt", signature)
        if complete is not None:
            self.classifier = joblib.load(destination / "classifier.pkl")
            return complete["metrics"]
        train_pairs, validation_pairs = train_test_split(np.arange(count), test_size=0.2,
                                                         random_state=self.seed + self._round)
        train = np.concatenate([train_pairs, train_pairs + count])
        validation = np.concatenate([validation_pairs, validation_pairs + count])
        features = self.features(texts)
        classifier = self.classifier
        family = type(classifier).__name__
        if family == "XGBClassifier":
            if self.specification["family"] == "diveye":
                classifier.set_params(n_estimators=200, max_depth=12, colsample_bytree=0.8,
                                      subsample=0.7, min_child_weight=5, gamma=1, random_state=self.seed)
            classifier.fit(features[train], labels[train], xgb_model=classifier.get_booster())
        elif family == "LGBMClassifier":
            classifier.fit(features[train], labels[train], init_model=classifier.booster_)
        elif family == "RandomForestClassifier":
            classifier.set_params(warm_start=True, n_estimators=classifier.n_estimators + 100)
            classifier.fit(features[train], labels[train])
        elif family == "LogisticRegression":
            classifier.set_params(warm_start=True)
            classifier.fit(features[train], labels[train])
        else:
            raise ValueError(f"Unsupported adaptive feature classifier: {family}")
        scores = classifier.predict_proba(features[validation])[:, list(classifier.classes_).index(self.ai_label)]
        truth = labels[validation] == self.ai_label
        metrics = {"accuracy": accuracy_score(truth, scores >= 0.5), "auroc": roc_auc_score(truth, scores),
                   "train_samples": len(train), "val_samples": len(validation), "round": self._round}
        atomic_joblib(classifier, destination / "classifier.pkl")
        atomic_save({"config": signature, "metrics": metrics}, destination / "completed.pt")
        return metrics


class DefenderPool:
    """Keep frozen defenders on CPU between batched queries on the single GPU."""

    def __init__(self, specifications, root, checkpoints=None, device="cuda", retain_on_cpu=True):
        self.specifications = specifications
        self.root = Path(root)
        self.checkpoints = checkpoints or {}
        self.device = device
        self.retain_on_cpu = retain_on_cpu
        self.keep_on_device = False
        self.defenders = {}

    def get(self, name):
        if name not in self.defenders:
            specification = self.specifications[name]
            cls = FeatureDefender if specification["family"] in {"diveye", "tfidf"} else TorchDefender
            defender = cls(specification, self.root / name, device=self.device)
            state = rng_state()
            try:
                defender.setup()
                if name in self.checkpoints:
                    path = Path(self.checkpoints[name])
                    if isinstance(defender, TorchDefender):
                        restore_model(defender.model, load_checkpoint(path / "completed.pt")["model"])
                    else:
                        import joblib
                        defender.classifier = joblib.load(path / "classifier.pkl")
            finally:
                restore_rng(state)
            self.defenders[name] = defender
        defender = self.defenders[name]
        defender.move(self.device)
        return defender

    def release(self, name):
        if name in self.defenders:
            self.defenders[name].move("cpu")
            if not self.retain_on_cpu:
                del self.defenders[name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def predict_batch(self, texts, names=None):
        result = {}
        for name in names or self.specifications:
            complete = False
            try:
                scores = np.asarray(self.get(name).predict_batch(texts), dtype=np.float64)
                if scores.shape != (len(texts),) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                    raise ValueError(f"Invalid scores from {name}")
                result[name] = scores.tolist()
                complete = True
            finally:
                if not self.keep_on_device or not complete:
                    self.release(name)
        return result

    def clear(self):
        self.defenders.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
